"""Проверка nf_fp4 на видеокарте (под, venv сервера, после apply_fp4qsa.py). Нужно ~1 ГБ памяти карты.

1) раскладка: reference_dequant (чистый torch) == библиотечный NVFP4KVQuantizeUtil.dequantize (flashinfer);
2) gather_rows (Triton) == reference на случайных слотах;
3) compact_kv (Triton) == отбор по req_to_token/topk + reference, в т.ч. невалидные позиции не пишутся;
4) ошибка округления fp4 против fp8 на тех же данных (для сравнения с пробой качества).
usage: /tmp/sgl-intel/venv/bin/python test_fp4qsa_gpu.py
"""
import torch

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nf_fp4  # из этой папки: установленный сервер не трогаем
from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

torch.manual_seed(0)
dev = "cuda"
M, H, D = 4096, 2, 256          # слотов в «пуле», голов KV, размер головы (как у Flash-Next: 2 x 256)
x = (torch.randn(M, H, D, device=dev) * torch.rand(M, H, 1, device=dev) * 4).to(torch.bfloat16)
gs = torch.ones(1, dtype=torch.float32, device=dev)

q, sf, _ = NVFP4KVQuantizeUtil.quantize(x, gs)
q = q.view(torch.uint8).contiguous()
sf_u8 = sf.view(torch.uint8).contiguous()
print("форма данных", tuple(q.shape), "масштабов", tuple(sf_u8.shape))

lib = NVFP4KVQuantizeUtil.dequantize(q, sf, gs, dtype=torch.bfloat16).float()   # библиотека отдаёт только bf16/fp16
ref = nf_fp4.reference_dequant(q, sf_u8, gs, D)
d1 = (lib - ref.to(torch.bfloat16).float()).abs().max().item()
print(f"1) torch-справка против flashinfer: макс. разница {d1:.3g}")
assert d1 == 0.0, "раскладка/порядок полубайтов не совпали с библиотекой"

raw = nf_fp4.FP4Raw(q, sf_u8, gs, D)
idx = torch.randint(0, M, (3000,), device=dev)
g = nf_fp4.gather_rows(raw, idx)
d2 = (g.float() - ref[idx].to(g.dtype).float()).abs().max().item()
print(f"2) gather_rows: {tuple(g.shape)} {g.dtype}, макс. разница {d2:.3g}")
assert d2 == 0.0

# 3) compact: 3 запроса, у каждого своя таблица слотов, topk 100 с невалидными позициями
B, topk, L = 3, 100, 700
req_to_token = torch.randint(0, M, (8, 1024), dtype=torch.int32, device=dev)
req_indices = torch.tensor([5, 1, 7], dtype=torch.int64, device=dev)
seq_lens = torch.tensor([700, 50, 300], dtype=torch.int32, device=dev)
indices = torch.randint(0, L, (B, topk), dtype=torch.int32, device=dev)
indices[0, 90:] = -1
from sglang.srt.layers.attention.qsa.sparse_attn import qwen_sparse_fa2_cu_seqlens_triton
counts = torch.empty(B, dtype=torch.int32, device=dev)
cu = torch.empty(B + 1, dtype=torch.int32, device=dev)
qwen_sparse_fa2_cu_seqlens_triton(seq_lens, indices, counts, cu, B, topk)
cap = B * topk
SENT = 7.0
out_k = torch.full((cap, H, D), SENT, dtype=raw.dtype, device=dev)
out_v = torch.full((cap, H, D), SENT, dtype=raw.dtype, device=dev)
nf_fp4.compact_kv(raw, raw, req_to_token, req_indices, indices, seq_lens, cu, out_k, out_v, B, topk)
# та же упаковка, но на обычном bf16-«кэше» из уже распакованных значений — эталон отбора
dense = ref.to(torch.bfloat16)
ek = torch.full_like(out_k, SENT)
ev = torch.full_like(out_v, SENT)
nf_fp4.compact_kv(dense, dense, req_to_token, req_indices, indices, seq_lens, cu, ek, ev, B, topk)
d3 = max((out_k.float() - ek.float()).abs().max().item(), (out_v.float() - ev.float()).abs().max().item())
print(f"3) compact_kv: валидных {counts.tolist()}, макс. разница с обычным путём {d3:.3g}")
assert d3 == 0.0

# 4) ошибка округления
e4 = ((ref - x.float()).norm() / x.float().norm()).item()
e8 = ((x.to(torch.float8_e4m3fn).float() - x.float()).norm() / x.float().norm()).item()
print(f"4) относительная ошибка: fp4 {e4:.4f}, fp8 {e8:.4f}")
print("ВСЁ СОВПАЛО")

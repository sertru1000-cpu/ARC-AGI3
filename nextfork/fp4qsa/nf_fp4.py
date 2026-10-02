"""nextfork: кэш KV в nvfp4 для разреженного внимания Qwen (QSA) — распаковка только отобранных токенов.

Ставится в sglang/srt/layers/attention/qsa/nf_fp4.py патчем apply_fp4qsa.py. Сам по себе ничего не включает:
QSA зовёт kv_buffers / compact_kv / gather_rows, и при обычном кэше (bf16 / fp8) они отдают прежнее поведение.

Формат nvfp4 в пуле (NVFP4KVCacheMethod): данные [m, H, D/2] uint8, младший полубайт — чётный элемент;
масштаб блока из 16 элементов — fp8 e4m3, [m, H, D/16] (или плоско [m, H*D/16] — порядок тот же);
глобальный масштаб слоя — quant_method.k_scales_gpu[layer_id]. Значение = e2m1 * масштаб_блока * глобальный.
"""
import os

import torch
import triton
import triton.language as tl

# в чём отдавать распакованные K/V ядрам внимания: bf16 (без второго округления) или fp8 (как обычный кэш fp8)
_OUT_DTYPE = torch.float8_e4m3fn if os.environ.get("NEXTFORK_FP4_OUT", "bf16") == "fp8" else torch.bfloat16


class FP4Raw:
    """Сырой слой кэша nvfp4, выдающий себя за тензор [m, H, D] нужного типа (shape/dtype/device для скретча)."""

    def __init__(self, q, sf, gscale, head_dim):
        self.q = q.view(torch.uint8)                     # [m, H, D/2]
        self.sf = sf.view(torch.uint8)                   # [m, H, D/16] или [m, H*D/16]
        self.gscale = gscale                             # тензор из 1 элемента, fp32, на устройстве
        self.shape = (q.shape[0], q.shape[1], head_dim)
        self.dtype = _OUT_DTYPE
        self.device = q.device
        assert self.q.is_contiguous() and self.sf.is_contiguous()
        assert self.sf.numel() == q.shape[0] * q.shape[1] * head_dim // 16, "неожиданная раскладка масштабов nvfp4"


def _base_pool(pool):
    return getattr(pool, "full_kv_pool", pool)


def is_nvfp4_pool(pool) -> bool:
    base = _base_pool(pool)
    qm = getattr(base, "quant_method", None)
    return getattr(base, "k_scale_buffer", None) is not None and getattr(qm, "name", None) == "nvfp4"


def kv_buffers(pool, layer):
    """(K, V) для QSA: обычные буферы, а при nvfp4 — FP4Raw вместо попытки .view(float4) всего слоя."""
    if not is_nvfp4_pool(pool):
        return pool.get_key_buffer(layer.layer_id), pool.get_value_buffer(layer.layer_id)
    k_q, v_q, k_sf, v_sf = pool.get_raw_kv_buffer(layer.layer_id)
    qm = _base_pool(pool).quant_method
    lid = layer.layer_id                                 # глобальный номер слоя: так индексирует k_scales_gpu
    head_dim = k_q.shape[2] * 2
    return (FP4Raw(k_q, k_sf, qm.k_scales_gpu[lid:lid + 1], head_dim),
            FP4Raw(v_q, v_sf, qm.v_scales_gpu[lid:lid + 1], head_dim))


@triton.jit
def _e2m1_x_scales(qbyte, d, sbyte, gs):
    """байт с двумя e2m1 (младший полубайт — чётный d), байт масштаба fp8 e4m3, глобальный масштаб -> fp32."""
    nib = tl.where((d % 2) == 1, (qbyte >> 4) & 0xF, qbyte & 0xF)
    e = (nib >> 1) & 0x3
    m = (nib & 0x1).to(tl.float32)
    mag = tl.where(e == 0, m * 0.5, (1.0 + 0.5 * m) * tl.exp2((e - 1).to(tl.float32)))
    val = tl.where((nib & 0x8) != 0, -mag, mag)
    sc = sbyte.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    return val * sc * gs


@triton.jit
def _compact_kv_nvfp4(
    kq, vq, ksf, vsf, kgs, vgs,
    req_to_token, req_indices, indices, seq_lens, cu_k,
    out_k, out_v,
    topk: tl.constexpr, heads: tl.constexpr, dim: tl.constexpr,
    req_stride: tl.constexpr, idx_stride: tl.constexpr,
    BLOCK_TOPK: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # та же логика отбора, что в qsa/sparse_attn.py::_compact_kv, плюс распаковка nvfp4
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    d = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(req_to_token + req * req_stride + tl.where(valid, positions, 0), mask=valid, other=0).to(tl.int64)
    mask = valid[:, None] & (d[None, :] < dim)
    src_q = slots[:, None] * (heads * dim // 2) + head * (dim // 2) + (d // 2)[None, :]
    src_s = slots[:, None] * (heads * dim // 16) + head * (dim // 16) + (d // 16)[None, :]
    dst = (pack_start + cols).to(tl.int64)[:, None] * heads * dim + head * dim + d[None, :]
    kgsv = tl.load(kgs)
    vgsv = tl.load(vgs)
    kk = _e2m1_x_scales(tl.load(kq + src_q, mask=mask, other=0).to(tl.int32), d[None, :],
                        tl.load(ksf + src_s, mask=mask, other=0), kgsv)
    tl.store(out_k + dst, kk.to(out_k.dtype.element_ty), mask=mask)
    vv = _e2m1_x_scales(tl.load(vq + src_q, mask=mask, other=0).to(tl.int32), d[None, :],
                        tl.load(vsf + src_s, mask=mask, other=0), vgsv)
    tl.store(out_v + dst, vv.to(out_v.dtype.element_ty), mask=mask)


@triton.jit
def _gather_nvfp4(
    q, sf, gs, rows, out, n_rows,
    heads: tl.constexpr, dim: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid, head = tl.program_id(0), tl.program_id(1)
    r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    d = tl.arange(0, BLOCK_D)
    ok = r < n_rows
    slots = tl.load(rows + r, mask=ok, other=0).to(tl.int64)
    mask = ok[:, None] & (d[None, :] < dim)
    src_q = slots[:, None] * (heads * dim // 2) + head * (dim // 2) + (d // 2)[None, :]
    src_s = slots[:, None] * (heads * dim // 16) + head * (dim // 16) + (d // 16)[None, :]
    x = _e2m1_x_scales(tl.load(q + src_q, mask=mask, other=0).to(tl.int32), d[None, :],
                       tl.load(sf + src_s, mask=mask, other=0), tl.load(gs))
    dst = r.to(tl.int64)[:, None] * heads * dim + head * dim + d[None, :]
    tl.store(out + dst, x.to(out.dtype.element_ty), mask=mask)


def compact_kv(k, v, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk):
    """Замена qwen_sparse_kv_extraction_compact_triton: при FP4Raw — отбор с распаковкой, иначе прежний путь."""
    if not isinstance(k, FP4Raw):
        from sglang.srt.layers.attention.qsa.sparse_attn import qwen_sparse_kv_extraction_compact_triton
        return qwen_sparse_kv_extraction_compact_triton(
            k, v, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk)
    _, heads, dim = k.shape
    assert out_k.dtype == k.dtype and out_k.is_contiguous() and out_v.is_contiguous()
    block_topk = 16
    _compact_kv_nvfp4[(batch, heads, triton.cdiv(topk, block_topk))](
        k.q, v.q, k.sf, v.sf, k.gscale, v.gscale,
        req_to_token, req_indices, indices, seq_lens, cu_k,
        out_k, out_v,
        topk, heads, dim, req_to_token.stride(0), indices.stride(0),
        BLOCK_TOPK=block_topk, BLOCK_D=triton.next_power_of_2(dim), num_warps=8,
    )


def gather_rows(buf, index):
    """Замена buf.index_select(0, index) в чанковом префилле QSA."""
    if not isinstance(buf, FP4Raw):
        return buf.index_select(0, index)
    _, heads, dim = buf.shape
    n = index.numel()
    out = torch.empty((n, heads, dim), dtype=buf.dtype, device=buf.device)
    if n:
        block_r = 32
        _gather_nvfp4[(triton.cdiv(n, block_r), heads)](
            buf.q, buf.sf, buf.gscale, index.contiguous(), out, n,
            heads, dim, BLOCK_R=block_r, BLOCK_D=triton.next_power_of_2(dim), num_warps=4,
        )
    return out


# ---- справка для проверки (не на горячем пути) ----
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def reference_dequant(q_rows, sf_rows, gscale, head_dim, dtype=torch.float32):
    """Чистый torch: q_rows [n, H, D/2] uint8, sf_rows [n, H*D/16] или [n, H, D/16] (байты fp8) -> [n, H, D]."""
    n, h, half = q_rows.shape
    lo, hi = q_rows & 0x0F, (q_rows >> 4) & 0x0F
    nib = torch.stack([lo, hi], dim=-1).reshape(n, h, half * 2)
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=q_rows.device)
    vals = lut[nib.long()]
    sc = sf_rows.reshape(n, h, head_dim // 16).view(torch.float8_e4m3fn).float()
    out = vals.view(n, h, head_dim // 16, 16) * sc.unsqueeze(-1) * gscale.float().reshape(())
    return out.view(n, h, head_dim).to(dtype)

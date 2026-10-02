"""nextfork fp4qsa: кэш KV nvfp4 для Qwen3.8-Flash-Next (разреженное внимание QSA) на стеке Pennyroyal / SGLang.

Что правит (в установленном пакете sglang; повторный запуск ничего не меняет):
  1. qsa/nf_fp4.py — новый файл: распаковка nvfp4 только отобранных токенов (Triton);
  2. qwen_sparse_attn_backend.py — три чтения кэша идут через nf_fp4 (при bf16/fp8 поведение прежнее);
  3. server_args.py — при NEXTFORK_FP4_QSA=1 снимается проверка «KV4 только с triton/trtllm_mha/...»
     (у этой модели полное внимание всегда идёт через QSA, метка механизма на него не влияет);
  4. memory_pool.py — масштаб записи берётся тензором с карты (иначе падает захват CUDA-графа);
  5. fp4_kv_cache_quant_method.py — при NEXTFORK_FP4_NOWS=1 не выделяется общий fp8-буфер распаковки
     размером со весь пул (+~15% токенов в пуле; QSA он не нужен).
Включение на сервере: --kv-cache-dtype nvfp4 и NEXTFORK_FP4_QSA=1 [NEXTFORK_FP4_NOWS=1] [NEXTFORK_FP4_OUT=fp8].

usage: python apply_fp4qsa.py <путь к пакету sglang>   (папка, где лежит srt/)
"""
import shutil
import sys
from pathlib import Path

MARK = "nextfork fp4qsa"


def edit(path: Path, pairs):
    s = path.read_text()
    if MARK in s:
        print(f"уже применено: {path.name}")
        return
    for old, new, count in pairs:
        n = s.count(old)
        assert n == count, f"{path.name}: ожидалось {count} вхождений, найдено {n}:\n{old}"
        s = s.replace(old, new)
    path.write_text(s)
    print(f"применено: {path.name}")


def main(root: Path):
    srt = root / "srt"
    att = srt / "layers" / "attention"
    shutil.copyfile(Path(__file__).with_name("nf_fp4.py"), att / "qsa" / "nf_fp4.py")
    print("скопировано: qsa/nf_fp4.py")

    edit(att / "qwen_sparse_attn_backend.py", [
        ("from sglang.srt.layers.attention.qsa.stall_diagnostics import QSAStallDiagnostics\n",
         "from sglang.srt.layers.attention.qsa.stall_diagnostics import QSAStallDiagnostics\n"
         f"from sglang.srt.layers.attention.qsa import nf_fp4 as _nf_fp4  # {MARK}\n", 1),
        ("        k_buffer = pool.get_key_buffer(layer.layer_id)\n"
         "        v_buffer = pool.get_value_buffer(layer.layer_id)\n",
         f"        k_buffer, v_buffer = _nf_fp4.kv_buffers(pool, layer)  # {MARK}\n", 2),
        ("qwen_sparse_kv_extraction_compact_triton(\n            k_buffer,",
         "_nf_fp4.compact_kv(\n            k_buffer,", 2),
        ("            k_buffer.index_select(0, gather_index),\n"
         "            v_buffer.index_select(0, gather_index),\n",
         "            _nf_fp4.gather_rows(k_buffer, gather_index),\n"
         "            _nf_fp4.gather_rows(v_buffer, gather_index),\n", 1),
    ])

    edit(srt / "server_args.py", [
        ('        if cfg.kv_cache_dtype not in ("nvfp4", "fp4_mx_block16"):\n            return\n\n'
         "        use_mla_backend = self.use_mla_backend()\n",
         '        if cfg.kv_cache_dtype not in ("nvfp4", "fp4_mx_block16"):\n            return\n'
         f'        if __import__("os").environ.get("NEXTFORK_FP4_QSA") == "1":  # {MARK}\n            return\n\n'
         "        use_mla_backend = self.use_mla_backend()\n", 1),
    ])

    # запись в кэш: QSA передаёт layer.k_scale числом -> torch.tensor(...) копирует CPU->GPU и роняет захват
    # CUDA-графа (под 02.10, s20fp4q). Берём тот же масштаб из k_scales_gpu (он и заполнен из layer.k_scale).
    edit(srt / "mem_cache" / "memory_pool.py", [
        ("    def _quantized_scales(self, global_layer_id: int, k_scale, v_scale):\n"
         '        if k_scale is None and hasattr(self.quant_method, "k_scales_gpu"):\n',
         "    def _quantized_scales(self, global_layer_id: int, k_scale, v_scale):\n"
         f"        # {MARK}: число вместо тензора на карте ломает захват CUDA-графа\n"
         '        if (k_scale is None or isinstance(k_scale, (int, float))) and hasattr(self.quant_method, "k_scales_gpu"):\n', 1),
    ])

    qm = srt / "layers" / "quantization" / "fp4_kv_cache_quant_method.py"
    edit(qm, [
        ("# Registry: explicit --kv-cache-dtype value -> method class.\n",
         f"# {MARK}: без общего fp8-буфера распаковки — QSA распаковывает отобранные токены сама\n"
         'if __import__("os").environ.get("NEXTFORK_FP4_NOWS") == "1":\n'
         "    KV_CACHE_ATTENTION_ACCESS_REGISTRY[NVFP4KVCacheMethod.name] = (\n"
         "        _native_fp4(_PREFILL, _ANY_BACKEND, _NVFP4_SCALE, _TORCH_FP4),\n"
         "        _native_fp4(_DECODE, _ANY_BACKEND, _NVFP4_SCALE, _TORCH_FP4),\n"
         "    )\n\n\n"
         "# Registry: explicit --kv-cache-dtype value -> method class.\n", 1),
    ])
    for p in att.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))

"""Побайтовая правка одного тензора внутри файла safetensors — без перезаписи заголовка (22.09).

Зачем. Второй путь доставки обученной модели: влить адаптер прямо в bf16-веса и подменить 4 файла из 206,
оставив 119 ГБ экспертов нетронутыми. Пересохранять файл целиком нельзя: порядок ключей и выравнивание в
заголовке могут измениться, размер поедет, и сверка обвязки («число файлов, суммарный размер») упадёт.
Поэтому правим ТОЛЬКО байты данных нужного тензора на месте: заголовок, смещения и размер файла сохраняются.

Устройство файла: 8 байт длины заголовка (little-endian) + JSON-заголовок + область данных. У каждого тензора
в заголовке есть dtype, shape и data_offsets [начало, конец] — смещения внутри области данных.

usage (как библиотека):
    from patch_safetensors import header, patch_tensor
    hdr, data_start = header(path)
    patch_tensor(path, "model...q_proj.weight", lambda t: t + noise)   # t -- torch.Tensor bf16
"""
import json, struct
from pathlib import Path

import torch

_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn}


def header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode())
    return hdr, 8 + n


def read_tensor(path: Path, name: str) -> torch.Tensor:
    hdr, base = header(path)
    row = hdr[name]
    dt = _DTYPES[row["dtype"]]
    a, b = row["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        raw = f.read(b - a)
    return torch.frombuffer(bytearray(raw), dtype=dt).reshape(row["shape"])


def patch_tensor(path: Path, name: str, fn) -> dict:
    """читает тензор, применяет fn, записывает обратно В ТЕ ЖЕ байты; возвращает отчёт"""
    hdr, base = header(path)
    row = hdr[name]
    dt = _DTYPES[row["dtype"]]
    a, b = row["data_offsets"]
    t = read_tensor(path, name)
    new = fn(t.clone()).to(dt).contiguous()
    if tuple(new.shape) != tuple(row["shape"]):
        raise RuntimeError("форма изменилась: %s против %s" % (tuple(new.shape), tuple(row["shape"])))
    raw = new.view(torch.uint8).numpy().tobytes() if new.dtype != torch.uint8 else new.numpy().tobytes()
    if len(raw) != b - a:
        raise RuntimeError("размер данных изменился: %d против %d" % (len(raw), b - a))
    with open(path, "r+b") as f:
        f.seek(base + a)
        f.write(raw)
    delta = float((new.float() - t.float()).abs().mean())
    return {"tensor": name, "shape": list(row["shape"]), "bytes": len(raw), "mean_abs_delta": delta}


if __name__ == "__main__":
    print(__doc__)

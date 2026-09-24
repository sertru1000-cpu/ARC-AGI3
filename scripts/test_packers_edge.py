"""Придирчивая проверка упаковщиков fp8 и 4 бит на крайних случаях (24.09).

Обычный тест гоняет «хорошие» нормальные веса. Здесь проверяется то, на чём упаковщики ломаются молча:
нулевые строки (деление на масштаб), выбросы, значения вне диапазона формата, несимметричные строки,
сохранение формы и обратимость упаковки битов. Молчаливая порча весов в одном слое из 48 не видна
ни по потере, ни по скорости — её ловить надо здесь.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from fp8_experts import _quantize as q8, _dequantize as dq8  # noqa: E402
from fp4_experts import quantize_fp4, dequantize_fp4, LEVELS  # noqa: E402

FAILS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print("  %-52s %s%s" % (name, "ок" if cond else "ПРОВАЛ", (" — " + detail) if detail else ""))
    if not cond:
        FAILS.append(name)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    d = b.abs().mean()
    return float((a - b).abs().mean() / d) if d > 0 else float((a - b).abs().mean())


def main() -> None:
    torch.manual_seed(0)
    print("=== fp8 ===")
    w = (torch.randn(2, 64, 256) * 0.02).to(torch.bfloat16)
    q, s = q8(w)
    r = dq8(q, s)
    check("форма сохранена", r.shape == w.shape, "%s" % (tuple(r.shape),))
    check("тип восстановления bf16", r.dtype == torch.bfloat16)
    check("ошибка на обычных весах < 4%%", rel(r.float(), w.float()) < 0.04,
          "%.2f%%" % (100 * rel(r.float(), w.float())))

    z = torch.zeros(1, 4, 32).to(torch.bfloat16)                     # нулевая строка: деление на 0
    rz = dq8(*q8(z))
    check("нулевые веса не дают NaN/inf", bool(torch.isfinite(rz).all()) and float(rz.abs().max()) == 0)

    big = (torch.randn(1, 8, 64) * 1e4).to(torch.bfloat16)           # значения много больше предела fp8 (448)
    rb = dq8(*q8(big))
    check("большие значения без переполнения", bool(torch.isfinite(rb).all()),
          "max %.1f -> %.1f" % (float(big.abs().max()), float(rb.abs().max())))
    check("большие значения: ошибка < 5%%", rel(rb.float(), big.float()) < 0.05,
          "%.2f%%" % (100 * rel(rb.float(), big.float())))

    tiny = (torch.randn(1, 8, 64) * 1e-6).to(torch.bfloat16)         # значения у нижней границы bf16
    rt = dq8(*q8(tiny))
    check("очень малые значения без NaN", bool(torch.isfinite(rt).all()),
          "ошибка %.2f%%" % (100 * rel(rt.float(), tiny.float())))

    out = torch.randn(1, 4, 128).to(torch.bfloat16) * 0.01
    out[0, 0, 0] = 5.0                                               # один выброс на строку
    ro = dq8(*q8(out))
    check("выброс не портит строку целиком", rel(ro[0, 1:].float(), out[0, 1:].float()) < 0.04,
          "остальные строки %.2f%%" % (100 * rel(ro[0, 1:].float(), out[0, 1:].float())))

    print("\n=== 4 бита ===")
    w4 = (torch.randn(2, 64, 256) * 0.02).to(torch.bfloat16)
    p, sc = quantize_fp4(w4)
    r4 = dequantize_fp4(p, sc)
    check("форма сохранена", r4.shape == w4.shape, "%s" % (tuple(r4.shape),))
    check("память 0.625 байта на вес",
          abs((p.numel() + sc.numel() * 2) / w4.numel() - 0.625) < 1e-6,
          "%.3f" % ((p.numel() + sc.numel() * 2) / w4.numel()))
    check("ошибка на обычных весах < 12%%", rel(r4.float(), w4.float()) < 0.12,
          "%.2f%%" % (100 * rel(r4.float(), w4.float())))

    exact = (LEVELS.repeat(4)[:32] * 0.5).reshape(1, 1, 32).to(torch.bfloat16)  # точно представимые
    re = dequantize_fp4(*quantize_fp4(exact))
    check("точно представимые значения восстановлены точно",
          float((re.float() - exact.float()).abs().max()) < 1e-3,
          "max откл %.5f" % float((re.float() - exact.float()).abs().max()))

    z4 = torch.zeros(1, 2, 32).to(torch.bfloat16)
    rz4 = dequantize_fp4(*quantize_fp4(z4))
    check("нулевые веса не дают NaN", bool(torch.isfinite(rz4).all()) and float(rz4.abs().max()) == 0)

    sgn = torch.tensor([[[-1.0, 1.0] * 16]]).to(torch.bfloat16)
    rs = dequantize_fp4(*quantize_fp4(sgn))
    check("знаки не перепутаны при упаковке по два",
          bool((rs.float().sign() == sgn.float().sign()).all()))

    try:
        quantize_fp4(torch.randn(1, 1, 30).to(torch.bfloat16))       # 30 не делится на 16
        check("некратная длина отвергается", False, "прошло молча")
    except AssertionError:
        check("некратная длина отвергается", True)

    print("\nИТОГ:", "ПРОЙДЕНО" if not FAILS else "ПРОВАЛЫ: %s" % ", ".join(FAILS))
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()

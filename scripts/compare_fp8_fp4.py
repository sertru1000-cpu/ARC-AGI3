"""Сравнение точности bf16 / fp8 / 4 бита (24.09) — чтобы выбирать карту по числам, а не по ощущению.

Две части, потому что одной мало:
 1. УМНОЖЕНИЕ С БОЕВЫМИ РАЗМЕРАМИ (2560 и 640). Ошибки отдельных весов случайны и гасятся при
    суммировании как корень из длины строки, поэтому на игрушечных размерностях результат выходит
    пессимистичным в несколько раз. Здесь размеры настоящие.
 2. ПОЛНЫЙ ПРОХОД крошечной модели той же архитектуры — проверяет, что накопление по слоям
    не ломает картину.

Запуск: .venv/bin/python scripts/compare_fp8_fp4.py
"""
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from fp8_experts import _quantize as q8, _dequantize as dq8, compress_experts  # noqa: E402
from fp4_experts import quantize_fp4, dequantize_fp4, compress_experts_fp4  # noqa: E402
from test_fp8_experts import tiny_config  # noqa: E402

from transformers import AutoModelForCausalLM  # noqa: E402


def part1() -> None:
    print("=== 1. умножение с боевыми размерами ===")
    torch.manual_seed(0)
    for out_dim, in_dim, label in ((1280, 2560, "gate_up_proj (1280x2560)"),
                                   (2560, 640, "down_proj (2560x640)")):
        # веса обученной сети близки к нормальным; масштаб не важен — ошибка относительная
        w = (torch.randn(out_dim, in_dim) * 0.02).to(torch.bfloat16)
        x = torch.randn(64, in_dim).to(torch.bfloat16)
        ref = nn.functional.linear(x.float(), w.float())

        q, s = q8(w.unsqueeze(0))
        w8 = dq8(q, s).squeeze(0)
        y8 = nn.functional.linear(x.float(), w8.float())

        p, sc = quantize_fp4(w)
        w4 = dequantize_fp4(p, sc)
        y4 = nn.functional.linear(x.float(), w4.float())

        def rel(a, b):
            return 100 * ((a - b).abs().mean() / b.abs().mean()).item()

        print("  %-26s веса: fp8 %.2f%%, 4 бита %.2f%%   |   выход: fp8 %.3f%%, 4 бита %.3f%%"
              % (label, rel(w8.float(), w.float()), rel(w4.float(), w.float()),
                 rel(y8, ref), rel(y4, ref)))
        print("      память на вес: bf16 2.000 б, fp8 %.3f б, 4 бита %.3f б"
              % ((q.numel() + s.numel() * 2) / w.numel(),
                 (p.numel() + sc.numel() * 2) / w.numel()))


def part2() -> None:
    print("\n=== 2. полный проход крошечной модели (размерности маленькие, оценка пессимистична) ===")
    torch.manual_seed(0)
    cfg = tiny_config()
    x = torch.randint(0, 100, (1, 24))
    y = x.clone(); y[:, :4] = -100

    def build():
        torch.manual_seed(0)
        return AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).eval()

    base = build()
    with torch.no_grad():
        ref = base(input_ids=x).logits.float()
    ref_loss = nn.functional.cross_entropy(ref[0, :-1], y[0, 1:], ignore_index=-100).item()
    print("  bf16: потеря %.5f" % ref_loss)

    for name, fn in (("fp8", compress_experts), ("4 бита", compress_experts_fp4)):
        m = build()
        info = fn(m, verbose=False)
        with torch.no_grad():
            got = m(input_ids=x).logits.float()
        loss = nn.functional.cross_entropy(got[0, :-1], y[0, 1:], ignore_index=-100).item()
        rel = 100 * ((got - ref).abs().mean() / ref.abs().mean()).item()
        print("  %-7s логиты %.3f%% | потеря %.5f (сдвиг %+.5f) | эксперты в %.2f раза меньше"
              % (name, rel, loss, loss - ref_loss, info["было_ГБ"] / info["стало_ГБ"]))


def part3() -> None:
    print("\n=== 3. что это значит для настоящей модели ===")
    experts_bf16 = 247.1
    for name, bytes_per_w in (("bf16", 2.0), ("fp8", 1.0 + 2 / 2560), ("4 бита", 0.5 + 2 / 16)):
        experts = experts_bf16 * bytes_per_w / 2.0
        total_ple_bf16 = experts + 102.5 + 9.6
        total_ple_fp8 = experts + 51.2 + 9.6
        print("  %-7s эксперты %6.1f ГБ | модель %6.1f ГБ | с таблицей PLE в fp8 %6.1f ГБ"
              % (name, experts, total_ple_bf16, total_ple_fp8))
    print("  карты: B300 288 ГБ | 2xH200 282 ГБ | 1xH200 141 ГБ")


if __name__ == "__main__":
    part1()
    part2()
    part3()

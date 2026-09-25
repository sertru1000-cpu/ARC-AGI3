"""Проверка умножения прямо в fp8 локально, без карты (25.09).

Проверяется:
 1. точность схемы на БОЕВЫХ размерах (2560 и 640) -- насколько выход отличается от честного bf16;
 2. выход модели и потеря на крошечной модели той же архитектуры;
 3. обратный проход: градиент доходит до обучаемого слоя внимания;
 4. что разжатия весов действительно НЕ происходит (веса остаются fp8).

Сам аппаратный вызов `torch._scaled_mm` есть только на Blackwell; здесь работает запасной путь с той же
арифметикой, поэтому проверяется ТОЧНОСТЬ И ЛОГИКА, а скорость меряется уже на карте.
"""
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from fp8_experts import _quantize, compress_experts  # noqa: E402
from fp8_matmul_experts import fp8_linear, quantize_rows, use_fp8_matmul  # noqa: E402
from test_fp8_experts import tiny_config  # noqa: E402

from transformers import AutoModelForCausalLM  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print("  %-50s %s%s" % (name, "ок" if cond else "ПРОВАЛ", (" — " + detail) if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> None:
    torch.manual_seed(0)
    print("=== 1. точность на боевых размерах ===")
    for out_dim, in_dim, label in ((1280, 2560, "gate_up (1280x2560)"), (2560, 640, "down (2560x640)")):
        w = (torch.randn(out_dim, in_dim) * 0.02).to(torch.bfloat16)
        x = (torch.randn(64, in_dim) * 0.5).to(torch.bfloat16)
        ref = (x.float() @ w.float().t())

        q, s = _quantize(w.unsqueeze(0))
        y = fp8_linear(x, q.squeeze(0), s.squeeze(0))
        rel = ((y.float() - ref).abs().mean() / ref.abs().mean()).item()

        # для сравнения: сколько даёт только сжатие весов (активации в bf16)
        w8 = (q.squeeze(0).float() * s.squeeze(0).float())
        ref_w8 = (x.float() @ w8.t())
        rel_w = ((ref_w8 - ref).abs().mean() / ref.abs().mean()).item()
        print("  %-22s веса fp8 %.2f%% | веса И активации fp8 %.2f%%" % (label, 100 * rel_w, 100 * rel))
        check("%s: ошибка ниже 6%%" % label, rel < 0.06, "%.2f%%" % (100 * rel))

    print("\n=== 2. модель целиком ===")
    def build():
        torch.manual_seed(0)
        return AutoModelForCausalLM.from_config(tiny_config()).to(torch.bfloat16).eval()

    x = torch.randint(0, 100, (1, 48))
    y = x.clone(); y[:, :4] = -100

    m1 = build(); compress_experts(m1, verbose=False)
    with torch.no_grad():
        ref = m1(input_ids=x).logits.float()
    ref_loss = nn.functional.cross_entropy(ref[0, :-1], y[0, 1:], ignore_index=-100).item()

    m2 = build(); compress_experts(m2, verbose=False)
    n = use_fp8_matmul(m2, verbose=False)
    with torch.no_grad():
        got = m2(input_ids=x).logits.float()
    loss = nn.functional.cross_entropy(got[0, :-1], y[0, 1:], ignore_index=-100).item()
    rel = ((got - ref).abs().mean() / ref.abs().mean()).item()
    print("  модулей переключено: %d" % n)
    print("  логиты: расхождение %.3f%% | потеря %.5f против %.5f" % (100 * rel, loss, ref_loss))
    check("расхождение логитов ниже 3%", rel < 0.03, "%.3f%%" % (100 * rel))
    check("потеря сдвинулась меньше чем на 0.05", abs(loss - ref_loss) < 0.05,
          "%.5f" % abs(loss - ref_loss))

    print("\n=== 3. обратный проход ===")
    for p in m2.parameters():
        p.requires_grad_(False)
    prm = dict(m2.named_parameters())
    name = next(k for k in prm if "q_proj.weight" in k)
    prm[name].requires_grad_(True)
    out = m2(input_ids=x).logits.float()
    nn.functional.cross_entropy(out[0, :-1], y[0, 1:], ignore_index=-100).backward()
    gn = prm[name].grad.norm().item()
    check("градиент доходит до внимания", gn > 0, "норма %.6f" % gn)

    print("\n=== 4. веса остались сжатыми ===")
    mod = next(mm for mm in m2.modules() if type(mm).__name__ == "Qwen4ExpTextExperts")
    check("gate_up хранится в fp8", mod.gate_up_q.dtype == torch.float8_e4m3fn, str(mod.gate_up_q.dtype))
    check("down хранится в fp8", mod.down_q.dtype == torch.float8_e4m3fn, str(mod.down_q.dtype))

    print("\nИТОГ:", "ПРОЙДЕНО" if not FAILS else "ПРОВАЛЫ: %s" % ", ".join(FAILS))
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()

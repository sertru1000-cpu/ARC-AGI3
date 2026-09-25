"""Проверка группового счёта экспертов локально, без карты (25.09).

Три вещи:
 1. выход СОВПАДАЕТ с обычным циклом (иначе мы просто испортим модель быстро);
 2. обратный проход жив и градиент доходит до обучаемого слоя;
 3. сколько запусков умножений экономится -- это и есть причина, по которой карта простаивала.

Запуск: .venv/bin/python scripts/test_grouped_experts.py
"""
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from fp8_experts import compress_experts  # noqa: E402
from fp8_experts_grouped import use_grouped  # noqa: E402
from test_fp8_experts import tiny_config  # noqa: E402

from transformers import AutoModelForCausalLM  # noqa: E402


def build():
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(tiny_config()).to(torch.bfloat16).eval()


def main() -> None:
    x = torch.randint(0, 100, (1, 48))
    y = x.clone(); y[:, :4] = -100

    m1 = build(); compress_experts(m1, verbose=False)
    with torch.no_grad():
        ref = m1(input_ids=x).logits.float()

    m2 = build(); compress_experts(m2, verbose=False)
    n = use_grouped(m2, verbose=False)
    with torch.no_grad():
        got = m2(input_ids=x).logits.float()

    rel = ((got - ref).abs().mean() / ref.abs().mean()).item()
    mx = (got - ref).abs().max().item()
    print("модулей переключено: %d" % n)
    print("расхождение с обычным циклом: среднее %.3e, максимум %.3e" % (rel, mx))

    # сколько запусков умножения экономится: цикл делает 2 на КАЖДОГО задетого эксперта, групповой -- 2 всего
    mod = next(mm for mm in m2.modules() if type(mm).__name__ == "Qwen4ExpTextExperts")
    n_exp = mod.num_experts
    layers = sum(1 for mm in m2.modules() if type(mm).__name__ == "Qwen4ExpTextExperts")
    print("в этой игрушке: экспертов %d, слоёв с экспертами %d" % (n_exp, layers))
    print("запусков умножения за прямой проход: цикл до %d, групповой %d"
          % (2 * n_exp * layers, 2 * layers))
    print("в настоящей модели (512 экспертов, 48 слоёв): цикл до %d, групповой %d"
          % (2 * 512 * 48, 2 * 48))

    # обратный проход
    for p in m2.parameters():
        p.requires_grad_(False)
    prm = dict(m2.named_parameters())
    name = next(k for k in prm if "q_proj.weight" in k)
    prm[name].requires_grad_(True)
    out = m2(input_ids=x).logits.float()
    loss = nn.functional.cross_entropy(out[0, :-1], y[0, 1:], ignore_index=-100)
    loss.backward()
    g = prm[name].grad
    print("обратный проход: норма градиента %.6f" % g.norm().item())

    ok = rel < 5e-3 and n > 0 and g.norm().item() > 0
    print("\nИТОГ:", "ПРОЙДЕНО" if ok else "НЕ ПРОЙДЕНО")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

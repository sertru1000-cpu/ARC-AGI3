"""Сжатие экспертов в fp8 (24.09) — чтобы модель влезала на карту ЦЕЛИКОМ, вместе с таблицей PLE.

Зачем. При обучении в bf16 модель занимает 360 ГБ: эксперты 247, PLE 102, остальное 11. В B300 (288 ГБ)
это не влезает, поэтому PLE держали в оперативной памяти — и карта простаивала на две трети, ожидая
процессор (измерено 24.09: загрузка карты 31–38%, скорость 70 токенов/с, эпоха выходила 19 часов).

Почему fp8 здесь законен, а не компромисс. В БОЕВОМ чекпойнте эксперты лежат в NVFP4, то есть в четырёх
битах. Обучая со сжатыми в восемь бит, мы не отходим от боевых условий, а приближаемся к ним: точность
выше боевой. Обучаемые модули (внимание) остаются bf16 — побитово те же, что в бою.

Почему не bitsandbytes. Он берёт только обычные линейные слои, а эксперты хранятся слитыми трёхмерными
тензорами (512, 1280, 2560) внутри Qwen4ExpTextExperts. Проверено 24.09 — отказывается.

Числа: эксперты 247 -> 123 ГБ, вся модель 184 ГБ. На B300 остаётся больше 100 ГБ на активации.

Масштаб хранится на каждую строку выходного измерения (не один на эксперта) — это втрое точнее при той же
памяти, поскольку масштабов всего 2I+H на эксперта против 6.5 млн весов.
"""
from __future__ import annotations

import torch
from torch import nn

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def _quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(E, out, in) bf16 -> fp8 + масштаб (E, out, 1). Масштаб на строку выходного измерения."""
    scale = w.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12) / FP8_MAX
    q = (w.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q, scale.to(torch.bfloat16)


def _dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.bfloat16) * scale


def _forward_fp8(self, hidden_states, top_k_index, top_k_weights):
    """Тот же цикл, что в Qwen4ExpTextExperts.forward, но срез эксперта разжимается на лету."""
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        w_gu = _dequantize(self.gate_up_q[expert_idx], self.gate_up_scale[expert_idx])
        gate, up = nn.functional.linear(current_state, w_gu).chunk(2, dim=-1)
        h = self.act_fn(gate) * up
        w_dn = _dequantize(self.down_q[expert_idx], self.down_scale[expert_idx])
        h = nn.functional.linear(h, w_dn)
        h = h * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, h.to(final_hidden_states.dtype))

    return final_hidden_states


def compress_experts(model: nn.Module, verbose: bool = True) -> dict:
    """Заменяет bf16-параметры экспертов на fp8-буферы. Возвращает сводку по памяти (в ГБ)."""
    targets = [m for m in model.modules() if type(m).__name__ == "Qwen4ExpTextExperts"]
    if not targets:
        raise SystemExit("модулей Qwen4ExpTextExperts не найдено — сжимать нечего, запуск отменён")

    before = after = 0
    for mod in targets:
        for src, qn, sn in (("gate_up_proj", "gate_up_q", "gate_up_scale"),
                            ("down_proj", "down_q", "down_scale")):
            w = getattr(mod, src).data
            before += w.numel() * w.element_size()
            q, s = _quantize(w)
            delattr(mod, src)                      # снимаем nn.Parameter целиком
            mod.register_buffer(qn, q, persistent=False)
            mod.register_buffer(sn, s, persistent=False)
            after += q.numel() * q.element_size() + s.numel() * s.element_size()
        mod.forward = _forward_fp8.__get__(mod, type(mod))

    out = {"модулей": len(targets), "было_ГБ": before / 1e9, "стало_ГБ": after / 1e9}
    if verbose:
        print("эксперты сжаты в fp8: %d модулей, %.1f -> %.1f ГБ (в %.1f раза)"
              % (out["модулей"], out["было_ГБ"], out["стало_ГБ"],
                 out["было_ГБ"] / max(out["стало_ГБ"], 1e-9)), flush=True)
    return out

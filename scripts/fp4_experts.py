"""Сжатие экспертов в четыре бита (24.09) — то же представление, что в боевом чекпойнте NVFP4.

Формат e2m1 с блочным масштабом, как в NVFP4: один бит знака, два бита порядка, один бит мантиссы.
Представимы только 0, 0.5, 1, 1.5, 2, 3, 4, 6 — умноженные на масштаб своего блока из 16 элементов.
Два кода упаковываются в один байт, поэтому вес стоит 0.5 байта плюс 0.125 байта на масштаб.

Зачем: эксперты 247 -> 77 ГБ (против 124 в fp8), вся модель 189 ГБ, а с таблицей PLE в fp8 — 138.
Это открывает одну H200 (141 ГБ) за $3.59/ч вместо редкой B300 за $7.89.

Важная оговорка, которую нельзя терять: боевой чекпойнт сжимала NVIDIA своим инструментом, и прочитать
его transformers не умеет («Loading pre-quantized NVFP4 checkpoints is not supported yet», проверено).
Значит наши четыре бита — СВОЁ приближение того же формата, а не те же самые байты.
"""
from __future__ import annotations

import torch
from torch import nn

BLOCK = 16
# уровни e2m1: знак хранится отдельно битом
LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def quantize_fp4(w: torch.Tensor, block: int = BLOCK) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., n) -> упакованные байты (..., n/2) и масштабы (..., n/block)."""
    shape = w.shape
    assert shape[-1] % block == 0, "последнее измерение %d не делится на блок %d" % (shape[-1], block)
    wb = w.reshape(-1, block).float()
    scale = (wb.abs().amax(-1, keepdim=True) / 6.0).clamp(min=1e-12)
    x = wb / scale
    lv = LEVELS.to(x.device)
    idx = (x.abs().unsqueeze(-1) - lv).abs().argmin(-1)          # ближайший уровень
    code = idx.to(torch.uint8) | ((x < 0).to(torch.uint8) << 3)  # старший бит — знак
    code = code.reshape(-1, 2)
    packed = (code[:, 0] | (code[:, 1] << 4)).reshape(*shape[:-1], shape[-1] // 2)
    return packed, scale.reshape(*shape[:-1], shape[-1] // block).to(torch.bfloat16)


def dequantize_fp4(packed: torch.Tensor, scale: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    flat = packed.reshape(-1)
    code = torch.stack([flat & 0xF, flat >> 4], dim=-1).reshape(-1)
    sign = torch.where((code >> 3) & 1 == 1, -1.0, 1.0)
    vals = LEVELS.to(packed.device)[(code & 7).long()] * sign
    out = vals.reshape(-1, block) * scale.reshape(-1, 1).float()
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2).to(torch.bfloat16)


def _forward_fp4(self, hidden_states, top_k_index, top_k_weights):
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
        w_gu = dequantize_fp4(self.gate_up_q[expert_idx], self.gate_up_scale[expert_idx])
        gate, up = nn.functional.linear(current_state, w_gu).chunk(2, dim=-1)
        h = self.act_fn(gate) * up
        w_dn = dequantize_fp4(self.down_q[expert_idx], self.down_scale[expert_idx])
        h = nn.functional.linear(h, w_dn)
        h = h * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, h.to(final_hidden_states.dtype))

    return final_hidden_states


def compress_experts_fp4(model: nn.Module, verbose: bool = True) -> dict:
    targets = [m for m in model.modules() if type(m).__name__ == "Qwen4ExpTextExperts"]
    if not targets:
        raise SystemExit("модулей Qwen4ExpTextExperts не найдено — сжимать нечего")

    before = after = 0
    for mod in targets:
        for src, pre in (("gate_up_proj", "gate_up"), ("down_proj", "down")):
            w = getattr(mod, src).data
            before += w.numel() * w.element_size()
            q, s = quantize_fp4(w)
            delattr(mod, src)
            mod.register_buffer(pre + "_q", q, persistent=False)
            mod.register_buffer(pre + "_scale", s, persistent=False)
            after += q.numel() * q.element_size() + s.numel() * s.element_size()
        mod.forward = _forward_fp4.__get__(mod, type(mod))

    out = {"модулей": len(targets), "было_ГБ": before / 1e9, "стало_ГБ": after / 1e9}
    if verbose:
        print("эксперты сжаты в 4 бита: %d модулей, %.4f -> %.4f ГБ (в %.2f раза)"
              % (out["модулей"], out["было_ГБ"], out["стало_ГБ"],
                 out["было_ГБ"] / max(out["стало_ГБ"], 1e-9)), flush=True)
    return out

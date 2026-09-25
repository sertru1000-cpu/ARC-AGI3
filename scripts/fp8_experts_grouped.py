"""Счёт экспертов ОДНИМ групповым умножением вместо перебора по одному (25.09).

Зачем. Измерено 24.09 на живой модели: обучение идёт 60 токенов/с, это 0.15% от возможностей B300.
Карта простаивает, потому что штатный forward Qwen4ExpTextExperts перебирает задетых экспертов в
питоновском цикле и для каждого делает отдельное крошечное умножение: при 10 тыс. токенов каждому
эксперту достаётся десяток-другой строк, и время уходит на накладные расходы запуска, а не на счёт.
За один прямой проход таких запусков набирается под двадцать пять тысяч (48 слоёв x до 512 экспертов).

Что делает этот вариант. Токены группируются по экспертам, дополняются до общей длины и считаются
ОДНИМ torch.bmm на все задетые эксперты сразу: один запуск вместо сотен. Деквантование fp8 тоже
делается пачкой.

Цена — память на дополнение: (задетых экспертов) x (максимум токенов на эксперта) x 2560 x 2 байта.
При 10 тыс. токенов и равномерном распределении это порядка гигабайта, при перекосе больше, поэтому
есть предохранитель: если дополнение выходит дороже порога, слой считается старым способом.

Проверяется локально, без карты: scripts/test_grouped_experts.py сверяет выход с обычным циклом.
"""
from __future__ import annotations

import torch
from torch import nn

from fp8_experts import _dequantize

PAD_LIMIT_GB = 4.0        # больше этого на дополнение не тратим -- откатываемся к циклу


def _grouped_forward(self, hidden_states, top_k_index, top_k_weights):
    with torch.no_grad():
        mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten()
        hit = hit[hit != self.num_experts]
        if hit.numel() == 0:
            return torch.zeros_like(hidden_states)
        # кого считает каждый эксперт
        per_expert = []
        for e in hit.tolist():
            pos, tok_idx = torch.where(mask[e])
            per_expert.append((e, pos, tok_idx))
        widest = max(int(t.numel()) for _, _, t in per_expert)
        need_gb = len(per_expert) * widest * hidden_states.shape[-1] * 2 / 1e9

    if need_gb > PAD_LIMIT_GB:                      # предохранитель: перекос по экспертам
        return self._forward_loop(hidden_states, top_k_index, top_k_weights)

    E = len(per_expert)
    H = hidden_states.shape[-1]
    dev, dt = hidden_states.device, hidden_states.dtype
    xb = torch.zeros(E, widest, H, device=dev, dtype=dt)
    keep = torch.zeros(E, widest, device=dev, dtype=torch.bool)
    for i, (_, _, tok_idx) in enumerate(per_expert):
        n = int(tok_idx.numel())
        xb[i, :n] = hidden_states[tok_idx]
        keep[i, :n] = True

    idx = torch.tensor([e for e, _, _ in per_expert], device=dev)
    w_gu = _dequantize(self.gate_up_q[idx], self.gate_up_scale[idx])       # (E, 2I, H)
    w_dn = _dequantize(self.down_q[idx], self.down_scale[idx])             # (E, H, I)

    h = torch.bmm(xb, w_gu.transpose(1, 2))                                # ОДИН запуск на все эксперты
    gate, up = h.chunk(2, dim=-1)
    h = self.act_fn(gate) * up
    out = torch.bmm(h, w_dn.transpose(1, 2))                               # и ещё один

    final = torch.zeros_like(hidden_states)
    for i, (_, pos, tok_idx) in enumerate(per_expert):
        n = int(tok_idx.numel())
        piece = out[i, :n] * top_k_weights[tok_idx, pos, None]
        final.index_add_(0, tok_idx, piece.to(final.dtype))
    return final


def use_grouped(model: nn.Module, verbose: bool = True) -> int:
    """Переключить уже сжатые модули экспертов на групповой счёт. Возвращает число переключённых."""
    n = 0
    for m in model.modules():
        if type(m).__name__ != "Qwen4ExpTextExperts" or not hasattr(m, "gate_up_q"):
            continue
        if not hasattr(m, "_forward_loop"):
            m._forward_loop = m.forward                  # оставляем старый путь для предохранителя
        m.forward = _grouped_forward.__get__(m, type(m))
        n += 1
    if verbose:
        print("групповой счёт экспертов включён в %d модулях" % n, flush=True)
    return n

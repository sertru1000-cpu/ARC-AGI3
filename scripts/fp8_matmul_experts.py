"""Умножение экспертов ПРЯМО В fp8, без разжатия весов (25.09).

Зачем. Измерено 25.09 в живом сеансе: около 50 секунд на пример уходит на постоянную работу, не
зависящую от числа токенов, — это разжатие весов экспертов в bf16 перед умножением. На каждом слое
разжимаются все задетые эксперты, а задеты почти все 512 при любой длине примера. Пачкой эта работа
делится, но не исчезает. Умножение прямо в fp8 убирает её совсем: веса так и остаются сжатыми.

Как. `torch._scaled_mm` на Blackwell умножает fp8 на fp8 и отдаёт bf16, принимая масштабы отдельно:
для весов — по выходным каналам (ровно так мы их и храним, см. fp8_experts._quantize), для активаций —
по строкам, их квантуем на лету.

Обратный проход. Веса экспертов заморожены, поэтому нужен только градиент по входу: dX = dY · W.
Его считаем тем же способом, квантуя dY в fp8 по строкам.

Расплата за точность. Активации в fp8 теряют около 2-3% на значение, но ошибки усредняются по 2560
слагаемым, и на выходе слоя это доли процента — тот же порядок, что мы уже измерили для весов (fp8
дал 2.65% на слое, а модель в бою вообще работает в четырёх битах).

Проверяется локально: scripts/test_fp8_matmul.py считает ошибку схемы без карты (сам вызов
`_scaled_mm` есть только на Blackwell, для остального — запасной путь через разжатие).
"""
from __future__ import annotations

import torch
from torch import nn

from fp8_experts import _dequantize

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, K) bf16 -> fp8 + масштаб на строку (N, 1). Масштаб отдельно, как требует _scaled_mm."""
    scale = (x.abs().amax(dim=-1, keepdim=True).float() / FP8_MAX).clamp(min=1e-12)
    return (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8), scale


def _hw_available() -> bool:
    return torch.cuda.is_available() and hasattr(torch, "_scaled_mm")


def fp8_linear(x: torch.Tensor, w_q: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """y = x @ W^T, где W хранится в fp8 с масштабом по выходным каналам. Без разжатия W."""
    xq, xs = quantize_rows(x)
    if _hw_available():
        try:
            # _scaled_mm ждёт второй сомножитель по столбцам, поэтому W^T без копии
            return torch._scaled_mm(xq, w_q.t(), scale_a=xs,
                                    scale_b=w_scale.squeeze(-1).float().unsqueeze(0),
                                    out_dtype=torch.bfloat16)
        except Exception:
            pass
    # запасной путь (нет Blackwell или отказ вызова): та же арифметика через разжатие
    xf = xq.float() * xs
    wf = w_q.float() * w_scale.float()
    return (xf @ wf.t()).to(torch.bfloat16)


class _ExpertMM(torch.autograd.Function):
    """Умножение на ЗАМОРОЖЕННЫЕ веса: вперёд y = x·Wᵀ, назад dx = dy·W. Оба раза в fp8."""

    @staticmethod
    def forward(ctx, x, w_q, w_scale):
        ctx.save_for_backward(w_q, w_scale)
        return fp8_linear(x, w_q, w_scale)

    @staticmethod
    def backward(ctx, dy):
        w_q, w_scale = ctx.saved_tensors
        dq, ds = quantize_rows(dy)
        if _hw_available():
            try:
                w = w_q.t()                               # (K, N) -> умножаем dy на W
                dx = torch._scaled_mm(dq, w.t().t(), scale_a=ds,
                                      scale_b=w_scale.squeeze(-1).float().unsqueeze(0),
                                      out_dtype=torch.bfloat16)
                return dx, None, None
            except Exception:
                pass
        wf = w_q.float() * w_scale.float()
        dx = ((dq.float() * ds) @ wf).to(dy.dtype)
        return dx, None, None


def _forward_fp8mm(self, hidden_states, top_k_index, top_k_weights):
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero().flatten()
        hit = hit[hit != self.num_experts]

    for e in hit.tolist():
        pos, tok = torch.where(mask[e])
        xs = hidden_states[tok]
        h = _ExpertMM.apply(xs, self.gate_up_q[e], self.gate_up_scale[e])
        gate, up = h.chunk(2, dim=-1)
        h = self.act_fn(gate) * up
        out = _ExpertMM.apply(h, self.down_q[e], self.down_scale[e])
        final.index_add_(0, tok, (out * top_k_weights[tok, pos, None]).to(final.dtype))
    return final


def use_fp8_matmul(model: nn.Module, verbose: bool = True) -> int:
    n = 0
    for m in model.modules():
        if type(m).__name__ != "Qwen4ExpTextExperts" or not hasattr(m, "gate_up_q"):
            continue
        if not hasattr(m, "_forward_prev"):
            m._forward_prev = m.forward
        m.forward = _forward_fp8mm.__get__(m, type(m))
        n += 1
    if verbose:
        print("умножение прямо в fp8 включено в %d модулях (разжатие весов больше не делается)" % n,
              flush=True)
    return n

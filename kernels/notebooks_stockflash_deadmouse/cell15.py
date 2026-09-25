
# =====================================================================
# МЁРТВАЯ МЫШЬ: харнесс сообщает, что клики в этой игре ничего не меняют.
#
# Основание: sk48 — 284 хода, 0 уровней из 8, из них 165 (58%) пустые клики; первый уровень
# берётся 14 ходами одними простыми действиями. Вред измерен на 25 партиях: порог достигается
# только там, полезных кликов не запрещает нигде.
# =====================================================================
import inference.agent.tool_agent as _dmta

_DM_AFTER = 12           # столько пустых кликов подряд -- и мы говорим об этом модели
_DM_STATS = {"fired": 0, "games": set(), "clicks_after": 0}
_DM_TEXT = (
    "\n\n[HOST] MOUSE has produced NO board change in the last {n} clicks of this game. "
    "In some games the mouse does nothing at all and levels are solved with ACTION1..ACTION5 only. "
    "Consider dropping MOUSE here and working the simple actions -- including their ORDER: "
    "an action may only take effect AFTER another one has been performed first."
)

_dm_orig_prompt = _dmta.ToolAgent._build_user_prompt


def _dm_build_user_prompt(self, action_num, **kw):
    text = _dm_orig_prompt(self, action_num, **kw)
    try:
        hist = list(getattr(self, "history_entries", None) or [])
        streak = 0
        for prev, cur in zip(hist, hist[1:]):
            action = (getattr(cur, "action", "") or "").strip()
            if not action.startswith("MOUSE("):
                continue
            before, after = getattr(prev, "frame", None), getattr(cur, "frame", None)
            if before is None or after is None:
                continue
            if getattr(before, "grid", None) != getattr(after, "grid", None):
                streak = 0
            else:
                streak += 1
        if streak >= _DM_AFTER:
            if _DM_STATS["fired"] == 0 or streak == _DM_AFTER:
                print("DEADMOUSE: подсказка выдана (ход %s, пустых кликов подряд %d)"
                      % (action_num, streak), flush=True)
            _DM_STATS["fired"] += 1
            return text + _DM_TEXT.format(n=streak)
    except Exception as exc:                      # слой не имеет права ронять прогон
        print("DEADMOUSE: сбой слоя, пропускаю: %r" % (exc,), flush=True)
    return text


_dmta.ToolAgent._build_user_prompt = _dm_build_user_prompt

if not TRUE_SUBMISSION:
    bm.solver.max_runtime_s_per_game = 1800.0    # проба 30 минут: точка сравнения base30 = 3.06

print("DEADMOUSE: слой включён (%s). Порог %d пустых кликов подряд. Вред измерен на 25 партиях: "
      "срабатывает только в sk48, полезных кликов не запрещает нигде."
      % ("бой" if TRUE_SUBMISSION else "проба 30 мин", _DM_AFTER), flush=True)

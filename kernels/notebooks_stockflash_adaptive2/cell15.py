
# =====================================================================
# ПРЕДУСЛОВИЯ ПО УСЛОВИЮ: подсказка только застрявшей игре.
#
# Первая пара 25.09: блок всегда -> мёртвые игры ожили (sp80 0->1, g50t 0->2), но сильные просели
# (ar25 −3, re86 −3). Здоровые игры берут первый уровень за 24 хода (медиана), 75% за 32.
# Порог 45 ходов без смены уровня оставляет их нетронутыми.
# =====================================================================
import inference.agent.tool_agent as _adta

_AD_STUCK_AFTER = 45          # ходов на одном уровне -- после этого считаем, что игра встала
_AD_STATS = {"fired": 0, "games": 0}
_AD_BLOCK = (
    "\n\n[HOST] You have spent {n} actions on this level without clearing it. Before continuing, "
    "reconsider the actions you judged useless: an action that produced no visible change is NOT proven "
    "useless -- it may have an unmet PRECONDITION and start working after the board changes. Some actions "
    "only act on the second press. Pick one such action and retest it now, deliberately, after changing "
    "the board with another action. Think in terms of (PRECONDITION, ACTION) -> EFFECT, not ACTION -> EFFECT."
)

_ad_orig_prompt = _adta.ToolAgent._build_user_prompt


def _ad_build_user_prompt(self, action_num, **kw):
    text = _ad_orig_prompt(self, action_num, **kw)
    try:
        hist = list(getattr(self, "history_entries", None) or [])
        if not hist:
            return text
        cur_level = None
        frame = kw.get("current_frame")
        if frame is not None:
            cur_level = getattr(frame, "level", None)
        if cur_level is None:
            cur_level = getattr(getattr(hist[-1], "frame", None), "level", None)
        if cur_level is None:
            return text
        # сколько ходов подряд мы на этом уровне
        n = 0
        for e in reversed(hist):
            lv = getattr(getattr(e, "frame", None), "level", None)
            if lv != cur_level:
                break
            n += 1
        if n >= _AD_STUCK_AFTER:
            if _AD_STATS["fired"] == 0 or n == _AD_STUCK_AFTER:
                print("ADAPTIVE: подсказка выдана (уровень %s, ходов на нём %d)" % (cur_level, n), flush=True)
            _AD_STATS["fired"] += 1
            return text + _AD_BLOCK.format(n=n)
    except Exception as exc:                        # слой не имеет права ронять прогон
        print("ADAPTIVE: сбой слоя, пропускаю: %r" % (exc,), flush=True)
    return text


_adta.ToolAgent._build_user_prompt = _ad_build_user_prompt

# вторая правка: снять запрет на пробы для действий без НАБЛЮДЁННОГО эффекта (системный промпт)
_NS_OLD = ("Once the important state variables and action effects are sufficiently understood, "
           "stop probing and search in the inferred state space.")
_NS_NEW = ("Once the important state variables are understood, prefer searching in the inferred state "
           "space over further probing -- but only stop probing an ACTION whose effect you have actually "
           "OBSERVED at least once. An action that has produced no visible change anywhere is still "
           "UNKNOWN, not known-useless: it may have an unmet precondition, or it may need to be pressed "
           "twice. Keep such actions on the list of things to retest after the board changes.")
_ad_orig_system = _adta._build_system_prompt


def _ad_build_system_prompt(**kw):
    try:
        text = _ad_orig_system(**kw)
        if _NS_OLD in text:
            return text.replace(_NS_OLD, _NS_NEW)
        return text
    except Exception as exc:
        print("ADAPTIVE2: сбой правки системного промпта: %r" % (exc,), flush=True)
        return _ad_orig_system(**kw)


_adta._build_system_prompt = _ad_build_system_prompt


if not TRUE_SUBMISSION:
    bm.solver.max_runtime_s_per_game = 7920.0       # полный прогон 132 мин; сравнение flash_v1_phaseA = 9.43

print("ADAPTIVE2: два слоя включены (%s). Порог %d ходов на уровне без его взятия. Здоровые игры берут "
      "первый уровень за 24 хода (медиана), 75%% за 32 -- их подсказка не коснётся. Плюс снята стоковая фраза про stop probing."
      % ("бой" if TRUE_SUBMISSION else "полный прогон 132 мин", _AD_STUCK_AFTER), flush=True)

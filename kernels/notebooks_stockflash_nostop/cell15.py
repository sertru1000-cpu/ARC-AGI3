
# =====================================================================
# БЕЗ ПРЕЖДЕВРЕМЕННОГО ЗАКРЫТИЯ: «перестань пробовать» — только про наблюдённые эффекты.
#
# Стоковая строка велит прекращать пробы, когда эффекты «достаточно поняты». Пустой исход модель
# засчитывает за понимание и больше к действию не возвращается. Перебор на локальном движке (25.09):
# 40% мёртвых в старте действий оживают после ОДНОГО хода.
# =====================================================================
import inference.agent.tool_agent as _nsta

_NS_OLD = ("Once the important state variables and action effects are sufficiently understood, "
           "stop probing and search in the inferred state space.")
_NS_NEW = ("Once the important state variables are understood, prefer searching in the inferred state "
           "space over further probing -- but only stop probing an ACTION whose effect you have actually "
           "OBSERVED at least once. An action that has produced no visible change anywhere is still "
           "UNKNOWN, not known-useless: it may have an unmet precondition, or it may need to be pressed "
           "twice. Keep such actions on the list of things to retest after the board changes.")
_NS_DONE = {"ok": False}

_ns_orig_system = _nsta._build_system_prompt


def _ns_build_system_prompt(**kw):
    try:
        text = _ns_orig_system(**kw)
        if _NS_OLD in text:
            if not _NS_DONE["ok"]:
                print("NOSTOP: строка найдена и заменена", flush=True)
                _NS_DONE["ok"] = True
            return text.replace(_NS_OLD, _NS_NEW)
        if not _NS_DONE["ok"]:
            print("NOSTOP: ВНИМАНИЕ — стоковая строка НЕ найдена, промпт отдан как есть", flush=True)
            _NS_DONE["ok"] = True
        return text
    except Exception as exc:                        # слой не имеет права ронять прогон
        print("NOSTOP: сбой слоя, отдаю стоковый промпт: %r" % (exc,), flush=True)
        return _ns_orig_system(**kw)


_nsta._build_system_prompt = _ns_build_system_prompt

if not TRUE_SUBMISSION:
    bm.solver.max_runtime_s_per_game = 7920.0       # ПОЛНЫЙ прогон 132 мин; сравнение с flash_v1_phaseA = 9.43

print("NOSTOP: слой включён (%s). Меняем ОДНО предложение стокового промпта: запрет прекращать пробы "
      "для действий без наблюдённого эффекта." % ("бой" if TRUE_SUBMISSION else "полный прогон 132 мин"),
      flush=True)

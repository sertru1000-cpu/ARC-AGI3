
# =====================================================================
# ПРЕДУСЛОВИЯ: пустой исход не доказывает бесполезность действия.
#
# Измерено перебором на локальном движке (25.09): 40% действий, не меняющих кадр в стартовом
# состоянии, оживают ПОСЛЕ ОДНОГО ХОДА (sk48: ACTION2 после ACTION1; g50t: три действия после
# ACTION1 — переключатель, срабатывающий со второго нажатия). Стоковый промпт учит связке
# «действие -> эффект» и велит прекращать пробы, когда эффекты поняты; из-за этого отвергнутое
# действие больше не проверяется никогда.
# =====================================================================
import inference.agent.tool_agent as _pcta

_PC_BLOCK = (
    "\n"
    "Conditional effects (important):\n"
    "- An action that produces NO visible change is NOT proven useless. It may have an unmet "
    "PRECONDITION: the same action can become effective after the state changes.\n"
    "- Model rules as (PRECONDITION P, ACTION A) -> EFFECT E, not as A -> E. When A yields nothing, "
    "record the condition C under which it did nothing, and keep the hypothesis that some P makes A work.\n"
    "- Before writing an action off, retest it at least once after the board has changed in a way you "
    "caused deliberately -- in particular after pressing each of the other simple actions once.\n"
    "- Some actions toggle: the first press arms something invisible and the second press acts. "
    "If an action seems dead, try it twice in a row before discarding it.\n"
    "- 'Stop probing' applies only to actions whose effect you have OBSERVED. An action with no "
    "observed effect anywhere is still unknown, not known-useless.\n"
)

_NS_OLD = ("Once the important state variables and action effects are sufficiently understood, "
           "stop probing and search in the inferred state space.")
_NS_NEW = ("Once the important state variables are understood, prefer searching in the inferred state "
           "space over further probing -- but only stop probing an ACTION whose effect you have actually "
           "OBSERVED at least once. An action that has produced no visible change anywhere is still "
           "UNKNOWN, not known-useless: it may have an unmet precondition, or it may need to be pressed "
           "twice. Keep such actions on the list of things to retest after the board changes.")

_pc_orig_system = _pcta.ToolAgent._build_system_prompt


def _pc_build_system_prompt(*args, **kw):
    try:
        text = _pc_orig_system(*args, **kw)
        if "Conditional effects (important)" in text:
            return text
        if _NS_OLD in text:                     # вторая правка: снять преждевременное закрытие
            text = text.replace(_NS_OLD, _NS_NEW)
        return text + _PC_BLOCK
    except Exception as exc:                       # слой не имеет права ронять прогон
        print("PRECOND: сбой слоя, отдаю стоковый промпт: %r" % (exc,), flush=True)
        return _pc_orig_system(*args, **kw)


_pcta.ToolAgent._build_system_prompt = _pc_build_system_prompt

if not TRUE_SUBMISSION:
    bm.solver.max_runtime_s_per_game = 7920.0     # ПОЛНЫЙ прогон 132 мин: точка сравнения
                                                  # runs/flash_v1_phaseA = 9.43, 40 уровней, первый в 21/25

print("COMBINED: оба слоя включены (%s). Основание: перебором на локальном движке 40%% мёртвых действий "
      "оживают после одного хода (sk48 ACTION2 после ACTION1; g50t три действия после ACTION1). "
      "Блок +%d знаков И замена фразы про stop probing."
      % ("бой" if TRUE_SUBMISSION else "полный прогон 132 мин", len(_PC_BLOCK)), flush=True)

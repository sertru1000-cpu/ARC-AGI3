
# =====================================================================
# ЦЕЛЬ КАК ПРАВИЛО: назвать вслух и перепроверить на новом уровне.
#
# Цель-картинка (кадр победы) закрыта 19.09: перенесённая даёт 19% против 18% случайного.
# Цель-правило словами — другое: она работает на ОПОЗНАНИЕ цели, а это и есть измеренное узкое место.
# =====================================================================
import inference.agent.tool_agent as _grta

_GR_BLOCK = (
    "\n"
    "Level goal as a rule (state it, then re-check it):\n"
    "- Keep ONE short sentence in your world model called GOAL RULE: what has to become true for the "
    "level to be cleared, phrased as a rule about the board, not as a list of moves. "
    "Examples of the right shape: 'bring every coloured shape onto the matching coloured tile', "
    "'remove all blocks of the same colour', 'move the agent to the exit after opening it'.\n"
    "- The GOAL RULE is about the board, never about which buttons you pressed. Do not carry move "
    "sequences between levels -- layouts change, rules usually do not.\n"
    "- When the level number changes, do NOT assume the rule still holds and do NOT relearn it from "
    "scratch either. State it as a hypothesis first: 'previous GOAL RULE was X; does the new board fit "
    "X?'. Confirm or correct it from the first frames, then continue.\n"
    "- If the new board clearly cannot fit the previous rule, say so explicitly and write a new GOAL "
    "RULE before planning any move.\n"
)

_gr_orig_system = _grta._build_system_prompt


def _gr_build_system_prompt(**kw):
    try:
        text = _gr_orig_system(**kw)
        if "Level goal as a rule" in text:
            return text
        return text + _GR_BLOCK
    except Exception as exc:                        # слой не имеет права ронять прогон
        print("GOALRULE: сбой слоя, отдаю стоковый промпт: %r" % (exc,), flush=True)
        return _gr_orig_system(**kw)


_grta._build_system_prompt = _gr_build_system_prompt

if not TRUE_SUBMISSION:
    bm.solver.max_runtime_s_per_game = 7920.0       # полный прогон 132 мин; сравнение flash_v1_phaseA = 9.43

print("GOALRULE: слой включён (%s). Цель уровня как ПРАВИЛО о доске, названное одной строкой и "
      "перепроверяемое при смене уровня. Путь прохождения НЕ переносится (это carry, закрыт 17.09). "
      "Блок +%d знаков." % ("бой" if TRUE_SUBMISSION else "полный прогон 132 мин", len(_GR_BLOCK)),
      flush=True)

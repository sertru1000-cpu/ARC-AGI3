
# =====================================================================
# ОРАКУЛ-ИНЪЕКЦИЯ: застрявшей игре выдаётся ОДНО правило УПРАВЛЕНИЯ.
#
# Вопрос замера: модель не умеет НАХОДИТЬ правила — или не умеет ПЛАНИРОВАТЬ с готовым?
# Выдаётся только схема управления (что делают действия, чем засчитывается уровень),
# не решение. Каждое правило восстановлено из настоящей успешной траектории в журналах.
#
# Включается с хода 50: до него промпт побайтово стоковый, поэтому сравнение идёт
# «та же игра до подсказки» и «та же игра в базовом прогоне».
#
# Цели — 9 игр, застрявших в базе на 0-1 уровне. Остальные 16 не трогаются: внутренний контроль.
# Только оффлайн: в боевой ветке ничего не меняется.
# =====================================================================
import inference.framework.solver as _os_solver

_ORACLE_ACTS = 50
_ORACLE_HINTS = {
    "tn36": "Mouse clicks toggle individual switches in the strip below the target grid; each click flips one switch between its two states. A separate button outside that strip submits the current setting and is what completes a level. The strip and the button move to new coordinates on every level, so locate them again after each transition.",
    "sk48": "Arrow keys are the controls: LEFT and RIGHT shift a coloured segment along its row, while UP and DOWN switch which row is active. The mouse is not needed. A level completes on an arrow move, not on a separate confirm action.",
    "sp80": "LEFT and RIGHT slide the blue bar along the orange track one step per press; SPACE commits the current position and is what completes a level. Pressing SPACE in the wrong position costs the attempt, so position first and commit once.",
    "g50t": "Arrow keys step a single small piece one cell at a time through the field; there is no jump and no mouse control. The level completes on the arrow press that brings the piece to its destination.",
    "lf52": "Mouse clicks on individual cells are the only control; each valid click advances the progress meter along the top edge of the board. Clicking the same cell again does not advance it. The level completes on a click, not on a keyboard action.",
    "cn04": "LEFT and RIGHT move the white block along its row one step per press; SPACE acts at the current position and changes the pattern beneath it. The meter along the top edge grows as the level progresses. A level can complete on an arrow move.",
    "bp35": "Arrow keys move the marked piece along its row one step per press. Mouse clicks on the side panel do not move the piece: they change the whole layout at once, which is a separate control. A level completes on an arrow move.",
    "wa30": "Arrow keys move a four-cell piece around the board; SPACE changes the state of that piece in place rather than moving it, cycling it through its available states. The level completes on SPACE with the piece in the right place and state.",
    "ls20": "Arrow keys drive a five-cell piece through the field; UP is the direction that makes progress and the meter along the bottom edge grows as it advances. The level completes on an arrow press, not on SPACE."
}

_oracle_stats = {"games": {}, "injected": 0}

if not TRUE_SUBMISSION:
    _o_orig_make = _os_solver.HarnessSolver._make_analyzer

    def _o_make(self, game, index, local_server=None):
        agent = _o_orig_make(self, game, index, local_server)
        try:
            gid = str(getattr(getattr(game, "game_run", None), "game_id", ""))[:4]
        except Exception:
            return agent
        hint = _ORACLE_HINTS.get(gid)
        if not hint:
            return agent
        _orig_build = agent._build_user_prompt

        def _build(action_num, **kw):
            text = _orig_build(action_num, **kw)
            try:
                if int(action_num) < _ORACLE_ACTS:
                    return text
            except Exception:
                return text
            seen = _oracle_stats["games"].setdefault(gid, 0)
            _oracle_stats["games"][gid] = seen + 1
            _oracle_stats["injected"] += 1
            if seen == 0:
                print("[ORACLE] %s: правило выдано на ходу %s" % (gid, action_num), flush=True)
            return text + "\n" + (
                "Known mechanic for this game, verified externally and reliable: "
                + hint
                + " Use it as given; do not spend turns re-deriving it. Everything else about this level "
                  "-- the goal, the target pattern, and the order of moves -- is still for you to work out."
            )

        agent._build_user_prompt = _build
        return agent

    _os_solver.HarnessSolver._make_analyzer = _o_make
    print("ORACLE: правило управления выдаётся %d играм с хода %d; остальные 16 игр не тронуты. "
          "Вопрос: модель не умеет находить правила или не умеет планировать с готовым. "
          "ПОРОГИ: взятых уровней в целевых играх > 9 (в базе 5 на 9 игр) при неизменных 16 контрольных -- "
          "правило усваивается; уровней 5-9 -- неразличимо; меньше 5 -- подсказка мешает. "
          "Механизм: строк [ORACLE] ровно 9, по одной на целевую игру."
          % (len(_ORACLE_HINTS), _ORACLE_ACTS), flush=True)


# =====================================================================
# НАКОПЛЕНИЕ ЗНАНИЯ МЕЖДУ ИГРАМИ: узнанное одной игрой достаётся следующим.
#
# В бою 110 игр идут в одном процессе четырьмя волнами; сейчас каждая начинается с нуля.
# Переносим только game-agnostic: словарь механик из наших журналов (статическая часть)
# и распределение того, чем засчитывались уровни у уже сыгравших игр (динамическая).
#
# Регистрация и выдача — в одном шве, лишних вызовов модели НОЛЬ.
# Два прохода нужны, чтобы у знания появился получатель: 28 запусков с нуля, 22 с 132-й минуты.
#
# Только оффлайн: в боевой ветке ничего не меняется.
# =====================================================================
import threading as _xt
import inference.framework.solver as _xs

_XG_MIN_FACTS = 3
_XG_STATIC = True      # словарь механик из наших журналов
_XG_DYNAMIC = True     # накопленное в этом прогоне

_XG_PRIOR = (
    "Cross-game notes (verified on other games of this same benchmark, not on this one):\n"
    "- A level completes on an acting move, not on an inspection call: in every solved case we "
    "recorded it was an arrow key, SPACE, or a click on a confirm control outside the puzzle area.\n"
    "- A changed board does not mean a useful move. Progress meters and budget bars along the "
    "edges change on almost every action; in one game 367 of 561 moves changed only such a bar "
    "and nothing in the play area. Separate edge/HUD changes from play-area changes before "
    "concluding that an action did something.\n"
    "- A strip of cells below or beside the play area is usually a control panel or a progress "
    "meter rather than part of the puzzle.\n"
    "- Control positions move between levels of the same game: re-locate the controls after every "
    "level transition instead of reusing coordinates that worked before.\n"
    "- If many moves pass with no level gained, the hypothesis class is wrong, not the "
    "coordinates: across games, once a run went 100 moves without a level, only one game in nine "
    "ever gained another one. Change what you believe the controls ARE, not where you click."
)

_xg = {"lock": _xt.Lock(), "levels": [], "games_done": set(), "games_scored": set(), "served": 0}


def _xg_note(gid, action_name):
    with _xg["lock"]:
        _xg["levels"].append(str(action_name or "?"))
        _xg["games_scored"].add(gid)


def _xg_digest():
    """Короткая сводка накопленного. Возвращает пустую строку, пока фактов мало."""
    with _xg["lock"]:
        levels = list(_xg["levels"])
        scored = len(_xg["games_scored"])
        done = len(_xg["games_done"])
    if len(levels) < _XG_MIN_FACTS:
        return ""
    kinds = {"arrow": 0, "space": 0, "mouse": 0, "other": 0}
    for name in levels:
        up = name.upper()
        if "MOUSE" in up:
            kinds["mouse"] += 1
        elif "SPACE" in up:
            kinds["space"] += 1
        elif any(d in up for d in ("UP", "DOWN", "LEFT", "RIGHT")):
            kinds["arrow"] += 1
        else:
            kinds["other"] += 1
    parts = ["%s %d" % (k, v) for k, v in kinds.items() if v]
    line = ("So far in this run, %d levels were completed across %d games; the completing action "
            "was: %s." % (len(levels), scored, ", ".join(parts)))
    if done:
        line += (" %d games have finished; %d of them completed at least one level."
                 % (done, scored))
    return line


if not TRUE_SUBMISSION:
    bm.n_passes = 2          # два прохода: у знания появляется получатель (вторая волна)

    _xg_orig_make = _xs.HarnessSolver._make_analyzer

    def _xg_make(self, game, index, local_server=None):
        agent = _xg_orig_make(self, game, index, local_server)
        try:
            gid = str(getattr(getattr(game, "game_run", None), "game_id", ""))[:4]
        except Exception:
            return agent
        _orig_build = agent._build_user_prompt

        def _build(action_num, **kw):
            summary = kw.get("previous_step_summary") or {}
            try:
                if summary.get("level_transition"):
                    acts = summary.get("executed_actions") or []
                    _xg_note(gid, acts[-1] if acts else "?")
                if summary.get("game_over") or summary.get("run_complete"):
                    with _xg["lock"]:
                        _xg["games_done"].add(gid)
            except Exception as _e:
                print("[XGAME] сбой учёта, играем как обычно: %r" % (_e,), flush=True)
            text = _orig_build(action_num, **kw)
            block = []
            if _XG_STATIC:
                block.append(_XG_PRIOR)
            if _XG_DYNAMIC:
                d = _xg_digest()
                if d:
                    block.append(d)
            if not block:
                return text
            with _xg["lock"]:
                _xg["served"] += 1
                first = _xg["served"] == 1
            if first:
                print("[XGAME] первая выдача знания: игра %s, ход %s" % (gid, action_num), flush=True)
            return text + "\n" + "\n".join(block)

        agent._build_user_prompt = _build
        return agent

    _xs.HarnessSolver._make_analyzer = _xg_make

    print("XGAME: накопление знания между играми включено; два прохода (28 с нуля, 22 с 132-й мин). "
          "Статический словарь: %s, динамическая сводка: %s (порог %d фактов). "
          "Лишних вызовов модели ноль. КОНТРОЛЬ — тот же расклад без накопления, runs/flash_passes_v1: "
          "средний балл запуска 7.10, первая волна 5.51, вторая 9.13. "
          "ПОРОГИ: вторая волна > 10.5 при неизменной первой — знание переносится; 8.5-10.5 — "
          "неразличимо; < 8.5 — мешает. Механизм: строка [XGAME] есть, сбоев учёта ноль."
          % (_XG_STATIC, _XG_DYNAMIC, _XG_MIN_FACTS), flush=True)

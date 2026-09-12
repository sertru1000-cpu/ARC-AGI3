"""Накопление знания между играми: то, что узнала одна игра, достаётся следующим.

ЗАЧЕМ. В бою 110 игр идут в ОДНОМ процессе четырьмя волнами по 28 мест. Сейчас каждая игра
начинается с чистого листа: ничего, узнанного в первой волне, вторая не получает. Это
единственный крупный рычаг из четырёх возможных (другая модель, другой обмен «качество против
вызовов», лучший цикл, накопление), который нам доступен и который мы не пробовали ни разу.

ЧТО ПЕРЕНОСИТСЯ. Правила конкретной игры бесполезны: в бою 110 невиданных игр. Переносится
только game-agnostic — и обе части ниже собраны из НАШИХ измерений, а не придуманы:

  СТАТИЧЕСКАЯ часть (словарь механик, добыт из журналов 25 публичных игр):
  * уровень засчитывается действием, а не осмотром; в разобранных нами девяти играх это были
    стрелки, SPACE или клик по кнопке подтверждения вне рабочего поля;
  * «доска изменилась» часто означает движение индикатора, а не поля: в tn36 367 ходов из 561
    меняли только полосу сверху, и агент считал их удачными;
  * полоса под полем или вдоль кромки — обычно органы управления либо счётчик прогресса,
    а не часть головоломки;
  * координаты органов управления МЕНЯЮТСЯ между уровнями одной игры (проверено на tn36:
    кнопка подтверждения переехала с (55,36) на (58,46)).

  ДИНАМИЧЕСКАЯ часть (накапливается прямо в прогоне): чем именно засчитывались уровни
  у уже сыгравших игр — распределение по типам действий и доля игр, взявших хоть один уровень.

ПОЧЕМУ ЗАМЕР ИДЁТ В ДВА ПРОХОДА. При 28 местах и 25 играх в Фазе A все стартуют разом,
и получателя знания нет. Раскладка пункта 3 (n_passes=2) даёт настоящие волны: 28 запусков
с нулевой минуты, 22 — с 132-й. Вторая волна и есть подопытная, и её контроль уже измерен
сегодня БЕЗ накопления: 9.13 против 5.51 у первой (runs/flash_passes_v1).

ШОВ. `HarnessSolver._make_analyzer` — единственное место создания агента; там есть игра.
Внутри оборачивается `_build_user_prompt`, и он же служит регистратором: в него приходит
`previous_step_summary` с `level_transition` и `executed_actions`, то есть факт взятия уровня
и действие, которым он взят. Лишних вызовов модели не делается ни одного — это важно,
потому что вызовы у нас самый дефицитный ресурс (51 на игру за все 132 минуты).

usage:  .venv/bin/python scripts/build_crossgame_notebook.py
"""
import ast
import json
import os

MIN_FACTS = 3          # раньше трёх записей динамическую часть не показываем
SLUG = "sergueimakarov/arc3-stock-flash-wm12"   # рабочий слаг (там отыграл пункт 3, выход скачан)

CELL = '''
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

_XG_MIN_FACTS = %d
_XG_STATIC = True      # словарь механик из наших журналов
_XG_DYNAMIC = True     # накопленное в этом прогоне

_XG_PRIOR = (
    "Cross-game notes (verified on other games of this same benchmark, not on this one):\\n"
    "- A level completes on an acting move, not on an inspection call: in every solved case we "
    "recorded it was an arrow key, SPACE, or a click on a confirm control outside the puzzle area.\\n"
    "- A changed board does not mean a useful move. Progress meters and budget bars along the "
    "edges change on almost every action; in one game 367 of 561 moves changed only such a bar "
    "and nothing in the play area. Separate edge/HUD changes from play-area changes before "
    "concluding that an action did something.\\n"
    "- A strip of cells below or beside the play area is usually a control panel or a progress "
    "meter rather than part of the puzzle.\\n"
    "- Control positions move between levels of the same game: re-locate the controls after every "
    "level transition instead of reusing coordinates that worked before.\\n"
    "- If many moves pass with no level gained, the hypothesis class is wrong, not the "
    "coordinates: across games, once a run went 100 moves without a level, only one game in nine "
    "ever gained another one. Change what you believe the controls ARE, not where you click."
)

_xg = {"lock": _xt.Lock(), "levels": [], "games_done": set(), "games_scored": set(), "served": 0}


def _xg_note(gid, action_name):
    with _xg["lock"]:
        _xg["levels"].append((gid, str(action_name or "?")))
        _xg["games_scored"].add(gid)


def _xg_digest(for_gid):
    """Сводка того, что узнали ДРУГИЕ игры. Собственные факты исключаются: при двух проходах
    та же игра встречается дважды, и без фильтра её первый проход подсказывал бы второму —
    это была бы подсказка самой себе, а не перенос между играми."""
    with _xg["lock"]:
        levels = [name for gid, name in _xg["levels"] if gid != for_gid]
        scored = len({gid for gid, _ in _xg["levels"] if gid != for_gid})
        done = len([g for g in _xg["games_done"] if g != for_gid])
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
    parts = ["%%s %%d" %% (k, v) for k, v in kinds.items() if v]
    line = ("So far in this run, %%d levels were completed across %%d games; the completing action "
            "was: %%s." %% (len(levels), scored, ", ".join(parts)))
    if done:
        line += (" %%d games have finished; %%d of them completed at least one level."
                 %% (done, scored))
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
                print("[XGAME] сбой учёта, играем как обычно: %%r" %% (_e,), flush=True)
            text = _orig_build(action_num, **kw)
            block = []
            if _XG_STATIC:
                block.append(_XG_PRIOR)
            if _XG_DYNAMIC:
                d = _xg_digest(gid)
                if d:
                    block.append(d)
            if not block:
                return text
            with _xg["lock"]:
                _xg["served"] += 1
                first = _xg["served"] == 1
            if first:
                print("[XGAME] первая выдача знания: игра %%s, ход %%s" %% (gid, action_num), flush=True)
            return text + "\\n" + "\\n".join(block)

        agent._build_user_prompt = _build
        return agent

    _xs.HarnessSolver._make_analyzer = _xg_make

    print("XGAME: накопление знания между играми включено; два прохода (28 с нуля, 22 с 132-й мин). "
          "Статический словарь: %%s, динамическая сводка: %%s (порог %%d фактов). "
          "Лишних вызовов модели ноль. КОНТРОЛЬ — тот же расклад без накопления, runs/flash_passes_v1: "
          "средний балл запуска 7.10, первая волна 5.51, вторая 9.13. "
          "ПОРОГИ: вторая волна > 10.5 при неизменной первой — знание переносится; 8.5-10.5 — "
          "неразличимо; < 8.5 — мешает. Механизм: строка [XGAME] есть, сбоев учёта ноль."
          %% (_XG_STATIC, _XG_DYNAMIC, _XG_MIN_FACTS), flush=True)
''' % MIN_FACTS


def main() -> None:
    src = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))
    nb = json.loads(json.dumps(src))
    cell = nb["cells"][15]
    body = "".join(cell["source"])
    anchor = "    seconds=budget - 600.0\n)"
    at = body.find(anchor)
    run_at = body.find("await bm.run(")
    if at < 0 or run_at < 0:
        raise SystemExit("не нашёл стоковый soft_end или запуск прогона — сборка остановлена")
    at += len(anchor)
    code = body[:at] + "\n" + CELL + body[at:]
    cell["source"] = code.splitlines(keepends=True)

    out = "kernels/notebooks_stockflash_xgame"
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(CELL)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    meta["id"] = SLUG
    meta["title"] = "arc3 stock flash wm12"
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)

    compile(code, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    diff = [i for i in range(18) if "".join(nb["cells"][i]["source"]) != "".join(src["cells"][i]["source"])]
    size = os.path.getsize(os.path.join(out, "submission.ipynb"))
    print("ok   изменена только ячейка 15:", diff == [15])
    print("ok   патч после стокового soft_end и до запуска прогона:",
          0 < code.find(anchor) < code.find("_XG_MIN_FACTS = ") < code.find("await bm.run("))
    print("ok   два прохода вне боя; порог динамической части %d факта" % MIN_FACTS)
    print("ok   лишних вызовов модели не делается:", "_chat_completion" not in CELL)
    print("ok   слаг для пуша:", meta["id"])
    print("ok   компилируется; размер ноутбука %.0f КБ, предел 1 МБ: %s" % (size / 1024, size < 1_000_000))


if __name__ == "__main__":
    main()

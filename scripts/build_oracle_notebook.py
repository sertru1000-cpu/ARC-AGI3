"""Оракул-инъекция: застрявшей игре выдаётся ОДНО правило управления, написанное человеком.

ЗАЧЕМ. Единственный вопрос, от которого зависит смысл всей оставшейся ветки (опровержение
ожиданий): модель не умеет НАХОДИТЬ правила игры — или не умеет ПЛАНИРОВАТЬ даже с готовым
правилом? Если с правилом застрявшая игра оживает, работать надо над добычей гипотез.
Если не оживает — ни опровержения, ни модели мира не помогут, и ветка закрывается.

ЧТО ИМЕННО ВЫДАЁТСЯ. Только СХЕМА УПРАВЛЕНИЯ: какие действия на что влияют и чем засчитывается
уровень. Не решение, не цель головоломки, не последовательность ходов. Каждое правило
восстановлено из НАСТОЯЩЕЙ успешной траектории в наших журналах (указано в комментарии),
а не придумано: выдуманное правило превратило бы замер в проверку моей формулировки.

КОГДА. Инъекция включается с хода 50 и дальше — до этого промпт побайтово стоковый. Так
измеряется именно «модель уже застряла, ей дают правило», а не «модель избавили от поиска».

ЦЕЛИ (9 игр). Все, кто в базовом прогоне `runs/flash_v1_phaseA` взял 0 или 1 уровень и у кого
в архиве есть хотя бы один наблюдённый успех. Остальные 16 игр не трогаются — это внутренний
контроль на снос прогона.

ШОВ. `HarnessSolver._make_analyzer` — единственное место создания агента (solver.py:1223),
и туда приходит и игра, и локальный сервер. Штатный `analyzer_factory` НЕ годится: он
вызывается без `local_server`, и агент ушёл бы мимо локального эндпоинта.

usage:  .venv/bin/python scripts/build_oracle_notebook.py
"""
import ast
import json
import os

ACTS_BEFORE = 50
SLUG = "sergueimakarov/arc3-stock-flash-loop"   # рабочий слаг; выход прошлых версий скачан

# Правила управления. Источник каждого — успешная траектория в наших журналах.
HINTS = {
    # runs/flash_input_v1: уровень 1 взят на ходу 39, уровень 2 на ходу 80
    "tn36": "Mouse clicks toggle individual switches in the strip below the target grid; each click flips one switch between its two states. A separate button outside that strip submits the current setting and is what completes a level. The strip and the button move to new coordinates on every level, so locate them again after each transition.",
    # runs/duck_harness_ref/example-run, проход 13: уровень взят на ходу 22
    "sk48": "Arrow keys are the controls: LEFT and RIGHT shift a coloured segment along its row, while UP and DOWN switch which row is active. The mouse is not needed. A level completes on an arrow move, not on a separate confirm action.",
    # runs/flash_input_v1: уровень взят на ходу 138
    "sp80": "LEFT and RIGHT slide the blue bar along the orange track one step per press; SPACE commits the current position and is what completes a level. Pressing SPACE in the wrong position costs the attempt, so position first and commit once.",
    # runs/public_flash_tufa: уровень взят на ходу 77
    "g50t": "Arrow keys step a single small piece one cell at a time through the field; there is no jump and no mouse control. The level completes on the arrow press that brings the piece to its destination.",
    # runs/flash_v1_phaseA: уровень взят на ходу 9
    "lf52": "Mouse clicks on individual cells are the only control; each valid click advances the progress meter along the top edge of the board. Clicking the same cell again does not advance it. The level completes on a click, not on a keyboard action.",
    # runs/flash_v1_phaseA: уровень взят на ходу 24
    "cn04": "LEFT and RIGHT move the white block along its row one step per press; SPACE acts at the current position and changes the pattern beneath it. The meter along the top edge grows as the level progresses. A level can complete on an arrow move.",
    # runs/flash_v1_phaseA: уровень взят на ходу 25
    "bp35": "Arrow keys move the marked piece along its row one step per press. Mouse clicks on the side panel do not move the piece: they change the whole layout at once, which is a separate control. A level completes on an arrow move.",
    # runs/flash_v1_phaseA: уровень взят на ходу 32
    "wa30": "Arrow keys move a four-cell piece around the board; SPACE changes the state of that piece in place rather than moving it, cycling it through its available states. The level completes on SPACE with the piece in the right place and state.",
    # runs/flash_v1_phaseA: уровень взят на ходу 23
    "ls20": "Arrow keys drive a five-cell piece through the field; UP is the direction that makes progress and the meter along the bottom edge grows as it advances. The level completes on an arrow press, not on SPACE.",
}

CELL = '''
# =====================================================================
# ОРАКУЛ-ИНЪЕКЦИЯ: застрявшей игре выдаётся ОДНО правило УПРАВЛЕНИЯ.
#
# Вопрос замера: модель не умеет НАХОДИТЬ правила — или не умеет ПЛАНИРОВАТЬ с готовым?
# Выдаётся только схема управления (что делают действия, чем засчитывается уровень),
# не решение. Каждое правило восстановлено из настоящей успешной траектории в журналах.
#
# Включается с хода %d: до него промпт побайтово стоковый, поэтому сравнение идёт
# «та же игра до подсказки» и «та же игра в базовом прогоне».
#
# Цели — 9 игр, застрявших в базе на 0-1 уровне. Остальные 16 не трогаются: внутренний контроль.
# Только оффлайн: в боевой ветке ничего не меняется.
# =====================================================================
import inference.framework.solver as _os_solver

_ORACLE_ACTS = %d
_ORACLE_HINTS = %s

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
                print("[ORACLE] %%s: правило выдано на ходу %%s" %% (gid, action_num), flush=True)
            return text + "\\n" + (
                "Known mechanic for this game, verified externally and reliable: "
                + hint
                + " Use it as given; do not spend turns re-deriving it. Everything else about this level "
                  "-- the goal, the target pattern, and the order of moves -- is still for you to work out."
            )

        agent._build_user_prompt = _build
        return agent

    _os_solver.HarnessSolver._make_analyzer = _o_make
    print("ORACLE: правило управления выдаётся %%d играм с хода %%d; остальные 16 игр не тронуты. "
          "Вопрос: модель не умеет находить правила или не умеет планировать с готовым. "
          "ПОРОГИ: взятых уровней в целевых играх > 9 (в базе 5 на 9 игр) при неизменных 16 контрольных -- "
          "правило усваивается; уровней 5-9 -- неразличимо; меньше 5 -- подсказка мешает. "
          "Механизм: строк [ORACLE] ровно 9, по одной на целевую игру."
          %% (len(_ORACLE_HINTS), _ORACLE_ACTS), flush=True)
''' % (ACTS_BEFORE, ACTS_BEFORE, json.dumps(HINTS, ensure_ascii=False, indent=4))


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

    out = "kernels/notebooks_stockflash_oracle"
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(CELL)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    meta["id"] = SLUG
    meta["title"] = "arc3 stock flash loop"
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)

    compile(code, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    diff = [i for i in range(18) if "".join(nb["cells"][i]["source"]) != "".join(src["cells"][i]["source"])]
    patch_at = code.find("_ORACLE_ACTS = ")
    size = os.path.getsize(os.path.join(out, "submission.ipynb"))
    print("ok   изменена только ячейка 15:", diff == [15])
    print("ok   патч после стокового soft_end и до запуска прогона:",
          0 < code.find(anchor) < patch_at < code.find("await bm.run("))
    print("ok   целей %d, инъекция с хода %d" % (len(HINTS), ACTS_BEFORE))
    print("ok   слаг для пуша:", meta["id"])
    print("ok   компилируется; размер ноутбука %.0f КБ, предел 1 МБ: %s" % (size / 1024, size < 1_000_000))


if __name__ == "__main__":
    main()

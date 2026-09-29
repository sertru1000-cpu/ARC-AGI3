"""Сборка `arc3-nextfork-twin`: исполняемая модель мира с реплей-проверкой — ЯЧЕЙКОЙ поверх датасета v3 (26.09).

По статьям Twin (arXiv:2608.14490: 7.8% -> 93.3% на публичных 25, 179/183 уровней) и Tycho (arXiv:2607.28287:
Opus 4.8 — 88.5 RHAE, GPT-5.6/Opus 5 — 100). Обе — на frontier-моделях; слабые не проверялись.

Наша история (08–15.09, девять конфигураций): программы модели предсказывают 56% новых переходов, при горизонте
раскатки ≥8 план найден в 6 играх из 9; в бою слой проигрывал, потому что синтез съедал вызовы. Непроверенное у
нас — именно то, на чём держится Twin: (1) ход по плану только после ПОЛНОГО воспроизведения наблюдённого,
(2) ремонт по контрпримеру, (3) исполнение найденного пути одним пакетом (много ходов за вызов — удар по очереди,
где у нас 87% задержки).

Что делает ячейка:
  * в каждый вызов python подставляются помощники (validate, plan, to_actions, level_transitions, parse_action)
    и функции модели, определённые ранее в этой игре (песочница каждый раз новая — без этого модель переписывала бы
    симулятор каждым вызовом и тратила выходные токены);
  * после вызова из кода модели вынимаются верхнеуровневые def, если среди них есть step или goal;
  * блок в системный промпт описывает цикл: step -> validate -> чинить по контрпримерам -> goal -> plan -> пакет.
Мерить: доля вызовов с validate/plan (слой работал?), игры с 0 расхождений, ходов на вызов, охват и уровни.
usage: .venv/bin/python scripts/build_twin_notebook.py
"""
import ast, json, os

HELPERS = open("kernels/notebooks_nextfork_twin/twin_helpers.py", encoding="utf-8").read()

CELL_TEMPLATE = r'''
# =====================================================================
# TWIN: исполняемая модель мира с реплей-проверкой и планом (26.09). Ячейкой на датасет v3.
# =====================================================================
import ast as _twast, sys as _twsys
import inference.agent.tool_agent as _twta

_TW_HELPERS = __HELPERS_REPR__

_TW_BLOCK = (
    "\n"
    "Executable world model (do this once the basic effects of actions are known):\n"
    "- Write `def step(grid, action):` returning the next grid (list of lists of ints) for an action string "
    "('UP', 'DOWN', 'LEFT', 'RIGHT', 'SPACE', or 'MOUSE(row=R, col=C)'). Use `parse_action(a)` for clicks.\n"
    "- Call `validate(step)`: it replays EVERY observed transition of this level through your step and prints "
    "counterexamples (cells predicted vs observed). Fix step until it prints 0 mismatches. Do not trust a plan "
    "from a step that does not reproduce all observations.\n"
    "- Write `def goal(grid):` -> True when the level would be cleared (your current goal hypothesis).\n"
    "- With 0 mismatches, call `path = plan(step, goal)` (breadth-first search inside your simulator; pass "
    "`actions=[...]` to include specific clicks) and execute the whole path at once: `action(to_actions(path))`.\n"
    "- `step`, `goal` and any helper functions you define are kept by the harness for later calls in this game: "
    "redefine them only to change them. Helpers available: validate, plan, to_actions, level_transitions, parse_action.\n"
    "- If the plan's moves do not produce the predicted result, the new transitions become counterexamples: "
    "revalidate and repair step, or revise goal.\n"
)
_tw_orig_prompt = _twta._build_system_prompt


def _tw_build_system_prompt(**kw):
    try:
        text = _tw_orig_prompt(**kw)
        return text if "Executable world model" in text else text + _TW_BLOCK
    except Exception as exc:
        print("TWIN: сбой промпта: %r" % (exc,), flush=True)
        return _tw_orig_prompt(**kw)


_tw_orig_run = _twta.ToolAgent._run_python_tool


def _tw_store(self, code):
    """Вынуть верхнеуровневые def из кода модели, если среди них есть step или goal."""
    try:
        tree = _twast.parse(code)
    except Exception:
        return
    defs = [n for n in tree.body if isinstance(n, _twast.FunctionDef)]
    if not any(n.name in ("step", "goal") for n in defs):
        return
    store = self.__dict__.setdefault("_tw_defs", {})
    for n in defs:
        src = _twast.get_source_segment(code, n)
        if src:
            store[n.name] = src


def _tw_run(self, state_path, arguments):
    try:
        key = str(getattr(state_path, "parent", state_path))
        if self.__dict__.get("_tw_game") != key:           # новая игра — забыть прежние функции
            self._tw_game = key; self._tw_defs = {}
        code = str((arguments or {}).get("code", ""))
        prefix = _TW_HELPERS + "\n" + "\n\n".join(self.__dict__.get("_tw_defs", {}).values()) + "\n"
        res = _tw_orig_run(self, state_path, dict(arguments or {}, code=prefix + code))
        _tw_store(self, code)
        if ("validate(" in code) or ("plan(" in code):
            print("TWIN: вызов с %s | сохранено функций: %s" % (
                ",".join(k for k in ("validate", "plan") if k + "(" in code),
                ",".join(sorted(self.__dict__.get("_tw_defs", {})))), file=_twsys.__stderr__, flush=True)
        return res
    except Exception as exc:
        print("TWIN: сбой слоя, вызов без него: %r" % (exc,), file=_twsys.__stderr__, flush=True)
        return _tw_orig_run(self, state_path, arguments)


_twta._build_system_prompt = _tw_build_system_prompt
_twta.ToolAgent._run_python_tool = _tw_run
print("TWIN: слой включён — помощники %d строк, блок +%d знаков, поверх датасета v3" % (
    len(_TW_HELPERS.splitlines()), len(_TW_BLOCK)), flush=True)
'''
CELL = CELL_TEMPLATE.replace("__HELPERS_REPR__", repr(HELPERS))


def main() -> None:
    src = json.load(open("kernels/notebooks_nextfork/submission.ipynb", encoding="utf-8"))
    nb = json.loads(json.dumps(src)); cell = nb["cells"][15]; body = "".join(cell["source"])
    anchor = "    seconds=budget - 600.0\n)"; at = body.find(anchor)
    if at < 0 or body.find("await bm.run(") < 0:
        raise SystemExit("не нашёл стоковый soft_end или запуск прогона — сборка остановлена")
    at += len(anchor); code = body[:at] + "\n" + CELL + body[at:]; cell["source"] = code.splitlines(keepends=True)
    out = "kernels/notebooks_nextfork_twin"; os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(CELL)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_nextfork/kernel-metadata.json"))
    meta["id"] = "sergueimakarov/arc3-nextfork-twin"; meta["title"] = "arc3 nextfork twin"
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    compile(code, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    print("ok   патч ДО запуска прогона:", code.find("_tw_run") < code.find("await bm.run("))
    print("собрано:", out, "| датасет", meta["dataset_sources"][0])


if __name__ == "__main__":
    main()

"""Проверки ворот цели для Duck на НАСТОЯЩИХ классах бандла (без модели, без движка).

Проверяется то, что может сломаться и чего не видно глазами:
  * помощники, подставляемые в код модели, исполняются в песочнице-подобном окружении: без цели
    пробы <= 2 ходов проходят (и только 4), длинная пачка блокируется; set_goal с исполняемым
    измерителем открывает action(); измеритель считается до/после; после 3 пачек без роста
    цель опровергнута и action() закрыт; та же цель дословно отвергается;
  * обёртка _run_python_tool разбирает маркеры из stdout, переносит состояние на агент, вырезает
    маркеры из ответа инструмента, считает статистику;
  * обёртка _build_user_prompt ставит протокол и статус ПЕРЕД стоковым текстом и не меняет его;
  * смена уровня подтверждает цель и сбрасывает состояние;
  * боевая ветка (TRUE_SUBMISSION) не тронута; ячейка компилируется.

usage:  .venv/bin/python scripts/test_goal_duck_patch.py --bundle <путь к бандлу keithtyser>
"""
import argparse
import ast
import json
import sys
import types
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True)
ap.add_argument("--cell", default="kernels/notebooks_stockflash_goal/cell15.py")
a = ap.parse_args()

SRC = Path(a.bundle) / "src" / "ARC3-Inference"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(a.bundle) / "src" / "tufa-arc-agi-framework" / "src"))
import inference.agent.tool_agent as wta  # noqa: E402

fails = []


def check(b, m):
    print(("ok   " if b else "СБОЙ ") + m)
    if not b:
        fails.append(m)


cell = open(a.cell, encoding="utf-8").read()
compile(cell, "cell15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
check(True, "ячейка компилируется")

# --- боевая ветка не тронута
orig_prompt, orig_run = wta.ToolAgent._build_user_prompt, wta.ToolAgent._run_python_tool
ns = {"TRUE_SUBMISSION": True}
exec(cell, ns)
check(wta.ToolAgent._build_user_prompt is orig_prompt and wta.ToolAgent._run_python_tool is orig_run,
      "TRUE_SUBMISSION=True: классы Duck не тронуты")

# --- оффлайн: обёртки встали
ns = {"TRUE_SUBMISSION": False}
exec(cell, ns)
check(wta.ToolAgent._build_user_prompt is not orig_prompt and wta.ToolAgent._run_python_tool is not orig_run,
      "TRUE_SUBMISSION=False: обе обёртки установлены")
HELPERS = ns["_GOAL_HELPERS"]


# --- помощники в окружении, похожем на песочницу Duck
class Frame:
    def __init__(self, col, level=0):
        self.grid = [[0] * 64 for _ in range(64)]; self.grid[5][col] = 5; self.level = level
        self.ascii = "\n".join("".join("%x" % v for v in row) for row in self.grid)


def make_env(state):
    env = {"col": 0, "level": 0, "calls": []}
    g = {"__builtins__": __builtins__}
    g["current_frame"] = Frame(0)

    def action(actions):
        acts = actions if isinstance(actions, list) else [actions]
        env["calls"].append(list(acts))
        for act in acts:
            if act == "RIGHT":
                env["col"] += 1
            if env["col"] >= 10 and env["level"] == 0:
                env["level"] = 1; env["col"] = 0
        g["current_frame"] = Frame(env["col"], env["level"])
        return {"executed": True, "level": env["level"], "board_changed": True}
    g["action"] = action
    exec(HELPERS % {"state": repr(state), "patience": 3, "probe_len": 2, "probe_batches": 4}, g)
    return env, g


import io, contextlib  # noqa: E402


def run(g, code):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            exec(code, g); err = None
        except Exception as e:  # noqa: BLE001
            err = repr(e)
    return buf.getvalue(), err


empty = {"text": "", "code": "", "closed": None, "values": [], "no_progress": 0, "probes_used": 0, "falsified_texts": []}
env, g = make_env(empty)
out, err = run(g, "action(['RIGHT'])")
check(err is None and "[[GOAL_PROBE]]" in out, "проба из 1 хода без цели проходит и печатает маркер пробы")
out, err = run(g, "action(['RIGHT','RIGHT','RIGHT'])")
check(err and "GOAL GATE: action() blocked" in err, "пачка из 3 без цели блокируется")
for _ in range(3):
    out, err = run(g, "action(['UP'])")
check(err is None, "пробы 2-4 проходят")
out, err = run(g, "action(['UP'])")
check(err and "blocked" in err and "4 of 4" in err, "пятая проба без цели блокируется (4 of 4)")
out, err = run(g, "set_goal('x', 'def progress(frame): return 0')")
check(err and "checkable condition" in err, "короткий текст цели отвергнут")
out, err = run(g, "set_goal('move the 5-cell to column 10', lambda f: 0)")
check(err and "SOURCE STRING" in err, "измеритель не строкой отвергнут")
out, err = run(g, "set_goal('move the 5-cell to column 10', 'def progress(frame): return frame.grid[5].index(5)')")
check(err is None and "[[GOAL_SET]]" in out, "цель с измерителем принята, маркер напечатан")
state_after_set = json.loads(out.split("[[GOAL_SET]] ")[1].splitlines()[0])
check(state_after_set["value"] == 1.0, "значение измерителя при регистрации = 1 (после одной пробы RIGHT)")
out, err = run(g, "r = action(['RIGHT','RIGHT']); print(r['goal_progress'])")
check(err is None and "'delta': 2.0" in out and "[[GOAL_BATCH]]" in out, "после пачки delta=2, маркер пачки напечатан")
for _ in range(3):
    out, err = run(g, "r = action(['UP','UP']); print(r['goal_progress'])")
check("'falsified': True" in out, "3 пачки без роста -> цель опровергнута")
out, err = run(g, "action(['RIGHT'])")
check(err and "FALSIFIED" in err, "после опровержения action() закрыт")

# --- состояние переносится литералом: новый вызов с falsified_texts отвергает ту же цель
env2, g2 = make_env({**empty, "falsified_texts": ["move the 5-cell to column 10"]})
out, err = run(g2, "set_goal('move the 5-cell to column 10', 'def progress(frame): return frame.grid[5].index(5)')")
check(err and "already FALSIFIED" in err, "та же цель дословно отвергается в новом вызове (перенос через литерал)")
out, err = run(g2, "set_goal('reach column 10 with the 5-cell', 'def progress(frame): return frame.grid[5].index(5)')")
check(err is None, "новая формулировка принята")

# --- обёртка _run_python_tool: разбор маркеров, перенос на агент, вырезание, статистика
agent = types.SimpleNamespace()
captured = {}


def fake_run(self, state_path, arguments):
    captured["code"] = arguments["code"]
    return wta._ToolDispatchResult(content='[[GOAL_SET]] {"text": "reach col 10", "code": "def progress(frame): return 1", "value": 1.0}\n'
                                           'hello\n[[GOAL_BATCH]] {"after": 1.0, "no_progress": 1, "falsified": false}\n', step_executed=True)


ns["_g_orig_run"] = fake_run
out = wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "print(1)"})
check("_GOAL = {" in captured["code"] and captured["code"].rstrip().endswith("print(1)"), "помощники подставлены перед кодом модели")
st = agent._goal_state
check(st["text"] == "reach col 10" and st["values"] == [1.0, 1.0] and st["no_progress"] == 1, "состояние цели перенесено на агент из маркеров")
check("[[GOAL_" not in out.content and "hello" in out.content, "маркеры вырезаны из ответа инструмента, остальное на месте")
check(ns["_goal_stats"]["set"] == 1 and ns["_goal_stats"]["batches"] == 1, "статистика: set=1, batches=1")
captured.clear()
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "print(2)"})
lit = ast.literal_eval(captured["code"].split("_GOAL = ", 1)[1].splitlines()[0])
check(lit["text"] == "reach col 10" and lit["values"] == [1.0, 1.0], "на следующий вызов состояние подставлено литералом")


def fake_run_fals(self, state_path, arguments):
    return wta._ToolDispatchResult(content='[[GOAL_BATCH]] {"after": 1.0, "no_progress": 3, "falsified": true}\n', step_executed=True)


ns["_g_orig_run"] = fake_run_fals
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "action(['UP'])"})
check(st["closed"] == "falsified" and st["falsified_texts"] == ["reach col 10"] and ns["_goal_stats"]["falsified"] == 1,
      "опровержение перенесено на агент, текст запомнен")

# --- обёртка промпта: протокол + статус перед стоковым текстом; смена уровня сбрасывает цель
ns["_g_orig_prompt"] = lambda self, action_num, *a, **k: "STOCK PROMPT"
f1 = types.SimpleNamespace(level=0)
text = wta.ToolAgent._build_user_prompt(agent, 5, current_frame=f1)
check(text.endswith("STOCK PROMPT") and text.startswith("GOAL GATE (mandatory)") and "FALSIFIED" in text,
      "промпт: протокол и статус (опровергнута) перед стоковым текстом")
f2 = types.SimpleNamespace(level=1)
text = wta.ToolAgent._build_user_prompt(agent, 6, current_frame=f2)
check(st["text"] == "" and st["closed"] is None and st["falsified_texts"] == [] and "no goal registered" in text,
      "смена уровня: состояние сброшено, статус «цели нет»")
check(ns["_goal_stats"]["levels"] == 1, "уровень засчитан в статистике")

wta.ToolAgent._build_user_prompt, wta.ToolAgent._run_python_tool = orig_prompt, orig_run
print("\nИТОГ: %d сбоев" % len(fails) if fails else "\nВСЕ ПРОВЕРКИ ПРОШЛИ")
sys.exit(1 if fails else 0)

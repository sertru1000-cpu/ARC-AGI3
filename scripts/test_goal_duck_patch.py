"""Проверки ворот цели для Duck на НАСТОЯЩИХ классах бандла (без модели, без движка).

Проверяется то, что может сломаться и чего не видно глазами:
  * помощники в песочнице БЕЗ exec (в песочнице Duck его нет): без цели пробы <= 2 ходов проходят
    (и только 4), длинная пачка блокируется; set_goal печатает маркер и открывает action(); после
    опровержения action() закрыт; та же цель дословно отвергается;
  * обёртка песочницы берёт маркеры из СЫРОГО stdout (в ответе они уже в JSON) и вырезает их;
  * обёртка _run_python_tool исполняет измеритель В ОБВЯЗКЕ на текущем кадре: регистрация с
    негодным измерителем отклоняется с пояснением; после вызова с ходами значение дописывается,
    3 вызова без роста -> опровержение, текст запомнен; статистика;
  * обёртка _build_user_prompt ставит протокол и статус ПЕРЕД стоковым текстом; смена уровня
    подтверждает цель и сбрасывает состояние;
  * боевая ветка (TRUE_SUBMISSION) не тронута; ячейка компилируется.

usage:  .venv/bin/python scripts/test_goal_duck_patch.py --bundle <путь к бандлу keithtyser>
"""
import argparse
import ast
import contextlib
import io
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

orig_prompt, orig_run, orig_sandbox = wta.ToolAgent._build_user_prompt, wta.ToolAgent._run_python_tool, wta.run_sandboxed_python
ns = {"TRUE_SUBMISSION": True}
exec(cell, ns)
check(wta.ToolAgent._build_user_prompt is orig_prompt and wta.ToolAgent._run_python_tool is orig_run
      and wta.run_sandboxed_python is orig_sandbox, "TRUE_SUBMISSION=True: классы и песочница Duck не тронуты")

ns = {"TRUE_SUBMISSION": False}
exec(cell, ns)
check(wta.ToolAgent._build_user_prompt is not orig_prompt and wta.ToolAgent._run_python_tool is not orig_run
      and wta.run_sandboxed_python is not orig_sandbox, "TRUE_SUBMISSION=False: три обёртки установлены")
HELPERS = ns["_GOAL_HELPERS"]
check("exec(" not in HELPERS and "eval(" not in HELPERS and "compile(" not in HELPERS, "помощники не используют exec/eval/compile (их нет в песочнице)")


# --- помощники в окружении, похожем на песочницу Duck (без exec в builtins)
class Frame:
    def __init__(self, col, level=0):
        self.grid = [[0] * 64 for _ in range(64)]; self.grid[5][col] = 5; self.level = level
        self.ascii = "\n".join("".join("%x" % v for v in row) for row in self.grid)


import builtins as _b  # noqa: E402


def make_env(state):
    env = {"col": 0, "level": 0}
    safe = {k: getattr(_b, k) for k in ("isinstance", "len", "str", "print", "ValueError", "RuntimeError", "dict", "list", "float", "int", "__import__")}
    g = {"__builtins__": safe, "current_frame": Frame(0)}

    def action(actions):
        acts = actions if isinstance(actions, list) else [actions]
        for act in acts:
            if act == "RIGHT":
                env["col"] += 1
        g["current_frame"] = Frame(env["col"], env["level"])
        return {"executed": True, "level": env["level"], "board_changed": True}
    g["action"] = action
    exec(HELPERS % {"state": repr(state), "patience": 3, "probe_len": 2, "probe_batches": 4}, g)
    return env, g


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
out, err = run(g, "r = set_goal('move the 5-cell to column 10', 'def progress(frame): return frame.grid[5].index(5)'); print(r['ok'])")
check(err is None and "[[GOAL_SET]]" in out, "цель принята в песочнице, маркер напечатан")
out, err = run(g, "r = action(['RIGHT','RIGHT']); print(r['goal_note'])")
check(err is None and "[[GOAL_BATCH]]" in out and "measured by the harness" in out, "после регистрации пачка проходит, маркер пачки напечатан")
env2, g2 = make_env({**empty, "text": "reach col 10", "code": "def progress(frame): return 1", "closed": "falsified", "values": [1, 1, 1, 1]})
out, err = run(g2, "action(['RIGHT'])")
check(err and "FALSIFIED" in err, "с опровергнутой целью в состоянии action() закрыт")
env3, g3 = make_env({**empty, "falsified_texts": ["move the 5-cell to column 10"]})
out, err = run(g3, "set_goal('move the 5-cell to column 10', 'def progress(frame): return 0')")
check(err and "already FALSIFIED" in err, "та же цель дословно отвергается (перенос через литерал)")

# --- обёртки обвязки: маркеры из сырого stdout; измеритель исполняется в обвязке
agent = types.SimpleNamespace()
captured = {}
fake_stdout = {"text": ""}
frame_now = {"f": Frame(1)}


def fake_sandbox(**kw):
    captured["code"] = kw["code"]
    return {"stdout": fake_stdout["text"], "action_results": []}


def fake_run(self, state_path, arguments):
    res = wta.run_sandboxed_python(code=arguments["code"], timeout_seconds=1, initial_state={}, action_handler=None)
    return wta._ToolDispatchResult(json.dumps({"tool": "python", "stdout": res["stdout"]}, indent=2), step_executed=True)


ns["_g_orig_sandbox"] = fake_sandbox
ns["_g_orig_run"] = fake_run
wta.load_runtime_state = lambda p: (frame_now["f"], [])
wta._ascii_frame_view_payload = lambda f: {"grid": f.grid, "ascii": f.ascii, "level": f.level, "step": 0, "shape": [64, 64]}

fake_stdout["text"] = '[[GOAL_SET]] {"text": "reach col 10", "code": "def progress(frame): return frame.grid[5].index(5)"}\nhello\n'
out = wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "print(1)"})
st = agent._goal_state
check("_GOAL = {" in captured["code"] and captured["code"].rstrip().endswith("print(1)"), "помощники подставлены перед кодом модели")
check(st["text"] == "reach col 10" and st["values"] == [1.0], "регистрация: измеритель исполнен обвязкой на текущем кадре, значение 1")
check("[[GOAL_" not in out.content and "hello" in out.content, "маркеры вырезаны до JSON-обёртки, остальное на месте")

fake_stdout["text"] = '[[GOAL_SET]] {"text": "bad measure goal", "code": "def progress(frame): return frame.nope"}\n'
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "x"})
check(st["text"] == "" and "failed on the current board" in st["rejected"], "негодный измеритель отклонён с пояснением, цели нет")
check("REJECTED" in ns["_g_status"](st), "статус показывает причину отклонения")

fake_stdout["text"] = '[[GOAL_SET]] {"text": "reach col 10", "code": "def progress(frame): return frame.grid[5].index(5)"}\n'
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "x"})
frame_now["f"] = Frame(4)
fake_stdout["text"] = "[[GOAL_BATCH]] {\"n\": 3}\n"
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "action(['RIGHT']*3)"})
check(st["values"] == [1.0, 4.0] and st["no_progress"] == 0, "после вызова с ходами значение измерено обвязкой: 1 -> 4, рост")
for _ in range(3):
    wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "action(['UP'])"})
check(st["closed"] == "falsified" and st["falsified_texts"] == ["reach col 10"] and st["no_progress"] == 3,
      "3 вызова с ходами без роста -> опровергнута, текст запомнен")
lit = ast.literal_eval(captured["code"].split("_GOAL = ", 1)[1].splitlines()[0])
check(lit["text"] == "reach col 10" and lit["values"][-1] == 4.0, "состояние подставляется в код литералом")
check(ns["_goal_stats"]["set"] == 2 and ns["_goal_stats"]["falsified"] == 1 and ns["_goal_stats"]["batches"] == 4 and ns["_goal_stats"].get("rejected") == 1,
      "статистика: set=2, rejected=1, batches=4, falsified=1")

# --- ступени после опровержения: требование в этом вызове, отказ исполнять осмотр, сброс в пробы
st["closed"] = "falsified"; st["since_fals"] = 0; st["fals_codes"] = ["def progress(frame): return frame.grid[5].index(5)"]
check("MUST call set_goal" in ns["_g_status"](st) and "IN THIS CODE BLOCK" in ns["_g_status"](st), "ступень 1: статус требует set_goal в этом вызове")
fake_stdout["text"] = "inspect\n"
r1 = wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "print(current_frame.ascii)"})
r2 = wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "print(current_frame.ascii)"})
check(r1.step_executed and r2.step_executed and st["since_fals"] == 2, "первые 2 вызова без новой цели исполняются (счётчик 2)")
r3 = wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "print(current_frame.ascii)"})
check((not r3.step_executed) and "not executed" in r3.content and ns["_goal_stats"]["inspect_refused"] == 1, "ступень 2: третий вызов без set_goal не исполняется")
for _ in range(3):
    r = wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "print(1)"})
check(st["closed"] is None and st["text"] == "" and st["probes_used"] == 0 and ns["_goal_stats"]["gate_reset"] == 1 and r.step_executed,
      "ступень 3: после 5 отказов ворота сброшены в режим проб, вызов исполнен")
# А: другой измеритель — сравнение с опровергнутыми в обвязке
st["fals_codes"] = ["def progress(frame): return frame.grid[5].index(5)"]
fake_stdout["text"] = '[[GOAL_SET]] {"text": "reach col 10 again", "code": "def progress(frame): return frame.grid[5].index(5) + 0"}\n'
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "set_goal(1,2)"})
check(st["text"] == "" and "same measure under a new name" in st["rejected"] and ns["_goal_stats"]["same_measure"] == 1,
      "А: измеритель с тем же числом на текущей доске отвергнут")
fake_stdout["text"] = '[[GOAL_SET]] {"text": "count filled holes", "code": "def progress(frame): return 100 + frame.grid[5].index(5)"}\n'
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "set_goal(1,2)"})
check(st["text"] == "count filled holes" and st["values"] == [104.0], "А: измеритель с другим числом принят")
st["closed"] = "falsified"; st["falsified_texts"] = ["reach col 10"]

# --- обёртка промпта
ns["_g_orig_prompt"] = lambda self, action_num, *a, **k: "STOCK PROMPT"
text = wta.ToolAgent._build_user_prompt(agent, 5, current_frame=types.SimpleNamespace(level=0))
check(text.endswith("STOCK PROMPT") and text.startswith("GOAL GATE (mandatory)") and "FALSIFIED" in text,
      "промпт: протокол и статус (опровергнута) перед стоковым текстом")
text = wta.ToolAgent._build_user_prompt(agent, 6, current_frame=types.SimpleNamespace(level=1))
check(st["text"] == "" and st["closed"] is None and st["falsified_texts"] == [] and "no goal registered" in text,
      "смена уровня: состояние сброшено, статус «цели нет»")
check(ns["_goal_stats"]["levels"] == 1, "уровень засчитан в статистике")

wta.ToolAgent._build_user_prompt, wta.ToolAgent._run_python_tool, wta.run_sandboxed_python = orig_prompt, orig_run, orig_sandbox
print("\nИТОГ: %d сбоев" % len(fails) if fails else "\nВСЕ ПРОВЕРКИ ПРОШЛИ")
sys.exit(1 if fails else 0)

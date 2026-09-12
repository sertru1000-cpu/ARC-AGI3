"""Проверка сборки 2б (перенос импортов и def между вызовами) на НАСТОЯЩЕЙ песочнице стокового бандла.

Исходный `_run_python_tool` заменён верной копией его ядра (синтаксис, запуск `run_sandboxed_python`
бандла, сборка полезной нагрузки с error/stdout, `_ToolDispatchResult`), чтобы не поднимать весь агент;
песочница — настоящая, со своим белым списком. Проверяется: перенос проходит белый список; номера строк
в ошибке возвращаются к коду модели; повторного запуска после исполненных действий нет никогда;
сломанная подстановка сбрасывается и вызов повторяется без неё; присваивания, классы и функции с
вычисляемыми значениями по умолчанию не переносятся (классов в песочнице нет вовсе — нет __build_class__); системная фраза заменена; список имён дописан.
"""
import importlib.util, json, sys, types
from pathlib import Path

B = Path("/tmp/duckbundle/src")
sys.path.insert(0, str(B / "ARC3-Inference")); sys.path.insert(0, str(B / "tufa-arc-agi-framework" / "src"))
import inference.agent.tool_agent as ta  # noqa: E402
import inference.agent.python_tool_sandbox as pts  # noqa: E402

ok = lambda b, m: print(("ok   " if b else "СБОЙ ") + m)
CALLS = []
ACTED = []
FRAME = {"ascii": "00\n00", "step": 1, "level": 1}
ST = {"current_frame": FRAME, "history": [{"action": "", "frame": FRAME}], "valid_actions": ["UP"], "last_action_result": {}}

def handler(actions):
    ACTED.extend(actions)
    res = {"executed": True, "action_num": 2, "level": 1, "board_changed": True}
    return {"action_result": res, "state": ST}

def fake_run(self, state_path, arguments):
    """Ядро настоящего _run_python_tool: синтаксис -> песочница -> полезная нагрузка."""
    code = str(arguments.get("code", "")).rstrip()
    CALLS.append(code)
    try:
        compile(code, "<python_tool>", "exec")
    except SyntaxError as exc:
        return ta._ToolDispatchResult(json.dumps({"error": f"Python syntax error: {exc}"}, indent=2))
    r = pts.run_sandboxed_python(code=code, timeout_seconds=30, initial_state=ST, action_handler=handler)
    acts = [x for x in r.get("action_results") or [] if isinstance(x, dict)]
    payload = {"tool": "python"}
    if r.get("error"):
        payload["error"] = str(r["error"])
        if r.get("stdout"): payload["stdout"] = r["stdout"]
    else:
        payload["returncode"] = 0
        if r.get("stdout"): payload["stdout"] = r["stdout"]
    return ta._ToolDispatchResult(json.dumps(payload, indent=2), step_executed=any(a.get("executed") for a in acts))

ta.ToolAgent._run_python_tool = fake_run
ta.ToolAgent._build_user_prompt = lambda self, action_num, **kw: "БАЗОВЫЙ ПРОМПТ"
ns = {"TRUE_SUBMISSION": False, "bm": types.SimpleNamespace(solver=types.SimpleNamespace(max_runtime_s_per_game=7920.0)),
      "print": lambda *a, **k: None}
exec(compile(Path("kernels/notebooks_stockflash_carry/cell15.py").read_text(encoding="utf-8"), "cell15", "exec"), ns)

ag = object.__new__(ta.ToolAgent)
run = lambda code: ta.ToolAgent._run_python_tool(ag, Path("/tmp/state.json"), {"code": code})

r1 = run("from collections import Counter\nimport math\ndef st(x):\n    return x + 1\ndef bad(g=current_frame):\n    return g\na = 5\nprint(st(1), Counter('aab')['a'], math.floor(2.5))")
ok('"error"' not in r1.content and "2 2 2" in r1.content, "вызов 1 прошёл в настоящей песочнице")
labels = [lab for _, lab in ag._carry_store.values()]
ok(set(labels) == {"Counter", "math", "st(x)"}, "перенесены только импорты и простой def: %s" % labels)

r2 = run("print(st(41), Counter('xyzz')['z'], math.sqrt(16))")
ok('"error"' not in r2.content and "42 2 4.0" in r2.content, "вызов 2: перенесённые Counter/math/st работают без переопределения")
ok(CALLS[-1].startswith("from collections import Counter"), "подстановка стоит в начале кода")

r3 = run("x = 1\ny = undefined_name\nprint(x)")
ok('"error"' in r3.content, "вызов 3: ошибка модели возвращается как ошибка")
ok('line 2, in <module>' in r3.content, "номер строки возвращён к коду модели (строка 2)")

run("def st(x):\n    return x * 10\nprint(st(2))")
r5 = run("print(st(3))")
ok("30" in r5.content and sum(1 for k in ag._carry_store if k == "def:st") == 1, "новое определение с тем же именем заменило старое")

sys_text = ta._build_system_prompt(tool_output_tokens=1024)
ok("starts fresh" not in sys_text and "automatically re-loaded" in sys_text, "системная фраза о свежем старте заменена")
up = ta.ToolAgent._build_user_prompt(ag, 0, valid_actions=["UP"], current_frame=None, history_entries=[], previous_step_summary=None)
ok("RE-LOADED in python" in up and "st(x)" in up, "в промпт дописан список перенесённых имён")

# сломанная подстановка: ошибка ДО кода модели, действий нет -> сброс и повтор без подстановки
ag._carry_store["import:bad"] = ("import os", "os")
n0 = len(CALLS)
r6 = run("print('after-reset')")
ok("after-reset" in r6.content and '"error"' not in r6.content, "сломанная подстановка: сброс и чистый повтор прошёл")
ok(len(CALLS) - n0 == 2 and not CALLS[-1].startswith("import"), "исходный метод вызван дважды, второй раз без подстановки")
ok(ns["_CARRY_STATS"]["prefix_errors"] == 1, "ошибка подстановки посчитана")

# действия исполнены, потом ошибка -> повторного запуска НЕТ
run("from collections import deque\nprint(1)")
ag._carry_store["import:bad2"] = ("import os", "os")   # подстановка упадёт — но пусть код успеет походить? нет: упадёт до кода
ag._carry_store.pop("import:bad2")
ACTED.clear(); n1 = len(CALLS)
r7 = run("action(['UP'])\nboom = undefined_after_action")
ok(len(CALLS) - n1 == 1, "после исполненного действия и ошибки повторного запуска нет")
ok(ACTED == ['UP'] or len(ACTED) == 1, "действие исполнено ровно один раз: %s" % ACTED)

n2 = len(CALLS)
r8 = run("def broken(:\n  pass")
ok(len(CALLS) - n2 == 1 and not CALLS[-1].startswith("from collections"), "синтаксическая ошибка модели: вызов без подстановки")
ok(ns["_CARRY_STATS"]["fail"] == 0 and ns["_CARRY_STATS"]["sys_fail"] == 0, "сбоев патча: %s" % ns["_CARRY_STATS"])

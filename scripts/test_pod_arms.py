"""Проверка ячеек вариантов a3/a4/a5 поверх AGENTFIX Скотта в НАСТОЯЩЕЙ песочнице (27.09), без модели."""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "reference/keith-upstream/src"
sys.path[:0] = [str(B / "ARC3-Inference"), str(B / "tufa-arc-agi-framework/src")]
os.environ.update(MULTIMODAL_UPSCALE="4", MULTIMODAL_CONTEXT="current_grid")
exec(open("/tmp/scott_cell10.py").read(), {"__name__": "scott"})
import inference.agent.tool_agent as ta
from inference.agent.runtime_state import RUNTIME_STATE_FILENAME
nb = {n: json.load(open(ROOT / f"pod_arms/{n}.ipynb")) for n in ("a3_noreason", "a4_persist", "a5_helpers")}
cell = lambda n: "".join(next(c for c in nb[n]["cells"] if ("ARM " + n.split("_")[0]) in "".join(c["source"]))["source"])
TMP = ROOT / ".tmp_podarms"; TMP.mkdir(exist_ok=True); STATE = TMP / RUNTIME_STATE_FILENAME
STATE.write_text(json.dumps({"current_frame": {"grid": [[0, 1, 1], [0, 0, 2], [3, 3, 3]], "step": 0, "level": 1}, "history": []}))
def agent():
    a = ta.ToolAgent.__new__(ta.ToolAgent)
    a.__dict__.update(_session_runtime_dir=TMP, _history_messages=[], _session_total_tokens=0, _session_generated_tokens=0,
                      _last_step_summary=None, _last_action_result=None, _summarized_knowledge=ta._empty_world_model(),
                      _step_env_callback=lambda r: {}, _current_valid_actions=["UP"], _python_timeout=30.0,
                      _tool_output_chars=20000, _context_budget_tokens=10 ** 6)
    return a
ok = bad = 0
def check(c, what):
    global ok, bad
    print(("ok   " if c else "FAIL ") + what); ok += bool(c); bad += (not c)
run = lambda a, code: json.loads(a._run_python_tool(STATE, {"code": code}).content)
orig = {k: getattr(ta.ToolAgent, k) for k in ("_persistent_history_messages", "_run_python_tool", "_build_user_prompt")}
orig_sb = ta.run_sandboxed_python
# a3
exec(cell("a3_noreason"), {"__name__": "a3"})
a = agent()
hist = a._persistent_history_messages([{"role": "system", "content": "S"}, {"role": "user", "content": "u"},
    {"role": "assistant", "content": "x", "reasoning": "long thoughts", "tool_calls": []}, {"role": "user", "content": "u2"}], tools=None)
check(all("reasoning" not in m for m in hist) and any(m.get("role") == "assistant" for m in hist), "a3: рассуждение вынуто, ответ модели на месте")
for k, v in orig.items(): setattr(ta.ToolAgent, k, v)
# a4
exec(cell("a4_persist"), {"__name__": "a4"})
a = agent()
r1 = run(a, "def f(x):\n    return x * 2\ndef g():\n    return 7\nprint(f(3))")
r2 = run(a, "print(f(5), g())")
check(r1.get("stdout", "").strip() == "6" and r2.get("stdout", "").strip() == "10 7", "a4: функция и класс из прошлого вызова доступны (%r / %r)" % (r1, r2))
r0 = run(a, "class Z:\n    pass")
r3 = run(a, "def f(x):\n    return x + 100\nprint(f(1))")
r4 = run(a, "print(f(1))")
check(r4.get("stdout", "").strip() == "101", "a4: переопределение побеждает старое; попытка класса не отравила память (%r)" % r4)
check("Currently kept: f, g" in a._build_user_prompt(1, valid_actions=["UP"], current_frame=None, history_entries=[]) or "Currently kept: g, f" in a._build_user_prompt(1, valid_actions=["UP"], current_frame=None, history_entries=[]), "a4: список сохранённого в сообщении")
r5 = run(a, "from collections import deque, Counter\nimport typing\nTG = 5\ndef h():\n    return len(deque([1, 2])) + Counter('aa')['a'] + TG\nprint(h())")
r6 = run(a, "print(h())")
check(r6.get("stdout", "").strip() == "9", "a4 v2: функция с импортом и константой из прошлого вызова работает (%r)" % r6)
r7 = run(a, "print(1)")
check(r7.get("stdout", "").strip() == "1", "a4 v2: запрещённый импорт (typing) не сохранён и не ломает следующие вызовы (%r)" % r7)
a._session_runtime_dir = TMP; a._pm_game = "другая игра"
check("NameError" in str(run(a, "print(f(1))")), "a4: новая игра — память пустая")
for k, v in orig.items(): setattr(ta.ToolAgent, k, v)
# a5
exec(cell("a5_helpers"), {"__name__": "a5"})
a = agent()
r = run(a, "import difflib, time\nprint(len(objs()), summ().splitlines()[0])")
check("error" not in r and r.get("stdout", "").startswith("4 "), "a5: помощники работают, difflib/time разрешены (%r)" % (r.get("stdout") or r.get("error"))[:80])
check("Predefined in every python call" in a._build_user_prompt(1, valid_actions=["UP"], current_frame=None, history_entries=[]), "a5: строка о помощниках в сообщении")
print("\nитого: ок %d, провалов %d" % (ok, bad))

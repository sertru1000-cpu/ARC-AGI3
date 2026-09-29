"""a8b в настоящей песочнице: заготовка base_step, check_step, память функций, подсказка после 8 ходов на уровне (27.09)."""
import json, logging, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "reference/keith-upstream/src"
sys.path[:0] = [str(B / "ARC3-Inference"), str(B / "tufa-arc-agi-framework/src")]
os.environ.update(MULTIMODAL_UPSCALE="4", MULTIMODAL_CONTEXT="current_grid")
logging.disable(logging.ERROR)
exec(open("/tmp/scott_cell10.py").read(), {"__name__": "scott"})
import inference.agent.tool_agent as ta
from inference.agent.runtime_state import RUNTIME_STATE_FILENAME, Frame, HistoryEntry
nb = json.load(open(ROOT / "pod_arms/a8b_buildwm.ipynb"))
exec("".join(next(c for c in nb["cells"] if "ARM a8b" in "".join(c["source"]))["source"]), {"__name__": "a8"})
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
env = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT / "environment_files")).make("ls20-9607627b")
fr = env.reset(); g = lambda f: [[int(v) for v in r] for r in f.frame[-1]]
hist = [{"action": "", "frame": {"grid": g(fr), "step": 0, "level": 1}}]
names = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT"}
for i, a in enumerate(["ACTION4", "ACTION4", "ACTION1", "ACTION3"]):
    fr = env.step(GameAction[a]); hist.append({"action": names[a], "frame": {"grid": g(fr), "step": i + 1, "level": 1}})
TMP = ROOT / ".tmp_a8b"; TMP.mkdir(exist_ok=True); STATE = TMP / RUNTIME_STATE_FILENAME
STATE.write_text(json.dumps({"current_frame": hist[-1]["frame"], "history": hist}))
a = ta.ToolAgent.__new__(ta.ToolAgent)
a.__dict__.update(_session_runtime_dir=TMP, _history_messages=[], _session_total_tokens=0, _session_generated_tokens=0,
                  _last_step_summary=None, _last_action_result=None, _summarized_knowledge=ta._empty_world_model(),
                  _step_env_callback=lambda r: {}, _current_valid_actions=["UP"], _python_timeout=30.0, _tool_output_chars=20000)
run = lambda code: json.loads(a._run_python_tool(STATE, {"code": code}).content)
r1 = run("def step(rows, action):\n    return list(rows)\nprint(len(level_transitions()))\nbad = check_step(step)")
print("вызов 1 (step = «ничего не меняй»):", (r1.get("stdout") or r1.get("error"))[-300:])
r2 = run("print(check_step(step))")
print("вызов 2 (step из памяти):", (r2.get("stdout") or r2.get("error"))[-160:])
ok1 = "check_step: 0/4" in (r1.get("stdout") or "") and "cells wrong" in (r1.get("stdout") or "")
ok2 = "check_step: 0/4" in (r2.get("stdout") or "")
fr_ = lambda h: Frame(grid=tuple(tuple(r) for r in h["frame"]["grid"]), step=h["frame"]["step"], level=1)
short = [HistoryEntry(action=h["action"], frame=fr_(h)) for h in hist]
short = short + [HistoryEntry(action="UP", frame=short[-1].frame)] * 2          # 7 записей уровня — подсказки ещё нет
long_ = short + [HistoryEntry(action="UP", frame=short[-1].frame)]              # 8 — есть
p1 = a._build_user_prompt(7, valid_actions=["UP"], current_frame=short[-1].frame, history_entries=short)
p2 = a._build_user_prompt(8, valid_actions=["UP"], current_frame=long_[-1].frame, history_entries=long_)
ok3 = "base_step(rows, action) is predefined" not in p1 and "base_step(rows, action) is predefined" in p2
print("\n%s check_step видит 4 перехода уровня и печатает контрпримеры" % ("ok  " if ok1 else "FAIL"))
print("%s step сохранился и работает в следующем вызове" % ("ok  " if ok2 else "FAIL"))
print("%s подсказка с заготовкой появляется на 8-м ходу уровня, не на 7-м" % ("ok  " if ok3 else "FAIL"))
r3 = run("print(check_step(base_step))")
ok4 = "check_step: 4/4" in (r3.get("stdout") or "")
print("%s base_step воспроизводит все 4 увиденных перехода: %s" % ("ok  " if ok4 else "FAIL", (r3.get("stdout") or r3.get("error"))[-120:].strip()))
# непредвиденный переход: ещё один RIGHT с текущей доски (такой доски с таким ходом не было) — сверка с настоящим движком
fr2 = env.step(GameAction["ACTION4"]); want = hist[-1]["frame"]["grid"]
r4 = run("rows = current_frame.ascii.split(chr(10))\np = base_step(rows, 'RIGHT')\nprint('SAME' if p == rows else 'CHANGED')\nprint(chr(10).join(p))")
out = (r4.get("stdout") or r4.get("error") or "")
print("     непредвиденный RIGHT: base_step %s доску" % ("ИЗМЕНИЛ" if out.startswith("CHANGED") else "НЕ изменил (%s)" % out[:120]))
r5 = run("print(check_step(tpl_step))\nprint(tpl_step(current_frame.ascii.split(chr(10)), 'RIGHT') != current_frame.ascii.split(chr(10)))")
o5 = (r5.get("stdout") or "") + (r5.get("error") or "")
ok5 = "check_step: 4/4" in o5 and "not allowed" not in o5 and "Error" not in o5
print("%s tpl_step (шаблоны, numpy + классы в песочнице) воспроизводит 4/4 и предсказывает новый ход: %s" % ("ok  " if ok5 else "FAIL", o5[-160:].strip().replace(chr(10), " | ")))
exit(0 if (ok1 and ok2 and ok3 and ok4 and ok5) else 1)

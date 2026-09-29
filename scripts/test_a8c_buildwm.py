"""a8c в настоящей песочнице (a8b + планировщик plan): заготовка base_step, check_step, память функций, подсказка после 8 ходов на уровне (27.09)."""
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
nb = json.load(open(ROOT / "pod_arms/a8c_buildwm.ipynb"))
exec("".join(next(c for c in nb["cells"] if "ARM a8c" in "".join(c["source"]))["source"]), {"__name__": "a8"})
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
env = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT / "environment_files")).make("ls20-9607627b")
fr = env.reset(); g = lambda f: [[int(v) for v in r] for r in f.frame[-1]]
hist = [{"action": "", "frame": {"grid": g(fr), "step": 0, "level": 1}}]
names = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT"}
for i, a in enumerate(["ACTION4", "ACTION4", "ACTION1", "ACTION3"]):
    fr = env.step(GameAction[a]); hist.append({"action": names[a], "frame": {"grid": g(fr), "step": i + 1, "level": 1}})
TMP = ROOT / ".tmp_a8c"; TMP.mkdir(exist_ok=True); STATE = TMP / RUNTIME_STATE_FILENAME
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
ok1 = "check_step: 0/4" in (r1.get("stdout") or "") and "actually changed" in (r1.get("stdout") or "")
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
# a8c: план внутри симулятора. Цель — доска после RIGHT, RIGHT по самому симулятору: план обязан найти ровно 2 хода.
r6 = run("cur = current_frame.ascii.split(chr(10))\ntarget = tpl_step(tpl_step(cur, 'RIGHT'), 'RIGHT')\n"
         "p = plan(tpl_step, lambda rows: rows == target, actions=['UP', 'DOWN', 'LEFT', 'RIGHT'])\nprint('PLAN', p)")
o6 = (r6.get("stdout") or "") + (r6.get("error") or "")
ok6 = "PLAN ['RIGHT', 'RIGHT']" in o6 or ("PLAN [" in o6 and o6.count("'") == 4 and "RIGHT" in o6)
print("%s plan находит кратчайший путь к цели в симуляторе: %s" % ("ok  " if ok6 else "FAIL", o6.strip().replace(chr(10), " | ")[-200:]))
r7 = run("p = plan(lambda rows, a: rows, lambda rows: False, actions=['UP'], clicks=[(3, 4)])\nprint('PLAN', p)\n"
         "print(_pl_to_action('MOUSE(row=3, col=4)'))")
o7 = (r7.get("stdout") or "") + (r7.get("error") or "")
ok7 = "PLAN None" in o7 and "{'action': 'MOUSE', 'row': 3, 'col': 4}" in o7 and "недостижима" in o7
print("%s недостижимая цель -> None без зависания; клик в формате action(): %s" % ("ok  " if ok7 else "FAIL", o7.strip().replace(chr(10), " | ")[-200:]))
r8 = run("p = plan(tpl_step, lambda rows: False, actions=['UP', 'DOWN', 'LEFT', 'RIGHT'], seconds=12.0)")
o8 = (r8.get("stdout") or "") + (r8.get("error") or "")
ok8 = "plan:" in o8 and "Timeout" not in o8 and "timed out" not in o8.lower()
print("%s скорость поиска на настоящей доске 64x64 с tpl_step, предел 12 с (песочнице дано 30): %s" % ("ok  " if ok8 else "FAIL", o8.strip()[-160:]))
code9 = ("def mv(rows, a):\n"
         "    g = [list(r) for r in rows]\n"
         "    r0 = next(i for i, r in enumerate(g) if '@' in r); c0 = g[r0].index('@')\n"
         "    dr, dc = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}[a]\n"
         "    r1, c1 = r0 + dr, c0 + dc\n"
         "    if 0 <= r1 < len(g) and 0 <= c1 < len(g[0]):\n"
         "        g[r0][c0], g[r1][c1] = '.', '@'\n"
         "    return [''.join(r) for r in g]\n"
         "start = ['.' * 64 for _ in range(64)]; start[0] = '@' + '.' * 63\n"
         "cf = current_frame\n"
         "class _F: pass\n"
         "current_frame = _F(); current_frame.ascii = chr(10).join(start)\n"
         "p = plan(mv, lambda rows: rows[63][63] == '@', actions=['UP', 'DOWN', 'LEFT', 'RIGHT'])\n"
         "print('LEN', None if p is None else len(p))")
r9 = run(code9)
o9 = (r9.get("stdout") or "") + (r9.get("error") or "")
ok9 = "LEN 126" in o9
print("%s поле 64x64, простой step модели: кратчайший путь 126 ходов через 4096 состояний: %s" % ("ok  " if ok9 else "FAIL", o9.strip().replace(chr(10), " | ")[-170:]))
# ---- части Tycho на НАСТОЯЩЕМ движке: ход из песочницы уходит в ls20, состояние пишется обратно ----
NAMES = {"UP": "ACTION1", "DOWN": "ACTION2", "LEFT": "ACTION3", "RIGHT": "ACTION4"}
class RealEngine:
    def __init__(self):
        self.env = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT / "environment_files")).make("ls20-9607627b")
        f = self.env.reset(); self.hist = [{"action": "", "frame": {"grid": g(f), "step": 0, "level": 1}}]; self.seen = []
        for a in ["RIGHT", "RIGHT", "UP", "LEFT"]:
            self._do(a)
        self.seen = []
    def _do(self, name):
        f = self.env.step(GameAction[NAMES[name]]); lv = 1 + (f.levels_completed or 0)
        self.hist.append({"action": name, "frame": {"grid": g(f), "step": len(self.hist), "level": lv}})
        STATE.write_text(json.dumps({"current_frame": self.hist[-1]["frame"], "history": self.hist}))
        return f
    def __call__(self, request):
        name = request["actions"][0]["action"]; self.seen.append(name)
        before = self.hist[-1]["frame"]["grid"]; f = self._do(name)
        return {"executed": True, "action_num": len(self.hist) - 1, "level": self.hist[-1]["frame"]["level"], "score": 0,
                "reward": 0.0, "state": "NOT_FINISHED", "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT"],
                "board_changed": g(f) != before, "frame_count": 1, "done": False, "level_completed": False,
                "game_over": False, "run_complete": False, "action_display": name}
eng = RealEngine(); a._step_env_callback = eng; a._current_valid_actions = ["UP", "DOWN", "LEFT", "RIGHT"]
if hasattr(a, "_noop_guard"):
    a._noop_guard = None
r10 = run("n = run_plan(['RIGHT', 'RIGHT', 'RIGHT'], lambda rows, act: rows)\nprint('DONE', n)")
o10 = (r10.get("stdout") or "") + (r10.get("error") or "")
ok10 = "DONE 1" in o10 and "diverged" in o10 and eng.seen == ["RIGHT"]
print("%s run_plan с неверным step: остановился после 1-го хода (в движок ушло %s): %s" % ("ok  " if ok10 else "FAIL", eng.seen, o10.strip().replace(chr(10), " | ")[-220:]))
eng.seen = []
r11 = run("n = run_plan(['LEFT'], tpl_step)\nprint('DONE', n)")
o11 = (r11.get("stdout") or "") + (r11.get("error") or "")
ok11 = ("DONE 1" in o11) and eng.seen == ["LEFT"] and ("matched" in o11 or "diverged" in o11)
print("%s run_plan с tpl_step: ход ушёл в движок и сверен (в движок %s): %s" % ("ok  " if ok11 else "FAIL", eng.seen, o11.strip().replace(chr(10), " | ")[-200:]))
r12 = run("print('T', check_goal(lambda rows: True))\nprint('F', check_goal(lambda rows: False))")
o12 = (r12.get("stdout") or "") + (r12.get("error") or "")
ok12 = "T False" in o12 and "F True" in o12 and "must be 0" in o12
print("%s check_goal: «всегда True» отвергнута, «всегда False» принята: %s" % ("ok  " if ok12 else "FAIL", o12.strip().replace(chr(10), " | ")[-220:]))
exit(0 if (ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7 and ok8 and ok9 and ok10 and ok11 and ok12) else 1)

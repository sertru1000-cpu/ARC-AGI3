"""Предсказание перед ходом (NEXTFORK_PREDICT=1, inference/agent/predict.py) в настоящей песочнице на настоящем движке (30.09).
Ставится импортом solver.py, как в ноутбуке v4-lite; ходы идут в локальный движок ls20, состояние пишется в файл обвязки.
usage: .venv/bin/python scripts/test_nextfork_predict.py
"""
import json, logging, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "nextfork/src"
sys.path[:0] = [str(B / "ARC3-Inference"), str(B / "tufa-arc-agi-framework/src")]
os.environ.update(MULTIMODAL_UPSCALE="4", MULTIMODAL_CONTEXT="current_grid", NEXTFORK_PREDICT="1", NEXTFORK_NOOPGUARD="0")
logging.disable(logging.ERROR)
import inference.framework.solver  # noqa: F401,E402
import inference.agent.predict as pd  # noqa: E402
import inference.agent.tool_agent as ta  # noqa: E402
from inference.agent.noop_guard import NoopGuard  # noqa: E402
from inference.agent.runtime_state import RUNTIME_STATE_FILENAME  # noqa: E402
import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

env = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT / "environment_files")).make("ls20-9607627b")
fr = env.reset()
g = lambda f: [[int(v) for v in r] for r in f.frame[-1]]
hist = [{"action": "", "frame": {"grid": g(fr), "step": 0, "level": 1}}]
TMP = ROOT / ".tmp_predict"; TMP.mkdir(exist_ok=True); STATE = TMP / RUNTIME_STATE_FILENAME
STATE.write_text(json.dumps({"current_frame": hist[-1]["frame"], "history": hist}))
CODE = {"UP": "ACTION1", "DOWN": "ACTION2", "LEFT": "ACTION3", "RIGHT": "ACTION4"}
steps = {"n": 0}


def step_cb(req):
    last = None
    for a in req["actions"]:
        before = hist[-1]["frame"]["grid"]
        f = env.step(GameAction[CODE[a["action"]]])
        steps["n"] += 1
        lvl = 1 + int(getattr(f, "levels_completed", 0) or 0)
        hist.append({"action": a["action"], "frame": {"grid": g(f), "step": steps["n"], "level": lvl}})
        STATE.write_text(json.dumps({"current_frame": hist[-1]["frame"], "history": hist}))
        last = {"executed": True, "board_changed": g(f) != before, "level": lvl, "level_completed": lvl > hist[-2]["frame"]["level"],
                "game_over": str(getattr(f, "state", "")).endswith("GAME_OVER"), "done": False, "valid_actions": list(CODE),
                "action_name": a["action"]}
    return last


a = ta.ToolAgent.__new__(ta.ToolAgent)
a.__dict__.update(_session_runtime_dir=TMP, _history_messages=[], _session_total_tokens=0, _session_generated_tokens=0,
                  _last_step_summary=None, _last_action_result=None, _summarized_knowledge=ta._empty_world_model(),
                  _step_env_callback=step_cb, _current_valid_actions=list(CODE), _python_timeout=30.0,
                  _tool_output_chars=20000, _noop_guard=NoopGuard(), _noop_guard_blocked=0)
run = lambda code: json.loads(a._run_python_tool(STATE, {"code": code}).content)
ok = True


def check(cond, text, detail=""):
    global ok
    ok &= bool(cond)
    print("%s %s%s" % ("ok  " if cond else "FAIL", text, ("   | " + detail.replace("\n", " / ")[:300]) if detail else ""))


check("PREDICT-BEFORE-YOU-ACT" in ta._build_system_prompt(tool_output_tokens=4000), "правило и словарь — в системном промпте")
n0 = steps["n"]
r = run("action(['RIGHT'])")
check("REFUSED (no move spent)" in r.get("error", "") and steps["n"] == n0, "ход без predict отклонён, ход не потрачен", r.get("error", ""))

# найти игрока: объект, который двигает RIGHT (узнаём честно — ходом с расплывчатым предсказанием)
r = run("r = action(['RIGHT'], predict='change')")
out = r.get("stdout", "")
check("PREDICT ✓" in out and "cells changed" in out and steps["n"] == n0 + 1, "расплывчатое 'change' оценено ✓, рассказ о переходе есть", out)

r = run("action(['RIGHT'], predict='noop')")
out = r.get("stdout", "")
check("PREDICT ✗" in out and "MISS noop" in out, "ложное 'noop' оценено ✗ с разбором", out)

# точное предсказание движения: объект, сдвинувшийся на последнем RIGHT — по разнице кадров
before, after = hist[-2]["frame"]["grid"], hist[-1]["frame"]["grid"]
moved = sorted({(r_, c_) for r_ in range(64) for c_ in range(64) if before[r_][c_] != after[r_][c_]})
code_find = (
    "rows = current_frame.ascii.split(chr(10)); prev = previous_frame.ascii.split(chr(10))\n"
    "cells = [(r, c) for r in range(len(rows)) for c in range(len(rows[r])) if rows[r][c] != prev[r][c]]\n"
    "bg = _pd_background(rows)\n"
    "obj = [(r, c) for (r, c) in cells if rows[r][c] != bg]\n"
    "print('OBJ', obj[:3], len(cells))\n"
)
r = run(code_find)
print("     ", (r.get("stdout") or r.get("error"))[:200])
r = run("rows = current_frame.ascii.split(chr(10)); prev = previous_frame.ascii.split(chr(10))\n"
        "cells = [(r, c) for r in range(len(rows)) for c in range(len(rows[r])) if rows[r][c] != prev[r][c] and rows[r][c] != _pd_background(rows)]\n"
        "r0, c0 = cells[0]\n"
        "dc = 0\n"
        "for d in range(1, 9):\n"
        "    if 0 <= c0 - d and prev[r0][c0 - d] == rows[r0][c0]:\n"
        "        dc = d; break\n"
        "print('guess', r0, c0, dc)\n"
        "action(['RIGHT', 'RIGHT'], predict=['move %d,%d 0,%d' % (r0, c0, dc), 'move %d,%d 0,%d' % (r0, c0 + dc, dc)])")
out = r.get("stdout", "") + r.get("error", "")
check("PREDICT" in out and ("MISS" in out or "OK" in out), "пачка из двух ходов с предсказанием на каждый оценена по шагам", out)

n1 = steps["n"]
r = run("action(['LEFT', 'LEFT', 'LEFT'], predict=['noop', 'change', 'change'])")
out = r.get("stdout", "")
check("batch stopped after move 1 of 3" in out and steps["n"] == n1 + 1, "пачка остановилась на первом промахе, 2 хода не потрачены", out)

r = run("action(['UP', 'DOWN'], predict='change')")
check("LIST of 2 predictions" in r.get("error", ""), "пачка с одним предсказанием на всё — отказ с объяснением", r.get("error", ""))

r = run("action(['UP'], predict='teleport 3,4')")
check("claim not understood" in r.get("stdout", ""), "непонятное утверждение — ✗ со словарём форм", r.get("stdout", ""))

r = run("notes('Verified: RIGHT moves the player (move 3). Assumed: goal is the key. Plan: go right.')\nprint('done')")
check("notes saved" in r.get("stdout", "") and "@@NOTES@@" not in r.get("stdout", ""), "заметки сохранены, маркер из вывода убран", r.get("stdout", ""))
p = a._build_user_prompt(5, valid_actions=list(CODE), current_frame=None, history_entries=[])
check("YOUR NOTES" in p and "RIGHT moves the player" in p, "страница заметок показывается в сообщении модели")

r = run("def go():\n    return action(['DOWN'], predict='change')\n")
r = run("go()")
check("PREDICT" in r.get("stdout", "") + r.get("error", ""), "функции модели из памяти вызывают обёрнутый action", r.get("stdout", "") + r.get("error", ""))
print("\nстатистика слоя:", pd.STATS)
print("ИТОГ:", "все проверки пройдены" if ok else "ЕСТЬ ПРОВАЛЫ")
sys.exit(0 if ok else 1)

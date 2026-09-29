"""Ячейка twin против исходников v3 с Kaggle: настоящая песочница, синтетическая игра «блок ходит стрелками»."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, "/tmp/nf_kaggle/src/ARC3-Inference")
import inference.agent.tool_agent as TA
from inference.agent.runtime_state import RUNTIME_STATE_FILENAME
exec(compile(open(ROOT/"kernels/notebooks_nextfork_twin/cell15.py", encoding="utf-8").read(), "cell15", "exec"), {})
TMP = ROOT/".tmp_twin"; TMP.mkdir(exist_ok=True); STATE = TMP/RUNTIME_STATE_FILENAME
CH = ".WwgGcBMPRbSYOrNp"
def grid(r, c):
    g = [[0]*8 for _ in range(8)]; g[r][c] = 3; return g
def fr(g, step=0): return {"grid": g, "step": step, "level": 1, "shape": [8, 8], "ascii": "\n".join("".join(CH[v] for v in row) for row in g)}
hist = [{"action": "", "frame": fr(grid(2, 2))}, {"action": "DOWN", "frame": fr(grid(3, 2), 1)},
        {"action": "RIGHT", "frame": fr(grid(3, 3), 2)}, {"action": "LEFT", "frame": fr(grid(3, 2), 3)}]
STATE.write_text(json.dumps({"current_frame": fr(grid(3, 2), 3), "history": hist}))
a = TA.ToolAgent.__new__(TA.ToolAgent)
a._session_runtime_dir=None; a._history_messages=[]; a._session_total_tokens=0; a._session_generated_tokens=0; a._last_step_summary=None; a._last_action_result=None
a._summarized_knowledge=TA._empty_world_model(); a._noop_guard=TA.NoopGuard(); a._noop_guard_blocked=0; a._python_timeout=30.0; a._tool_output_chars=20000
a._current_valid_actions=["UP","DOWN","LEFT","RIGHT"]; sent=[]; a._step_env_callback=lambda req: (sent.append(req), {"executed": True, "board_changed": True})[1]; a._ensure_session(STATE)
def run(code):
    out = json.loads(a._run_python_tool(STATE, {"code": code}).content); return out.get("stdout", "") + ("\nERROR " + out["error"] if out.get("error") else "")
ok = True
def check(l, c):
    global ok; ok &= bool(c); print("%s %s" % ("ОК  " if c else "ПРОВАЛ", l))
BAD = '''
def step(grid, action):
    return grid
print("bad", validate(step))'''
o = run(BAD); check("1. неверный step: контрпримеры напечатаны, 1/3 воспроизведено (LEFT? нет — все двигают)", "counterexample" in o and "0/3" in o)
GOOD = '''
def find(g):
    for r, row in enumerate(g):
        for c, v in enumerate(row):
            if v == 3: return r, c
def step(grid, action):
    r, c = find(grid); dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}.get(action, (0, 0))
    nr, nc = r + dr, c + dc
    if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])): return grid
    grid[r][c] = 0; grid[nr][nc] = 3; return grid
def goal(grid):
    return find(grid) == (6, 6)
print("good", validate(step))'''
o = run(GOOD); check("2. верный step: 3/3 воспроизведено, 0 расхождений", "3/3" in o and "good 0" in o)
o = run("path = plan(step, goal)\nprint(path)\naction(to_actions(path))")
check("3. без повторного def: step/goal взяты из сохранённых, план найден и ушёл одним пакетом",
      "plan: found 7 moves" in o and len(sent) == 1 and len(sent[0]["actions"]) == 7)
print("   пакет:", [x["action"] for x in sent[0]["actions"]] if sent else None)
a._tw_game = "другая игра"; o = run("print(validate(step))")
check("4. новая игра: сохранённые функции забыты", "NameError" in o or "not defined" in o)
import shutil; shutil.rmtree(TMP, ignore_errors=True)
print("\nИТОГ:", "все проверки пройдены" if ok else "ЕСТЬ ПРОВАЛЫ"); sys.exit(0 if ok else 1)

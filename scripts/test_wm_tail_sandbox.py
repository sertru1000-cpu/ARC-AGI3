"""Хвост слоя «модель мира» — В ТОЧНОСТИ ТОТ, ЧТО ПОЕДЕТ В KAGGLE, — в настоящей песочнице бандла.

Проверка, которой не было у v9 и которая поймала бы её дефект за минуту:
хвост исполняется ПОСЛЕ кода модели, внутри белого списка имён песочницы.
"""
import ast, importlib.util, sys
from pathlib import Path
import os
NB = Path(os.environ.get("WM_NB", "kernels/notebooks_stockflash_wm10/submission.ipynb"))
ROOT = Path('/Users/sergeimakarov/Projects/ARC-AGI-3'); ARC3 = ROOT/"atlas_src/src/ARC3-Inference"
sys.path.insert(0, str(ARC3))
spec = importlib.util.spec_from_file_location("pts", ARC3/"inference/agent/python_tool_sandbox.py")
pts = importlib.util.module_from_spec(spec); spec.loader.exec_module(pts)

import json
cell = "".join("".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"]
                if c["cell_type"] == "code")
tree = ast.parse(cell)
TAIL = HELPERS = None
for n in tree.body:
    if isinstance(n, ast.Assign):
        name = getattr(n.targets[0], "id", "")
        if name == "_WM_TAIL": TAIL = ast.literal_eval(n.value)
        if name == "_WM_HELPERS": HELPERS = ast.literal_eval(n.value)
assert TAIL and HELPERS, "не найдены _WM_TAIL/_WM_HELPERS в собранной ячейке"

def grid(col):
    g = [[0]*8 for _ in range(4)]; g[1][col] = 9; return g
def fp(g, step, level=1):
    return {"ascii": "\n".join("".join(str(c) for c in r) for r in g), "step": step,
            "level": level, "shape": [4, 8], "grid": g}
hist = [{"action": "", "frame": fp(grid(1), 0)}]
for i, c in enumerate([2, 3, 4], 1): hist.append({"action": "RIGHT", "frame": fp(grid(c), i)})
ST = {"current_frame": hist[-1]["frame"], "history": hist, "valid_actions": ["RIGHT", "LEFT"],
      "last_action_result": {}}
ACTED = []
def handler(actions, level=1, col=5):
    ACTED.extend(actions)
    nh = list(hist) + [{"action": "RIGHT", "frame": fp(grid(col), 4, level)}]
    return {"ok": True, "executed": True,
            "state": {"current_frame": nh[-1]["frame"], "history": nh,
                      "valid_actions": ["RIGHT", "LEFT"], "last_action_result": {}}}
def go(code, h=handler):
    ACTED.clear()
    r = pts.run_sandboxed_python(code=HELPERS + "\n" + code + TAIL, timeout_seconds=30,
                                 initial_state=ST, action_handler=h)
    return (r.get("stdout") or "") + "\nERR:" + str(r.get("error") or "")

ok = lambda b, m: print(("ok   " if b else "СБОЙ ") + m)
MODEL = """
def state_of(f):
    row = f.ascii.split('\\n')[1]
    return (wm_level(f), row.index('9') if '9' in row else -1)
def predict(s, a):
    if a == 'RIGHT': return (s[0], s[1] + 1)
    if a == 'LEFT':  return (s[0], s[1] - 1)
action(['RIGHT'])
"""
out = go(MODEL)
ok("WM_CHECK admitted=1" in out and "is not defined" not in out,
   "модель без goal: проверка ОТРАБОТАЛА (это и есть дефект v9, теперь его нет)")
ok("total=4" in out, "проверка идёт по буферу ПОСЛЕ хода: 4 перехода, включая только что сделанный")
ok(len(ACTED) == 1 and "RIGHT" in str(ACTED[0]),
   f"ход исполнен, проверка действий не отняла (шлюз получил {ACTED})")
ok("reward_unchecked" not in out or "WM_REWARD" not in out, "награды не было — про цель не спрашиваем")

out = go("action(['RIGHT'])\nprint('без модели')")
ok("WM_CHECK" not in out and "ERR:" in out and "Traceback" not in out,
   "ход без модели: хвост молчит и ничего не ломает")

out = go(MODEL.replace("action(['RIGHT'])", "") + "def goal(s): return s[1] >= 5\naction(['RIGHT'])")
ok("WM_CHECK admitted=1" in out and "WM_GOAL ok=" in out,
   "модель вместе с целью: обе проверки отработали в одном ходу")

BROKEN = "def state_of(f): return 0\ndef predict(s,a):\n    raise ValueError('падаю')\naction(['RIGHT'])\n"
out = go(BROKEN)
ok("Traceback" not in out and ("WM_CHECK" in out or "WM_TAIL сбой" in out),
   "битая модель: страж не пускает трассировку в ответ модели")

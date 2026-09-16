"""Сухой прогон слоя перебора (kernels/notebooks_stockflash_bfs/cell15.py) на НАСТОЯЩЕМ модуле solver из бандла и
настоящем локальном движке: подделка сессии даёт step_env / _execute_auto_reset / should_stop / game.current_state.
usage: test_bfs_patch.py --bundle <бандл> [--games lp85,ft09,sc25] [--moves 8000]"""
import argparse, json, sys, time, types
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--bundle", required=True); ap.add_argument("--games", default="lp85,ft09,sc25"); ap.add_argument("--moves", type=int, default=12000); ap.add_argument("--seconds", type=float, default=600.0); a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(a.bundle) / "src" / "ARC3-Inference")); sys.path.insert(0, str(Path(a.bundle) / "src" / "tufa-arc-agi-framework" / "src")); sys.path.insert(0, str(ROOT))
import logging; logging.disable(logging.ERROR)
import numpy as np
import inference.framework.solver as solver
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
fails = []
def check(b, m):
    print(("ok   " if b else "СБОЙ ") + m)
    if not b: fails.append(m)

cell = open("kernels/notebooks_stockflash_bfs/cell15.py", encoding="utf-8").read()
seg = cell[cell.index("import time as _bf_time"):]
orig_play = solver._HarnessGameSession.play
ns = {"TRUE_SUBMISSION": True}; exec(seg, ns)
check(solver._HarnessGameSession.play is orig_play, "TRUE_SUBMISSION=True: боевая ветка не тронута")
ns = {"TRUE_SUBMISSION": False}; exec(seg, ns)
check(solver._HarnessGameSession.play is not orig_play, "TRUE_SUBMISSION=False: обёртка play установлена")
solver._HarnessGameSession.play = orig_play


class Raw:  # .state
    def __init__(self, st): self.state = st
class Frame:
    def __init__(self, data): self.data = data
class State:
    def __init__(self, fr):
        self.frame = Frame(np.asarray(fr.frame[-1]) if fr is not None and fr.frame else np.zeros((64, 64), int))
        self.levels_completed = int(fr.levels_completed or 0) if fr is not None else 0
        self.raw = Raw(str(fr.state).split(".")[-1] if fr is not None else "GAME_OVER")
        self.available_actions = [int(x) for x in (fr.available_actions or [])] if fr is not None else []
        self.won = self.raw.state == "WIN"
class Game:
    def __init__(self, env, gid): self.env = env; self.game_id = gid; self.current_state = State(None); self.n = 0
    def step(self, action, data=None):
        fr = self.env.step(action, data=data) if action != GameAction.RESET else self.env.reset(); self.n += 1
        self.current_state = State(fr)
class Sess:
    def __init__(self, game): self.game = game
    def should_stop(self): return False
    def _execute_auto_reset(self): self.game.step(GameAction.RESET)
    def step_env(self, args):
        name = str(args.get("action")).upper(); act = {"UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "SPACE": 5, "MOUSE": 6}[name]
        ga = GameAction.from_id(act)
        if act not in self.game.current_state.available_actions:
            return {"executed": False, "error": "not valid"}
        data = {"x": int(args["col"]), "y": int(args["row"])} if act == 6 else None
        self.game.step(ga, data); return {"executed": True, "grid": self.game.current_state.frame.data.tolist()}

ns["_BF_MOVES"] = a.moves; ns["_BF_SECONDS"] = a.seconds
arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir="environment_files")
envs = {e.game_id[:4]: e.game_id for e in arc.get_environments()}
for g in a.games.split(","):
    env = arc.make(envs[g]); game = Game(env, envs[g]); sess = Sess(game)
    t0 = time.time(); rec = ns["_bf_prephase"](sess); dt = time.time() - t0
    lvl = game.current_state.levels_completed
    print(f"  {g}: {json.dumps(rec, ensure_ascii=False)}; уровень после слоя {lvl}, ходов движка {game.n}, {dt:.0f} с")
    check(rec["moves"] == game.n, f"{g}: счётчик ходов слоя совпадает с числом ходов движка")
    check((rec["found"] and lvl == 1) or (not rec["found"] and lvl == 0), f"{g}: found согласован с уровнем движка")
    check(rec["moves"] <= a.moves + 64, f"{g}: бюджет ходов соблюдён")
print("ИТОГ:", "все проверки пройдены" if not fails else f"сбоев {len(fails)}: {fails}")

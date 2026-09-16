"""Сухой прогон слоя перебора на НАСТОЯЩЕМ модуле solver из бандла (публичный Duck или atlas_src) и локальном движке.
Подделка сессии даёт game.execute_action / current_state / timing_payload / should_stop / analyzer, как в _HarnessGameSession.
usage: test_bfs_patch.py --bundle <бандл> --cell kernels/notebooks_stockflash_bfstail/cell15.py [--games ...] [--moves 12000]"""
import argparse, json, sys, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--bundle", required=True); ap.add_argument("--cell", default="kernels/notebooks_stockflash_bfstail/cell15.py")
ap.add_argument("--games", default="lp85,ft09,sc25,sp80"); ap.add_argument("--moves", type=int, default=12000); ap.add_argument("--seconds", type=float, default=600.0); a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(a.bundle) / "src" / "ARC3-Inference")); sys.path.insert(0, str(Path(a.bundle) / "src" / "tufa-arc-agi-framework" / "src")); sys.path.insert(0, str(ROOT))
import logging; logging.disable(logging.ERROR)
import numpy as np
import inference.framework.solver as solver
import arc_agi, arcengine
from arc_agi import OperationMode
from arcengine import GameAction
fails = []
def check(b, m):
    print(("ok   " if b else "СБОЙ ") + m)
    if not b: fails.append(m)

cell = open(a.cell, encoding="utf-8").read()
seg = cell[cell.index("import time as _bf_time"):]
orig_play = solver._HarnessGameSession.play
import os; os.environ["BFS_LAYER"] = "0"; ns = {"TRUE_SUBMISSION": True}; exec(seg, ns)
check(solver._HarnessGameSession.play is orig_play, "BFS_LAYER=0: play не тронут")
os.environ["BFS_LAYER"] = "1"; ns = {"TRUE_SUBMISSION": True}; exec(seg, ns)
check(solver._HarnessGameSession.play is not orig_play, "TRUE_SUBMISSION=True: обёртка play установлена (слой боевой)")
solver._HarnessGameSession.play = orig_play
mode = ns["_BF_MODE"]; print("режим", mode)


class Raw:
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
    def __init__(self, env, gid): self.env = env; self.game_id = gid; self.current_state = State(None); self.n = 0; self.game_run = type("R", (), {"state": "playing"})()
    def execute_action(self, action, generated_tokens=0, uncached_input_tokens=0):
        if action.id != GameAction.RESET and action.id.value not in self.current_state.available_actions:
            raise ValueError("not available")
        fr = self.env.reset() if action.id == GameAction.RESET else self.env.step(action.id, data=dict(action.data) or None)
        self.n += 1; self.current_state = State(fr); return self.current_state
class FakeAnalyzer:
    def __init__(self, sess, rng): self.sess = sess; self.rng = rng; self.calls = 0
    def analyze(self, *a, **k):
        self.calls += 1; av = self.sess.game.current_state.available_actions
        simple = [x for x in av if 1 <= x <= 5]
        if simple: self.sess.game.execute_action(arcengine.ActionInput(id=GameAction.from_id(self.rng.choice(simple)), data={}))
        return {"ok": True}
class Sess:
    def __init__(self, game, cap): self.game = game; self.cap = cap; self.t0 = time.monotonic(); self.last_engine_action = None; self.analyzer = None; self.history_entries = []
    def should_stop(self): return (time.monotonic() - self.t0) >= self.cap or self.game.current_state.raw.state == "WIN"
    def timing_payload(self): return {"run_elapsed_seconds": time.monotonic() - self.t0, "time_remaining_seconds": max(0.0, self.cap - (time.monotonic() - self.t0))}

ns["_BF_MOVES"] = a.moves; ns["_BF_SECONDS"] = a.seconds
arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir="environment_files")
envs = {e.game_id[:4]: e.game_id for e in arc.get_environments()}
for g in a.games.split(","):
    env = arc.make(envs[g]); game = Game(env, envs[g]); sess = Sess(game, cap=1e9)
    game.execute_action(arcengine.ActionInput(id=GameAction.RESET, data={}))
    rec = ns["_bf_prephase"](sess); lvl = game.current_state.levels_completed
    print(f"  {g}: {json.dumps(rec, ensure_ascii=False)}; уровень {lvl}, ходов движка {game.n}")
    check(rec["moves"] + 1 == game.n, f"{g}: счётчик ходов слоя = ходы движка (без стартового RESET)")
    check((rec["found"] and lvl == 1) or (not rec["found"] and lvl == 0), f"{g}: found согласован с уровнем движка")
    check(rec["moves"] <= a.moves + 64, f"{g}: бюджет ходов соблюдён")
if mode == "tail":
    # хвост: игра с потолком 12 с, хвост 6 с, застой 3 с; поддельный play гоняет анализатор случайными стрелками
    import random
    ns["_BF_TAIL_S"] = 6.0; ns["_BF_STALL_S"] = 3.0
    def fake_orig_play(self):
        while not self.should_stop():
            self.analyzer.analyze(); time.sleep(0.2)
    ns["_bf_orig_play"] = fake_orig_play
    for g in ("lp85", "ls20"):
        env = arc.make(envs[g]); game = Game(env, envs[g]); sess = Sess(game, cap=12.0)
        game.execute_action(arcengine.ActionInput(id=GameAction.RESET, data={})); sess.analyzer = FakeAnalyzer(sess, random.Random(0))
        n0 = len(ns["_bf_stats"]["per_game"]); ns["_bf_play"](sess); rec = ns["_bf_stats"]["per_game"].get(envs[g])
        print(f"  хвост {g}: вызовов анализатора {sess.analyzer.calls}, запись {json.dumps(rec, ensure_ascii=False) if rec else None}, уровень {game.current_state.levels_completed}")
        check(rec is not None and rec["seconds"] <= 7.0, f"{g}: перебор в хвосте сработал один раз и уложился в остаток")
        check(isinstance(sess.analyzer, ns["_BfAnalyzerProxy"]) and sess.analyzer.calls > 5, f"{g}: анализатор проксирован и вызывался до и после")
print("ИТОГ:", "все проверки пройдены" if not fails else f"сбоев {len(fails)}: {fails}")

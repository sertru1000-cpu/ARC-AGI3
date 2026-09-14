"""Повтор боевой траектории на локальном движке: проверка детерминизма (доска после каждого хода совпадает с
записанной?) и заготовка контрфактического набора. usage: replay_battle_local.py <run> <game4> [--branch K]"""
import argparse, glob, json, os, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
import numpy as np
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

def load_actions(run, g):
    p = glob.glob(f"{run}/artifacts/{g}-*_p0_events.jsonl")[0]
    acts = []
    for l in open(p, encoding="utf-8"):
        e = json.loads(l)
        if e.get("type") != "action": continue
        name = str(e.get("action_name")); disp = str(e.get("action_display") or "")
        m = re.match(r"MOUSE\(row=(\d+), col=(\d+)\)", disp)
        payload = {"x": int(m.group(2)), "y": int(m.group(1))} if m else None
        acts.append({"name": name, "payload": payload, "board": np.asarray(e["board"], dtype=np.int16), "level": int(e.get("level") or 0),
                     "changed": e.get("board_changed") in (True, "True"), "lvl_done": e.get("level_completed") in (True, "True")})
    return acts

class Scripted(Agent):
    MAX_ACTIONS = 10**6
    def __init__(self, *a, script=None, **k):
        super().__init__(*a, **k); self.script = list(script or []); self.i = 0; self.trace = []
    def is_done(self, frames, latest):
        return self.i >= len(self.script)
    def choose_action(self, frames, latest):
        if latest.state in (GameState.NOT_PLAYED,) and self.i == 0 and self.script[0]["name"] != "RESET":
            return GameAction.RESET
        s = self.script[self.i]; self.i += 1
        a = GameAction.RESET if s["name"] == "RESET" else GameAction[s["name"]]
        if s.get("payload"): a.set_data(s["payload"])
        return a
    def take_action(self, action):
        fr = super().take_action(action)
        if fr is not None and fr.frame:
            self.trace.append((np.asarray(fr.frame[-1], dtype=np.int16), int(fr.levels_completed or 0), str(fr.state)))
        return fr

def play(arc, gid, script):
    env = arc.make(gid)
    ag = Scripted(card_id="replay", game_id=gid, agent_name=f"replay.{gid}", ROOT_URL="http://localhost", record=False, arc_env=env, tags=["replay"], script=script)
    ag.main(); return ag.trace

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("game"); a = ap.parse_args()
    import logging; logging.disable(logging.WARNING)
    import arc_agi
    from arc_agi import OperationMode
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    gid = [e.game_id.split("-")[0] for e in arc.get_environments() if e.game_id.startswith(a.game)][0]
    acts = load_actions(a.run, a.game)
    t0 = time.time(); trace = play(arc, gid, acts); dt = time.time() - t0
    # trace[i] соответствует acts[i] (первый RESET, если добавлен, даёт лишний кадр в начале)
    off = len(trace) - len(acts)
    match = 0; first_mismatch = None; lv_match = 0
    for i, s in enumerate(acts):
        tr = trace[i + off] if i + off < len(trace) else None
        if tr is None: break
        ok = tr[0].shape == s["board"].shape and bool((tr[0] == s["board"]).all())
        match += ok
        if not ok and first_mismatch is None: first_mismatch = i
        lv_match += (tr[1] == s["level"] - 1) or (tr[1] == s["level"])
    print(f"{a.game}: ходов {len(acts)}, локально {len(trace)} кадров за {dt:.1f} с ({len(trace)/max(dt,1e-9):.0f} ходов/с); "
          f"доска совпала {match}/{len(acts)}; первое расхождение на ходу {first_mismatch}; уровней в записи {sum(s['lvl_done'] for s in acts)}, локально {trace[-1][1]}")

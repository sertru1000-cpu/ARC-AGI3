"""Контрфактическая оценка слоя «перебор после застоя» на прогонах базы (локальный движок детерминирован, ходы базы
повторяются точно -- см. память arc-agi-3-local-engine-replays-battle).

Для каждой игры прогона базы, обрезанного на --cap секунд:
  * уровень L начинается в момент взятия предыдущего (или в 0); если модель не взяла его за --stall секунд и к этому моменту
    до конца остаётся >= --min-left + 30 секунд, перебор включается (как в ячейке bfscarry: один раз на уровень);
  * ходы базы до момента включения повторяются на локальном движке, с этого состояния запускается НАСТОЯЩИЙ _bf_prephase
    из ячейки (бюджет --moves ходов / --seconds секунд, RESET текущего уровня + повтор пути);
  * ПОТЕРЯ (ИЗМЕРЕНО по записи базы): балл всех уровней, которые база взяла ПОСЛЕ момента включения (уровень L обнуляется
    ходами перебора в любом случае, дальнейшая траектория модели расходится с записью);
  * НАХОДКА (ИЗМЕРЕНО перебором): найден ли уровень L, сколько секунд и ходов ушло;
  * ВЫГОДА (ПРЕДПОЛОЖЕНИЕ, сценарий): если найден -- модель берёт уровень L+1 своими ходами с вероятностью --p-next
    и баллом уровня L+1, равным медиане балла уровня того же номера у базы (по трём прогонам); уровни дальше L+1 не считаются.
Итог по прогону в единицах RHAE (среднее по 25 играм): потеря, выгода при --p-next, нетто.

usage:  .venv/bin/python scripts/bfs_stall_counterfactual.py runs/flash_v1_phaseA [--stall 1800 --min-left 900 --cap 3600]
"""
import argparse, glob, json, re, statistics, sys, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--bundle", default="runs/peer_kernels/duck_base/x")
ap.add_argument("--cell", default="kernels/notebooks_stockflash_bfscarry/cell15.py")
ap.add_argument("--stall", type=float, default=1800.0); ap.add_argument("--min-left", type=float, default=900.0); ap.add_argument("--cap", type=float, default=3600.0)
ap.add_argument("--moves", type=int, default=12000); ap.add_argument("--seconds", type=float, default=600.0); ap.add_argument("--p-next", type=float, default=0.39)
ap.add_argument("--out", default=None); a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(a.bundle) / "src" / "ARC3-Inference")); sys.path.insert(0, str(Path(a.bundle) / "src" / "tufa-arc-agi-framework" / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import logging; logging.disable(logging.ERROR)
import os; os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "Qwen/Qwen3.8-Flash-Next-NVFP4")
import numpy as np
import arc_agi, arcengine
from arc_agi import OperationMode
from arcengine import GameAction
from truncate_run import actions as ev_actions, score as rhae_score

cell = open(ROOT / a.cell, encoding="utf-8").read()
seg = cell[cell.index("import time as _bf_time"):cell.index("_bf_orig_play = _bf_solver._HarnessGameSession.play")]
ns = {"TRUE_SUBMISSION": False}; exec(seg, ns)
ns["_BF_MOVES"] = a.moves

class Raw:
    def __init__(self, st): self.state = st
class EFrame:
    def __init__(self, data): self.data = data
class State:
    def __init__(self, fr):
        self.frame = EFrame(np.asarray(fr.frame[-1]) if fr is not None and fr.frame else np.zeros((64, 64), int))
        self.levels_completed = int(fr.levels_completed or 0) if fr is not None else 0
        self.raw = Raw(str(fr.state).split(".")[-1] if fr is not None else "GAME_OVER")
        self.available_actions = [int(x) for x in (fr.available_actions or [])] if fr is not None else []
        self.won = self.raw.state == "WIN"
class Game:
    def __init__(self, env, gid): self.env = env; self.game_id = gid; self.current_state = State(None)
    def execute_action(self, action, generated_tokens=0, uncached_input_tokens=0):
        if action.id != GameAction.RESET and action.id.value not in self.current_state.available_actions:
            raise ValueError("not available")
        fr = self.env.reset() if action.id == GameAction.RESET else self.env.step(action.id, data=dict(action.data) or None)
        self.current_state = State(fr); return self.current_state
class Sess:
    def __init__(self, game): self.game = game; self.last_engine_action = None
    def should_stop(self): return False
    def timing_payload(self): return {"time_remaining_seconds": 1e9}

run = Path(a.run)
games = json.load(open(run / "benchmark.json", encoding="utf-8"))["game_runs"]
arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT / "environment_files"))
envs = {e.game_id[:4]: e.game_id for e in arc.get_environments()}

def level_scores(r, per, done):
    """вклад каждого взятого уровня в балл игры (h115)."""
    base = json.loads(str(r["base_actions_per_level"])); n = int(r["number_of_levels"]); den = n * (n + 1) / 2
    return [min(115.0, (base[i] / per[i]) ** 2 * 100) * (i + 1) / den if i < done and per[i] else 0.0 for i in range(n)]

rows = []
for r in games:
    gid = str(r["game_id"]); g4 = gid[:4]; h = r["history"]
    evs = []
    for l in open(glob.glob(str(run / "artifacts" / f"{gid}_p0_events.jsonl"))[0], encoding="utf-8"):
        e = json.loads(l)
        if e.get("type") != "action": continue
        i = int(e["action_num"]) - 1
        t = float(h[i].get("wallclock_seconds") or 0.0) if i < len(h) else 1e9
        m = re.match(r"MOUSE\(row=(\d+), col=(\d+)\)", str(e.get("action_display") or ""))
        evs.append({"name": str(e.get("action_name")), "data": ({"x": int(m.group(2)), "y": int(m.group(1))} if m else {}), "t": t,
                    "done": e.get("level_completed") in (True, "True")})
    evs = [e for e in evs if e["t"] <= a.cap]
    # по уровням: действия на уровень и время взятия
    per = [0] * int(r["number_of_levels"]); lvl = 0; times = []
    for e in evs:
        if lvl < len(per): per[lvl] += 1
        if e["done"]: times.append(e["t"]); lvl += 1
    done = len(times); ls = level_scores(r, per, done); game_score = sum(ls)
    # момент включения
    start = 0.0; trig = None; L = None
    for k in range(done + 1):
        end = times[k] if k < done else a.cap
        if end - start >= a.stall and start + a.stall <= a.cap - a.min_left - 30.0:
            trig = start + a.stall; L = k; break
        start = end
    rec = {"game": g4, "levels": done, "score": round(game_score, 3), "trigger_s": trig, "level_idx": L}
    if trig is None:
        rec.update({"lost": 0.0, "found": None}); rows.append(rec); print(json.dumps(rec, ensure_ascii=False), flush=True); continue
    rec["lost"] = round(sum(ls[L:]), 3); rec["lost_levels"] = done - L
    # повтор ходов базы до момента включения
    env = arc.make(envs[g4]); game = Game(env, envs[g4]); sess = Sess(game)
    game.execute_action(arcengine.ActionInput(id=GameAction.RESET, data={}))
    for e in evs:
        if e["t"] > trig: break
        try:
            act = GameAction.RESET if e["name"] == "RESET" else GameAction[e["name"]]
            game.execute_action(arcengine.ActionInput(id=act, data=e["data"]))
        except Exception:
            pass
    rec["replay_level_ok"] = game.current_state.levels_completed == L
    t0 = time.time(); bf = ns["_bf_prephase"](sess, budget_s=a.seconds)
    rec.update({"found": bf["found"], "path_len": bf["path_len"], "bfs_moves": bf["moves"], "bfs_seconds": bf["seconds"], "stopped_by": bf["stopped_by"]})
    rows.append(rec); print(json.dumps(rec, ensure_ascii=False), flush=True)

json.dump(rows, open(a.out or f"/dev/null", "w"), ensure_ascii=False, indent=1) if a.out else None
trig = [x for x in rows if x["trigger_s"] is not None]
print("\nИТОГ %s: застой %.0f с, остаток %.0f с, потолок %.0f с" % (run.name, a.stall, a.min_left, a.cap))
print("  игр с включением перебора: %d из %d; повтор до момента включения совпал по уровню: %d" % (len(trig), len(rows), sum(1 for x in trig if x.get("replay_level_ok"))))
print("  перебор нашёл уровень: %d (%s)" % (sum(1 for x in trig if x["found"]), ", ".join("%s L%d" % (x["game"], x["level_idx"] + 1) for x in trig if x["found"])))
print("  ПОТЕРЯ (балл уровней базы после включения): %.2f RHAE; уровней %d; игр с потерей %d" % (sum(x["lost"] for x in trig) / len(rows), sum(x.get("lost_levels", 0) for x in trig), sum(1 for x in trig if x["lost"] > 0)))

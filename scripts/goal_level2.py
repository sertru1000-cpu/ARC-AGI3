"""Второй уровень: помогает ли ВИД ЦЕЛИ, перенесённый с первого (17.09, тезис владельца «все уровни в игре подобны,
разница только в сложности»).

Измерено раньше (goal_cross_level.py): тип цели совпал у уровней 1 и 2 в обеих играх, где до уровня 2 удалось дойти,
а параметры совпали лишь в одной. Значит, переносить надо ВИД цели, а число подбирать заново на новом уровне.

Три руки на одних и тех же играх и бюджетах, старт -- состояние сразу после взятия уровня 1 по известному пути:
  bfs       -- слепой перебор в ширину (контроль);
  goal      -- поиск по всем кандидатам словаря (как goal_search.py);
  transfer  -- поиск ТОЛЬКО по тем типам целей, которые отделяли цель уровня 1 в этой же игре (параметры заново).
Победа переноса: transfer решает больше игр, чем bfs, ИЛИ дешевле по ходам на общих играх, и при этом не хуже goal.

usage:  .venv/bin/python scripts/goal_level2.py [--timeout 240] [--out runs/goal_level2_17_09.json]
"""
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import logging; logging.disable(logging.ERROR)
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
from engine_bfs2 import Snap, Counter, simple_actions, active_clicks, clock_mask
from goal_search import state_features, candidates, run_search
from goal_predicates import features as pfeatures, templates

# тип цели из индукции (T*) -> семейства кандидатов поиска (C*)
T2C = {"T1": {"C1"}, "T2": {"C1"}, "T3": {"C3"}, "T4": {"C3"}, "T5": {"C5"}, "T6": {"C3"}, "T7": {"C1"},
       "T8": {"C8"}, "T9": {"C9"}, "T10": {"C10"}, "T11": {"C11"}, "T12": {"C12"}, "T13": {"C13"}, "T14": {"C1"}}


def level1_goal_types(arc, gid, path, timeout=60.0):
    """типы целей, отделяющие цель уровня 1 от состояний уровня 1 (индукция как в goal_predicates.py)."""
    env = arc.make(gid); env.reset()
    negs = []
    for name, payload in path[:-1]:
        fr = env.step(GameAction[name] if name != "RESET" else GameAction.RESET, data=payload)
        negs.append(np.asarray(fr.frame[-1], dtype=np.int16))
    env_at_end = copy.deepcopy(env)
    fr = env.step(GameAction[path[-1][0]], data=path[-1][1])
    goal = [np.asarray(x, dtype=np.int16) for x in fr.frame][:-1] or [np.asarray(fr.frame[0], dtype=np.int16)]
    mask = np.zeros_like(goal[-1], dtype=bool)
    f_goal = pfeatures(goal[-1], mask)
    f_negs = [pfeatures(g, mask) for g in negs if g.shape == goal[-1].shape]
    types = set()
    for nm, tid, fn in templates(f_goal):
        try:
            if fn(f_goal) and not any(fn(x) for x in f_negs):
                types |= T2C.get(tid, set())
        except Exception:
            pass
    return types, env_at_end, fr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--max-states", type=int, default=3000); ap.add_argument("--max-moves", type=int, default=60000)
    ap.add_argument("--out", default="runs/goal_level2_17_09.json"); a = ap.parse_args()
    sol = json.load(open(ROOT / "runs/bfs_originals.json", encoding="utf-8"))
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT / "environment_files"))
    rows = []
    for gid, rec in sol.items():
        if not rec.get("solved"):
            continue
        types, env_end, fr_end = level1_goal_types(arc, gid, rec["path"])
        row = {"game": gid[:4], "types_from_level1": sorted(types)}
        for mode in ("bfs", "goal", "transfer"):
            t0 = time.time(); cnt = Counter(a.max_moves, t0 + a.timeout)
            env2 = copy.deepcopy(env_end)
            fr = env2.step(GameAction[rec["path"][-1][0]], data=rec["path"][-1][1])   # входим на уровень 2
            root = Snap(env2, fr)
            mask = clock_mask(root, cnt)
            acts = simple_actions(root.avail) + active_clicks(root, cnt)
            f0 = state_features(root.grid, mask)
            if mode == "bfs":
                cands = []
            else:
                allowed = None if mode == "goal" else types
                cands = [(nm, fn) for nm, fn in candidates(f0) if fn(f0) > 0 and (allowed is None or nm.split()[0] in allowed)]
            r = run_search(root, mask, acts, cands, cnt, a.max_states, "bfs" if mode == "bfs" else "goal")
            r["seconds"] = round(time.time() - t0, 1)
            row[mode] = {k: r[k] for k in ("solved", "path_len", "states", "moves", "seconds", "candidates", "best_candidate")}
            print("%s %-9s %s" % (gid[:4], mode, json.dumps(row[mode], ensure_ascii=False)), flush=True)
        rows.append(row)
    print("\nИТОГ уровень 2 (игр %d):" % len(rows))
    for mode in ("bfs", "goal", "transfer"):
        won = [r for r in rows if r[mode]["solved"]]
        print("  %-9s решено %d; ходов на решённую медиана %s" % (mode, len(won), int(np.median([r[mode]["moves"] for r in won])) if won else "—"))
    both = [r for r in rows if r["bfs"]["solved"] and r["transfer"]["solved"]]
    if both:
        print("  на общих %d играх ходов: перенос %d, слепой %d" % (len(both), sum(r["transfer"]["moves"] for r in both), sum(r["bfs"]["moves"] for r in both)))
    json.dump(rows, open(ROOT / a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print("записано:", a.out)


if __name__ == "__main__":
    main()

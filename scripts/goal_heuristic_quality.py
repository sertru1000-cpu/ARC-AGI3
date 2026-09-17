"""Есть ли в остатке кандидата СИГНАЛ (17.09). Проверка ДО всякой настройки поиска: если остаток верной цели
не убывает по мере приближения к решению, то поиск по нему настраивать бессмысленно (см. память
feedback-measure-behaviour-before-building-a-tool).

Данные: 10 игр с известным решением уровня 1 (runs/bfs_originals.json). Идём по решающему пути, на каждом шаге t
считаем остатки ВСЕХ кандидатов (goal_search.candidates, кандидаты, истинные на старте, отброшены) и меряем:
  * ро Спирмена между остатком и числом ОСТАВШИХСЯ ходов (k - t): ро > 0 -- остаток убывает к цели;
  * долю шагов, где остаток не вырос (монотонность);
  * ранг верного кандидата (того, у кого остаток на последнем шаге минимален) по ро среди всех кандидатов.
Контроль: те же меры на СЛУЧАЙНОМ блуждании той же длины (там приближения к цели нет, ро должно быть около нуля).

usage:  .venv/bin/python scripts/goal_heuristic_quality.py [--out runs/goal_heuristic_17_09.json]
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import logging; logging.disable(logging.ERROR)
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
from goal_search import state_features, candidates


def spearman(x, y):
    if len(x) < 4:
        return 0.0
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float))); ry = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / d) if d else 0.0


def clock_mask_from_walk(env, steps=100, seed=0):
    rng = random.Random(seed); acts = [GameAction[x] for x in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")]
    fr = env.reset(); prev = np.asarray(fr.frame[-1], dtype=np.int16); ch = np.zeros_like(prev, dtype=np.int32); n = 0; walk = []
    for _ in range(steps):
        fr = env.step(rng.choice(acts), data=None)
        g = np.asarray(fr.frame[-1], dtype=np.int16)
        if int(fr.levels_completed or 0) > 0 or str(fr.state).split(".")[-1] == "GAME_OVER":
            fr = env.reset(); prev = np.asarray(fr.frame[-1], dtype=np.int16); continue
        if g.shape == prev.shape and (g != prev).any():
            ch += (g != prev); n += 1
        walk.append(g); prev = g
    return (ch >= 0.8 * max(1, n)) & (ch >= 3), walk


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="runs/goal_heuristic_17_09.json"); a = ap.parse_args()
    sol = json.load(open(ROOT / "runs/bfs_originals.json", encoding="utf-8"))
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT / "environment_files"))
    rows = []
    for gid, rec in sol.items():
        if not rec.get("solved"):
            continue
        env = arc.make(gid)
        mask, walk = clock_mask_from_walk(env)
        fr = env.reset(); f0 = state_features(np.asarray(fr.frame[-1], dtype=np.int16), mask)
        cands = [(nm, fn) for nm, fn in candidates(f0) if fn(f0) > 0]
        path = rec["path"]; states = []
        for name, payload in path[:-1]:
            fr = env.step(GameAction[name] if name != "RESET" else GameAction.RESET, data=payload)
            states.append(np.asarray(fr.frame[-1], dtype=np.int16))
        fr = env.step(GameAction[path[-1][0]], data=path[-1][1])
        goal_frames = [np.asarray(x, dtype=np.int16) for x in fr.frame][:-1] or [np.asarray(fr.frame[0], dtype=np.int16)]
        seq = [np.asarray(x, dtype=np.int16) for x in [np.asarray(f0["grid"])]]  # старт
        seq = [np.asarray(arc and f0["grid"])] if False else states
        feats = [state_features(g, mask) for g in seq] + [state_features(goal_frames[-1], mask)]
        left = list(range(len(feats) - 1, -1, -1))
        # верный кандидат: минимальный остаток в целевом кадре
        vals_goal = [(fn(feats[-1]), nm) for nm, fn in cands]
        true_res, true_name = min(vals_goal, key=lambda x: x[0]) if vals_goal else (1.0, None)
        rho_all = []
        for nm, fn in cands:
            r = [fn(f) for f in feats]
            rho_all.append((spearman(r, left), nm, r))
        rho_all.sort(key=lambda x: -x[0])
        true_rho = next((r for r, nm, _ in rho_all if nm == true_name), 0.0)
        rank = next((i + 1 for i, (_, nm, _) in enumerate(rho_all) if nm == true_name), None)
        r_true = next((r for _, nm, r in rho_all if nm == true_name), [])
        mono = float(np.mean([r_true[i + 1] <= r_true[i] + 1e-9 for i in range(len(r_true) - 1)])) if len(r_true) > 1 else 0.0
        # контроль: блуждание той же длины
        wf = [state_features(g, mask) for g in walk[:len(feats)]]
        ctrl = spearman([next(fn for nm, fn in cands if nm == true_name)(f) for f in wf], list(range(len(wf) - 1, -1, -1))) if wf and true_name else 0.0
        rows.append({"game": gid[:4], "path": len(path), "candidates": len(cands), "true_candidate": true_name,
                     "true_residual_at_goal": round(true_res, 3), "rho_true": round(true_rho, 3), "rank_of_true": rank,
                     "monotone_share": round(mono, 2), "rho_best": round(rho_all[0][0], 3) if rho_all else 0.0,
                     "best_by_rho": rho_all[0][1] if rho_all else None, "rho_control_walk": round(ctrl, 3)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    if rows:
        print("\nИТОГ: игр %d; медиана ро верного кандидата %.2f (контроль на блуждании %.2f); медиана ранга верного кандидата %s из %d; медиана монотонности %.2f"
              % (len(rows), float(np.median([r["rho_true"] for r in rows])), float(np.median([r["rho_control_walk"] for r in rows])),
                 int(np.median([r["rank_of_true"] or 999 for r in rows])), int(np.median([r["candidates"] for r in rows])),
                 float(np.median([r["monotone_share"] for r in rows]))))
        json.dump(rows, open(ROOT / a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print("записано:", a.out)


if __name__ == "__main__":
    main()

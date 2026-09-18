"""Что дал бы список гипотез о цели ВО ВРЕМЯ боя, без единого лишнего хода (18.09, слово владельца «делай»).

Проверка полностью офлайн, на НАСТОЯЩИХ траекториях модели из наших прогонов (локальный движок их повторяет точно).
Моделируется то, что обвязка могла бы делать в бою даром: она и так видит каждое состояние, поэтому может
  * с начала уровня держать список беспараметрических кандидатов в цель (goal_search.candidates);
  * ВЫБРАСЫВАТЬ опровергнутых — тех, чей остаток дошёл до нуля там, где уровень НЕ взят;
  * класть выживших во вход модели и сообщать после каждого хода, приблизил он к ближайшей гипотезе или отдалил.

Считаются три числа на каждый уровень:
  1. ПОПАЛА ЛИ ЦЕЛЬ В СПИСОК: есть ли среди выживших к моменту взятия хоть один с остатком 0 в целевом кадре;
  2. ДЛИНА СПИСКА выживших в этот момент (сколько гипотез пришлось бы показать модели);
  3. СИГНАЛ ПРОГРЕССА: ро Спирмена между минимальным остатком по выжившим и числом ходов, оставшихся до взятия
     уровня (ро > 0 -- «приблизил/отдалил» несёт смысл).
Контроль: то же на перемешанном порядке состояний уровня (ожидание ро около нуля).

usage:  .venv/bin/python scripts/goal_online_check.py [--runs ...] [--out runs/goal_online_18_09.json]
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import logging; logging.disable(logging.ERROR)
import arc_agi
from arc_agi import OperationMode
from goal_cross_level_replay import load_events, replay_collect, DEFAULT_RUNS
from goal_search import state_features, candidates
from goal_heuristic_quality import spearman


def online_level(states, goal, lib=None):
    """прогон одного уровня «как в бою»: список гипотез, опровержение, сигнал прогресса."""
    mask = np.zeros_like(goal, dtype=bool)
    feats = [state_features(g, mask) for g in states if g.shape == goal.shape]
    if not feats:
        return None
    cands = [(nm, fn) for nm, fn in candidates(feats[0], lib) if fn(feats[0]) > 0]
    alive = list(cands); mins = []
    for f in feats:
        keep = []
        for nm, fn in alive:
            try:
                v = fn(f)
            except Exception:
                continue
            if v <= 0.0:          # истинно там, где уровень не взят -> опровергнут
                continue
            keep.append((nm, fn))
        alive = keep
        vals = []
        for nm, fn in alive:
            try:
                vals.append(fn(f))
            except Exception:
                pass
        mins.append(min(vals) if vals else 1.0)
    f_goal = state_features(goal, mask)
    hit = []
    for nm, fn in alive:
        try:
            if fn(f_goal) <= 0.0:
                hit.append(nm)
        except Exception:
            pass
    left = list(range(len(mins) - 1, -1, -1))
    rho = spearman(mins, left)
    sh = list(mins); random.Random(0).shuffle(sh)
    return {"in_list": bool(hit), "hits": hit[:3], "alive": len(alive), "cands0": len(cands),
            "rho_progress": round(rho, 3), "rho_shuffled": round(spearman(sh, left), 3), "states": len(feats)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    ap.add_argument("--max-states", type=int, default=120); ap.add_argument("--out", default="runs/goal_online_18_09.json")
    a = ap.parse_args()
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT / "environment_files"))
    rows = []
    for e in arc.get_environments():
        gid = e.game_id
        best = {}
        for run in a.runs.split(","):
            evs = load_events(run, gid)
            if not evs:
                continue
            for k, v in replay_collect(arc, gid, evs).items():
                if k not in best or len(v["negs"]) > len(best[k]["negs"]):
                    best[k] = v
        for k in sorted(best):
            negs = best[k]["negs"][-a.max_states:]
            r = online_level(negs, best[k]["goal"])
            if r is None:
                continue
            r.update({"game": gid[:4], "level": k})
            rows.append(r); print(json.dumps(r, ensure_ascii=False), flush=True)
    if rows:
        inl = sum(1 for r in rows if r["in_list"])
        print("\nИТОГ: уровней %d в %d играх" % (len(rows), len({r['game'] for r in rows})))
        print("  цель ПОПАЛА в список выживших гипотез: %d (%.0f%%)" % (inl, 100 * inl / len(rows)))
        print("  длина списка в момент взятия: медиана %d (было в начале уровня медиана %d)"
              % (int(np.median([r["alive"] for r in rows])), int(np.median([r["cands0"] for r in rows]))))
        print("  сигнал прогресса ро: медиана %.2f (контроль на перемешанном порядке %.2f); ро > 0.3 на %d уровнях"
              % (float(np.median([r["rho_progress"] for r in rows])), float(np.median([r["rho_shuffled"] for r in rows])),
                 sum(1 for r in rows if r["rho_progress"] > 0.3)))
    json.dump(rows, open(ROOT / a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print("записано:", a.out)


if __name__ == "__main__":
    main()

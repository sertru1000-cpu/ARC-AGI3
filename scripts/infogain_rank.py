"""Пункт 4 критика раунда 10: выбор хода по тому, насколько он РАЗЛИЧАЕТ гипотезы цели (19.09).

Гипотезы: кандидаты-предикаты цели уровня из нашего словаря (scripts/goal_search.candidates, считаются от кадра
старта уровня), из которых выброшены те, что уже были истинны в каком-то пройденном состоянии уровня (цель к тому
моменту не взята — значит, это не цель). Предикат истинен, когда его плотный остаток <= 0.

Две меры хода a из состояния s (разворачиваются на снимке локального движка, настоящих ходов не тратится):
  * РАЗЛИЧЕНИЕ (пункт 4): сколько гипотез ход переключает (ложь <-> истина) -- эксперимент, который больше всего
    двигает набор гипотез;
  * ПУЧОК (пункт 2): средний сдвиг остатка по ВСЕМУ набору живых гипотез -- «прогресс к цели, не выбирая одну».
Выигрышные ходы -- кратчайшие решения первого уровня (runs/bfs_originals.json), соседи -- прочие простые ходы.
Меряем место выигрышного хода (ничьи разбиты случайно) против случайного угадывания, как в пункте 3.

Порог (задан до замера, тот же, что критик дал для пункта 3): top-1 >= 0.50 -- мера пригодна для выбора хода,
строить; top-1 в пределах +0.05 от случайного -- пункт закрыт; между -- неопределённо.

usage:  .venv/bin/python scripts/infogain_rank.py
"""
import copy, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import logging; logging.disable(logging.ERROR)
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
from goal_search import state_features, candidates


def feats(g):
    return state_features(g, np.zeros_like(g, dtype=bool))


def truth(cands, f):
    out = []
    for _nm, fn in cands:
        try:
            out.append(fn(f))
        except Exception:
            out.append(None)
    return out


def exp_topk(score_win, score_sibs, k, higher_better=True):
    """ожидаемое попадание выигрышного хода в top-k при случайном разбиении ничьих"""
    if higher_better:
        better = sum(1 for s in score_sibs if s > score_win + 1e-12)
    else:
        better = sum(1 for s in score_sibs if s < score_win - 1e-12)
    tied = sum(1 for s in score_sibs if abs(s - score_win) <= 1e-12)
    places = range(better + 1, better + tied + 2)
    return float(np.mean([p <= k for p in places]))


def main():
    sol = json.load(open(ROOT / "runs/bfs_originals.json", encoding="utf-8"))
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT / "environment_files"))
    alphabet = ["ACTION%d" % i for i in range(1, 6)]
    res = {"различение": {1: [], 2: [], 3: []}, "пучок": {1: [], 2: [], 3: []}, "случайный": {1: [], 2: [], 3: []}}
    live_sizes = []; per_game = []
    for gid, rec in sol.items():
        if not rec.get("solved"):
            continue
        path = [(n, p) for n, p in rec["path"]]
        cur = arc.make(gid); fr = cur.reset()
        if fr is None or not fr.frame:
            continue
        g0 = np.asarray(fr.frame[-1], dtype=np.int16)
        f0 = feats(g0)
        cands = candidates(f0)
        if not cands:
            continue
        alive = [v is not None and v > 0 for v in truth(cands, f0)]   # ложные в старте -- живые гипотезы
        f_now = f0; steps = 0
        for n, p in path:
            t_now = truth(cands, f_now)
            idx = [i for i, a in enumerate(alive) if a and t_now[i] is not None]
            if not idx:
                break
            live_sizes.append(len(idx))
            def measure(act, data):
                e = copy.deepcopy(cur)
                try:
                    r = e.step(GameAction[act] if act != "RESET" else GameAction.RESET, data=data)
                except Exception:
                    return None
                if r is None or not r.frame:
                    return None
                t1 = truth(cands, feats(np.asarray(r.frame[-1], dtype=np.int16)))
                flips = sum(1 for i in idx if t1[i] is not None and ((t_now[i] > 0) != (t1[i] > 0)))
                dres = float(np.mean([(t1[i] - t_now[i]) for i in idx if t1[i] is not None])) if idx else 0.0
                return flips, dres, e, r
            mw = measure(n, p)
            if mw is None:
                break
            sibs = [measure(s, None) for s in alphabet if s != n]
            sibs = [s for s in sibs if s is not None]
            if sibs:
                for k in (1, 2, 3):
                    res["различение"][k].append(exp_topk(mw[0], [s[0] for s in sibs], k, higher_better=True))
                    res["пучок"][k].append(exp_topk(mw[1], [s[1] for s in sibs], k, higher_better=False))
                    res["случайный"][k].append(min(1.0, k / (1 + len(sibs))))
            steps += 1
            # шаг по решению; гипотезы, ставшие истинными без взятия уровня, выбывают
            _, _, cur, r = mw
            f_now = feats(np.asarray(r.frame[-1], dtype=np.int16))
            if int(r.levels_completed or 0) >= 1:
                break
            t_after = truth(cands, f_now)
            for i in idx:
                if t_after[i] is not None and t_after[i] <= 0:
                    alive[i] = False
        per_game.append({"game": gid[:4], "steps": steps, "hypotheses": len(cands)})
        print("%s: шагов %d, гипотез в словаре %d" % (gid[:4], steps, len(cands)), flush=True)
    n = len(res["случайный"][1])
    print("\nсостояний %d, живых гипотез на состояние: медиана %d" % (n, int(np.median(live_sizes)) if live_sizes else 0))
    for name in ("различение", "пучок", "случайный"):
        print("  %-11s top-1 %.2f, top-2 %.2f, top-3 %.2f" % (name, *[float(np.mean(res[name][k])) if res[name][k] else 0 for k in (1, 2, 3)]))
    ch = float(np.mean(res["случайный"][1])) if n else 0
    for name in ("различение", "пучок"):
        t1 = float(np.mean(res[name][1])) if n else 0
        verdict = "СТРОИТЬ" if t1 >= 0.5 else ("ЗАКРЫТ" if t1 <= ch + 0.05 else "НЕОПРЕДЕЛЁННО")
        print("  порог для «%s»: top-1 %.2f при случайном %.2f -> %s" % (name, t1, ch, verdict))
    json.dump({"res": {k: {kk: vv for kk, vv in v.items()} for k, v in res.items()}, "per_game": per_game},
              open(ROOT / "runs/infogain_rank_19_09.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

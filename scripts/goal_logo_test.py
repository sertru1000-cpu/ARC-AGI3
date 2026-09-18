"""Leave-one-game-out: предсказание цели СЛЕДУЮЩЕГО уровня по цели предыдущего (18.09, замер №1 критика раунда 9).

Зачем. Наши прежние числа (вид цели переносится в 80% переходов, конкретный предикат в 33%) получены на тех же играх,
на которых отбирался словарь, поэтому могут быть внутриигровым сходством. Критик: «если при leave-game-out 80% -> 30%,
весь сигнал — внутриигровое сходство». Здесь доля переноса по видам оценивается на ОСТАЛЬНЫХ 14 играх, а проверяется
на отложенной, и отдельно сравниваются три способа переноса ПАРАМЕТРА:
  point         -- то же число, что на уровне k;
  direction     -- монотонное направление (меньше/больше, чем на уровне k);
  distribution  -- небольшой набор значений вокруг наблюдённого, ранжированный (k, k-1, k+1, 0, 1, 2).
Мера: Coverage@1 / @3 (верный предикат среди первых 1/3 предсказаний) и доля ложных (предсказание истинно в
каком-то не-целевом состоянии уровня k+1).

Плюс КОНТРОЛЬ ложного отделения (критик, п.4): случайное не-целевое состояние объявляется «псевдоцелью», и по нему
запускается ПОЛНЫЙ подбор словаря с параметрами; считается, как часто хоть один предикат отделяет псевдоцель.
Это честнее прежних 8%, где параметры брались из настоящей цели.

Структурные предикаты (только переносимые семейства, каждый проверяется напрямую, без замыканий):
  gone(c)        -- клеток цвета c нет
  count(c)=k     -- клеток цвета c ровно k
  comps(c)=m     -- областей цвета c ровно m
  one(c)         -- клетки цвета c образуют одну область
  joined(c)      -- весь цвет c в одной области
  rect(c)        -- каждая область цвета c -- прямоугольник
  ncolors=k      -- различных цветов ровно k
  fewer(c,k0)    -- областей цвета c меньше, чем k0

usage:  .venv/bin/python scripts/goal_logo_test.py [--trials 20] [--out runs/goal_logo_18_09.json]
"""
import argparse, json, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import logging; logging.disable(logging.ERROR)
import arc_agi
from arc_agi import OperationMode
from goal_cross_level_replay import load_events, replay_collect, DEFAULT_RUNS
from goal_predicates import features


def feats(g):
    return features(g, np.zeros_like(g, dtype=bool))


def check(pred, f):
    """истинность структурного предиката на состоянии."""
    t = pred[0]
    if t == "gone":
        return f["colors"].get(pred[1], 0) == 0
    if t == "count":
        return f["colors"].get(pred[1], 0) == pred[2]
    if t == "comps":
        return len(f["comps"].get(pred[1], [])) == pred[2]
    if t == "one":
        return len(f["comps"].get(pred[1], [])) == 1
    if t == "joined":
        cs = f["comps"].get(pred[1], [])
        return bool(cs) and max(x["n"] for x in cs) == f["colors"].get(pred[1], 0)
    if t == "rect":
        cs = f["comps"].get(pred[1], [])
        return bool(cs) and all(x["n"] == (x["bbox"][2] - x["bbox"][0] + 1) * (x["bbox"][3] - x["bbox"][1] + 1) for x in cs)
    if t == "ncolors":
        return f["ncolors"] == pred[1]
    if t == "fewer":
        n = len(f["comps"].get(pred[1], []))
        return 0 < n < pred[2]
    return False


def candidates_from(f_goal, f_start):
    """все структурные предикаты, которые МОГУТ описывать эту цель (параметры берутся из целевого кадра)."""
    out = []
    bg = f_goal["bg"]
    cols = set(f_goal["colors"]) | set(f_start["colors"] if f_start else {})
    for c in cols:
        if c == bg:
            continue
        k = f_goal["colors"].get(c, 0)
        if k == 0:
            out.append(("gone", c))
        else:
            out.append(("count", c, k))
            out.append(("comps", c, len(f_goal["comps"].get(c, []))))
            if len(f_goal["comps"].get(c, [])) == 1:
                out.append(("one", c))
            cs = f_goal["comps"].get(c, [])
            if cs and max(x["n"] for x in cs) == k:
                out.append(("joined", c))
            if cs and all(x["n"] == (x["bbox"][2] - x["bbox"][0] + 1) * (x["bbox"][3] - x["bbox"][1] + 1) for x in cs):
                out.append(("rect", c))
            if f_start:
                k0 = len(f_start["comps"].get(c, []))
                if k0 > 1 and len(cs) < k0:
                    out.append(("fewer", c, k0))
    out.append(("ncolors", f_goal["ncolors"]))
    return out


def separating(f_goal, f_negs, cands):
    return [p for p in cands if check(p, f_goal) and not any(check(p, x) for x in f_negs)]


def predict(pred, mode):
    """список предсказаний для следующего уровня из предиката предыдущего, по способу переноса параметра."""
    t = pred[0]
    if t in ("gone", "one", "joined", "rect"):
        return [pred]
    if t == "count":
        c, k = pred[1], pred[2]
        if mode == "point":
            return [("count", c, k)]
        if mode == "direction":
            return [("gone", c), ("count", c, 1)] if k <= 2 else [("count", c, k)]
        return [("count", c, k), ("count", c, max(0, k - 1)), ("count", c, k + 1), ("gone", c)]
    if t == "comps":
        c, m = pred[1], pred[2]
        if mode == "point":
            return [("comps", c, m)]
        if mode == "direction":
            return [("fewer", c, m + 1), ("one", c)]
        return [("comps", c, m), ("comps", c, m + 1), ("comps", c, max(1, m - 1)), ("one", c)]
    if t == "fewer":
        c, k0 = pred[1], pred[2]
        return [("fewer", c, k0)] if mode == "point" else [("fewer", c, k0), ("one", c), ("fewer", c, k0 + 2)]
    if t == "ncolors":
        k = pred[1]
        if mode == "point":
            return [("ncolors", k)]
        return [("ncolors", k), ("ncolors", k - 1), ("ncolors", k + 1)]
    return [pred]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", default="runs/goal_logo_18_09.json"); a = ap.parse_args()
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT / "environment_files"))
    rng = random.Random(0)
    per_game = {}
    for e in arc.get_environments():
        gid = e.game_id; best = {}
        for run in DEFAULT_RUNS:
            evs = load_events(run, gid)
            if not evs:
                continue
            for k, v in replay_collect(arc, gid, evs).items():
                if k not in best or len(v["negs"]) > len(best[k]["negs"]):
                    best[k] = v
        if len(best) < 1:
            continue
        lv = {}
        for k, v in best.items():
            goal = v["goal"]; negs = [g for g in v["negs"][-200:] if g.shape == goal.shape]
            if not negs:
                continue
            f_goal = feats(goal); f_negs = [feats(g) for g in negs]
            cands = candidates_from(f_goal, f_negs[0])
            lv[k] = {"f_goal": f_goal, "f_negs": f_negs, "sep": separating(f_goal, f_negs, cands)}
        if lv:
            per_game[gid[:4]] = lv
        print("собрано %s: уровней %d" % (gid[:4], len(lv)), flush=True)

    # --- доли переноса по видам, оцениваемые на ОСТАЛЬНЫХ играх ---
    def rates(exclude):
        n = defaultdict(int); s = defaultdict(int)
        for g, lv in per_game.items():
            if g == exclude:
                continue
            for k in sorted(lv):
                if k + 1 not in lv:
                    continue
                for p in lv[k]["sep"]:
                    n[p[0]] += 1
                    if any(check(q, lv[k + 1]["f_goal"]) and not any(check(q, x) for x in lv[k + 1]["f_negs"])
                           for q in predict(p, "point")):
                        s[p[0]] += 1
        # апостериорная оценка Beta(1+s, 1+n-s): средняя доля со сглаживанием (критик, п.3)
        return {t: (1 + s[t]) / (2 + n[t]) for t in n}

    rows = []
    for mode in ("point", "direction", "distribution"):
        hit1 = hit3 = tot = fp = 0
        for g, lv in per_game.items():
            r = rates(g)
            for k in sorted(lv):
                if k + 1 not in lv or not lv[k]["sep"]:
                    continue
                tot += 1
                ranked = sorted(lv[k]["sep"], key=lambda p: -r.get(p[0], 0.3))
                preds = []
                for p in ranked:
                    for q in predict(p, mode):
                        if q not in preds:
                            preds.append(q)
                ok = [i for i, q in enumerate(preds)
                      if check(q, lv[k + 1]["f_goal"]) and not any(check(q, x) for x in lv[k + 1]["f_negs"])]
                if ok and ok[0] == 0:
                    hit1 += 1
                if ok and min(ok) < 3:
                    hit3 += 1
                if preds and not (check(preds[0], lv[k + 1]["f_goal"])):
                    fp += 1
        rows.append({"mode": mode, "pairs": tot, "cov1": hit1, "cov3": hit3, "false_top1": fp})
        print("%-12s пар %d | Coverage@1 %d (%.0f%%) | Coverage@3 %d (%.0f%%) | первый ложен %d (%.0f%%)"
              % (mode, tot, hit1, 100 * hit1 / max(tot, 1), hit3, 100 * hit3 / max(tot, 1), fp, 100 * fp / max(tot, 1)), flush=True)

    # --- контроль: полный подбор словаря по ПСЕВДОЦЕЛИ (критик, п.4) ---
    tot = hit = 0
    for g, lv in per_game.items():
        for k, d in lv.items():
            if len(d["f_negs"]) < 5:
                continue
            for _ in range(a.trials):
                i = rng.randrange(len(d["f_negs"]))
                pseudo = d["f_negs"][i]; rest = [x for j, x in enumerate(d["f_negs"]) if j != i]
                cands = candidates_from(pseudo, rest[0] if rest else None)
                tot += 1
                if separating(pseudo, rest, cands):
                    hit += 1
    print("\nКОНТРОЛЬ (полный подбор словаря по случайному не-целевому состоянию): отделено %d из %d проб (%.0f%%)"
          % (hit, tot, 100 * hit / max(tot, 1)))
    # для сравнения: настоящая цель
    tg = sum(1 for lv in per_game.values() for d in lv.values() if d["sep"]); ta = sum(len(lv) for lv in per_game.values())
    print("для сравнения: настоящая цель отделена на %d уровнях из %d (%.0f%%)" % (tg, ta, 100 * tg / max(ta, 1)))
    json.dump({"modes": rows, "control_false_separation": {"hits": hit, "trials": tot},
               "real_goal_separated": {"levels": tg, "total": ta}},
              open(ROOT / a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("записано:", a.out)


if __name__ == "__main__":
    main()

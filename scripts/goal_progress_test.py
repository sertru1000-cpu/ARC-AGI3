"""Различает ли мера близости к цели ВЫИГРЫШНЫЕ ходы от прочих (18.09, решающий замер критика раунда 9).

Постановка критика: вопрос не «как угадать цель», а «превращается ли известная цель в измеримый прогресс».
Для каждого перехода s_t -> s_t+1 на известном решающем пути считаем d(s_t, G) и d(s_t+1, G), где G -- НАСТОЯЩИЙ
кадр взятия уровня, а d -- предметная мера (задача о назначениях по объектам: несовпадение формы + расстояние
центров, scripts/goal_distance.py). Сравниваем три набора:
  * выигрышный ход (тот, что стоит в решении);
  * ходы-братья из того же состояния (все прочие действия алфавита, разворачиваются на снимке среды);
  * случайный ход из того же состояния.
Меры: P(d уменьшилась) для каждого набора и AUC (насколько Δd отделяет выигрышный ход от братьев).

Критерий критика: если P(Δd > 0 | выигрышный) СИЛЬНО выше, чем у случайного, — сигнал для планировщика есть,
и надо строить дешёвый поиск по этой мере. Если разницы нет — линию целей закрывать и идти в исследование
пространства состояний.

usage:  .venv/bin/python scripts/goal_progress_test.py [--out runs/goal_progress_18_09.json]
"""
import argparse, copy, json, random, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import logging; logging.disable(logging.ERROR)
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
from goal_search import state_features
from goal_distance import match_cost


def dist(f, f_goal):
    """предметное расстояние до целевого кадра: среднее по цветам от стоимости сопоставления объектов."""
    cols = set(f["comps"]) | set(f_goal["comps"])
    if not cols:
        return 1.0
    return float(np.mean([match_cost(f["comps"].get(c, []), f_goal["comps"].get(c, [])) for c in cols]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="runs/goal_progress_18_09.json")
    ap.add_argument("--siblings", type=int, default=8); a = ap.parse_args()
    sol = json.load(open(ROOT / "runs/bfs_originals.json", encoding="utf-8"))
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT / "environment_files"))
    rng = random.Random(0)
    rows = []
    win_better = win_n = sib_better = sib_n = rnd_better = rnd_n = 0
    pairs_auc = []
    ranks = []   # (место оптимистично, место пессимистично, число вариантов)
    for gid, rec in sol.items():
        if not rec.get("solved"):
            continue
        path = [(n, p) for n, p in rec["path"]]
        env = arc.make(gid); fr = env.reset()
        # целевой кадр: кадры завершающего хода до переключения
        e2 = copy.deepcopy(env)
        for n, p in path[:-1]:
            e2.step(GameAction[n] if n != "RESET" else GameAction.RESET, data=p)
        fr_end = e2.step(GameAction[path[-1][0]], data=path[-1][1])
        frames = [np.asarray(x, dtype=np.int16) for x in (fr_end.frame or [])]
        if len(frames) < 2:
            continue
        mask = np.zeros_like(frames[0], dtype=bool)
        f_goal = state_features(frames[:-1][-1], mask)
        # проход по решению со снимками
        cur = arc.make(gid); fr = cur.reset()
        g = np.asarray(fr.frame[-1], dtype=np.int16)
        alphabet = [("ACTION%d" % i, None) for i in range(1, 6)]
        per_game = {"game": gid[:4], "steps": 0, "win_better": 0, "sib_better": 0, "sib_n": 0}
        for t, (n, p) in enumerate(path):
            f_now = state_features(g, mask); d0 = dist(f_now, f_goal)
            # выигрышный ход
            e_win = copy.deepcopy(cur)
            fr_w = e_win.step(GameAction[n] if n != "RESET" else GameAction.RESET, data=p)
            if fr_w is None or not fr_w.frame:
                break
            g_w = np.asarray(fr_w.frame[-1], dtype=np.int16)
            d_w = dist(state_features(g_w, mask), f_goal)
            win_n += 1; per_game["steps"] += 1
            if d_w < d0 - 1e-9:
                win_better += 1; per_game["win_better"] += 1
            # братья
            sibs = [x for x in alphabet if x[0] != n][:a.siblings]
            d_sibs = []
            for sn, sp in sibs:
                e_s = copy.deepcopy(cur)
                fr_s = e_s.step(GameAction[sn], data=sp)
                if fr_s is None or not fr_s.frame:
                    continue
                d_s = dist(state_features(np.asarray(fr_s.frame[-1], dtype=np.int16), mask), f_goal)
                d_sibs.append(d_s); sib_n += 1; per_game["sib_n"] += 1
                if d_s < d0 - 1e-9:
                    sib_better += 1; per_game["sib_better"] += 1
            if d_sibs:
                rd = rng.choice(d_sibs); rnd_n += 1
                if rd < d0 - 1e-9:
                    rnd_better += 1
                pairs_auc += [(1.0 if d_w < ds else 0.5 if d_w == ds else 0.0) for ds in d_sibs]
                # 19.09, пункт 3 критика раунда 10: МЕСТО выигрышного хода среди всех ходов из состояния по Δd.
                # Оптимистично -- ничьи в пользу выигрышного, пессимистично -- против; случайный угад = k/(n+1).
                better = sum(1 for ds in d_sibs if ds < d_w - 1e-9)
                ties = sum(1 for ds in d_sibs if abs(ds - d_w) <= 1e-9)
                ranks.append((1 + better, 1 + better + ties, 1 + len(d_sibs)))
            cur = e_win; g = g_w
        rows.append(per_game)
        print("%s: шагов %d, выигрышный ход приближает в %d, братья в %d из %d"
              % (per_game["game"], per_game["steps"], per_game["win_better"], per_game["sib_better"], per_game["sib_n"]), flush=True)
    auc = float(np.mean(pairs_auc)) if pairs_auc else 0.5
    print("\nИТОГ (%d игр):" % len(rows))
    print("  P(мера уменьшилась | ВЫИГРЫШНЫЙ ход)  = %d/%d = %.2f" % (win_better, win_n, win_better / max(win_n, 1)))
    print("  P(мера уменьшилась | ход-брат)        = %d/%d = %.2f" % (sib_better, sib_n, sib_better / max(sib_n, 1)))
    print("  P(мера уменьшилась | случайный ход)   = %d/%d = %.2f" % (rnd_better, rnd_n, rnd_better / max(rnd_n, 1)))
    print("  AUC (Δd отделяет выигрышный ход от братьев) = %.3f" % auc)
    if ranks:
        print("  МЕСТО выигрышного хода по Δd среди %d состояний (вариантов на состояние: медиана %d):"
              % (len(ranks), int(np.median([r[2] for r in ranks]))))
        for k in (1, 2, 3):
            opt = np.mean([r[0] <= k for r in ranks]); pes = np.mean([r[1] <= k for r in ranks])
            chance = np.mean([min(1.0, k / r[2]) for r in ranks])
            print("    top-%d: оптимистично %.2f, пессимистично %.2f, случайный угад %.2f" % (k, opt, pes, chance))
    json.dump({"ranks": ranks, "per_game": rows, "win": [win_better, win_n], "sib": [sib_better, sib_n],
               "rnd": [rnd_better, rnd_n], "auc": auc}, open(ROOT / a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("записано:", a.out)


if __name__ == "__main__":
    main()

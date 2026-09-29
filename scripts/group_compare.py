"""Сравнение двух СБОРОК по нескольким прогонам каждой (27.09, ночь v4 против v4-lite на поде).

paired_delta.py сравнивает прогон с прогоном; здесь у каждой сборки несколько прогонов на тех же 25 играх.
Два теста, оба без допущения о нормальности:
  1. перестановка меток прогонов: балл прогона = среднее final_score по играм; все разбиения прогонов на группы
     тех же размеров (или 20 000 случайных) -> доля разбиений с |Δ| не меньше наблюдённой;
  2. парный бутстрэп по играм: для каждой игры среднее по прогонам сборки A и сборки B, Δ игры = A − B;
     10 000 пересборок игр с возвратом -> 95% интервал среднего Δ. Мёртвые игры (0 у всех прогонов обеих сборок)
     в среднем остаются (балл прогона считается по всем 25), но в строке «живых игр» показаны отдельно.
Калибровка: одна и та же сборка, разбитая на две группы, должна давать p > 0.05 и интервал через 0.
usage: .venv/bin/python scripts/group_compare.py --a runs/pod_v4_a runs/pod_v4_b --b runs/pod_a4_persist runs/pod_a4b_persist
"""
from __future__ import annotations
import argparse, itertools, json, random, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(d):
    p = Path(d) if Path(d).is_absolute() else ROOT / d
    return {r["game_id"]: (r.get("final_score") or 0.0) for r in json.loads((p / "benchmark.json").read_text())["game_runs"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True); ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rnd = random.Random(a.seed)
    A = [load(d) for d in a.a]; B = [load(d) for d in a.b]
    games = sorted(set.intersection(*(set(x) for x in A + B)))
    run_score = lambda r: sum(r[g] for g in games) / len(games)
    sa = [run_score(r) for r in A]; sb = [run_score(r) for r in B]
    obs = statistics.mean(sa) - statistics.mean(sb)

    allr = sa + sb; n = len(allr); k = len(sa)
    splits = list(itertools.combinations(range(n), k)) if __import__("math").comb(n, k) <= 20000 else \
        [tuple(rnd.sample(range(n), k)) for _ in range(20000)]
    ge = 0
    for idx in splits:
        s = set(idx); xa = [allr[i] for i in s]; xb = [allr[i] for i in range(n) if i not in s]
        ge += abs(statistics.mean(xa) - statistics.mean(xb)) >= abs(obs) - 1e-12
    p_perm = ge / len(splits)

    dg = [statistics.mean(r[g] for r in A) - statistics.mean(r[g] for r in B) for g in games]
    live = [g for g in games if any(r[g] > 0 for r in A + B)]
    boots = sorted(statistics.mean(rnd.choice(dg) for _ in dg) for _ in range(10000))
    lo, hi = boots[249], boots[9749]; p_pos = sum(x > 0 for x in boots) / len(boots)

    print("A: %d прогонов  %s  среднее %.2f" % (len(sa), " ".join("%.2f" % x for x in sa), statistics.mean(sa)))
    print("B: %d прогонов  %s  среднее %.2f" % (len(sb), " ".join("%.2f" % x for x in sb), statistics.mean(sb)))
    print("Δ = A − B = %+.2f | перестановка прогонов: p = %.3f (%d разбиений)" % (obs, p_perm, len(splits)))
    print("бутстрэп по %d играм (живых %d): 95%% интервал Δ [%+.2f, %+.2f], P(Δ>0) = %.2f" % (len(games), len(live), lo, hi, p_pos))
    wins = sum(x > 0 for x in dg); losses = sum(x < 0 for x in dg)
    print("игры: A лучше %d, B лучше %d, поровну %d" % (wins, losses, len(dg) - wins - losses))
    verdict = "РАЗЛИЧИМО" if p_perm < 0.05 and (lo > 0 or hi < 0) else "НЕ РАЗЛИЧИМО при этом числе прогонов"
    print("вердикт: %s" % verdict)


if __name__ == "__main__":
    main()

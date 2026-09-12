"""Сколько шагов программа держится верной, прежде чем разойтись с реальностью?

ЗАЧЕМ. Замер правил приёма (09.09) дал пару чисел, которые не сходятся между собой: программы,
принятые на буфере в 12 переходов, верны на 92% на ЧУЖОЙ траектории, но поиск внутри них
находит путь к взятию уровня лишь в 3 играх из 12. Если бы ошибки были независимы, при 92%
на шаг модель держалась бы около двенадцати шагов подряд, и планирование работало бы. Значит
ошибки копятся не как независимые.

ЧТО МЕРЯЕТСЯ. Раскатка: берём состояние на шаге i, дальше ведём его ТОЛЬКО программой по
фактически сделанным действиям и сверяем с настоящим состоянием на каждом шаге. Горизонт —
число шагов до первого расхождения (или до None). Так меряется ровно то, чем пользуется
планировщик: цепочка предсказаний, а не одиночный переход.

ЧТО ЭТО РЕШАЕТ. Если горизонт два-три шага, то планировать весь путь до цели бессмысленно,
и правильная форма — короткий план с перепланированием каждый ход. Вся наша линия строила
полный маршрут, и тогда она была неверна по форме, а не по идее.

usage:
    .venv/bin/python scripts/wm_horizon.py
    .venv/bin/python scripts/wm_horizon.py --k 12 --max-steps 12
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_acceptance_rules import (HELPERS, candidates, pts, trajectory)  # noqa: E402

OWN_CHECK = r'''
_all = _wm_pairs(10000, level=_LEVEL)
_pre = _all[:_K]
_cov = _cor = 0
for _b, _a, _f in _pre:
    try:
        _p = predict(state_of(_b), _a); _t = state_of(_f)
    except Exception:
        continue
    if _p is None: continue
    _cov += 1
    if _wm_norm(_p) == _wm_norm(_t): _cor += 1
print("PRE n=%d cov=%d cor=%d" % (len(_pre), _cov, _cor))
'''

HORIZON = r'''
_all = _wm_pairs(10000, level=_LEVEL)
_hor = []
for _i in range(len(_all)):
    try:
        _s = state_of(_all[_i][0])
    except Exception:
        continue
    _steps = 0
    for _j in range(_i, min(_i + _MAX, len(_all))):
        _b, _a, _f = _all[_j]
        try:
            _n = predict(_s, _a); _t = state_of(_f)
        except Exception:
            _n = None
        if _n is None or _wm_norm(_n) != _wm_norm(_t):
            break
        _steps += 1; _s = _n
    _hor.append(_steps)
print("HOR n=%d list=%s" % (len(_hor), ",".join(str(x) for x in _hor)))
'''


def sandbox(code: str, state: dict, level: int, k: int, max_steps: int) -> str:
    head = HELPERS + f"\n_LEVEL = {level}\n_K = {k}\n_MAX = {max_steps}\n"
    out = pts.run_sandboxed_python(code=head + code, timeout_seconds=180, initial_state=state,
                                   action_handler=lambda a: (_ for _ in ()).throw(RuntimeError("нельзя")))
    return str(out.get("stdout", "") or "") + str(out.get("error", "") or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12, help="буфер приёма: первые K переходов своей траектории")
    ap.add_argument("--max-steps", type=int, default=12, help="докуда вести раскатку")
    ap.add_argument("--runs", nargs="*", default=["runs/flash_wm_v5", "runs/flash_wm_v9", "runs/flash_wm_v10"])
    ap.add_argument("--foreign", default="runs/flash_v1_phaseA")
    a = ap.parse_args()

    pool = candidates(a.runs)
    own_ev = {os.path.basename(f).split('_p0')[0][:4]: f
              for r in a.runs for f in glob.glob(os.path.join(r, "artifacts", "*_events.jsonl"))}
    frn_ev = {os.path.basename(f).split('_p0')[0][:4]: f
              for f in glob.glob(os.path.join(a.foreign, "artifacts", "*_events.jsonl"))}

    accepted, hor_all, per_game = [], [], {}
    for run, game, code in pool:
        if game not in own_ev or game not in frn_ev:
            continue
        own = trajectory(own_ev[game]); frn = trajectory(frn_ev[game])
        if own is None or frn is None:
            continue
        lvl_own = own["history"][0]["frame"]["level"]
        m = re.search(r"PRE n=(\d+) cov=(\d+) cor=(\d+)", sandbox(code + OWN_CHECK, own, lvl_own, a.k, a.max_steps))
        if not m:
            continue
        n, cov, cor = map(int, m.groups())
        if not (n >= 1 and cov == n and cor == n):     # тот же точный приём на буфере K
            continue
        accepted.append((game, code))
        lvl_frn = frn["history"][0]["frame"]["level"]
        mh = re.search(r"HOR n=(\d+) list=(\S*)", sandbox(code + HORIZON, frn, lvl_frn, a.k, a.max_steps))
        if not mh or not mh.group(2):
            continue
        hs = [int(x) for x in mh.group(2).split(",") if x != ""]
        hor_all += hs
        per_game.setdefault(game, []).extend(hs)

    print(f"программ, принятых точным приёмом на буфере {a.k}: {len(accepted)} в {len(per_game)} играх")
    if not hor_all:
        print("раскаток нет")
        return 1
    hor_all.sort()
    n = len(hor_all)
    print(f"раскаток с разных стартов: {n}\n")
    print("СКОЛЬКО ШАГОВ ПРОГРАММА ДЕРЖИТСЯ ВЕРНОЙ (чужая траектория)")
    print(f"  среднее {sum(hor_all)/n:.2f}, медиана {hor_all[n//2]}")
    for k in (0, 1, 2, 3, 5, 8, a.max_steps):
        share = sum(1 for x in hor_all if x >= k) / n
        print(f"  выдержали {k:2d} шагов и больше: {100*share:5.1f}%")
    one = sum(1 for x in hor_all if x >= 1) / n
    print(f"\nдля сверки: доля верных ОДИНОЧНЫХ переходов {100*one:.0f}%")
    if one < 1:
        import math
        exp = 1 / (1 - one) - 1 if one < 1 else float("inf")
        print(f"если бы ошибки были независимы, средний горизонт был бы {exp:.1f} шага; "
              f"измерено {sum(hor_all)/n:.2f}")
    print("\nпо играм (медиана горизонта):")
    for g in sorted(per_game):
        v = sorted(per_game[g])
        print(f"  {g}: медиана {v[len(v)//2]}, раскаток {len(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

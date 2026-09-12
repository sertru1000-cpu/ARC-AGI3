"""Оценка пакета программ: горизонт, старое правило, находится ли путь. Локально, без пода.

ЧЕСТНОСТЬ ЗАМЕРА. Программы пишутся по сводке наблюдений ОДНОГО прогона (по умолчанию базовый,
`runs/flash_v1_phaseA`), а меряются на ДРУГОЙ траектории тех же игр (по умолчанию `runs/flash_wm_v13`).
Так проверяется то, ради чего всё делается: описывает программа игру или только тот путь,
который ей показали.

ЧТО СЧИТАЕТСЯ ПО КАЖДОЙ ПРОГРАММЕ:
  * старое правило (точный приём) на данных, которые модель видела — приняли бы её раньше;
  * горизонт раскатки на ОТЛОЖЕННОЙ траектории — новое правило;
  * находит ли поиск внутри программы путь к состоянию, из которого взят уровень.

Сводка отвечает на вопрос дня: отбирает ли приём по горизонту программы, по которым
планирование работает, лучше, чем точный приём.

usage:
    .venv/bin/python scripts/wm_eval_programs.py programs.json
    .venv/bin/python scripts/wm_eval_programs.py programs.json --held-out runs/flash_wm_v11
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_acceptance_rules import HELPERS, definitions_only, pts, trajectory  # noqa: E402

EXACT = r'''
_all = _wm_pairs(400, level=_LEVEL)
_cov = _cor = 0
for _b, _a, _f in _all:
    try:
        _p = predict(state_of(_b), _a); _t = state_of(_f)
    except Exception:
        continue
    if _p is None: continue
    _cov += 1
    if _wm_norm(_p) == _wm_norm(_t): _cor += 1
print("EXACT n=%d cov=%d cor=%d" % (len(_all), _cov, _cor))
'''

HELD = r'''
wm_horizon(predict, state_of, max_steps=_STEPS)
_all = _wm_pairs(400, level=_LEVEL)
_win = None
for _i, (_b, _a, _f) in enumerate(_all):
    if wm_level(_f) != wm_level(_b):
        _win = _i; break
_found = 0
if _win is not None:
    _entry = state_of(_all[0][0]); _target = _wm_norm(state_of(_all[_win][0]))
    _acts = []
    for _b, _a, _f in _all:
        if _a not in _acts: _acts.append(_a)
    _seen = {_wm_norm(_entry)}; _q = _wm_deque([(_entry, 0)]); _nodes = 0
    while _q and _nodes < 60000:
        _s, _d = _q.popleft()
        if _d >= 20: continue
        for _a in _acts:
            _nodes += 1
            try: _n = predict(_s, _a)
            except Exception: _n = None
            if _n is None: continue
            _k = _wm_norm(_n)
            if _k == _target: _found = 1; break
            if _k in _seen: continue
            _seen.add(_k); _q.append((_n, _d + 1))
        if _found: break
print("PLAN win=%s found=%d" % ("-" if _win is None else _win, _found))
'''


def run(code: str, state: dict, level: int, steps: int) -> str:
    head = HELPERS + f"\n_LEVEL = {level}\n_STEPS = {steps}\n"
    out = pts.run_sandboxed_python(code=head + code, timeout_seconds=180, initial_state=state,
                                   action_handler=lambda a: (_ for _ in ()).throw(RuntimeError("нельзя")))
    return str(out.get("stdout", "") or "") + str(out.get("error", "") or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("programs")
    ap.add_argument("--source", default="runs/flash_v1_phaseA", help="откуда брались сводки")
    ap.add_argument("--held-out", default="runs/flash_wm_v13", help="на чём меряем")
    ap.add_argument("--steps", type=int, default=8, help="потолок раскатки")
    ap.add_argument("--hor-min", type=int, default=8, help="порог приёма по горизонту")
    ap.add_argument("--out", default="docs/wm_batch_rows.json")
    a = ap.parse_args()

    src = {os.path.basename(f).split('_p0')[0][:4]: f
           for f in glob.glob(os.path.join(a.source, "artifacts", "*_events.jsonl"))}
    held = {os.path.basename(f).split('_p0')[0][:4]: f
            for f in glob.glob(os.path.join(a.held_out, "artifacts", "*_events.jsonl"))}

    progs = [p for p in json.load(open(a.programs, encoding="utf-8"))
             if "def predict" in (p.get("code") or "") and "def state_of" in (p.get("code") or "")]
    print(f"программ на входе: {len(progs)} в {len({p['game'] for p in progs})} играх")

    rows, seen = [], set()
    for p in progs:
        g, code = p["game"], definitions_only(p["code"])
        if not code or g not in src or g not in held:
            continue
        key = (g, code)
        if key in seen:
            continue
        seen.add(key)
        s_state, h_state = trajectory(src[g]), trajectory(held[g])
        if s_state is None or h_state is None:
            continue
        me = re.search(r"EXACT n=(\d+) cov=(\d+) cor=(\d+)",
                       run(code + EXACT, s_state, s_state["history"][0]["frame"]["level"], a.steps))
        so = run(code + HELD, h_state, h_state["history"][0]["frame"]["level"], a.steps)
        mh = re.search(r"WM_HORIZON med=(-?\d+)", so)
        mp = re.search(r"PLAN win=(\S+) found=(\d)", so)
        if not (me and mh and mp):
            continue
        n, cov, cor = map(int, me.groups())
        rows.append({"game": g, "exact": bool(n and cov == n and cor == n),
                     "horizon": int(mh.group(1)), "win": mp.group(1) != "-",
                     "found": mp.group(2) == "1"})

    json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    ev = [r for r in rows if r["win"]]
    print(f"оценено программ (без повторов): {len(rows)}; из них на играх со взятым уровнем: {len(ev)}\n")
    print("%-30s %-9s %-8s %s" % ("правило приёма", "принято", "игр", "план найден"))
    for name, sel in (("точный приём (старое)", lambda r: r["exact"]),
                      (f"горизонт >= {a.hor_min} (новое)", lambda r: r["horizon"] >= a.hor_min),
                      ("оба сразу", lambda r: r["exact"] and r["horizon"] >= a.hor_min),
                      ("всё подряд (контроль)", lambda r: True)):
        acc = [r for r in ev if sel(r)]
        if not acc:
            print("%-30s %-9d %-8s %s" % (name, 0, "-", "-"))
            continue
        print("%-30s %-9d %-8d %d из %d (%.0f%%)" % (
            name, len(acc), len({r["game"] for r in acc}),
            sum(1 for r in acc if r["found"]), len(acc),
            100 * sum(1 for r in acc if r["found"]) / len(acc)))
    h = collections.Counter(r["horizon"] for r in rows)
    print("\nраспределение горизонтов:", dict(sorted(h.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

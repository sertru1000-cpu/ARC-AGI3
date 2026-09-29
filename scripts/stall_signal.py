"""Порог «застрял» для слоя сброса гипотезы — по записанным прогонам форка v3, без модели (26.09).

Реплей на локальном движке: для каждого хода — повтор ли пары (доска, ход) на этом уровне. Сигнал в момент хода m
уровня: ходов на уровне >= MIN_MOVES и доля повторов среди последних WIN ходов >= FRAC. Меряем:
  ложные тревоги — доля ВЗЯТЫХ уровней, где сигнал сработал до взятия (сброс помешал бы);
  охват — доля НЕвзятых последних уровней, где сигнал сработал (было бы что лечить).
usage: .venv/bin/python scripts/stall_signal.py
"""
import hashlib, json, logging, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
logging.disable(logging.ERROR)
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

def H(fr):
    return hashlib.blake2b(np.asarray(fr.frame[-1], dtype=np.int16).tobytes(), digest_size=8).hexdigest()

def levels(game_id, history):
    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT / "environment_files"))
    env = arc.make(game_id); fr = env.reset(); lv = fr.levels_completed or 0
    out = []; cur = []; seen = set(); h = H(fr)
    for rec in history:
        a = rec.get("action") or {}
        if not a.get("id") or a["id"] == "RESET":
            continue
        key = (h, a["id"], json.dumps(a.get("data"), sort_keys=True))
        cur.append(key in seen); seen.add(key)
        fr = env.step(GameAction[a["id"]], data=a.get("data"))
        if fr is None or not fr.frame:
            break
        h = H(fr); lv2 = fr.levels_completed or 0
        if lv2 != lv:
            out.append((True, cur)); cur = []; seen = set(); lv = lv2
    out.append((False, cur))
    return out

rows = []
for run in ("b1", "b2", "b4", "b5"):
    for g in json.loads((ROOT / f"runs/night_nextfork-{run}/benchmark.json").read_text())["game_runs"]:
        rows += levels(g["game_id"], g.get("history") or [])
print("уровней: взятых %d, невзятых %d" % (sum(t for t, _ in rows), sum(not t for t, _ in rows)))
for mn in (15, 20, 30):
    for win, frac in ((10, .5), (10, .7), (15, .6)):
        def fires(rep):
            return any(m + 1 >= mn and m + 1 >= win and sum(rep[m + 1 - win:m + 1]) >= frac * win for m in range(len(rep)))
        fa = [fires(r) for t, r in rows if t]; hit = [fires(r) for t, r in rows if not t]
        print("ходов>=%d окно %d доля>=%.1f: ложные тревоги %.0f%% | охват невзятых %.0f%%" % (mn, win, frac, 100 * sum(fa) / len(fa), 100 * sum(hit) / len(hit)))

# Порог только по числу ходов на уровне (повторы пар «доска+ход» v3 почти исключила: запрет повтора пустого хода)
tk = sorted(len(r) for t, r in rows if t); st_ = [len(r) for t, r in rows if not t]
for mn in (30, 40, 50, 60, 80):
    print("ходов на уровне >= %d: ложные тревоги %.0f%% | охват невзятых %.0f%%" % (
        mn, 100 * sum(x >= mn for x in tk) / len(tk), 100 * sum(x >= mn for x in st_) / len(st_)))

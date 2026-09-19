"""Петли по СМЫСЛОВОЙ похожести состояний против точного совпадения доски (19.09, пункт 6 критика раунда 10).

Вопрос критика: наш слой «список испробованного» (10.09, эффекта нет) узнавал возврат по ТОЧНОМУ хешу доски.
Если на доске есть часы/счётчики, меняющиеся почти каждым ходом, точный хеш не повторяется никогда, и слой
петель просто не видел (в пробе v2 он сработал всего 15 раз за прогон). Тогда провал слоя измерял не идею,
а слепоту ключа.

Что считаем по записанным партиям базы (бесплатно, модель не нужна):
  * маска «шумных» клеток игры: клетки, меняющиеся больше чем в половине переходов внутри уровня (часы, счётчики);
  * точный ключ состояния = (уровень, вся доска); смысловой = (уровень, доска без шумных клеток);
  * возврат = состояние после хода уже встречалось раньше на этом уровне (по соответствующему ключу);
  * отдельно «холостой ход» = доска не изменилась.
Сравниваем застрявшие партии (0-1 уровень) и здоровые (2+), в последней трети ходов -- там, где идёт застревание.

Решение: если доля возвратов по смыслу сильно выше, чем по точному ключу, -- слой анти-петли надо переделать
на смысловой ключ (правка для форка); если близка -- пункт 6 закрыт.

usage:  .venv/bin/python scripts/semantic_loops.py [--runs flash_v1_phaseA,public_flash_keithtyser,public_flash_tufa]
"""
import argparse, glob, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_game(path):
    seq = []
    for line in open(path, encoding="utf-8"):
        e = json.loads(line)
        if e.get("type") != "action" or not isinstance(e.get("board"), list):
            continue
        seq.append((int(e.get("level") or 0), np.asarray(e["board"], dtype=np.int16), str(e.get("action_display", ""))))
    return seq


def noisy_mask(seq, frac=0.5):
    """клетки, меняющиеся больше чем в frac переходов внутри одного уровня"""
    if len(seq) < 3:
        return None
    ch = np.zeros_like(seq[0][1], dtype=np.int32); n = 0
    for (l0, b0, _), (l1, b1, _) in zip(seq, seq[1:]):
        if l0 == l1 and b0.shape == b1.shape:
            ch += (b0 != b1); n += 1
    return (ch > frac * n) if n else None


def analyse(seq, mask):
    exact_seen = set(); sem_seen = set()
    rows = []
    prev_exact = prev_sem = None
    for lvl, b, act in seq:
        bs = b.copy()
        if mask is not None and mask.shape == bs.shape:
            bs[mask] = -1
        ke = (lvl, b.tobytes()); ks = (lvl, bs.tobytes())
        rows.append({"exact_rev": ke in exact_seen, "sem_rev": ks in sem_seen,
                     "exact_noop": prev_exact == ke, "sem_noop": prev_sem == ks})
        exact_seen.add(ke); sem_seen.add(ks); prev_exact, prev_sem = ke, ks
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="flash_v1_phaseA,public_flash_keithtyser,public_flash_tufa")
    ap.add_argument("--out", default="runs/semantic_loops_19_09.json")
    a = ap.parse_args()
    agg = defaultdict(lambda: defaultdict(int)); per = []
    for run in a.runs.split(","):
        rd = ROOT / "runs" / run
        levels = {}
        bj = rd / "benchmark.json"
        if bj.exists():
            for r in json.load(open(bj))["game_runs"]:
                levels[r["game_id"][:4]] = r.get("levels_completed") or 0
        for f in sorted(glob.glob(str(rd / "artifacts" / "*_p0_events.jsonl"))):
            g = os.path.basename(f)[:4]
            seq = load_game(f)
            if len(seq) < 9:
                continue
            mask = noisy_mask(seq)
            rows = analyse(seq, mask)
            tail = rows[len(rows) * 2 // 3:]
            cls = "застрявшие (0-1 ур.)" if levels.get(g, 0) <= 1 else "здоровые (2+ ур.)"
            d = {k: sum(r[k] for r in tail) for k in ("exact_rev", "sem_rev", "exact_noop", "sem_noop")}
            d["sem_only"] = sum(1 for r in tail if r["sem_rev"] and not r["exact_rev"])
            d["n"] = len(tail)
            for k, v in d.items():
                agg[cls][k] += v
            per.append({"run": run, "game": g, "levels": levels.get(g), "class": cls, "noisy_cells": int(mask.sum()) if mask is not None else 0, **d})
    print("последняя треть ходов каждой партии (3 прогона базы):")
    for cls, d in agg.items():
        n = max(1, d["n"])
        print("  %-22s ходов %5d | возврат: точный %4.0f%%, смысловой %4.0f%%, только смысловой %4.0f%% | холостых: точно %3.0f%%, по смыслу %3.0f%%"
              % (cls, d["n"], 100 * d["exact_rev"] / n, 100 * d["sem_rev"] / n, 100 * d["sem_only"] / n,
                 100 * d["exact_noop"] / n, 100 * d["sem_noop"] / n))
    noisy = [p for p in per if p["noisy_cells"] > 0]
    print("игр с «шумными» клетками (часы/счётчики): %d из %d партий; медиана шумных клеток %s"
          % (len(noisy), len(per), int(np.median([p["noisy_cells"] for p in noisy])) if noisy else 0))
    top = sorted(per, key=lambda p: -(p["sem_only"] / max(1, p["n"])))[:8]
    print("больше всего петель, невидимых точному ключу:")
    for p in top:
        print("  %s %-24s уровней %s | ходов в хвосте %3d | только смысловой возврат %3d (%.0f%%) | шумных клеток %d"
              % (p["game"], p["run"], p["levels"], p["n"], p["sem_only"], 100 * p["sem_only"] / max(1, p["n"]), p["noisy_cells"]))
    json.dump({"per": per, "agg": {k: dict(v) for k, v in agg.items()}}, open(ROOT / a.out, "w"), ensure_ascii=False, indent=1)
    print("записано:", a.out)


if __name__ == "__main__":
    main()

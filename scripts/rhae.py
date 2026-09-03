"""RHAE of a run, read from every source and reconciled -- never from summary.txt.

WHY THIS EXISTS. `summary.txt` carries the benchmark's own periodic snapshot,
and that snapshot is FLUSHED ON A TIMER. When a run outlives the last flush,
games that finished afterwards are missing from it. On 02.09 that cost us a
correct conclusion twice in one evening:

  * the batch-hint arm reported `mean score: 0.36` in summary.txt while its
    own `[finished]` lines summed to 5.87 -- the run took 50 minutes because
    games were reaching level 4, so the snapshot was two-thirds stale. The
    change was declared a failure and nearly reverted;
  * earlier the same evening, `per-level=` counters from `[finished]` (which
    rollback() wipes) produced "zero actions on level 2" and a whole false
    hypothesis about the level transition breaking.

Neither source is trustworthy alone, and they fail in OPPOSITE directions:
the snapshot misses late games, `[finished]` gets its counters rewound. So
this reports both and takes the max per game, the same rule conversion.py
already uses -- and it prints the disagreement rather than hiding it, because
a large gap is itself a signal that something about the run needs looking at.

usage:
    python scripts/rhae.py runs/stocklab_batch [runs/stocklab_own30 ...]
    python scripts/rhae.py --per-game runs/stocklab_batch
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FIN = re.compile(
    r"\[finished\] (?P<gid>\S+) state=(?P<state>\S+) level=(?P<lvl>\d+)/(?P<tot>\d+) "
    r"score=(?P<score>[\d.]+)"
)
SNAP = re.compile(r"^\s{2}(?P<gid>\S+): score=(?P<score>[\d.]+), levels=(?P<lvl>[\d.]+)/")
MEAN = re.compile(r"mean score:\s+([\d.]+)")
LOG_NAMES = ("stdout.log", "kernel_stdout.txt", "harness_stdout.log")


def short(game_id: str) -> str:
    return game_id.split("-")[0]


def read(run: Path) -> dict:
    log = next((run / n for n in LOG_NAMES if (run / n).exists()), None)
    fin: dict[str, tuple[float, int]] = {}
    snap: dict[str, tuple[float, int]] = {}
    if log is not None:
        for line in log.open(encoding="utf-8", errors="replace"):
            m = FIN.search(line)
            if m:
                g = short(m["gid"])
                cur = fin.get(g, (0.0, 0))
                fin[g] = (max(cur[0], float(m["score"])), max(cur[1], int(m["lvl"])))
                continue
            m = SNAP.match(line.rstrip("\n"))
            if m:
                g = short(m["gid"])
                cur = snap.get(g, (0.0, 0))
                snap[g] = (max(cur[0], float(m["score"])),
                           max(cur[1], int(float(m["lvl"]))))

    games = sorted(set(fin) | set(snap))
    best = {g: (max(fin.get(g, (0.0, 0))[0], snap.get(g, (0.0, 0))[0]),
                max(fin.get(g, (0.0, 0))[1], snap.get(g, (0.0, 0))[1]))
            for g in games}

    reported = None
    summary = run / "summary.txt"
    if summary.exists():
        m = MEAN.search(summary.read_text(encoding="utf-8"))
        if m:
            reported = float(m.group(1))

    n = len(games) or 1
    return {
        "run": run.name,
        "games": len(games),
        "rhae": sum(v[0] for v in best.values()) / n,
        "rhae_finished": sum(v[0] for v in fin.values()) / n,
        "rhae_snapshot": sum(v[0] for v in snap.values()) / n,
        "rhae_summary": reported,
        "scoring": sum(1 for v in best.values() if v[0] > 0),
        "levels": sum(v[1] for v in best.values()),
        "deepest": max((v[1] for v in best.values()), default=0),
        "per_game": best,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--per-game", action="store_true")
    args = ap.parse_args()

    out = [read(r) for r in args.runs if r.is_dir()]
    if not out:
        sys.exit("нет каталогов прогонов")

    print(f"{'прогон':24}{'игр':>5}{'RHAE':>8}{'взяли':>7}{'уровней':>9}{'глубже всего':>14}")
    for s in out:
        print(f"{s['run']:24}{s['games']:5d}{s['rhae']:8.2f}{s['scoring']:7d}"
              f"{s['levels']:9d}{s['deepest']:14d}")

    print(f"\n{'прогон':24}{'сводка':>9}{'снимки':>9}{'finished':>10}{'принято':>9}")
    for s in out:
        rep = f"{s['rhae_summary']:.2f}" if s["rhae_summary"] is not None else "—"
        flag = ""
        if s["rhae_summary"] is not None and abs(s["rhae_summary"] - s["rhae"]) > 0.05:
            flag = "   <- сводка расходится, доверять ей нельзя"
        print(f"{s['run']:24}{rep:>9}{s['rhae_snapshot']:9.2f}"
              f"{s['rhae_finished']:10.2f}{s['rhae']:9.2f}{flag}")

    if args.per_game:
        for s in out:
            print(f"\n{s['run']} — по играм (балл, уровней):")
            for g, (sc, lv) in sorted(s["per_game"].items(), key=lambda x: -x[1][0]):
                if sc > 0:
                    print(f"   {g:8} {sc:6.2f}  {lv} ур.")


if __name__ == "__main__":
    main()

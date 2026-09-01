"""Level-1 conversion for a run: the share of games where the first level was taken.

WHY THIS METRIC. The leaderboard averages over games, and score is very nearly
conversion x 3.52 (taking level 1 in every game at human baseline scores 3.52).
RHAE-mean is not usable for comparing builds: it is dominated by rare deep
games -- one lp85 with five levels moved a 25-game mean by 1.67 all by itself.

WHY IT IS AFFORDABLE. Conversion is a proportion whose unit of observation is a
GAME, not a run. One 49-game run yields 49 Bernoulli trials, so an A/B needs
2-3 runs per arm instead of the ~50 submissions the score's own spread demands.

Reads a run directory (Kaggle kernel output or pod experiment dir) and reports
per-game outcome plus the run's conversion. Two independent sources, because
each has a known failure mode:
  * `[finished]` lines in the log carry the session's own tally, but rollback()
    rewinds the engine and wipes those counters (dc22 showed actions=0 after 11
    real actions);
  * the periodic benchmark snapshot is a stale copy -- it misses games that
    finished after the last flush (ar25 and re86 in V41).
Taking the max of both is what makes the number honest.

usage:
    python scripts/conversion.py runs/kaggle_v41 [runs/kaggle_v40 ...]
    python scripts/conversion.py --json runs/kaggle_v41 > conv.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

FIN = re.compile(
    r"^\[finished\] (?P<gid>\S+) state=(?P<state>\S+) level=(?P<lvl>\d+)/(?P<tot>\d+) "
    r"score=(?P<score>[\d.]+) actions=(?P<act>\d+) tokens=(?P<tok>\d+)"
)
SNAP = re.compile(
    r"^\s{2}(?P<gid>\S+): score=(?P<score>[\d.]+), levels=(?P<lvl>[\d.]+)/\d+, "
    r"actions=(?P<act>\d+), tokens=(?P<tok>\d+)"
)
LOG_NAMES = ("kernel_stdout.txt", "stdout.log", "harness_stdout.log", "arc3-atlas.log")


def find_log(run: Path) -> Path | None:
    for name in LOG_NAMES:
        p = run / name
        if p.exists() and p.stat().st_size > 1000:
            return p
    # pod experiment dirs keep the log a level up, named after the run
    for p in sorted(run.parent.glob(f"{run.name}*.log")):
        if p.stat().st_size > 1000:
            return p
    return None


def short(game_id: str) -> str:
    """ar25-0c556536 -> ar25. Clones keep their own id."""
    return game_id.split("-")[0]


def read_run(run: Path) -> dict[str, dict]:
    """game -> {lvl, score, actions, tokens}, max over both accountings."""
    games: dict[str, dict] = {}

    def bump(gid: str, lvl: int, score: float, act: int, tok: int) -> None:
        d = games.setdefault(gid, {"lvl": 0, "score": 0.0, "actions": 0, "tokens": 0})
        d["lvl"] = max(d["lvl"], lvl)
        d["score"] = max(d["score"], score)
        d["actions"] = max(d["actions"], act)
        d["tokens"] = max(d["tokens"], tok)

    log = find_log(run)
    if log is not None:
        for line in log.open(encoding="utf-8", errors="replace"):
            m = FIN.match(line.strip())
            if m:
                bump(short(m["gid"]), int(m["lvl"]), float(m["score"]),
                     int(m["act"]), int(m["tok"]))
                continue
            m = SNAP.match(line.rstrip("\n"))
            if m:
                bump(short(m["gid"]), int(float(m["lvl"])), float(m["score"]),
                     int(m["act"]), int(m["tok"]))

    # third source, authoritative on levels when present: per-game event logs.
    # NOTE: `level_completed` is a key present on EVERY event line, so a
    # substring test for the key name matches everything -- match the VALUE.
    # (Cost that bug 20 minutes on 31.08: every run reported 100% conversion.)
    art = run / "artifacts"
    for ev in (art.glob("*_events.jsonl") if art.is_dir() else []):
        gid = short(ev.name.split("_events")[0])
        levels = 0
        try:
            for line in ev.open(encoding="utf-8", errors="replace"):
                if '"level_completed":true' in line.replace(" ", ""):
                    levels += 1
        except OSError:
            continue
        if levels:
            bump(gid, levels, 0.0, 0, 0)

    return games


def summarise(run: Path) -> dict:
    games = read_run(run)
    took = {g: v for g, v in games.items() if v["lvl"] > 0 or v["score"] > 0}
    n = len(games)
    return {
        "run": str(run),
        "games": n,
        "took_l1": len(took),
        "conversion": (len(took) / n) if n else 0.0,
        "actions": sum(v["actions"] for v in games.values()),
        "tokens": sum(v["tokens"] for v in games.values()),
        "per_game": {g: {"took_l1": bool(v["lvl"] > 0 or v["score"] > 0), **v}
                     for g, v in sorted(games.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--per-game", action="store_true", help="list every game")
    args = ap.parse_args()

    out = []
    for run in args.runs:
        if not run.is_dir():
            print(f"нет каталога: {run}", file=sys.stderr)
            continue
        out.append(summarise(run))

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return

    print(f"{'прогон':34} {'игр':>4} {'взяли L1':>9} {'конверсия':>10} {'действий':>9}")
    for s in out:
        print(f"{Path(s['run']).name:34} {s['games']:4d} {s['took_l1']:9d} "
              f"{s['conversion']*100:9.0f}% {s['actions']:9d}")
        if args.per_game:
            for g, v in s["per_game"].items():
                mark = "+" if v["took_l1"] else "."
                print(f"      {mark} {g:8} действий {v['actions']:4d}  токенов {v['tokens']:7d}")
    if out:
        print(f"\nбалл ≈ конверсия × 3.52  ->  " + ", ".join(
            f"{Path(s['run']).name}: {s['conversion']*3.52:.2f}" for s in out))


if __name__ == "__main__":
    main()

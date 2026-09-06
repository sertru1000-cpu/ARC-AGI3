"""Retune the ASCII short games to the PUBLIC difficulty curve.

Why. Measured 03.09 against runs/calib_1: the public 25 need a median of 30
actions on level 1 and carry 6-10 levels, doubling to ~54 on level 2. Our 24
short games need a median of 6.5 on level 1 and carry five flat levels. An
agent that clears a 6-action level has shown nothing about a 30-action one,
and a 5-level game scores 6.7 for one level where a 9-level game scores 1.8 --
so neither the difficulty nor the RHAE arithmetic was comparable.

How. Each level is an ASCII map, so a level can be MUTATED (flip a floor to
wall or back, move the start) and re-measured. The optimum comes from
scripts/verify_short_baselines.py -- a BFS through the real engine -- so a
level is accepted only when its true optimum is known, never estimated. That
is the same rule the long batch follows: no baseline is ever hand-counted.

Mutation, not generation from scratch, is deliberate: it keeps each game's own
alphabet and object counts intact, so a tuned level still means what the game
means. A level whose optimum comes back None is simply rejected.

usage:
    python scripts/tune_short_levels.py --game cv01 --dry-run
    python scripts/tune_short_levels.py --game cv01 --write
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import verify_short_baselines as V  # noqa: E402

CURVE = [30, 54, 51, 54, 96, 80, 86, 90, 95]

# The four hostile games carry semantic guarantees a blind mutation can break
# (hc01's combo must not be bypassable, tr01 must keep a dead pocket behind
# every gate). scripts/hostile_baselines.py owns those checks; until they are
# wired in here, these are left alone rather than quietly corrupted.
HOSTILE = {"ch01", "hc01", "tr01", "vn01"}

FLOOR, WALL = ".", "#"


def load(gid: str):
    py = next((ROOT / "our_games" / gid).glob(f"*/{gid}.py"))
    md = json.loads((py.parent / "metadata.json").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location(f"tune_{gid}", py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, md["class_name"]), md, mod, py


def measure(mod, cls, rows_by_level, level: int, limit: int) -> int | None:
    """Optimum of `level` with LEVELS temporarily set to `rows_by_level`."""
    saved = [dict(s) for s in mod.LEVELS]
    try:
        for i, rows in enumerate(rows_by_level):
            if i < len(mod.LEVELS):
                mod.LEVELS[i]["rows"] = rows
        got, _ = V.optimum(cls, mod, level, limit)
        return got
    finally:
        for i, s in enumerate(saved):
            mod.LEVELS[i] = s


def mutate(rows: list[str], rng: random.Random) -> list[str]:
    """Flip one interior cell between floor and wall, keeping the border."""
    grid = [list(r) for r in rows]
    h, w = len(grid), len(grid[0])
    for _ in range(40):
        r = rng.randrange(1, h - 1)
        c = rng.randrange(1, w - 1)
        ch = grid[r][c]
        if ch == FLOOR:
            grid[r][c] = WALL
        elif ch == WALL:
            grid[r][c] = FLOOR
        else:
            continue          # never move a game's own objects
        return ["".join(x) for x in grid]
    return rows


def tune_level(mod, cls, base_rows: list[str], level: int, target: int,
               rng: random.Random, steps: int, limit: int):
    """Hill-climb the map toward `target`, keeping only measured optima."""
    cur = list(base_rows)
    cur_opt = measure(mod, cls, [cur], level, limit)
    if cur_opt is None:
        return None, None
    best, best_opt = cur, cur_opt
    for _ in range(steps):
        cand = mutate(cur, rng)
        opt = measure(mod, cls, [cand], level, limit)
        if opt is None:
            continue
        if abs(opt - target) < abs(best_opt - target):
            best, best_opt = cand, opt
            cur, cur_opt = cand, opt
        elif abs(opt - target) <= abs(cur_opt - target):
            cur, cur_opt = cand, opt        # sideways move, keeps it moving
        if best_opt == target:
            break
    return best, best_opt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--game", action="append", required=True)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--limit", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    for gid in args.game:
        if gid in HOSTILE:
            print(f"{gid}: пропуск -- враждебная игра, у неё свои гарантии")
            continue
        cls, md, mod, py = load(gid)
        if not all("rows" in s for s in mod.LEVELS):
            print(f"{gid}: пропуск -- уровни не ASCII-картами")
            continue
        rng = random.Random(args.seed)
        print(f"\n{gid}: {md['baseline_actions']}  ->  цель {CURVE[:len(mod.LEVELS)]}")
        got = []
        for lv in range(len(mod.LEVELS)):
            rows, opt = tune_level(mod, cls, mod.LEVELS[lv]["rows"], lv,
                                   CURVE[lv], rng, args.steps, args.limit)
            got.append((rows, opt))
            print(f"   уровень {lv+1}: цель {CURVE[lv]:3} -> получено {opt}")
        if args.write and all(o for _, o in got):
            print(f"   (запись пока не реализована -- сначала посмотреть, что выходит)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

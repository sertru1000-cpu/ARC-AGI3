"""Rebuild the COORDINATE short games on a full-size board, along the public curve.

Five games give their levels as coordinates rather than as an ASCII map, so
scripts/regen_short_levels.py skips them: lz01 and mr01 (laser, mirrors),
pi01 (pipes), mn01 (manipulator), rg01 (Rube). Together they are a sixth of
the testbed, and they carry the same defect as the rest -- level 1 costs 1-15
actions where the public median is 30, on a board filling a quarter of the frame.

The recipe is the same as for the map games, and it works for the same reason:
keep WHAT the level contains, randomise WHERE, and let a BFS through the real
engine (scripts/verify_short_baselines.py) decide whether the result is
solvable and what it truly costs. Nothing is hand-counted, and an unsolvable
layout is discarded rather than shipped.

The tuple layouts below were READ out of each game's own parser, not inferred:
lz01/mr01 unpack mirrors as `(o, c, r)` and the emitter as `ec, er, (dx, dy)`;
pi01 unpacks `sc, sr, sport` and keys `tiles` by `(c, r)`; rg01 unpacks pieces
as `(kind, orient, c, r)`; mn01 unpacks shapes as `(form, c0, r0, period)`.
Guessing here would write coordinates into fields that mean something else and
still produce a game that runs.

Border membership is preserved: lz01's emitter sits at column 0 and pi01's
source and sink sit on opposite edges, so a piece that was on the boundary is
placed on the new boundary and an interior piece stays interior. Without that
rule almost every candidate comes back unsolvable.

`solution=` is DROPPED from regenerated levels. No game reads it -- it is the
author's note of a known-good answer, confirmed by grep -- and a note that no
longer matches its layout is worse than no note at all.

usage:
    python scripts/regen_coord_levels.py --game lz01 --tries 300
    python scripts/regen_coord_levels.py --game lz01 --tries 300 --write
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import verify_short_baselines as V  # noqa: E402

CURVE = [30, 54, 51, 54, 96, 80, 86, 90, 95]
NEW_GRID = 21
NEW_CELL = 3
DROP_FIELDS = {"solution"}

# field -> (shape, index of column, index of row) inside each tuple.
#   "one"  a single tuple
#   "many" a list of tuples
#   "keys" a dict keyed by (c, r)
COORD_FIELDS: dict[str, dict[str, tuple[str, int, int]]] = {
    "lz01": {"emitter": ("one", 0, 1), "mirrors": ("many", 1, 2), "targets": ("many", 0, 1)},
    "mr01": {"emitter": ("one", 0, 1), "mirrors": ("many", 1, 2), "targets": ("many", 0, 1)},
    "pi01": {"source": ("one", 0, 1), "sink": ("one", 0, 1), "tiles": ("keys", 0, 1)},
    "rg01": {"ball": ("one", 0, 1), "goal": ("one", 0, 1), "pieces": ("many", 2, 3)},
    "mn01": {"grip": ("one", 0, 1), "shapes": ("many", 1, 2)},
}


def on_border(c: int, r: int, grid: int) -> bool:
    return c in (0, grid - 1) or r in (0, grid - 1)


def pick(rng: random.Random, border: bool, grid: int, taken: set) -> tuple[int, int] | None:
    """A free cell, on the boundary or strictly inside it, as asked."""
    for _ in range(200):
        if border:
            if rng.random() < 0.5:
                c, r = rng.choice([0, grid - 1]), rng.randrange(1, grid - 1)
            else:
                c, r = rng.randrange(1, grid - 1), rng.choice([0, grid - 1])
        else:
            c, r = rng.randrange(1, grid - 1), rng.randrange(1, grid - 1)
        if (c, r) not in taken:
            taken.add((c, r))
            return c, r
    return None


def relocate(gid: str, spec: dict, rng: random.Random, old_grid: int) -> dict | None:
    """Same objects, same counts, new positions on the bigger board."""
    out = {k: v for k, v in spec.items() if k not in DROP_FIELDS}
    taken: set = set()
    for field, (shape, ic, ir) in COORD_FIELDS[gid].items():
        if field not in out:
            continue
        val = out[field]
        if shape == "one":
            t = list(val)
            p = pick(rng, on_border(t[ic], t[ir], old_grid), NEW_GRID, taken)
            if p is None:
                return None
            t[ic], t[ir] = p
            out[field] = tuple(t)
        elif shape == "many":
            new = []
            for item in val:
                t = list(item)
                p = pick(rng, on_border(t[ic], t[ir], old_grid), NEW_GRID, taken)
                if p is None:
                    return None
                t[ic], t[ir] = p
                new.append(tuple(t))
            out[field] = new
        elif shape == "keys":
            new = {}
            for key, v in val.items():
                p = pick(rng, on_border(key[ic], key[ir], old_grid), NEW_GRID, taken)
                if p is None:
                    return None
                new[p] = v
            out[field] = new
    return out


def write_game(gid: str, results, name: str) -> None:
    py = next((ROOT / "our_games" / gid).glob(f"*/{gid}.py"))
    src = py.read_text(encoding="utf-8")
    body = []
    for spec, _opt in results:
        fields = ",\n".join(f"         {k}={v!r}" for k, v in spec.items())
        body.append("    dict(\n" + fields + "),")
    header = (
        f"{name} = [\n"
        "    # 03.09: regenerated by scripts/regen_coord_levels.py on the full\n"
        "    # 64x64 board (GRID 8 -> 21, CELL 4 -> 3), nine levels along the\n"
        "    # public difficulty curve. The old five cost 1-15 actions where the\n"
        "    # public median level 1 is 30, so clearing them said nothing about\n"
        "    # the games we are scored on. Same objects in the same numbers, new\n"
        "    # positions; every level PROVEN by BFS through the engine.\n"
        "    # `solution=` dropped: no game reads it, and a stale note is worse\n"
        "    # than none.\n")
    block = header + "\n".join(body) + "\n]"
    src = re.sub(rf"^{name} = \[.*?^\]", lambda _m: block, src, count=1, flags=re.S | re.M)
    src = re.sub(r"^GRID = \d+$", f"GRID = {NEW_GRID}", src, count=1, flags=re.M)
    src = re.sub(r"^CELL = \d+$", f"CELL = {NEW_CELL}", src, count=1, flags=re.M)
    py.write_text(src, encoding="utf-8")

    md_path = py.parent / "metadata.json"
    md = json.loads(md_path.read_text(encoding="utf-8"))
    md["baseline_actions"] = [o for _, o in results]
    md_path.write_text(json.dumps(md, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--game", action="append", required=True)
    ap.add_argument("--tries", type=int, default=300)
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--limit", type=int, default=25_000)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    for gid in args.game:
        if gid not in COORD_FIELDS:
            print(f"{gid}: пропуск -- нет описания координатных полей")
            continue
        cls, md, mod = V.load(gid)
        name = "LEVELS" if hasattr(mod, "LEVELS") else "LAYOUTS"
        levels = getattr(mod, name)
        old_grid = mod.GRID
        # The module must believe in the bigger board while we search, or the
        # game clips every move to the old 8x8 and we measure new layouts
        # through a quarter-sized window.
        mod.GRID, mod.CELL = NEW_GRID, NEW_CELL
        rng = random.Random(args.seed)
        print(f"\n{gid}: было {md['baseline_actions']}")
        results = []
        for lvl, target in enumerate(CURVE):
            src = levels[min(lvl, len(levels) - 1)]
            best, deadline = None, time.time() + args.seconds
            for _ in range(args.tries):
                if time.time() > deadline:
                    break
                cand = relocate(gid, src, rng, old_grid)
                if cand is None:
                    continue
                saved = levels[:]
                try:
                    while len(levels) <= lvl:
                        levels.append(levels[-1])
                    levels[lvl] = cand
                    opt, _ = V.optimum(cls, mod, lvl, args.limit)
                finally:
                    levels[:] = saved
                if opt is None:
                    continue
                if best is None or abs(opt - target) < abs(best[1] - target):
                    best = (cand, opt)
                if best[1] == target:
                    break
            results.append(best)
            print(f"   уровень {lvl+1}: цель {target:3} -> {best[1] if best else 'нет решения'}")
        ok = [r for r in results if r]
        print(f"   получено уровней: {len(ok)}/{len(CURVE)}")
        if args.write and len(ok) == len(CURVE):
            write_game(gid, results, name)
            print(f"   записано: {[o for _, o in results]}")
        elif args.write:
            print("   НЕ ПИШУ: не все уровни получены")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

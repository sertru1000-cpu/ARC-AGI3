"""Rewrite hand-written sprite pixels as expressions in CELL.

The short games draw themselves with literal indices -- `px[0][1] = px[0][2] =
AGENT`, `px[1][3] = color`, `r in (0, 3)`. Every one of those silently assumes
CELL == 4, which pins the board at 16 cells (16*4 = 64, the frame limit in
arcengine.camera.Camera.MAX_DIMENSION) and with it pins how long a level can
be: a 16-cell maze tops out near a 70-action path, against public level-5
baselines of 96.

The literals are not replaced number-for-number -- that would be wrong.
`px[0][1] = px[0][2]` means "the middle of the top row", which is two pixels
at CELL 4 and one at CELL 3. Each pattern below is rewritten to the SHAPE it
was drawing, so the same idea renders at any cell size.

Proof of equivalence is external and mechanical: scripts/frame_fingerprint.py
hashes every level's rendered frame before and after, and at CELL == 4 the
hashes must match exactly. A shifted sprite passes every other check we have
-- the game still plays, the baseline still verifies -- so nothing here is
trusted to reading.

usage:
    python scripts/cellify_sprites.py --dry-run
    python scripts/cellify_sprites.py --write
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (pattern, replacement, what shape it draws). Order matters: the longest
# assignment chains are matched before their prefixes.
RULES: list[tuple[str, str, str]] = [
    # the inner block, written as a chain or as a nested loop over (1, 2)
    (r"px\[1\]\[1\] = px\[2\]\[2\] = px\[1\]\[2\] = px\[2\]\[1\] = (\w+)",
     r"_fill_inner(px, \1)", "сплошная середина"),
    (r"px\[1\]\[1\] = px\[1\]\[2\] = px\[2\]\[1\] = px\[2\]\[2\] = (\w+)",
     r"_fill_inner(px, \1)", "сплошная середина"),
    (r"for r in \(1, 2\):\n(\s+)for c in \(1, 2\):\n\s+px\[r\]\[c\] = (\w+)",
     r"_fill_inner(px, \2)", "сплошная середина"),
    # middle of an edge row/column
    (r"px\[0\]\[1\] = px\[0\]\[2\] = (\w+)", r"_fill_edge(px, 'top', \1)", "середина верхнего ряда"),
    (r"px\[3\]\[1\] = px\[3\]\[2\] = (\w+)", r"_fill_edge(px, 'bottom', \1)", "середина нижнего ряда"),
    (r"px\[1\]\[3\] = px\[2\]\[3\] = (\w+)", r"_fill_edge(px, 'right', \1)", "середина правого столбца"),
    (r"px\[1\]\[0\] = px\[2\]\[0\] = (\w+)", r"_fill_edge(px, 'left', \1)", "середина левого столбца"),
    # a full inner row spanning the sprite
    (r"px\[1\]\[i\] = px\[2\]\[i\] = (\w+)", r"_fill_row(px, i, \1)", "средние ряды по столбцу i"),
    # border tests written against the last index
    (r"\(0, 3\)", r"(0, CELL - 1)", "край"),
    (r"r in \(0, CELL - 1\) or c in \(0, CELL - 1\)",
     r"r in (0, CELL - 1) or c in (0, CELL - 1)", "край"),
]

HELPERS = '''

# --- pixel helpers, added 03.09 by scripts/cellify_sprites.py ---------------
# These replace literal pixel indices so the sprite renders at any CELL. The
# shapes are exactly what the literals drew at CELL == 4, which
# scripts/frame_fingerprint.py verifies by hashing every frame before and
# after the rewrite.

def _fill_inner(px, color):
    """Everything except the one-pixel border."""
    for _r in range(1, CELL - 1):
        for _c in range(1, CELL - 1):
            px[_r][_c] = color
    return px


def _fill_edge(px, side, color):
    """The middle stretch of one edge -- the part that is not a corner."""
    span = range(1, CELL - 1) if CELL > 2 else range(CELL)
    for _i in span:
        if side == "top":
            px[0][_i] = color
        elif side == "bottom":
            px[CELL - 1][_i] = color
        elif side == "left":
            px[_i][0] = color
        else:
            px[_i][CELL - 1] = color
    return px


def _fill_row(px, col, color):
    """The middle rows of one column."""
    for _r in range(1, CELL - 1):
        px[_r][col] = color
    return px
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--game", action="append")
    args = ap.parse_args()

    gids = args.game or sorted(
        p.split(os.sep)[-3] for p in glob.glob(str(ROOT / "our_games" / "*" / "*" / "metadata.json")))
    touched = 0
    for gid in gids:
        py = next((ROOT / "our_games" / gid).glob(f"*/{gid}.py"))
        src = orig = py.read_text(encoding="utf-8")
        hits = []
        for pat, rep, what in RULES:
            new, n = re.subn(pat, rep, src)
            if n:
                hits.append(f"{what} x{n}")
                src = new
        if src == orig:
            continue
        if "_fill_inner" in src and "def _fill_inner" not in src:
            anchor = re.search(r"^(CELL = \d+.*?\n(?:[A-Z_]+ = .*\n)*)", src, re.M)
            at = anchor.end() if anchor else 0
            src = src[:at] + HELPERS + src[at:]
        touched += 1
        print(f"{gid}: {', '.join(hits)}")
        if args.write:
            py.write_text(src, encoding="utf-8")
    print(f"\nигр затронуто: {touched}" + ("" if args.write else "  (пробный прогон, ничего не записано)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

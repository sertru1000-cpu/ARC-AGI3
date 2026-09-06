"""Re-pick the long batch's per-level layouts so no two levels are the same.

Why. The first tuner searched (ring inset, mark count, offset) per level and
kept whatever landed closest to that level's target. Nothing stopped it from
choosing the SAME triple for several levels, and it did: ky01 shipped nine
levels of which only four were distinct, with levels 5-9 byte-identical.

That is not a cosmetic flaw. RHAE weights a level by its index, so levels 5-9
carry 35 of a nine-level game's 45 total weight -- an agent that cracked level
5 collected three quarters of the game by replaying one solution, with nothing
left to discover. It is the same defect as padding a short game with copies of
its last level, which is what this rebuild exists to remove.

Per-level baselines were never wrong: each is a true optimum, proven by BFS
over the game's own pure model. That is exactly why the baseline check could
not catch this -- a duplicate level is individually correct. The property that
has to hold is between levels, so it is enforced here.

usage:
    python scripts/retune_long_layouts.py            # print the table
    python scripts/retune_long_layouts.py --write    # write it into gen_long_games.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("glg", ROOT / "scripts" / "gen_long_games.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

COUNTS = {
    "gk01": range(2, 7),
    "ac01": range(2, 9),
    "ky01": range(2, 6),
    "fr01": range(1, 6),
    "rl01": range(2, 6),
}
OFFSETS = (0, 2, 3, 5, 7, 9, 11)


def spec_for(gid: str, inset: int, count: int, off: int) -> dict:
    walls = G.ring_walls(inset)
    ring = G.perimeter(walls)
    n = len(ring)
    base = dict(walls=walls, start=ring[0])
    if gid == "gk01":
        base.update(pads=G.spread(ring, count, off), exit=ring[(n // 2 + off) % n])
    elif gid == "ac01":
        base.update(tokens=G.spread(ring, count, off), exit=ring[(n // 2 + off) % n])
    elif gid == "ky01":
        base.update(doors=G.spread(ring, count, off), exit=ring[(n - 2 - off) % n])
    elif gid == "fr01":
        base.update(crates=G.spread(ring, count, off + n // 4), depot=ring[2],
                    exit=ring[(n // 2 + off) % n])
    elif gid == "rl01":
        base.update(gates=G.spread(ring, count, off), switch=ring[(off + 3) % n],
                    exit=ring[(off + n // 2) % n])
    return base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    table: dict[str, list] = {}
    for gid in ("gk01", "ac01", "ky01", "fr01", "rl01"):
        mod = G.load(gid, G.render(gid))
        chosen: list = []
        used: set = set()
        want = G.LEVEL_PLAN.get(gid, len(G.PUBLIC_CURVE))
        curve = (G.PUBLIC_CURVE * 2)[:want]   # план может просить больше девяти
        for target in curve:
            best = None
            for inset in range(1, 9):
                for count in COUNTS[gid]:
                    for off in OFFSETS:
                        if (inset, count, off) in used:
                            continue          # a repeat is not a new level
                        opt = G.optimum(mod, spec_for(gid, inset, count, off))
                        if opt is None:
                            continue
                        d = abs(opt - target)
                        if best is None or d < best[0]:
                            best = (d, inset, count, off, opt)
            if best is None:
                chosen.append(None)
                continue
            used.add((best[1], best[2], best[3]))
            chosen.append([best[1], best[2], best[3], best[4]])
        table[gid] = chosen
        got = [c[3] if c else None for c in chosen]
        uniq = len({(c[0], c[1], c[2]) for c in chosen if c})
        print(f"{gid}: {got}")
        print(f"      различных раскладок {uniq} из {len([c for c in chosen if c])}")

    if not args.write:
        print("\n(пробный прогон, таблица не записана)")
        return 0

    lines = "\n".join(
        f'    "{g}": {[(c[0], c[1], c[2]) for c in v if c]},' for g, v in table.items())
    p = ROOT / "scripts" / "gen_long_games.py"
    src = p.read_text(encoding="utf-8")
    src = re.sub(r"^LAYOUT = \{.*?^\}", "LAYOUT = {\n" + lines + "\n}", src,
                 count=1, flags=re.S | re.M)
    p.write_text(src, encoding="utf-8")
    print("\nтаблица вписана в scripts/gen_long_games.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

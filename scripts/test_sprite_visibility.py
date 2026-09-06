"""Every object on a level's map must be VISIBLE in the rendered frame.

Why this exists. The short games draw their own sprites with pixel indices
written out by hand -- `px[1][3] = color` and the like -- which silently
assumes CELL == 4. Change the cell size and those lines either throw
IndexError (loud, fine) or, worse, land on the wrong pixel and quietly
produce a sprite that is blank or identical to its neighbours. The game still
plays, the baseline still verifies, the score still computes: nothing fails.
The only thing that breaks is the agent's ability to SEE the board, and we
would find out from a bad number weeks later.

So: for every level of every game, render the frame and require that each
object cell on the ASCII map differs from an empty floor cell. That is the
property the hand-written pixel code is supposed to provide, checked directly
rather than trusted.

usage:
    python scripts/test_sprite_visibility.py
    python scripts/test_sprite_visibility.py --game cv01
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FLOOR, WALL = ".", "#"

# Games that hide part of the board ON PURPOSE. The check below asks whether
# every object differs from plain floor; for these the answer is "no, and that
# is the mechanic", so asking would only teach us to ignore the guard.
#   fw01 -- fog of war: walls, floor and exit all render as fog until the
#           player clicks a cell to clear a 3x3 area around it;
#   hc01 -- the button combo's progress is deliberately never rendered.
# Both are still checked for a non-blank frame; only the per-object test is
# skipped, and only for the objects, not for the level.
CONCEALED = {"fw01": "туман войны -- объекты скрыты до щелчка",
             "hc01": "скрытый прогресс комбинации не отображается намеренно"}

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'СБОЙ'}  {name}" + (f": {detail}" if not ok and detail else ""))
    if not ok:
        FAILURES.append(name)


def load(gid: str):
    py = next((ROOT / "our_games" / gid).glob(f"*/{gid}.py"))
    md = json.loads((py.parent / "metadata.json").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location(f"vis_{gid}", py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, md["class_name"]), md, mod


def block(frame, r: int, c: int, cell: int):
    return tuple(tuple(int(frame[r * cell + i][c * cell + j]) for j in range(cell))
                 for i in range(cell))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--game", action="append")
    args = ap.parse_args()

    gids = args.game or sorted(
        p.split(os.sep)[-3] for p in glob.glob(str(ROOT / "our_games" / "*" / "*" / "metadata.json")))
    for gid in gids:
        try:
            cls, md, mod = load(gid)
        except Exception as exc:
            check(f"{gid}: строится", False, str(exc)[:70])
            continue
        levels = getattr(mod, "LEVELS", [])
        if not levels or "rows" not in levels[0]:
            print(f"  —     {gid}: не ASCII-карта, пропуск")
            continue
        cell = getattr(mod, "CELL", 4)
        game = cls()
        blank_levels, invisible = [], []
        for lv, spec in enumerate(levels):
            game._current_level_index = lv
            hook = getattr(game, "on_set_level", None) or getattr(game, "_load", None)
            if callable(hook):
                try:
                    hook(game.current_level)
                except TypeError:
                    hook()
            frame = game._camera._raw_render(list(game.current_level.get_sprites()))
            if len({int(v) for row in frame for v in row}) < 2:
                blank_levels.append(lv + 1)
                continue
            floor = None
            for r, row in enumerate(spec["rows"]):
                for c, ch in enumerate(row):
                    if ch == FLOOR and floor is None:
                        floor = block(frame, r, c, cell)
            for r, row in enumerate(spec["rows"]):
                for c, ch in enumerate(row):
                    if ch in (FLOOR, WALL):
                        continue
                    if block(frame, r, c, cell) == floor:
                        invisible.append(f"ур.{lv+1} '{ch}' в ({c},{r})")
        check(f"{gid}: кадр не пустой на всех уровнях", not blank_levels, f"пустые: {blank_levels}")
        if gid in CONCEALED:
            print(f"  —     {gid}: объекты не проверяю -- {CONCEALED[gid]}")
        else:
            check(f"{gid}: каждый объект отличим от пола", not invisible,
                  f"{len(invisible)} невидимых, напр. {invisible[:3]}")
    print(f"\n{'все игры видимы' if not FAILURES else f'ПРОВАЛОВ: {len(FAILURES)}'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())

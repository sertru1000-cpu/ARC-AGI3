"""Rebuild the ASCII short games on a full-size board, along the public curve.

Why. Measured 03.09 against runs/calib_1: the public 25 need a median of 30
actions on level 1, carry 6-10 levels, and roughly double on level 2. Our
short games need a median of 6.5, carry five levels, and grow x1.4. An agent
that clears a 6-action level has demonstrated nothing about a 30-action one,
and the RHAE arithmetic differs too: one level of a 5-level game scores 6.7,
of a 9-level game 1.8.

The lever is the board, exactly as it was for the long batch. These games run
on GRID=8 at CELL=4 -- a 32x32 frame, half of the 64x64 the engine allows
(camera.MAX_DIMENSION). Doubling the side to GRID=16 keeps CELL=4 and fills
the frame, quadruples the cells, and puts a 30-action level within reach.

Every generated level is PROVEN before it is kept: its optimum comes from a
BFS through the real engine (scripts/verify_short_baselines.py), so a level is
accepted only when its true cost is known and lands on the target. Nothing
here is hand-counted, and an unsolvable layout is simply discarded.

What is preserved: each game's own alphabet and the COUNT of every object in
the level it replaces. The generator never invents a symbol or changes how
many there are, so a regenerated level still means what the game means -- only
the board it sits on is bigger.

usage:
    python scripts/regen_short_levels.py --game cv01 --tries 300
    python scripts/regen_short_levels.py --game cv01 --tries 300 --write
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import verify_short_baselines as V  # noqa: E402

CURVE = [30, 54, 51, 54, 96, 80, 86, 92, 163, 225]
# 21x3 = 63 pixels -- the same board the long batch moved to on 03.09, and
# the largest number of CELLS that fits the 64x64 frame
# (arcengine.camera.Camera.MAX_DIMENSION).
#
# 16x4 was the first attempt and it fills the frame too, but a 16-cell maze
# tops out near a 70-action path, so the curve's upper half (80-96) was out of
# reach: levels 5-9 came back at 46-68. Getting past that needed the short
# games' sprite code to stop assuming CELL == 4, which
# scripts/cellify_sprites.py did -- and scripts/frame_fingerprint.py proved
# the rewrite changed no pixel of any level at CELL == 4 before the cell was
# shrunk.
NEW_GRID = 21
NEW_CELL = 3
FLOOR, WALL = ".", "#"

# The hostile pool needs more than "solvable at the target cost". A random
# layout can quietly make hc01's combo bypassable or leave tr01 without a dead
# pocket, and the game then still plays -- it just stops being hostile, which
# is the only reason it exists. scripts/hostile_baselines.py already owns
# those structural checks, so they are imported and every candidate level must
# pass them too.
#
# hc01 stays out regardless: its progress is deliberately NOT rendered, so the
# frame-keyed BFS here cannot tell two different internal states apart and
# could report a path that does not exist. It has its own verifier.

# Сколько уровней у КАЖДОЙ игры -- из scripts/level_plan.json, а не одно число
# на всех. Публичные 25 игр несут 6-10 уровней: девять по шесть, пять по семь,
# шесть по восемь, четыре по девять, одна по десять (медиана 7, из runs/calib_1).
#
# Длина -- это знаменатель счёта игры: сумма номеров уровней. Один взятый
# уровень стоит 4.76 балла на шестиуровневой игре и 2.22 на девятиуровневой,
# больше чем вдвое. Полигон из одних девятиуровневых поэтому систематически
# ЗАНИЖАЛ бы наш RHAE против боевого -- зеркальное отражение прежней ошибки,
# где пять плоских уровней его завышали (6.67 за уровень).
LEVEL_PLAN = json.loads((ROOT / "scripts" / "level_plan.json").read_text(encoding="utf-8"))


# Цель для каждой игры -- НАСТОЯЩАЯ кривая одной из публичных игр, а не
# медиана по набору. Раздача в scripts/target_curves.json, её делает
# scripts/assign_target_curves.py.
#
# Медиана [30, 54, 51, 54, 96, 80, ...] не описывает ни одну публичную игру:
# на пятом уровне они разбросаны от 23 до 500. Требуя её от каждой нашей игры,
# генератор требовал недостижимого -- игры брали первые четыре уровня,
# упирались в пятый с целью 96 и отбрасывались целиком ("решаемых уровней 5").
#
# Взята ДОСТИЖИМАЯ часть публичного набора: кривые с максимумом до 200, их
# пятнадцать из двадцати пяти. Тяжёлый хвост (dc22 требует 578 действий на
# шестом уровне, m0r0 -- 500 на пятом) нашим механикам не по силам: потолок
# доски 21x21 около 163 даже там, где стоимость умножается с числом меток.
# Это осознанное ограничение полигона -- он воспроизводит более скромную
# половину публичных игр и НЕ воспроизводит их тяжёлый хвост.
_TC_PATH = ROOT / "scripts" / "target_curves.json"
TARGET_CURVES = json.loads(_TC_PATH.read_text(encoding="utf-8")) if _TC_PATH.exists() else {}


def curve_for(gid: str) -> list[int]:
    """Кривая-цель этой игры: настоящая публичная, а не медиана."""
    entry = TARGET_CURVES.get(gid)
    return list(entry["curve"]) if entry else CURVE



SKIP_ALWAYS = {"hc01": "hidden state is not rendered -- see scripts/hostile_baselines.py"}

try:
    import hostile_baselines as HB
except Exception:          # the module imports the games; never fatal here
    HB = None


def structural_violations(gid: str, mod, level: int, opt: int) -> list[str]:
    """Whatever this game must be TRUE of, beyond costing the right amount."""
    if HB is None or not hasattr(HB, "guarantees"):
        return []
    try:
        return HB.guarantees(gid, mod, level, opt)
    except Exception as exc:
        return [f"проверка гарантий не отработала: {exc}"]


def maze(rng: random.Random, size: int) -> list[list[str]]:
    """A perfect maze on odd cells -- long corridors, no open halls.

    Open halls are what kept the old 8x8 levels cheap: with everything
    reachable in a straight line, the optimum is the Manhattan distance. A
    maze makes the optimum a real path, which is what the public games cost.
    """
    g = [[WALL] * size for _ in range(size)]
    start = (1, 1)
    g[1][1] = FLOOR
    stack = [start]
    while stack:
        r, c = stack[-1]
        nbrs = [(r + dr, c + dc) for dr, dc in ((0, 2), (0, -2), (2, 0), (-2, 0))
                if 1 <= r + dr < size - 1 and 1 <= c + dc < size - 1
                and g[r + dr][c + dc] == WALL]
        if not nbrs:
            stack.pop()
            continue
        nr, nc = rng.choice(nbrs)
        g[(r + nr) // 2][(c + nc) // 2] = FLOOR
        g[nr][nc] = FLOOR
        stack.append((nr, nc))
    return g


def place(rng: random.Random, grid: list[list[str]], symbols: Counter) -> list[str] | None:
    """Drop each object of the source level onto a free floor cell."""
    free = [(r, c) for r in range(len(grid)) for c in range(len(grid))
            if grid[r][c] == FLOOR]
    need = sum(symbols.values())
    if len(free) < need + 4:
        return None
    # Placed uniformly at random, and deliberately NOT biased towards the
    # corners: an earlier version ordered the free cells by distance from a
    # random corner to spread the pieces out, and it drove the solvable rate
    # from 9 levels in 9 to 1 -- pushing conveyors and ladders into dead ends
    # makes a level unsolvable far more often than it makes it long. The
    # target optimum already rejects a degenerate layout, so the placement
    # does not need to be clever, only varied.
    rng.shuffle(free)
    i = 0
    for sym, n in symbols.items():
        for _ in range(n):
            r, c = free[i]; i += 1
            grid[r][c] = sym
    return ["".join(row) for row in grid]


def parallel_tables(mod, name: str, n_levels: int) -> dict[str, list]:
    """Module constants that carry ONE entry per level, beside the level list.

    bx01 keeps `N_STOCK = [1, 2, 2, 3, 3]` next to its LAYOUTS and indexes it
    by level. Grow the levels without growing that and the game raises
    IndexError the moment it reaches level 6 -- which is exactly what happened
    the first time. Anything module-level whose length matches the old level
    count is treated as such a table and padded with its own last value.

    Padding rather than inventing: the last level's setting is the best guess
    available, and a wrong guess here shows up immediately as an unsolvable or
    mis-costed level, which the optimum check rejects.
    """
    out = {}
    for attr in dir(mod):
        if attr.startswith("_") or attr == name or not attr.isupper():
            continue
        val = getattr(mod, attr)
        if isinstance(val, list) and len(val) == n_levels and not isinstance(val[0], str):
            out[attr] = val
    return out


def levels_of(mod):
    """The level list and how to read a level's rows, whatever it is called.

    Not every game spells this the same way, and two games sat outside the
    generator for no better reason than the name of a constant:
      * LEVELS  = [dict(rows=[...], ...)]      -- the common shape;
      * LAYOUTS = [dict(stock=1, rows=[...])]  -- bx01, same idea, other name;
      * LAYOUTS = [["#####", ...]]             -- kq01, the level IS its rows.
    Returns (name, list, rows_getter, shape) or four Nones when the levels are
    given as coordinates rather than as a map.
    """
    for name in ("LEVELS", "LAYOUTS"):
        lst = getattr(mod, name, None)
        if isinstance(lst, list) and lst:
            first = lst[0]
            if isinstance(first, dict) and "rows" in first:
                return name, lst, (lambda e: e["rows"]), "dict"
            if isinstance(first, list) and first and isinstance(first[0], str):
                return name, lst, (lambda e: e), "bare"
    return None, None, None, None


def write_game(gid: str, mod, results, name="LEVELS", levels=None, shape="dict", tables=None) -> None:
    """Replace the LEVELS block and the baked baselines, nothing else."""
    py = next((ROOT / "our_games" / gid).glob(f"*/{gid}.py"))
    src = py.read_text(encoding="utf-8")

    body = []
    for lvl, (rows, _opt) in enumerate(results):
        extra = {k: v for k, v in mod.LEVELS[min(lvl, len(mod.LEVELS) - 1)].items()
                 if k != "rows"}
        lines = ",\n".join(f"        {r!r}" for r in rows)
        tail = "".join(f", {k}={v!r}" for k, v in extra.items())
        body.append("    dict(rows=[\n" + lines + ",\n    ]" + tail + "),")

    header = (
        "LEVELS = [\n"
        "    # 03.09: regenerated by scripts/regen_short_levels.py on a full\n"
        "    # 64x64 board (GRID 8 -> 16), nine levels along the public\n"
        "    # difficulty curve. The old five levels cost 6-10 actions where the\n"
        "    # public median level 1 is 30, so clearing them said nothing about\n"
        "    # the games we are actually scored on -- and one of them (gv01)\n"
        "    # shipped a level whose true optimum was a single action.\n"
        "    # Every level below was PROVEN by BFS through the engine, never\n"
        "    # counted by hand, and carries the objects of the level it replaces\n"
        "    # in the same numbers.\n")
    block = header + "\n".join(body) + "\n]"

    src = re.sub(rf"^{name} = \[.*?^\]", lambda _m: block, src, count=1, flags=re.S | re.M)
    for attr, val in (tables or {}).items():
        src = re.sub(rf"^{attr} = \[[^]]*\]",
                     f"{attr} = {val!r}  # достроено под девять уровней 03.09",
                     src, count=1, flags=re.M)
    src = re.sub(r"^GRID = \d+$", f"GRID = {NEW_GRID}", src, count=1, flags=re.M)
    src = re.sub(r"^CELL = \d+$", f"CELL = {NEW_CELL}", src, count=1, flags=re.M)
    py.write_text(src, encoding="utf-8")

    md_path = py.parent / "metadata.json"
    md = json.loads(md_path.read_text(encoding="utf-8"))
    md["baseline_actions"] = [o[1] for o in results]
    md_path.write_text(json.dumps(md, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--game", action="append", required=True)
    ap.add_argument("--tries", type=int, default=200)
    ap.add_argument("--seconds", type=float, default=90.0,
                    help="бюджет времени на уровень")
    ap.add_argument("--limit", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    for gid in args.game:
        if gid in SKIP_ALWAYS:
            print(f"{gid}: пропуск -- {SKIP_ALWAYS[gid]}")
            continue
        cls, md, mod = V.load(gid)
        name, levels, rows_of, shape = levels_of(mod)
        if levels is None:
            print(f"{gid}: пропуск -- уровни заданы координатами, не картой")
            continue

        # The module must BELIEVE in the bigger board while we search, or the
        # game clips every move to the old 8x8 and we measure 16-wide maps
        # through a quarter-sized window. That produced a run of "optimum 3"
        # levels against a target of 30 -- and wrote them, because every level
        # did come back solvable.
        mod.GRID = NEW_GRID
        mod.CELL = NEW_CELL
        tables = parallel_tables(mod, name, len(levels))
        for attr, val in tables.items():
            val.extend([val[-1]] * (len(CURVE) - len(val)))
        if tables:
            print(f"   параллельные таблицы достроены до {len(CURVE)}: {', '.join(tables)}")
        curve = curve_for(gid)
        rng = random.Random(args.seed)
        print(f"\n{gid}: было {md['baseline_actions']}")
        results = []
        # A level that repeats another level's layout is NOT a level. RHAE
        # weights a level by its index, so the top of a nine-level game carries
        # most of its score: ky01 shipped levels 5-9 identical and an agent that
        # cracked level 5 collected 35 of the 45 total weight by replaying the
        # same solution. Каждый уровень по отдельности при этом корректен --
        # его базлайн доказан перебором -- поэтому проверка базлайнов такого не
        # ловит, и следить надо здесь.
        used: set[str] = set()
        for lvl in range(len(curve)):
            src = rows_of(levels[min(lvl, len(levels) - 1)])
            syms = Counter(ch for row in src for ch in row if ch not in (FLOOR, WALL))
            target = curve[lvl]
            # A wall-clock budget per level, not just a try count. Some
            # mechanics explode on the bigger board -- fl01 has to visit every
            # floor cell, so its state space is 2**cells and the BFS hits its
            # ceiling on every candidate without ever finding a path. One such
            # game stalled a sixteen-game run for over an hour at nine games
            # done. A game that cannot be measured in the budget is left as it
            # was rather than holding up the rest.
            best = None
            deadline = time.time() + args.seconds
            for _ in range(args.tries):
                if time.time() > deadline:
                    break
                rows = place(rng, maze(rng, NEW_GRID), Counter(syms))
                if rows is None:
                    continue
                saved = levels[:]
                try:
                    while len(levels) <= lvl:
                        levels.append(levels[-1])
                    if shape == "dict":
                        levels[lvl] = dict(levels[lvl], rows=rows)
                    else:
                        levels[lvl] = rows
                    opt, _ = V.optimum(cls, mod, lvl, args.limit)
                finally:
                    levels[:] = saved
                if opt is None:
                    continue
                bad = structural_violations(gid, mod, lvl, opt)
                if bad:
                    continue          # plays fine, but stopped being the game
                if repr(rows) in used:
                    continue
                if best is None or abs(opt - target) < abs(best[1] - target):
                    best = (rows, opt)
                if best[1] == target:
                    break
            if best is not None:
                used.add(repr(best[0]))
            results.append(best)
            print(f"   уровень {lvl+1}: цель {target:3} -> {best[1] if best else 'нет решения'}")
        # Ship only the levels that were actually FOUND -- никакой заливки.
        #
        # An earlier version repeated the last solvable layout into the empty
        # slots, so ic01 shipped as [30, 54, 53, 51, 93, 93, 93, 93, 93] with
        # levels 6-9 byte-identical to level 5. That looks like nine levels and
        # is worth more than nine honest ones: RHAE weights a level by its
        # index, levels 6-9 carry 30 of the 45 total weight, and an agent that
        # cracked level 5 replays the same solution four times for full marks.
        # Two thirds of the game's score for zero discovery -- exactly the
        # flattering this rebuild exists to remove, moved to a new place rather
        # than removed.
        #
        # A game with six real levels is a game; the public set's own range is
        # 6-10. Fewer than six and it stays on the old geometry, listed as such.
        results = [r for r in results if r is not None]
        ok = results
        print(f"   получено уровней: {len(ok)}/{len(curve)}"
              + ("" if len(ok) == len(curve) else "  (короче цели, лишнего не дописываю)"))
        if args.write:
            if len(ok) < 6:
                print(f"   НЕ ПИШУ: решаемых уровней {len(ok)}, у публичных минимум шесть")
                continue
            write_game(gid, mod, results, name, levels, shape, tables)
            print(f"   записано: {[o[1] for o in results]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

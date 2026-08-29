"""Testbed level factory (backlog 18, user order 30.08: «запускай генератор»).

Generates NEW validated level packs for the own-game mechanics whose
validators are battle-tested: fl01 (floor paint / Hamiltonian tour) and
ph01 (pharmacy / exact counting). Each pack is a standalone game dir in
our_games_gen/ (same clone recipe as build_clone_env_110.py: the original
game .py with only its LEVELS block replaced, metadata.json carrying the
new game_id + exact baselines from the validator).

Every level ships PROVEN: solvable by the validator, with a reachable
trap (fl01) / a real surplus (ph01), and an exact optimal baseline.

Usage:  python scripts/gen_testbed_levels.py [--per-game 6] [--seed 472901]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "our_games_gen"
GRID = 8
DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]


# ---------------------------------------------------------------- shared
def neighbors(cell, cells):
    return [(cell[0] + dx, cell[1] + dy) for dx, dy in DIRS
            if (cell[0] + dx, cell[1] + dy) in cells]


def connected(cells, anchor):
    if not cells:
        return True
    seen = {anchor}
    q = deque([anchor])
    while q:
        cur = q.popleft()
        for nb in neighbors(cur, cells):
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(cells)


# ---------------------------------------------------------------- fl01
def ham_path(floor, start, budget=400_000):
    path = [start]
    visited = {start}
    calls = [0]

    def dfs(cur):
        calls[0] += 1
        if calls[0] > budget:
            raise RuntimeError("budget")
        if len(path) == len(floor):
            return True
        rest = floor - visited
        nbs = [nb for nb in neighbors(cur, floor) if nb not in visited]
        if not nbs:
            return False
        if not connected(rest, next(iter(rest))):
            return False
        nbs.sort(key=lambda nb: sum(1 for x in neighbors(nb, floor) if x not in visited))
        for nb in nbs:
            path.append(nb)
            visited.add(nb)
            if dfs(nb):
                return True
            path.pop()
            visited.remove(nb)
        return False

    try:
        return dfs(start)
    except RuntimeError:
        return False


def trap_exists(floor, start, max_depth=12):
    q = deque([(start, frozenset({start}))])
    seen = {(start, frozenset({start}))}
    while q:
        pos, vis = q.popleft()
        if len(vis) > max_depth:
            continue
        nbs = [nb for nb in neighbors(pos, floor) if nb not in vis]
        if not nbs and len(vis) < len(floor):
            return True
        for nb in nbs:
            st = (nb, vis | {nb})
            if st not in seen:
                seen.add(st)
                q.append(st)
    return False


def gen_fl01_level(rng, floor_n):
    interior = [(c, r) for c in range(1, 7) for r in range(1, 7)]
    for _ in range(3000):
        # carve a connected region of floor_n cells by frontier growth --
        # connected by construction, so only Hamiltonicity needs searching
        seed_cell = rng.choice(interior)
        floor = {seed_cell}
        frontier = set(neighbors(seed_cell, set(interior)))
        while len(floor) < floor_n and frontier:
            cell = rng.choice(sorted(frontier))
            frontier.discard(cell)
            floor.add(cell)
            for nb in neighbors(cell, set(interior)):
                if nb not in floor:
                    frontier.add(nb)
        if len(floor) < floor_n:
            continue
        pillars = set(interior) - floor
        start = rng.choice(sorted(floor))
        if not ham_path(floor, start):
            continue
        if not trap_exists(floor, start):
            continue
        rows = []
        for r in range(GRID):
            row = ""
            for c in range(GRID):
                if r in (0, 7) or c in (0, 7) or (c, r) in pillars:
                    row += "#"
                elif (c, r) == start:
                    row += "P"
                else:
                    row += "."
            rows.append(row)
        return rows, len(floor) - 1
    raise RuntimeError(f"fl01: no layout at pillars={pillars_n}")


def gen_fl01_pack(rng):
    levels, baselines = [], []
    for floor_n in (12, 16, 20, 24, 28):    # escalating tour sizes
        rows, base = gen_fl01_level(rng, floor_n)
        levels.append({"rows": rows})
        baselines.append(base)
    return levels, baselines


# ---------------------------------------------------------------- ph01
PH_COLORS = ["R", "B", "Y", "M"]


def ph01_solve(rows, need):
    walls, pills, start, door = set(), {}, None, None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "D":
                door = (c, r)
            elif ch in PH_COLORS:
                pills[(c, r)] = ch

    def counts(taken):
        out = {}
        for cell in taken:
            ch = pills[cell]
            out[ch] = out.get(ch, 0) + 1
        return out

    q = deque([((start, frozenset()), 0)])
    seen = {(start, frozenset())}
    while q:
        (pos, taken), dist = q.popleft()
        got = counts(taken)
        exact = all(got.get(ch, 0) == n for ch, n in need.items())
        for dx, dy in DIRS:
            np_ = (pos[0] + dx, pos[1] + dy)
            if np_ == door:
                if exact:
                    return dist + 1
                continue
            if np_ in walls or not (0 <= np_[0] < GRID and 0 <= np_[1] < GRID):
                continue
            st = (np_, taken)
            if st not in seen:
                seen.add(st)
                q.append((st, dist + 1))
        if pos in pills and pos not in taken:
            ch = pills[pos]
            if got.get(ch, 0) < need.get(ch, 0):
                st = (pos, taken | {pos})
                if st not in seen:
                    seen.add(st)
                    q.append((st, dist + 1))
    return None


def gen_ph01_level(rng, n_colors, per_color_max, n_walls, trap_colors_n):
    for _ in range(3000):
        interior = [(c, r) for c in range(1, 7) for r in range(1, 7)]
        walls = set(rng.sample(interior, n_walls)) if n_walls else set()
        free = [cell for cell in interior if cell not in walls]
        if len(free) < 10 or not connected(set(free), free[0]):
            continue
        rng.shuffle(free)
        start = free.pop()
        door_col = rng.randrange(1, 7)
        colors = rng.sample(PH_COLORS, n_colors + trap_colors_n)
        need_colors, trap_colors = colors[:n_colors], colors[n_colors:]
        need = {ch: rng.randrange(1, per_color_max + 1) for ch in need_colors}
        for ch in trap_colors:
            need[ch] = 0
        placements = {}
        ok = True
        for ch in need_colors:
            supply = need[ch] + rng.randrange(1, 3)
            for _ in range(supply):
                if not free:
                    ok = False
                    break
                placements[free.pop()] = ch
        for ch in trap_colors:
            for _ in range(rng.randrange(1, 3)):
                if not free:
                    ok = False
                    break
                placements[free.pop()] = ch
        if not ok:
            continue
        rows = []
        for r in range(GRID):
            row = ""
            for c in range(GRID):
                if (c, r) == (door_col, 7):
                    row += "D"
                elif r in (0, 7) or c in (0, 7) or (c, r) in walls:
                    row += "#"
                elif (c, r) == start:
                    row += "P"
                elif (c, r) in placements:
                    row += placements[(c, r)]
                else:
                    row += "."
            rows.append(row)
        base = ph01_solve(rows, need)
        if base is None:
            continue
        return {"need": need, "rows": rows}, base
    raise RuntimeError("ph01: no layout")


def gen_ph01_pack(rng):
    levels, baselines = [], []
    for n_colors, per_max, n_walls, traps in (
        (1, 2, 0, 0), (2, 2, 0, 0), (3, 2, 4, 0), (2, 2, 4, 1), (3, 3, 5, 1),
    ):
        spec, base = gen_ph01_level(rng, n_colors, per_max, n_walls, traps)
        levels.append(spec)
        baselines.append(base)
    return levels, baselines


# ---------------------------------------------------------------- sk01
def sk01_solve(rows, node_cap=400_000):
    """Exact optimal move count (agent steps incl. pushes) via BFS over
    (agent, frozenset(boxes)); None if unsolvable / cap hit."""
    walls, boxes, targets, start = set(), set(), set(), None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "B":
                boxes.add((c, r))
            elif ch == "T":
                targets.add((c, r))
            elif ch == "*":
                boxes.add((c, r))
                targets.add((c, r))
    state0 = (start, frozenset(boxes))
    q = deque([(state0, 0)])
    seen = {state0}
    nodes = 0
    while q:
        (pos, bx), dist = q.popleft()
        nodes += 1
        if nodes > node_cap:
            return None
        for dx, dy in DIRS:
            np_ = (pos[0] + dx, pos[1] + dy)
            if np_ in walls or not (0 <= np_[0] < GRID and 0 <= np_[1] < GRID):
                continue
            if np_ in bx:
                nb = (np_[0] + dx, np_[1] + dy)
                if nb in walls or nb in bx or not (0 <= nb[0] < GRID and 0 <= nb[1] < GRID):
                    continue
                nbx = frozenset((nb if cell == np_ else cell) for cell in bx)
                if targets <= nbx:
                    return dist + 1
                st = (np_, nbx)
            else:
                st = (np_, bx)
            if st not in seen:
                seen.add(st)
                q.append((st, dist + 1))
    return None


def gen_sk01_level(rng, n_boxes, n_walls, min_base):
    interior = [(c, r) for c in range(1, 7) for r in range(1, 7)]
    for _ in range(4000):
        cells = list(interior)
        rng.shuffle(cells)
        walls = set(cells[:n_walls])
        free = [c for c in cells[n_walls:]]
        if len(free) < 2 * n_boxes + 1:
            continue
        boxes = free[:n_boxes]
        targets = free[n_boxes:2 * n_boxes]
        start = free[2 * n_boxes]
        # boxes on the border ring push-lock instantly against the outer
        # wall unless already on target -- keep generated boxes interior
        if any(c in (1, 6) or r in (1, 6) for c, r in boxes):
            continue
        rows = []
        for r in range(GRID):
            row = ""
            for c in range(GRID):
                if r in (0, 7) or c in (0, 7) or (c, r) in walls:
                    row += "#"
                elif (c, r) == start:
                    row += "P"
                elif (c, r) in boxes and (c, r) in targets:
                    row += "*"
                elif (c, r) in boxes:
                    row += "B"
                elif (c, r) in targets:
                    row += "T"
                else:
                    row += "."
            rows.append(row)
        base = sk01_solve(rows)
        if base is None or base < min_base:
            continue
        return {"rows": rows}, base
    raise RuntimeError(f"sk01: no layout at boxes={n_boxes}")


def gen_sk01_pack(rng):
    levels, baselines = [], []
    for n_boxes, n_walls, min_base in (
        (1, 2, 6), (1, 4, 10), (2, 3, 12), (2, 5, 16), (3, 4, 18),
    ):
        spec, base = gen_sk01_level(rng, n_boxes, n_walls, min_base)
        levels.append(spec)
        baselines.append(base)
    return levels, baselines


# ---------------------------------------------------------------- packaging
def fmt_fl01_levels(levels):
    parts = ["LEVELS = ["]
    for lv in levels:
        parts.append("    dict(rows=[")
        for row in lv["rows"]:
            parts.append(f'        "{row}",')
        parts.append("    ]),")
    parts.append("]")
    return "\n".join(parts)


def fmt_ph01_levels(levels):
    parts = ["LEVELS = ["]
    for lv in levels:
        parts.append(f"    dict(need={lv['need']!r}, rows=[")
        for row in lv["rows"]:
            parts.append(f'        "{row}",')
        parts.append("    ]),")
    parts.append("]")
    return "\n".join(parts)


LEVELS_RE = re.compile(r"\nLEVELS = \[.*?\n\]", re.DOTALL)

MECHANICS = {
    "fl01": {
        "src": ROOT / "our_games" / "fl01" / "b8c9d0e1" / "fl01.py",
        "class_name": "Fl01",
        "prefix": "fg",
        "gen": gen_fl01_pack,
        "fmt": fmt_fl01_levels,
        "title": "Floor Paint (generated)",
    },
    "ph01": {
        "src": ROOT / "our_games" / "ph01" / "a7b8c9d0" / "ph01.py",
        "class_name": "Ph01",
        "prefix": "pg",
        "gen": gen_ph01_pack,
        "fmt": fmt_ph01_levels,
        "title": "Pharmacy (generated)",
    },
    "sk01": {
        "src": ROOT / "our_games" / "sk01" / "c9d0e1f2" / "sk01.py",
        "class_name": "Sk01",
        "prefix": "sg",
        "gen": gen_sk01_pack,
        "fmt": fmt_fl01_levels,
        "title": "Sokoban (generated)",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-game", type=int, default=6)
    ap.add_argument("--seed", type=int, default=472901)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    ids = []
    for mech, cfg in MECHANICS.items():
        source = cfg["src"].read_text(encoding="utf-8")
        assert LEVELS_RE.search(source), f"{mech}: LEVELS block not found"
        for i in range(args.per_game):
            rng = random.Random(args.seed + hash(mech) % 10_000 + i * 131)
            levels, baselines = cfg["gen"](rng)
            body = LEVELS_RE.sub("\n" + cfg["fmt"](levels), source, count=1)
            ver = hashlib.md5(body.encode()).hexdigest()[:8]
            prefix = f"{cfg['prefix']}{i:02d}"
            game_id = f"{prefix}-{ver}"
            pack = OUT / prefix / ver
            pack.mkdir(parents=True, exist_ok=True)
            (pack / f"{cfg['class_name'].lower()}.py").write_text(body, encoding="utf-8", newline="\n")
            (pack / "metadata.json").write_text(json.dumps({
                "game_id": game_id,
                "class_name": cfg["class_name"],
                "title": f"{cfg['title']} #{i}",
                "baseline_actions": baselines,
            }, indent=4) + "\n", encoding="utf-8", newline="\n")
            ids.append(game_id)
            print(f"{game_id}: baselines={baselines}")
    (OUT / "game_ids.txt").write_text(",".join(ids), encoding="utf-8", newline="\n")
    print(f"wrote {len(ids)} pack(s) -> {OUT}")


if __name__ == "__main__":
    main()

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


# ---------------------------------------------------------------- pt01
def pt01_solve(rows):
    """(steps, used_portal) via BFS over agent position with portal edges."""
    walls, portals, start, exit_cell = set(), {}, None, None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "E":
                exit_cell = (c, r)
            elif ch in "123":
                portals.setdefault(ch, []).append((c, r))
    hop = {}
    for cells in portals.values():
        if len(cells) == 2:
            hop[cells[0]] = cells[1]
            hop[cells[1]] = cells[0]
    q = deque([((start, False), 0)])
    seen = {(start, False)}
    while q:
        ((pos, used), dist) = q.popleft()
        for dx, dy in DIRS:
            np_ = (pos[0] + dx, pos[1] + dy)
            if np_ in walls or not (0 <= np_[0] < GRID and 0 <= np_[1] < GRID):
                continue
            u = used
            if np_ in hop:
                np_ = hop[np_]
                u = True
            if np_ == exit_cell:
                return dist + 1, u
            st = (np_, u)
            if st not in seen:
                seen.add(st)
                q.append((st, dist + 1))
    return None, False


def gen_pt01_level(rng, n_pairs, n_walls, min_base):
    for _ in range(4000):
        interior = [(c, r) for c in range(1, 7) for r in range(1, 7)]
        rng.shuffle(interior)
        walls = set(interior[:n_walls])
        free = interior[n_walls:]
        need = 2 * n_pairs + 2
        if len(free) < need:
            continue
        cells = free[:need]
        start, exit_cell = cells[0], cells[1]
        rows = []
        portal_cells = {}
        for k in range(n_pairs):
            ch = "123"[k]
            portal_cells[cells[2 + 2 * k]] = ch
            portal_cells[cells[3 + 2 * k]] = ch
        for r in range(GRID):
            row = ""
            for c in range(GRID):
                if r in (0, 7) or c in (0, 7) or (c, r) in walls:
                    row += "#"
                elif (c, r) == start:
                    row += "P"
                elif (c, r) == exit_cell:
                    row += "E"
                elif (c, r) in portal_cells:
                    row += portal_cells[(c, r)]
                else:
                    row += "."
            rows.append(row)
        base, used_portal = pt01_solve(rows)
        if base is None or base < min_base:
            continue
        if not used_portal:
            continue  # the optimal path MUST thread a portal (the point)
        return {"rows": rows}, base
    raise RuntimeError(f"pt01: no layout at pairs={n_pairs}")


def gen_pt01_pack(rng):
    levels, baselines = [], []
    for n_pairs, n_walls, min_base in (
        (1, 4, 4), (1, 8, 6), (2, 8, 6), (2, 10, 8), (3, 10, 8),
    ):
        spec, base = gen_pt01_level(rng, n_pairs, n_walls, min_base)
        levels.append(spec)
        baselines.append(base)
    return levels, baselines


# ---------------------------------------------------------------- gv01
ACTS = {(0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4}


def gv01_settle(pos, walls, ladders):
    c, r = pos
    while True:
        if (c, r) in ladders:
            return (c, r)
        below = (c, r + 1)
        if below in walls or below in ladders or r + 1 >= GRID:
            return (c, r)
        r += 1


def gv01_solve_actions(spec):
    rows = spec["rows"]
    walls, ladders = set(), set()
    start = exit_cell = None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "H":
                ladders.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "E":
                exit_cell = (c, r)
    start = gv01_settle(start, walls, ladders)
    q = deque([(start, [])])
    seen = {start}
    while q:
        pos, path = q.popleft()
        c, r = pos
        for a in (1, 2, 3, 4):
            nc, nr = c, r
            if a == 1:
                if (c, r) in ladders and (c, r - 1) not in walls and r - 1 >= 0:
                    nr = r - 1
            elif a == 2:
                if (c, r + 1) not in walls and r + 1 < GRID:
                    nr = r + 1
            elif a == 3:
                if (c - 1, r) not in walls and c - 1 >= 0:
                    nc = c - 1
            else:
                if (c + 1, r) not in walls and c + 1 < GRID:
                    nc = c + 1
            st = gv01_settle((nc, nr), walls, ladders)
            if st == exit_cell:
                return path + [a]
            if st not in seen:
                seen.add(st)
                q.append((st, path + [a]))
    return None


def gen_gv01_level(rng, n_platforms, min_base):
    for _ in range(4000):
        walls = set()
        for c in range(GRID):
            walls.add((c, 0)); walls.add((c, 7))
        for r in range(GRID):
            walls.add((0, r)); walls.add((7, r))
        # random platform segments on interior rows
        for _p in range(n_platforms):
            r = rng.randrange(2, 7)
            c0 = rng.randrange(1, 5)
            ln = rng.randrange(2, 4)
            for c in range(c0, min(c0 + ln, 7)):
                walls.add((c, r))
        # 1-2 ladders
        ladders = set()
        for _l in range(rng.randrange(1, 3)):
            c = rng.randrange(1, 7)
            r0 = rng.randrange(2, 6)
            h = rng.randrange(2, 4)
            for r in range(r0, min(r0 + h, 7)):
                if (c, r) not in walls:
                    ladders.add((c, r))
        free = [(c, r) for c in range(1, 7) for r in range(1, 7)
                if (c, r) not in walls and (c, r) not in ladders]
        if len(free) < 2:
            continue
        rng.shuffle(free)
        start, exit_cell = free[0], free[1]
        # exit must be a SETTLED cell (reachable resting spot)
        if gv01_settle(exit_cell, walls, ladders) != exit_cell:
            continue
        rows = []
        for r in range(GRID):
            row = ""
            for c in range(GRID):
                if (c, r) == start:
                    row += "P"
                elif (c, r) == exit_cell:
                    row += "E"
                elif (c, r) in walls:
                    row += "#"
                elif (c, r) in ladders:
                    row += "H"
                else:
                    row += "."
            rows.append(row)
        spec = {"rows": rows}
        plan = gv01_solve_actions(spec)
        if plan is None or len(plan) < min_base:
            continue
        if 1 not in plan:
            continue  # must use a ladder climb (asymmetry is the point)
        return spec, len(plan)
    raise RuntimeError("gv01: no layout")


def gen_gv01_pack(rng):
    levels, baselines = [], []
    for n_platforms, min_base in ((2, 4), (3, 6), (3, 8), (4, 9), (4, 11)):
        spec, base = gen_gv01_level(rng, n_platforms, min_base)
        levels.append(spec)
        baselines.append(base)
    return levels, baselines


# ---------------------------------------------------------------- sw01
def sw01_solve_actions(spec):
    rows = spec["rows"]
    walls, a_walls, b_walls, levers = set(), set(), set(), set()
    start = exit_cell = None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "A":
                a_walls.add((c, r))
            elif ch == "B":
                b_walls.add((c, r))
            elif ch == "L":
                levers.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "E":
                exit_cell = (c, r)
    state0 = (start, 0)
    q = deque([(state0, [])])
    seen = {state0}
    while q:
        ((pos, mode), path) = q.popleft()
        solid = walls | (a_walls if mode == 0 else b_walls)
        if pos in levers:
            st = (pos, 1 - mode)
            if st not in seen:
                seen.add(st)
                q.append((st, path + [5]))
        for d, a in ACTS.items():
            np_ = (pos[0] + d[0], pos[1] + d[1])
            if np_ in solid or not (0 <= np_[0] < GRID and 0 <= np_[1] < GRID):
                continue
            if np_ == exit_cell:
                return path + [a]
            st = (np_, mode)
            if st not in seen:
                seen.add(st)
                q.append((st, path + [a]))
    return None


def gen_sw01_level(rng, n_a, n_b, min_base):
    for _ in range(4000):
        interior = [(c, r) for c in range(1, 7) for r in range(1, 7)]
        rng.shuffle(interior)
        a_walls = set(interior[:n_a])
        b_walls = set(interior[n_a:n_a + n_b])
        rest = interior[n_a + n_b:]
        if len(rest) < 3:
            continue
        start, exit_cell, lever = rest[0], rest[1], rest[2]
        rows = []
        for r in range(GRID):
            row = ""
            for c in range(GRID):
                if r in (0, 7) or c in (0, 7):
                    row += "#"
                elif (c, r) == start:
                    row += "P"
                elif (c, r) == exit_cell:
                    row += "E"
                elif (c, r) == lever:
                    row += "L"
                elif (c, r) in a_walls:
                    row += "A"
                elif (c, r) in b_walls:
                    row += "B"
                else:
                    row += "."
            rows.append(row)
        spec = {"rows": rows}
        plan = sw01_solve_actions(spec)
        if plan is None or len(plan) < min_base:
            continue
        if 5 not in plan:
            continue  # solution must toggle the world at least once
        return spec, len(plan)
    raise RuntimeError("sw01: no layout")


def gen_sw01_pack(rng):
    levels, baselines = [], []
    for n_a, n_b, min_base in ((4, 0, 5), (5, 2, 7), (6, 3, 8), (6, 4, 10), (7, 5, 11)):
        spec, base = gen_sw01_level(rng, n_a, n_b, min_base)
        levels.append(spec)
        baselines.append(base)
    return levels, baselines


# ---------------------------------------------------------------- lz01
def lz01_trace(spec, orient_map):
    walls = set(spec["walls"])
    ec, er, (dx, dy) = spec["emitter"]
    c, r = ec + dx, er + dy
    lit = []
    for _ in range(64):
        if not (0 < c < GRID - 1 and 0 < r < GRID - 1):
            break
        if (c, r) in walls:
            break
        lit.append((c, r))
        if (c, r) in orient_map:
            if orient_map[(c, r)] == 0:
                dx, dy = -dy, -dx
            else:
                dx, dy = dy, dx
        c, r = c + dx, r + dy
    return lit


def lz01_solve_actions(spec):
    """Min clicks to light all targets; returns [(6, cx, cy), ...]."""
    mirrors = [(c, r) for _, c, r in spec["mirrors"]]
    start = tuple(o for o, _, _ in spec["mirrors"])
    targets = set(spec["targets"])

    def lit_ok(orients):
        omap = {mirrors[i]: orients[i] for i in range(len(mirrors))}
        lit = set(lz01_trace(spec, omap))
        return targets <= lit

    if lit_ok(start):
        return []
    q = deque([(start, [])])
    seen = {start}
    while q:
        orients, path = q.popleft()
        for i, (c, r) in enumerate(mirrors):
            no = tuple((1 - o if j == i else o) for j, o in enumerate(orients))
            if no in seen:
                continue
            npath = path + [(6, c, r)]
            if lit_ok(no):
                return npath
            seen.add(no)
            q.append((no, npath))
    return None


def gen_lz01_level(rng, n_mirrors, n_targets, min_clicks):
    for _ in range(6000):
        side = rng.choice(["L", "R", "T", "B"])
        if side == "L":
            emitter = (0, rng.randrange(1, 7), (1, 0))
        elif side == "R":
            emitter = (7, rng.randrange(1, 7), (-1, 0))
        elif side == "T":
            emitter = (rng.randrange(1, 7), 0, (0, 1))
        else:
            emitter = (rng.randrange(1, 7), 7, (0, -1))
        interior = [(c, r) for c in range(1, 7) for r in range(1, 7)]
        rng.shuffle(interior)
        mirror_cells = interior[:n_mirrors]
        sol_orients = [rng.randrange(2) for _ in mirror_cells]
        spec0 = {"emitter": emitter, "walls": [],
                 "mirrors": [(sol_orients[i], c, r) for i, (c, r) in enumerate(mirror_cells)],
                 "targets": []}
        omap = {mirror_cells[i]: sol_orients[i] for i in range(n_mirrors)}
        lit = [cell for cell in lz01_trace(spec0, omap) if cell not in omap]
        if len(set(lit)) < n_targets:
            continue
        targets = rng.sample(sorted(set(lit)), n_targets)
        # scramble some mirrors away from the solution
        scr = list(sol_orients)
        flips = rng.sample(range(n_mirrors), min(n_mirrors, max(1, min_clicks)))
        for i in flips:
            scr[i] = 1 - scr[i]
        spec = {"emitter": emitter, "walls": [],
                "mirrors": [(scr[i], c, r) for i, (c, r) in enumerate(mirror_cells)],
                "targets": targets}
        plan = lz01_solve_actions(spec)
        if plan is None or len(plan) < min_clicks:
            continue
        return spec, len(plan)
    raise RuntimeError("lz01: no layout")


def gen_lz01_pack(rng):
    levels, baselines = [], []
    for n_mirrors, n_targets, min_clicks in (
        (1, 1, 1), (2, 1, 1), (3, 1, 2), (3, 2, 2), (4, 2, 3),
    ):
        spec, base = gen_lz01_level(rng, n_mirrors, n_targets, min_clicks)
        levels.append(spec)
        baselines.append(base)
    return levels, baselines


def fmt_lz01_levels(levels):
    parts = ["LEVELS = ["]
    for lv in levels:
        parts.append(f"    dict(emitter={lv['emitter']!r}, walls={lv['walls']!r},")
        parts.append(f"         mirrors={lv['mirrors']!r},")
        parts.append(f"         targets={lv['targets']!r}),")
    parts.append("]")
    return "\n".join(parts)


# ---------------------------------------------------------------- cm01
CM_RED, CM_BLUE, CM_PURPLE, CM_NONE = 2, 8, 6, 3
CM_DOORS = {"1": CM_RED, "2": CM_BLUE, "3": CM_PURPLE}


def cm01_mix(color, pad):
    if pad == "r":
        return CM_PURPLE if color == CM_BLUE else CM_RED
    return CM_PURPLE if color == CM_RED else CM_BLUE


def cm01_solve_actions(spec, want_meta=False):
    rows = spec["rows"]
    walls, pads, doors = set(), {}, {}
    start = exit_cell = None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "E":
                exit_cell = (c, r)
            elif ch in ("r", "b"):
                pads[(c, r)] = ch
            elif ch in CM_DOORS:
                doors[(c, r)] = CM_DOORS[ch]
    state0 = (start, CM_NONE)
    q = deque([(state0, [], False)])
    seen = {state0}
    while q:
        ((pos, color), path, used_door) = q.popleft()
        for d, a in ACTS.items():
            np_ = (pos[0] + d[0], pos[1] + d[1])
            if np_ in walls or not (0 <= np_[0] < GRID and 0 <= np_[1] < GRID):
                continue
            ud = used_door
            if np_ in doors:
                if doors[np_] != color:
                    continue
                ud = True
            ncolor = color
            if np_ in pads:
                ncolor = cm01_mix(color, pads[np_])
            if np_ == exit_cell:
                return (path + [a], ud) if want_meta else path + [a]
            st = (np_, ncolor)
            if st not in seen:
                seen.add(st)
                q.append((st, path + [a], ud))
    return (None, False) if want_meta else None


def gen_cm01_level(rng, n_doors, use_purple, n_walls, min_base):
    for _ in range(6000):
        interior = [(c, r) for c in range(1, 7) for r in range(1, 7)]
        rng.shuffle(interior)
        walls = set(interior[:n_walls])
        rest = interior[n_walls:]
        if len(rest) < 4 + n_doors:
            continue
        start, exit_cell = rest[0], rest[1]
        pad_cells = {rest[2]: "r", rest[3]: "b"}
        door_cells = {}
        kinds = (["3"] if use_purple else []) + ["1", "2", "1", "2"]
        for k in range(n_doors):
            door_cells[rest[4 + k]] = kinds[k % len(kinds)]
        rows = []
        for r in range(GRID):
            row = ""
            for c in range(GRID):
                if r in (0, 7) or c in (0, 7) or (c, r) in walls:
                    row += "#"
                elif (c, r) == start:
                    row += "P"
                elif (c, r) == exit_cell:
                    row += "E"
                elif (c, r) in pad_cells:
                    row += pad_cells[(c, r)]
                elif (c, r) in door_cells:
                    row += door_cells[(c, r)]
                else:
                    row += "."
            rows.append(row)
        spec = {"rows": rows}
        plan, used_door = cm01_solve_actions(spec, want_meta=True)
        if plan is None or len(plan) < min_base or not used_door:
            continue
        return spec, len(plan)
    raise RuntimeError("cm01: no layout")


def gen_cm01_pack(rng):
    levels, baselines = [], []
    for n_doors, use_purple, n_walls, min_base in (
        (2, False, 6, 6), (3, False, 8, 8), (3, True, 8, 8),
        (4, True, 10, 10), (5, True, 10, 11),
    ):
        spec, base = gen_cm01_level(rng, n_doors, use_purple, n_walls, min_base)
        levels.append(spec)
        baselines.append(base)
    return levels, baselines


# ---------------------------------------------------------------- cv01
CV_ARROWS = {"^": (0, -1), "v": (0, 1), "<": (-1, 0), ">": (1, 0)}


def cv01_slide(pos, walls, arrows):
    c, r = pos
    for _ in range(8):
        if (c, r) not in arrows:
            break
        dx, dy = arrows[(c, r)]
        nc, nr = c + dx, r + dy
        if (nc, nr) in walls or not (0 <= nc < GRID and 0 <= nr < GRID):
            break
        c, r = nc, nr
    return (c, r)


def cv01_solve_actions(spec, want_meta=False):
    rows = spec["rows"]
    walls, arrows = set(), {}
    start = exit_cell = None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "E":
                exit_cell = (c, r)
            elif ch in CV_ARROWS:
                arrows[(c, r)] = CV_ARROWS[ch]
    q = deque([(start, [], False)])
    seen = {start}
    while q:
        pos, path, rode = q.popleft()
        for d, a in ACTS.items():
            np_ = (pos[0] + d[0], pos[1] + d[1])
            if np_ in walls or not (0 <= np_[0] < GRID and 0 <= np_[1] < GRID):
                continue
            rd = rode or np_ in arrows
            np_ = cv01_slide(np_, walls, arrows)
            if np_ == exit_cell:
                return (path + [a], rd) if want_meta else path + [a]
            if np_ not in seen:
                seen.add(np_)
                q.append((np_, path + [a], rd))
    return (None, False) if want_meta else None


def gen_cv01_level(rng, n_arrows, n_walls, min_base):
    for _ in range(6000):
        interior = [(c, r) for c in range(1, 7) for r in range(1, 7)]
        rng.shuffle(interior)
        walls = set(interior[:n_walls])
        rest = interior[n_walls:]
        if len(rest) < 2 + n_arrows:
            continue
        start, exit_cell = rest[0], rest[1]
        arrow_cells = {cell: rng.choice("^v<>") for cell in rest[2:2 + n_arrows]}
        rows = []
        for r in range(GRID):
            row = ""
            for c in range(GRID):
                if r in (0, 7) or c in (0, 7) or (c, r) in walls:
                    row += "#"
                elif (c, r) == start:
                    row += "P"
                elif (c, r) == exit_cell:
                    row += "E"
                elif (c, r) in arrow_cells:
                    row += arrow_cells[(c, r)]
                else:
                    row += "."
            rows.append(row)
        spec = {"rows": rows}
        plan, rode = cv01_solve_actions(spec, want_meta=True)
        if plan is None or len(plan) < min_base or not rode:
            continue
        return spec, len(plan)
    raise RuntimeError("cv01: no layout")


def gen_cv01_pack(rng):
    levels, baselines = [], []
    for n_arrows, n_walls, min_base in ((3, 4, 5), (4, 6, 6), (5, 6, 7), (6, 8, 8), (7, 8, 9)):
        spec, base = gen_cv01_level(rng, n_arrows, n_walls, min_base)
        levels.append(spec)
        baselines.append(base)
    return levels, baselines


# ---------------------------------------------------------------- sn01
SN_TRAIL = 3


def sn01_solve_actions(spec):
    rows = spec["rows"]
    walls = set()
    start = exit_cell = None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "E":
                exit_cell = (c, r)
    state0 = (start, (start,) * SN_TRAIL)
    q = deque([(state0, [])])
    seen = {state0}
    while q:
        ((pos, trail), path) = q.popleft()
        for d, a in ACTS.items():
            np_ = (pos[0] + d[0], pos[1] + d[1])
            if np_ in walls or np_ in trail or not (0 <= np_[0] < GRID and 0 <= np_[1] < GRID):
                continue
            if np_ == exit_cell:
                return path + [a]
            st = (np_, trail[1:] + (pos,))
            if st not in seen:
                seen.add(st)
                q.append((st, path + [a]))
    return None


def sn01_plain_bfs(rows):
    walls = set()
    start = exit_cell = None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "E":
                exit_cell = (c, r)
    q = deque([(start, 0)])
    seen = {start}
    while q:
        pos, dist = q.popleft()
        for d in DIRS:
            np_ = (pos[0] + d[0], pos[1] + d[1])
            if np_ in walls or not (0 <= np_[0] < GRID and 0 <= np_[1] < GRID):
                continue
            if np_ == exit_cell:
                return dist + 1
            if np_ not in seen:
                seen.add(np_)
                q.append((np_, dist + 1))
    return None


def gen_sn01_level(rng, n_walls, min_base):
    for _ in range(6000):
        interior = [(c, r) for c in range(1, 7) for r in range(1, 7)]
        rng.shuffle(interior)
        walls = set(interior[:n_walls])
        rest = interior[n_walls:]
        if len(rest) < 2:
            continue
        start, exit_cell = rest[0], rest[1]
        rows = []
        for r in range(GRID):
            row = ""
            for c in range(GRID):
                if r in (0, 7) or c in (0, 7) or (c, r) in walls:
                    row += "#"
                elif (c, r) == start:
                    row += "P"
                elif (c, r) == exit_cell:
                    row += "E"
                else:
                    row += "."
            rows.append(row)
        spec = {"rows": rows}
        plan = sn01_solve_actions(spec)
        if plan is None or len(plan) < min_base:
            continue
        # NOTE: an optimal path never self-intersects, so the trail cannot
        # lengthen the baseline -- it punishes WANDERING agents (dead ends
        # become one-way pockets). Dense walls keep corridors narrow so
        # that punishment is real.
        return spec, len(plan)
    raise RuntimeError("sn01: no layout")


def gen_sn01_pack(rng):
    levels, baselines = [], []
    for n_walls, min_base in ((14, 6), (16, 8), (17, 8), (18, 9), (19, 10)):
        spec, base = gen_sn01_level(rng, n_walls, min_base)
        levels.append(spec)
        baselines.append(base)
    return levels, baselines


# ---------------------------------------------------------------- wf01
WF_SPREAD = 2


def wf01_parse(rows):
    walls, sources = set(), set()
    start = exit_cell = None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "F":
                sources.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "E":
                exit_cell = (c, r)
    return walls, sources, start, exit_cell


def wf01_burn_dist(rows):
    walls, sources, start, exit_cell = wf01_parse(rows)
    dist = {s: 0 for s in sources}
    q = deque(sources)
    while q:
        c, r = q.popleft()
        for dx, dy in DIRS:
            nb = (c + dx, r + dy)
            if nb in walls or nb in dist or nb == exit_cell:
                continue
            if not (0 <= nb[0] < GRID and 0 <= nb[1] < GRID):
                continue
            dist[nb] = dist[(c, r)] + 1
            q.append(nb)
    return dist


def wf01_solve_actions(spec, t0=0, t_cap=40):
    rows = spec["rows"]
    walls, sources, start, exit_cell = wf01_parse(rows)
    bd = wf01_burn_dist(rows)

    def burning(cell, t):
        d = bd.get(cell)
        return d is not None and d * WF_SPREAD <= t

    if burning(start, t0):
        return None
    state0 = (start, t0)
    q = deque([(state0, [])])
    seen = {state0}
    while q:
        ((pos, t), path) = q.popleft()
        if t - t0 > t_cap:
            continue
        moves = [(d, a) for d, a in ACTS.items()] + [((0, 0), 5)]
        for d, a in moves:
            np_ = (pos[0] + d[0], pos[1] + d[1])
            if np_ != pos:
                if np_ in walls or not (0 <= np_[0] < GRID and 0 <= np_[1] < GRID):
                    continue
            if np_ == exit_cell:
                return path + [a]
            if burning(np_, t + 1):
                continue
            st = (np_, t + 1)
            if st not in seen:
                seen.add(st)
                q.append((st, path + [a]))
    return None


def gen_wf01_level(rng, n_walls, min_base):
    for _ in range(6000):
        interior = [(c, r) for c in range(1, 7) for r in range(1, 7)]
        rng.shuffle(interior)
        walls = set(interior[:n_walls])
        rest = interior[n_walls:]
        if len(rest) < 3:
            continue
        start, exit_cell, source = rest[0], rest[1], rest[2]
        rows = []
        for r in range(GRID):
            row = ""
            for c in range(GRID):
                if r in (0, 7) or c in (0, 7) or (c, r) in walls:
                    row += "#"
                elif (c, r) == start:
                    row += "P"
                elif (c, r) == exit_cell:
                    row += "E"
                elif (c, r) == source:
                    row += "F"
                else:
                    row += "."
            rows.append(row)
        spec = {"rows": rows}
        plan = wf01_solve_actions(spec)
        if plan is None or len(plan) < min_base:
            continue
        # time pressure must be REAL: dawdling 6 actions at the start
        # makes the level unsolvable
        if wf01_solve_actions(spec, t0=6) is not None:
            continue
        return spec, len(plan)
    raise RuntimeError("wf01: no layout")


def gen_wf01_pack(rng):
    levels, baselines = [], []
    for n_walls, min_base in ((4, 5), (6, 6), (6, 7), (8, 8), (8, 9)):
        spec, base = gen_wf01_level(rng, n_walls, min_base)
        levels.append(spec)
        baselines.append(base)
    return levels, baselines


# ---------------------------------------------------------------- cl01
CL_LCM = 6


def cl01_solve_actions(spec, want_meta=False):
    rows = spec["rows"]
    walls, gates = set(), {}
    start = exit_cell = None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "E":
                exit_cell = (c, r)
            elif ch in ("2", "3"):
                gates[(c, r)] = int(ch)
    state0 = (start, 0)
    q = deque([(state0, [], False)])
    seen = {state0}
    while q:
        ((pos, t), path, used_gate) = q.popleft()
        if len(path) > 60:
            continue
        moves = [(d, a) for d, a in ACTS.items()] + [((0, 0), 5)]
        for d, a in moves:
            np_ = (pos[0] + d[0], pos[1] + d[1])
            ug = used_gate
            if np_ != pos:
                if np_ in walls or not (0 <= np_[0] < GRID and 0 <= np_[1] < GRID):
                    continue
                if np_ in gates:
                    if (t + 1) % gates[np_] != 0:
                        continue
                    ug = True
            if np_ == exit_cell:
                return (path + [a], ug) if want_meta else path + [a]
            st = (np_, (t + 1) % CL_LCM)
            if st not in seen:
                seen.add(st)
                q.append((st, path + [a], ug))
    return (None, False) if want_meta else None


def gen_cl01_level(rng, n_gates, n_walls, min_base):
    for _ in range(6000):
        interior = [(c, r) for c in range(1, 7) for r in range(1, 7)]
        rng.shuffle(interior)
        walls = set(interior[:n_walls])
        rest = interior[n_walls:]
        if len(rest) < 2 + n_gates:
            continue
        start, exit_cell = rest[0], rest[1]
        gate_cells = {cell: rng.choice("23") for cell in rest[2:2 + n_gates]}
        rows = []
        for r in range(GRID):
            row = ""
            for c in range(GRID):
                if r in (0, 7) or c in (0, 7) or (c, r) in walls:
                    row += "#"
                elif (c, r) == start:
                    row += "P"
                elif (c, r) == exit_cell:
                    row += "E"
                elif (c, r) in gate_cells:
                    row += gate_cells[(c, r)]
                else:
                    row += "."
            rows.append(row)
        spec = {"rows": rows}
        plan, used_gate = cl01_solve_actions(spec, want_meta=True)
        if plan is None or len(plan) < min_base or not used_gate:
            continue
        return spec, len(plan)
    raise RuntimeError("cl01: no layout")


def gen_cl01_pack(rng):
    levels, baselines = [], []
    for n_gates, n_walls, min_base in ((1, 8, 5), (2, 8, 6), (2, 10, 7), (3, 10, 8), (3, 12, 9)):
        spec, base = gen_cl01_level(rng, n_gates, n_walls, min_base)
        levels.append(spec)
        baselines.append(base)
    return levels, baselines


SOLVERS = {
    "gv01": gv01_solve_actions,
    "sw01": sw01_solve_actions,
    "lz01": lz01_solve_actions,
    "cm01": cm01_solve_actions,
    "cv01": cv01_solve_actions,
    "sn01": sn01_solve_actions,
    "wf01": wf01_solve_actions,
    "cl01": cl01_solve_actions,
}


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
    "pt01": {
        "src": ROOT / "our_games" / "pt01" / "d0e1f2a3" / "pt01.py",
        "class_name": "Pt01",
        "prefix": "tg",
        "gen": gen_pt01_pack,
        "fmt": fmt_fl01_levels,
        "title": "Portals (generated)",
    },
    "gv01": {
        "src": ROOT / "our_games" / "gv01" / "e1f2a3b4" / "gv01.py",
        "class_name": "Gv01",
        "prefix": "gg",
        "gen": gen_gv01_pack,
        "fmt": fmt_fl01_levels,
        "title": "Gravity (generated)",
    },
    "sw01": {
        "src": ROOT / "our_games" / "sw01" / "f2a3b4c5" / "sw01.py",
        "class_name": "Sw01",
        "prefix": "wg",
        "gen": gen_sw01_pack,
        "fmt": fmt_fl01_levels,
        "title": "Switches (generated)",
    },
    "lz01": {
        "src": ROOT / "our_games" / "lz01" / "a3b4c5d6" / "lz01.py",
        "class_name": "Lz01",
        "prefix": "lg",
        "gen": gen_lz01_pack,
        "fmt": fmt_lz01_levels,
        "title": "Laser Rotate (generated)",
    },
    "cm01": {
        "src": ROOT / "our_games" / "cm01" / "b4c5d6e7" / "cm01.py",
        "class_name": "Cm01",
        "prefix": "cg",
        "gen": gen_cm01_pack,
        "fmt": fmt_fl01_levels,
        "title": "Color Algebra (generated)",
    },
    "cv01": {
        "src": ROOT / "our_games" / "cv01" / "c5d6e7f8" / "cv01.py",
        "class_name": "Cv01",
        "prefix": "vg",
        "gen": gen_cv01_pack,
        "fmt": fmt_fl01_levels,
        "title": "Conveyors (generated)",
    },
    "sn01": {
        "src": ROOT / "our_games" / "sn01" / "d6e7f8a9" / "sn01.py",
        "class_name": "Sn01",
        "prefix": "ng",
        "gen": gen_sn01_pack,
        "fmt": fmt_fl01_levels,
        "title": "Snake Trail (generated)",
    },
    "wf01": {
        "src": ROOT / "our_games" / "wf01" / "e7f8a9b0" / "wf01.py",
        "class_name": "Wf01",
        "prefix": "hg",
        "gen": gen_wf01_pack,
        "fmt": fmt_fl01_levels,
        "title": "Wildfire (generated)",
    },
    "cl01": {
        "src": ROOT / "our_games" / "cl01" / "f8a9b0c1" / "cl01.py",
        "class_name": "Cl01",
        "prefix": "kg",
        "gen": gen_cl01_pack,
        "fmt": fmt_fl01_levels,
        "title": "Clockgates (generated)",
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

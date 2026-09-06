"""Levels with a PLANTED solution, for the games a random layout cannot carry.

Why. scripts/regen_short_levels.py drops a level's objects onto a random maze
and lets a BFS through the engine decide what it costs. That works where the
task IS the path (cv01, cl01, cm01, ic01 hit their targets almost exactly) and
fails where the cost is a CONFIGURATION: sk01 must push a box onto a mark and
gave 0 solvable layouts in 12; bx01 needs targets on the blast rays; gv01 needs
ladders that actually climb to the exit; fl01 must paint every floor cell, a
state space of 2**cells that no search can settle on a 21x21 board; lz01 and
pi01 need a dozen mirrors or pipe tiles on ONE beam or trace, which chance
never lines up. Measured 03.09 with scripts/probe_regen_fitness.py, not
guessed -- the guess had been wrong three times that day.

The recipe here is the other way round: build the SOLUTION first (a push run,
a climb, a blast spot with targets on its rays, a corridor to paint, a beam
staircase, a pipe trace), then derive the level from it. Solvability is then
true by construction, and what remains is to learn what the level COSTS:

  * sk01, gv01, bx01, fl01 -- the cost is still measured by BFS through the
    real engine (scripts/verify_short_baselines.optimum). The construction
    only guarantees the search will find something; the number is the search's,
    never the builder's;
  * lz01, pi01 -- the search cannot settle a 30-click level (2**30 states), so
    the optimum is proven by construction instead, with a structural check the
    builder enforces and a replay of the planted answer through the engine:
      lz01: every mirror's WRONG deflection ray leaves the board without meeting
            another mirror or a target. Then the first wrong mirror along the
            beam kills the target whatever the rest do, so every mirror must be
            flipped once -- optimum == number of mirrors;
      pi01: the trace is non-self-touching (no two non-consecutive tiles are
            neighbours), so the only route from source to sink is the trace and
            every tile has exactly one orientation that works -- optimum == sum
            of the rotations planted.
    For small levels the claim is ALSO checked exhaustively over every
    click subset (--exhaustive), which is how the argument itself was tested.

Targets come from scripts/target_curves.json (a real public game's curve),
levels are kept only if found, never padded, and a game ships only with six
or more levels -- same rules as the random generator, same guards afterwards.

usage:
    python scripts/plant_levels.py --game sk01 --game gv01
    python scripts/plant_levels.py --game lz01 --exhaustive
    python scripts/plant_levels.py --game sk01 --write
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import regen_short_levels as R  # noqa: E402
import verify_short_baselines as V  # noqa: E402

G = R.NEW_GRID
FLOOR, WALL = R.FLOOR, R.WALL
DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))


# ----------------------------------------------------------------- grid helpers
def floor_cells(g):
    return [(c, r) for r in range(G) for c in range(G) if g[r][c] == FLOOR]


def bfs(g, start, blocked=frozenset()):
    """Distance from start over FLOOR cells, and the parent map for paths."""
    dist, parent = {start: 0}, {}
    q = deque([start])
    while q:
        c, r = q.popleft()
        for dx, dy in DIRS:
            n = (c + dx, r + dy)
            if not (0 <= n[0] < G and 0 <= n[1] < G) or g[n[1]][n[0]] != FLOOR:
                continue
            if n in blocked or n in dist:
                continue
            dist[n] = dist[(c, r)] + 1
            parent[n] = (c, r)
            q.append(n)
    return dist, parent


def path_to(parent, start, end):
    p = [end]
    while p[-1] != start:
        p.append(parent[p[-1]])
    return p[::-1]


def straight_runs(g, min_len=4):
    """Maximal straight FLOOR runs, row-wise and column-wise."""
    out = []
    for r in range(G):
        run = []
        for c in range(G + 1):
            if c < G and g[r][c] == FLOOR:
                run.append((c, r))
            else:
                if len(run) >= min_len:
                    out.append(run)
                run = []
    for c in range(G):
        run = []
        for r in range(G + 1):
            if r < G and g[r][c] == FLOOR:
                run.append((c, r))
            else:
                if len(run) >= min_len:
                    out.append(run)
                run = []
    return out


def rows_of(grid):
    return ["".join(row) for row in grid]


def stamp(g, marks):
    grid = [row[:] for row in g]
    for ch, cells in marks.items():
        for c, r in cells:
            grid[r][c] = ch
    return rows_of(grid)


# ------------------------------------------------------------ ASCII builders
# Each returns a level spec (dict with rows=..., plus any per-level extras) or
# None when this attempt did not come together. The cost is NOT theirs to
# claim: it is measured afterwards by the engine BFS.

def build_sk01(rng, target):
    """Boxes on straight corridor runs, each pushed k cells onto its mark.

    In a one-wide maze corridor a box can only travel straight, so a run
    holds the whole story of one box: the agent must reach the cell behind it
    and push. Several boxes add up; the walk between them is the rest."""
    g = R.maze(rng, G)
    nb = max(1, min(5, -(-target // 35)))          # ceil: a box is worth ~35
    nb = rng.choice([max(1, nb - 1), nb, nb, min(5, nb + 1)])
    runs = straight_runs(g)
    rng.shuffle(runs)
    used, boxes, marks = set(), [], []
    for run in runs:
        if len(boxes) >= nb:
            break
        if any(cell in used for cell in run):
            continue
        L = len(run)
        kmax = min(L - 2, max(2, int(target / nb * 0.7)))
        if kmax < 2:
            continue
        k = rng.randint(2, kmax)
        i = rng.randint(1, L - 1 - k)
        if rng.random() < 0.5:          # push the other way along the run
            run = run[::-1]
        boxes.append(run[i])
        marks.append(run[i + k])
        used.update(run)
    if not boxes:
        return None
    free = [c for c in floor_cells(g) if c not in used]
    if not free:
        return None
    # start within reach of the first box, or the walk alone eats the target
    near = [c for c in free if abs(c[0] - boxes[0][0]) + abs(c[1] - boxes[0][1]) <= max(4, target // (3 * nb))]
    start = rng.choice(near or free)
    return dict(rows=stamp(g, {"B": boxes, "T": marks, "P": [start]}))


def build_gv01(rng, target):
    """A maze path from start to exit, with ladders wherever the path climbs
    or would otherwise fall through the floor."""
    g = R.maze(rng, G)
    fl = floor_cells(g)
    fs = set(fl)
    supported = [c for c in fl if (c[0], c[1] + 1) not in fs]
    if len(supported) < 2:
        return None
    start = rng.choice(supported)
    dist, parent = bfs(g, start)
    want = target * rng.uniform(1.0, 1.5)     # falls are one action each
    cand = [c for c in supported if c != start and c in dist
            and abs(dist[c] - want) <= 0.15 * want + 3]
    if not cand:
        return None
    exit_cell = rng.choice(cand)
    path = path_to(parent, start, exit_cell)
    ladders = set()
    for a, b in zip(path, path[1:]):
        if b[1] < a[1]:
            ladders.add(a)                    # climbing: the cell you leave
    for cell in path[1:-1]:
        if (cell[0], cell[1] + 1) in fs:
            ladders.add(cell)                 # a shaft below: stand on a ladder
    ladders -= {start, exit_cell}
    return dict(rows=stamp(g, {"H": sorted(ladders), "E": [exit_cell], "P": [start]}))


def build_bx01(rng, target):
    """k blast spots at the right distance from the start, targets at the far
    ends of the corridors each spot looks down. The ball respawns at the start
    after every blast, so each spot is a trip from the start."""
    g = R.maze(rng, G)
    fl = floor_cells(g)
    start = rng.choice(fl)
    dist, _ = bfs(g, start)
    k = max(1, min(4, round(target / 25)))
    k = rng.choice([k, max(1, k - 1), min(4, k + 1)])
    per = (target - k) / k
    spots, targets = [], set()
    for _ in range(k):
        cand = [c for c in fl if c in dist and c != start and c not in spots
                and abs(dist[c] - per) <= max(2, 0.2 * per)]
        if not cand:
            return None
        s = rng.choice(cand)
        spots.append(s)
        rays = []
        for dx, dy in DIRS:
            x, y = s[0] + dx, s[1] + dy
            ray = []
            while 0 <= x < G and 0 <= y < G and g[y][x] == FLOOR:
                ray.append((x, y))
                x += dx
                y += dy
            if ray:
                rays.append(ray)
        if not rays:
            return None
        rng.shuffle(rays)
        for ray in rays[:rng.randint(1, min(3, len(rays)))]:
            targets.add(ray[-1])
    targets -= {start}
    targets -= set(spots)
    if not targets:
        return None
    return dict(stock=k, rows=stamp(g, {"o": sorted(targets), "P": [start]}))


def build_fl01(rng, target):
    """The floor IS a corridor along a maze path, with a few 2x2 bulges so the
    agent still has a wrong turn available. Painting every cell then costs
    cells - 1, and the search stays linear."""
    g = R.maze(rng, G)
    fl = floor_cells(g)
    start = rng.choice(fl)
    dist, parent = bfs(g, start)
    nb = rng.randint(0, 3)
    want = target - 2 * nb
    cand = [c for c in fl if dist.get(c) == want]
    if not cand:
        return None
    path = path_to(parent, start, rng.choice(cand))
    floor = set(path)
    added = 0
    order = list(range(len(path) - 1))
    rng.shuffle(order)
    for i in order:
        if added >= nb:
            break
        a, b = path[i], path[i + 1]
        if a[1] != b[1]:
            continue                          # horizontal pairs only
        for dr in (-1, 1):
            p, q = (a[0], a[1] + dr), (b[0], b[1] + dr)
            if p in floor or q in floor or not (0 < p[1] < G - 1):
                continue
            touch = any((cell[0] + dx, cell[1] + dy) in floor
                        and (cell[0] + dx, cell[1] + dy) not in (a, b)
                        for cell in (p, q) for dx, dy in DIRS)
            if touch:
                continue
            floor.update((p, q))
            added += 1
            break
    grid = [[WALL] * G for _ in range(G)]
    for c, r in floor:
        grid[r][c] = FLOOR
    grid[start[1]][start[0]] = "P"
    return dict(rows=rows_of(grid))


def build_hc01(rng, target):
    """Buttons to press in a hidden order, a door sealing a dead end that
    holds the exit, and one decoy button that resets progress. The exit is
    reachable ONLY through the door (hostile_baselines checks it)."""
    g = R.maze(rng, G)
    fl = floor_cells(g)
    fs = set(fl)
    dead = [c for c in fl if sum((c[0] + dx, c[1] + dy) in fs for dx, dy in DIRS) == 1]
    if len(dead) < 2:
        return None
    exit_cell = rng.choice(dead)
    door = next((exit_cell[0] + dx, exit_cell[1] + dy) for dx, dy in DIRS
                if (exit_cell[0] + dx, exit_cell[1] + dy) in fs)
    k = rng.choice([2, 2, 3]) if target < 40 else 3
    letters = list("abc")[:k]
    rng.shuffle(letters)
    order = "".join(letters)
    per = max(3, target // (k + 1))
    start = rng.choice([c for c in fl if c not in (exit_cell, door)])
    cur = start
    buttons = {}
    for ch in order:
        dist, _ = bfs(g, cur)
        cand = [c for c in fl if c in dist and c not in buttons and c not in (exit_cell, door, start)
                and abs(dist[c] - per) <= max(2, per // 4)]
        if not cand:
            return None
        cur = rng.choice(cand)
        buttons[cur] = ch
    dist, _ = bfs(g, cur)
    if door not in dist:
        return None
    # a decoy of the FIRST letter somewhere off the tour: stepping on it at
    # the wrong moment silently resets (that is what makes the game hostile)
    decoys = [c for c in fl if c not in buttons and c not in (exit_cell, door, start)]
    if decoys:
        buttons[rng.choice(decoys)] = order[-1] if k > 1 else order[0]
    marks = {ch: [c for c, l in buttons.items() if l == ch] for ch in set(buttons.values())}
    marks.update({"P": [start], "E": [exit_cell], "D": [door]})
    return dict(rows=stamp(g, marks), order=order)


def build_ch01(rng, target):
    """Chaser: a plain maze walk with the pursuer dropped somewhere else. The
    layout is random (the probe measured 6/6 solvable); what needed replacing
    was the VERIFIER, not the builder -- see ch01_optimum."""
    g = R.maze(rng, G)
    fl = floor_cells(g)
    start = rng.choice(fl)
    dist, _ = bfs(g, start)
    cand = [c for c in fl if c in dist and abs(dist[c] - target) <= max(1, target // 12)]
    if not cand:
        return None
    exit_cell = rng.choice(cand)
    others = [c for c in fl if c not in (start, exit_cell)]
    chaser = rng.choice(others)
    return dict(rows=stamp(g, {"E": [exit_cell], "C": [chaser], "P": [start]}))


def _maze_pair(rng, target, tol=None):
    """A maze with a start and an exit at maze distance ~target."""
    g = R.maze(rng, G)
    fl = floor_cells(g)
    start = rng.choice(fl)
    dist, parent = bfs(g, start)
    tol = tol if tol is not None else max(1, target // 12)
    cand = [c for c in fl if c in dist and c != start and abs(dist[c] - target) <= tol]
    if not cand:
        return None
    exit_cell = rng.choice(cand)
    return g, fl, start, exit_cell, dist, parent


def build_vn01(rng, target):
    """Visual noise: a plain maze walk; the noise is cosmetic."""
    out = _maze_pair(rng, target)
    if out is None:
        return None
    g, fl, start, exit_cell, _d, _p = out
    return dict(rows=stamp(g, {"E": [exit_cell], "P": [start]}))


def build_sn01(rng, target):
    """Snake trail: a shortest maze path never steps on its own trail."""
    out = _maze_pair(rng, target)
    if out is None:
        return None
    g, fl, start, exit_cell, _d, _p = out
    return dict(rows=stamp(g, {"E": [exit_cell], "P": [start]}))


def build_fw01(rng, target):
    """Fog of war: reveal clicks are optional, walking costs the same."""
    out = _maze_pair(rng, target)
    if out is None:
        return None
    g, fl, start, exit_cell, _d, _p = out
    return dict(rows=stamp(g, {"E": [exit_cell], "P": [start]}))


def build_wf01(rng, target):
    """Wildfire: 1-2 sources placed away from the planted path; whether the
    shortest path outruns the fire is decided by wf01_optimum."""
    out = _maze_pair(rng, target)
    if out is None:
        return None
    g, fl, start, exit_cell, dist, parent = out
    path = set(path_to(parent, start, exit_cell))
    far = [c for c in fl if c not in path and c != start and dist.get(c, 0) >= 6]
    if not far:
        return None
    n = rng.choice([1, 1, 2])
    sources = rng.sample(far, min(n, len(far)))
    return dict(rows=stamp(g, {"E": [exit_cell], "F": sources, "P": [start]}))


def build_tr01(rng, target):
    """One-way trap: the exit path is clean; a gate seals a dead-end side
    branch off it (enter and you are stuck, no game_over). A second gate may
    sit in another branch. hostile_baselines checks all of that."""
    out = _maze_pair(rng, target)
    if out is None:
        return None
    g, fl, start, exit_cell, dist, parent = out
    path = path_to(parent, start, exit_cell)
    pset = set(path)
    fs = set(fl)
    gates = []
    branches = []
    for cell in path[1:-1]:
        for dx, dy in DIRS:
            n = (cell[0] + dx, cell[1] + dy)
            if n in fs and n not in pset:
                branches.append(n)
    rng.shuffle(branches)
    for n in branches[:rng.choice([1, 1, 2])]:
        gates.append(n)
    if not gates:
        return None
    return dict(rows=stamp(g, {"E": [exit_cell], ">": gates, "P": [start]}))


def build_sw01(rng, target):
    """Switches: an A-wall blocks the planted path; a lever earlier on the
    path (or in a side pocket) flips the mode; B-walls close elsewhere."""
    out = _maze_pair(rng, target - 1)
    if out is None:
        return None
    g, fl, start, exit_cell, dist, parent = out
    path = path_to(parent, start, exit_cell)
    if len(path) < 8:
        return None
    pset = set(path)
    fs = set(fl)
    i_wall = rng.randint(len(path) // 2, len(path) - 2)
    a_wall = path[i_wall]
    lever = path[rng.randint(1, i_wall - 1)]
    others = [c for c in fl if c not in pset]
    b_walls = rng.sample(others, min(len(others), rng.randint(2, 5)))
    return dict(rows=stamp(g, {"A": [a_wall], "B": b_walls, "L": [lever],
                               "E": [exit_cell], "P": [start]}))


# ------------------------------------------------------- coordinate builders
def turn_orient(din, dout):
    """Mirror orientation that sends `din` into `dout` (lz01.trace_beam)."""
    dx, dy = din
    return 0 if (-dy, -dx) == dout else 1


def build_lz01(rng, target):
    """A rectilinear beam with a mirror at every bend; all mirrors start
    flipped. The wrong-ray check below is what makes `cost == mirrors` a proof."""
    lo, hi = 1, G - 2
    # A staircase (right, down, right, down ...) with unit steps packs the
    # most bends the wrong-ray rule allows -- two mirrors per row and column,
    # about 36 on this board. Long targets get that; short ones wander.
    stair = target > 14 or rng.random() < 0.3
    r0 = rng.randint(lo, lo + 3) if stair else rng.randint(lo, hi)
    pos, d = (1, r0), (1, 0)
    cells = [pos]
    mirrors = []          # (orient_correct, c, r)
    m = 0
    while m < target:
        L = 1 if stair and rng.random() < 0.85 else rng.randint(1, 3)
        nxt = (pos[0] + d[0] * L, pos[1] + d[1] * L)
        if not (lo <= nxt[0] <= hi and lo <= nxt[1] <= hi):
            break
        seg = [(pos[0] + d[0] * i, pos[1] + d[1] * i) for i in range(1, L + 1)]
        if any(c in cells for c in seg):
            break
        cells.extend(seg)
        pos = nxt
        if stair:
            opts = [(0, 1)] if d == (1, 0) else [(1, 0)]
        else:
            opts = [(d[1], d[0]), (-d[1], -d[0])]
        opts = [o for o in opts if lo <= pos[0] + o[0] <= hi and lo <= pos[1] + o[1] <= hi
                and (pos[0] + o[0], pos[1] + o[1]) not in cells]
        if not opts:
            break
        nd = rng.choice(opts)
        mirrors.append((turn_orient(d, nd), pos[0], pos[1]))
        m += 1
        d = nd
    if m < 4:
        return None
    # final segment and the target on its last cell
    L = rng.randint(1, 3)
    tgt = None
    for i in range(1, L + 1):
        c = (pos[0] + d[0] * i, pos[1] + d[1] * i)
        if not (lo <= c[0] <= hi and lo <= c[1] <= hi) or c in cells:
            break
        cells.append(c)
        tgt = c
    if tgt is None:
        return None
    mcells = {(c, r) for _, c, r in mirrors}
    spec = dict(emitter=(0, r0, (1, 0)), walls=[],
                mirrors=[(1 - o, c, r) for o, c, r in mirrors], targets=[tgt])
    # wrong-ray check: from each mirror, the deflection it must NOT make goes
    # to the border without meeting a mirror or the target
    d = (1, 0)
    for o, c, r in mirrors:
        out = (-d[1], -d[0]) if o == 0 else (d[1], d[0])
        wrong = (-out[0], -out[1])
        x, y = c + wrong[0], r + wrong[1]
        while lo <= x <= hi and lo <= y <= hi:
            if (x, y) in mcells or (x, y) == tgt:
                return None
            x += wrong[0]
            y += wrong[1]
        d = out
    return spec, m


def build_mr01(rng, target):
    """Movable mirrors, but inside a WALLED corridor: the beam has exactly one
    way to go, each bend needs a mirror, and every mirror starts parked d cells
    down a dead-end pocket off its bend. Cost = sum(select 1 + d moves + rotate)
    by construction; moving a mirror anywhere else is impossible, so nothing
    cheaper exists. Walls do the proving that lz01's wrong-ray rule did."""
    lo, hi = 1, G - 2
    r0 = rng.randint(lo + 2, hi - 2)
    pos, d = (1, r0), (1, 0)
    corridor = [pos]
    bends = []            # (cell, din, dout)
    taken = {pos}
    budget = target
    while budget > 0:
        L = rng.randint(2, 4)
        seg = [(pos[0] + d[0] * i, pos[1] + d[1] * i) for i in range(1, L + 1)]
        if any(not (lo <= c[0] <= hi and lo <= c[1] <= hi) or c in taken for c in seg):
            break
        # keep the corridor from touching itself (pockets need the room)
        if any((c[0] + dx, c[1] + dy) in taken and (c[0] + dx, c[1] + dy) != pos
               and (c[0] + dx, c[1] + dy) not in seg for c in seg for dx, dy in DIRS):
            break
        corridor.extend(seg)
        taken.update(seg)
        pos = seg[-1]
        opts = [o for o in ((d[1], d[0]), (-d[1], -d[0]))
                if lo <= pos[0] + o[0] <= hi and lo <= pos[1] + o[1] <= hi
                and (pos[0] + o[0], pos[1] + o[1]) not in taken]
        if not opts:
            break
        nd = rng.choice(opts)
        bends.append((pos, d, nd))
        budget -= 3
        d = nd
    if len(bends) < 2:
        return None
    # the target sits at the end of one last straight run
    tail = []
    for i in range(1, rng.randint(2, 4) + 1):
        c = (pos[0] + d[0] * i, pos[1] + d[1] * i)
        if not (lo <= c[0] <= hi and lo <= c[1] <= hi) or c in taken:
            break
        if any((c[0] + dx, c[1] + dy) in taken and (c[0] + dx, c[1] + dy) != pos
               and (c[0] + dx, c[1] + dy) not in tail for dx, dy in DIRS):
            break
        tail.append(c)
        taken.add(c)
    if not tail:
        return None
    corridor.extend(tail)
    target_cell = tail[-1]
    # pockets: a dead end off each bend, straight ahead or back from the turn
    mirrors, plan_bits, pockets = [], [], set()
    cost = 0
    for cell, din, dout in bends:
        placed = False
        for pdir in rng.sample([din, (-dout[0], -dout[1])], 2):
            depths = [1, 2, 3]
            rng.shuffle(depths)
            for depth in depths:
                pk = [(cell[0] + pdir[0] * i, cell[1] + pdir[1] * i) for i in range(1, depth + 1)]
                if any(not (lo <= c[0] <= hi and lo <= c[1] <= hi) or c in taken or c in pockets for c in pk):
                    continue
                if any((c[0] + dx, c[1] + dy) in (taken | pockets) and (c[0] + dx, c[1] + dy) != cell
                       and (c[0] + dx, c[1] + dy) not in pk for c in pk for dx, dy in DIRS):
                    continue
                rot = rng.randint(0, 1)
                o_correct = turn_orient(din, dout)
                mirrors.append((1 - o_correct if rot else o_correct, pk[-1][0], pk[-1][1]))
                plan_bits.append((pk[-1], (-pdir[0], -pdir[1]), depth, rot))
                pockets.update(pk)
                cost += 1 + depth + rot
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    open_cells = taken | pockets
    walls = [(c, r) for r in range(lo, hi + 1) for c in range(lo, hi + 1) if (c, r) not in open_cells]
    spec = dict(emitter=(0, r0, (1, 0)), walls=walls, mirrors=mirrors, targets=[target_cell])
    return spec, cost, plan_bits


def mr01_plan(mod, spec, plan_bits):
    plan = []
    for cell, step, depth, rot in plan_bits:
        plan.append(click(mod, *cell))
        plan.extend([(ARROW[step], {})] * depth)
        if rot:
            plan.append(("ACTION5", {}))
    return plan


def mr01_min_actions(mod, spec, limit=200_000):
    """Exhaustive BFS over (mirror positions+orientations, selection) with
    mr01's own move rules -- the check the corridor argument is tested against."""
    walls = set(spec["walls"])
    targets = set(spec["targets"])
    start = tuple((o, c, r) for o, c, r in spec["mirrors"])
    key0 = (start, None)
    seen = {key0}
    q = deque([(key0, 0)])
    while q and len(seen) < limit:
        (ms, sel), dist = q.popleft()
        if mod.solved(spec, list(ms)):
            return dist
        nxt = []
        for i in range(len(ms)):
            nxt.append((ms, i))                                   # click a mirror
        if sel is not None:
            o, c, r = ms[sel]
            occ = {(mc, mr) for j, (_, mc, mr) in enumerate(ms) if j != sel} | walls | targets
            for dx, dy in DIRS:
                nc, nr = c + dx, r + dy
                if 0 < nc < G - 1 and 0 < nr < G - 1 and (nc, nr) not in occ:
                    nxt.append((ms[:sel] + ((o, nc, nr),) + ms[sel + 1:], sel))
            nxt.append((ms[:sel] + ((1 - o, c, r),) + ms[sel + 1:], sel))  # rotate
        for key in nxt:
            if key not in seen:
                seen.add(key)
                q.append((key, dist + 1))
    return None


PORT_OF = {(0, -1): 1, (1, 0): 2, (0, 1): 4, (-1, 0): 8}


def rot_cw(ports, k=1):
    for _ in range(k % 4):
        ports = ((ports << 1) | (ports >> 3)) & 0b1111
    return ports


def build_pi01(rng, target):
    """A non-self-touching trace from the left edge to the right edge; every
    tile is rotated away from its one working orientation by a planted count."""
    lo, hi = 1, G - 2
    r0 = rng.randint(lo, hi)
    start = (1, r0)
    ok = False
    path = [start]
    for _attempt in range(40):
        path = [start]
        occupied = {start}
        ok = False
        for _ in range(120):
            c, r = path[-1]
            if c == hi and rng.random() < 0.7:
                ok = True
                break
            opts = []
            for dx, dy in DIRS:
                n = (c + dx, r + dy)
                if not (lo <= n[0] <= hi and lo <= n[1] <= hi) or n in occupied:
                    continue
                # must not touch any path cell except the one we come from
                if any((n[0] + ex, n[1] + ey) in occupied and (n[0] + ex, n[1] + ey) != (c, r)
                       for ex, ey in DIRS):
                    continue
                w = 3 if dx == 1 else (0.3 if dx == -1 else 1)
                opts.append((w, n))
            if not opts:
                break
            tot = sum(w for w, _ in opts)
            pick = rng.uniform(0, tot)
            chosen = opts[-1][1]
            for w, n in opts:
                pick -= w
                if pick <= 0:
                    chosen = n
                    break
            path.append(chosen)
            occupied.add(chosen)
        if ok and len(path) >= 4:
            break
    if not ok or path[-1][0] != hi:
        return None
    end = path[-1]
    tiles = {}
    cost = 0
    budget = target
    for i, cell in enumerate(path):
        pin = 8 if i == 0 else PORT_OF[(path[i - 1][0] - cell[0], path[i - 1][1] - cell[1])]
        pout = 2 if i == len(path) - 1 else PORT_OF[(path[i + 1][0] - cell[0], path[i + 1][1] - cell[1])]
        correct = pin | pout
        straight = correct in (5, 10)
        remaining = len(path) - i
        k_max = 1 if straight else 3
        k = min(k_max, max(0, round(budget / remaining))) if budget > 0 else 0
        if k > 0 and rng.random() < 0.25:
            k = max(0, k - 1)
        tiles[cell] = rot_cw(correct, 4 - k) if k else correct
        cost += k
        budget -= k
    spec = dict(source=(0, r0, 2), sink=(G - 1, end[1], 8), tiles=tiles)
    return spec, cost


# --------------------------------------------------------------- verification
def engine_root(cls, level_index):
    root = cls()
    root._state = type(root._state).NOT_FINISHED
    root._current_level_index = level_index
    for hook in ("on_set_level", "_load"):
        fn = getattr(root, hook, None)
        if callable(fn):
            try:
                fn(root.current_level) if hook == "on_set_level" else fn()
            except TypeError:
                fn()
            break
    return root


def replay(cls, mod, level_index, actions):
    """Play `actions` [(ACTIONn, data)] through the engine; True if the level clears."""
    import arcengine
    game = engine_root(cls, level_index)
    base = game._score
    for name, data in actions:
        act = getattr(arcengine.GameAction, name)
        game._action = type(game._action)(id=act, data=data)
        game._action_complete = False
        game.step()
        if game._score > base or game._state.name == "WIN":
            return True
    return False


def click(mod, c, r):
    scale = max(1, 64 // (mod.GRID * mod.CELL))
    off = (64 - mod.GRID * mod.CELL * scale) // 2
    return ("ACTION6", {"x": off + c * mod.CELL * scale, "y": off + r * mod.CELL * scale})


def lz01_min_clicks(mod, spec):
    """Exhaustive: fewest mirror flips that light every target (small m only)."""
    mirrors = spec["mirrors"]
    m = len(mirrors)
    for k in range(m + 1):
        for flip in itertools.combinations(range(m), k):
            orients = {(c, r): (1 - o if i in flip else o)
                       for i, (o, c, r) in enumerate(mirrors)}
            lit = set(mod.trace_beam(spec, orients))
            if all(t in lit for t in spec["targets"]):
                return k
    return None


def pi01_min_clicks(mod, spec):
    """Exhaustive over every rotation of every tile (small levels only)."""
    tiles = sorted(spec["tiles"])
    best = None
    for ks in itertools.product(range(4), repeat=len(tiles)):
        total = sum(ks)
        if best is not None and total >= best:
            continue
        ports = {t: rot_cw(spec["tiles"][t], k) for t, k in zip(tiles, ks)}
        if mod.connected(spec, ports):
            best = total
    return best


def pi01_plan(mod, spec):
    """Clicks that rotate every tile into the trace's orientation, found by
    walking the (unique) trace from the source."""
    tiles = dict(spec["tiles"])
    sc, sr, sport = spec["source"]
    kc, kr, kport = spec["sink"]
    dx, dy = mod.DELTA[sport]
    cur = (sc + dx, sr + dy)
    prev_port = mod.OPP[sport]
    order, seen = [], set()
    while cur in tiles and cur not in seen:
        seen.add(cur)
        sdx, sdy = mod.DELTA[mod.OPP[kport]]
        if (cur[0] + sdx, cur[1] + sdy) == (kc, kr):
            order.append((cur, prev_port | mod.OPP[kport]))
            break
        nbrs = [(p, (cur[0] + ddx, cur[1] + ddy)) for p, (ddx, ddy) in mod.DELTA.items()
                if (cur[0] + ddx, cur[1] + ddy) in tiles and (cur[0] + ddx, cur[1] + ddy) not in seen]
        if len(nbrs) != 1:
            return None
        out_port, nxt = nbrs[0]
        order.append((cur, prev_port | out_port))
        prev_port = mod.OPP[out_port]
        cur = nxt
    if len(order) != len(tiles):
        return None
    plan = []
    for cell, want in order:
        have = tiles[cell]
        k = next((k for k in range(4) if rot_cw(have, k) == want), None)
        if k is None:
            return None
        plan.extend([click(mod, *cell)] * k)
    return plan


# ------------------------------------------------------- model verifiers
ARROW = {(0, -1): "ACTION1", (0, 1): "ACTION2", (-1, 0): "ACTION3", (1, 0): "ACTION4"}


def sk01_optimum(cls, mod, lvl, spec, limit):
    """Shortest push solution by BFS over (agent, boxes) with sk01's own rules,
    confirmed by replaying the plan through the engine. None if unsolved."""
    walls, boxes, targets, start = mod._parse(spec)
    boxes = frozenset(boxes)
    key0 = (start, boxes)
    parent = {key0: None}
    q = deque([key0])
    found = None
    while q and len(parent) < limit:
        agent, bx = q.popleft()
        for d, name in ARROW.items():
            n = (agent[0] + d[0], agent[1] + d[1])
            if n in walls or not (0 <= n[0] < G and 0 <= n[1] < G):
                continue
            nb = bx
            if n in bx:
                beyond = (n[0] + d[0], n[1] + d[1])
                if beyond in walls or beyond in bx or not (0 <= beyond[0] < G and 0 <= beyond[1] < G):
                    continue
                nb = (bx - {n}) | {beyond}
            key = (n, nb)
            if key in parent:
                continue
            parent[key] = ((agent, bx), name)
            if nb >= targets:
                found = key
                break
            q.append(key)
        if found:
            break
    if not found:
        return None
    plan = []
    k = found
    while parent[k] is not None:
        prev, name = parent[k]
        plan.append((name, {}))
        k = prev
    plan.reverse()
    return len(plan) if replay(cls, mod, lvl, plan) else None


def hc01_optimum(cls, mod, lvl, spec, limit):
    """hc01 hides its progress, so a frame-keyed search cannot tell states
    apart; hostile_baselines searches over (position, progress) instead."""
    import arcengine
    plan, _nodes = R.HB.bfs_level("hc01", cls, arcengine, lvl, max_nodes=min(limit, 400_000))
    if plan is None:
        return None
    return len(plan) if R.HB.verify("hc01", cls, arcengine, lvl, plan) else None


def ch01_optimum(cls, mod, lvl, spec, limit):
    """Exact by sandwich: the maze distance is a lower bound on any solution,
    and a shortest path that the chaser never intercepts is a solution of that
    length. Search only over shortest paths (moves that reduce the distance),
    simulating the pursuer with the game's own chaser_step; give up (None)
    when every shortest path is caught -- unknown is not a baseline."""
    walls, start, exit_cell, chaser0 = mod._parse(spec)
    g = [[WALL if (c, r) in walls else FLOOR for c in range(G)] for r in range(G)]
    dist_exit, _ = bfs(g, exit_cell)
    if start not in dist_exit:
        return None
    best = dist_exit[start]
    sys.setrecursionlimit(10000)
    plan = []

    def dfs(agent, chaser, t, depth):
        if agent == exit_cell:
            return True
        if depth > limit:
            return False
        for d, name in ARROW.items():
            n = (agent[0] + d[0], agent[1] + d[1])
            if n not in dist_exit or dist_exit[n] != dist_exit[agent] - 1:
                continue
            if n == exit_cell:
                plan.append((name, {}))
                return True
            nc = mod.chaser_step(t, chaser, n, walls)
            if nc == n:
                continue
            plan.append((name, {}))
            if dfs(n, nc, t + 1, depth + 1):
                return True
            plan.pop()
        return False

    if not dfs(start, chaser0, 0, 0):
        return None
    if len(plan) != best:
        return None
    return best if replay(cls, mod, lvl, plan) else None


def _walls_and_ends(mod, spec):
    walls = {(c, r) for r, row in enumerate(spec["rows"]) for c, ch in enumerate(row) if ch == "#"}
    start = next((c, r) for r, row in enumerate(spec["rows"]) for c, ch in enumerate(row) if ch == "P")
    exit_cell = next((c, r) for r, row in enumerate(spec["rows"]) for c, ch in enumerate(row) if ch == "E")
    return walls, start, exit_cell


def _shortest_paths(walls, start, exit_cell, survive, limit):
    """DFS over shortest paths only; `survive(cell, t)` vetoes a step."""
    g = [[WALL if (c, r) in walls else FLOOR for c in range(G)] for r in range(G)]
    dist_exit, _ = bfs(g, exit_cell)
    if start not in dist_exit:
        return None
    plan = []
    budget = [limit]

    def dfs(cell, t):
        if cell == exit_cell:
            return True
        budget[0] -= 1
        if budget[0] < 0:
            return False
        for d, name in ARROW.items():
            n = (cell[0] + d[0], cell[1] + d[1])
            if n not in dist_exit or dist_exit[n] != dist_exit[cell] - 1:
                continue
            if n != exit_cell and not survive(n, t + 1):
                continue
            plan.append((name, {}))
            if dfs(n, t + 1):
                return True
            plan.pop()
        return False

    return plan if dfs(start, 0) else None


def path_optimum(cls, mod, lvl, spec, limit):
    """Sandwich: maze distance is a lower bound; a shortest path that the
    engine accepts is a solution of that length."""
    walls, start, exit_cell = _walls_and_ends(mod, spec)
    plan = _shortest_paths(walls, start, exit_cell, lambda cell, t: True, limit)
    if plan is None:
        return None
    return len(plan) if replay(cls, mod, lvl, plan) else None


def wf01_optimum(cls, mod, lvl, spec, limit):
    """Same sandwich, but a step onto a burning cell is vetoed (the game's own
    fire_mask); the level is kept only if SOME shortest path outruns the fire."""
    walls, start, exit_cell = _walls_and_ends(mod, spec)
    masks = {}

    def alive(cell, t):
        if t not in masks:
            masks[t] = mod.fire_mask(spec, t)
        return cell not in masks[t]

    if not alive(start, 0):
        return None
    plan = _shortest_paths(walls, start, exit_cell, alive, limit)
    if plan is None:
        return None
    return len(plan) if replay(cls, mod, lvl, plan) else None


VERIFY = {"sk01": sk01_optimum, "hc01": hc01_optimum, "ch01": ch01_optimum,
          "vn01": path_optimum, "sn01": path_optimum, "fw01": path_optimum,
          "wf01": wf01_optimum}


# -------------------------------------------------------------------- writers
HEADER = (
    "    # 03.09: built by scripts/plant_levels.py -- each level is derived from\n"
    "    # a PLANTED solution on the full 64x64 board (GRID {grid}, CELL {cell}),\n"
    "    # along a real public game's difficulty curve. A random layout could\n"
    "    # not carry this mechanic (measured, scripts/probe_regen_fitness.py).\n"
    "    # {proof}\n")
PROOF_BFS = "Every baseline below is the optimum found by BFS through the engine."
PROOF_LZ = ("Baseline == mirrors: each mirror's wrong deflection leaves the board "
            "without meeting another mirror or the target, so every mirror must be flipped.")
PROOF_MR = ("Baseline == sum(select + pocket depth + rotation): the beam runs in a walled "
            "corridor with one bend per mirror, and each mirror can only leave its dead-end pocket "
            "onto its bend.")
PROOF_PI = ("Baseline == planted rotations: the trace never touches itself, so it is "
            "the only route and every tile has one working orientation.")


def write_levels(gid, name, results, proof):
    py = next((ROOT / "our_games" / gid).glob(f"*/{gid}.py"))
    src = py.read_text(encoding="utf-8")
    body = []
    for spec, _opt in results:
        if "rows" in spec:
            lines = ",\n".join(f"        {r!r}" for r in spec["rows"])
            tail = "".join(f", {k}={v!r}" for k, v in spec.items() if k != "rows")
            body.append("    dict(rows=[\n" + lines + ",\n    ]" + tail + "),")
        else:
            fields = ",\n".join(f"         {k}={v!r}" for k, v in spec.items())
            body.append("    dict(\n" + fields + "),")
    block = (f"{name} = [\n" + HEADER.format(grid=G, cell=R.NEW_CELL, proof=proof)
             + "\n".join(body) + "\n]")
    src = re.sub(rf"^{name} = \[.*?^\]", lambda _m: block, src, count=1, flags=re.S | re.M)
    src = re.sub(r"^GRID = \d+$", f"GRID = {G}", src, count=1, flags=re.M)
    src = re.sub(r"^CELL = \d+$", f"CELL = {R.NEW_CELL}", src, count=1, flags=re.M)
    py.write_text(src, encoding="utf-8")
    md_path = py.parent / "metadata.json"
    md = json.loads(md_path.read_text(encoding="utf-8"))
    md["baseline_actions"] = [o for _, o in results]
    md_path.write_text(json.dumps(md, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------- main
ASCII = {"sk01": build_sk01, "gv01": build_gv01, "bx01": build_bx01, "fl01": build_fl01,
         "hc01": build_hc01, "ch01": build_ch01, "vn01": build_vn01, "sn01": build_sn01,
         "fw01": build_fw01, "wf01": build_wf01, "tr01": build_tr01, "sw01": build_sw01}
COORD = {"lz01": build_lz01, "pi01": build_pi01, "mr01": build_mr01}


def run_ascii(gid, args):
    cls, md, mod = V.load(gid)
    name, levels, _rows_of, shape = R.levels_of(mod)
    mod.GRID, mod.CELL = G, R.NEW_CELL
    curve = R.curve_for(gid)
    rng = random.Random(args.seed)
    print(f"\n{gid}: было {md['baseline_actions']}")
    results, used = [], set()
    for lvl, target in enumerate(curve):
        best = None
        deadline = time.time() + args.seconds
        tries = 0
        while time.time() < deadline and tries < args.tries:
            tries += 1
            spec = ASCII[gid](rng, target)
            if spec is None or repr(spec["rows"]) in used:
                continue
            saved = levels[:]
            saved_tables = {}
            try:
                while len(levels) <= lvl:
                    levels.append(levels[-1])
                levels[lvl] = dict(levels[lvl], **spec) if shape == "dict" else spec["rows"]
                # bx01 derives two module tables from LAYOUTS at import time;
                # they must follow the candidate or the ball respawns on the
                # old board and the stock is the old level's
                if gid == "bx01":
                    start = next((c, r) for r, row in enumerate(spec["rows"])
                                 for c, ch in enumerate(row) if ch == "P")
                    for attr, val in (("N_STOCK", spec["stock"]), ("START_CELLS", start)):
                        tbl = getattr(mod, attr)
                        saved_tables[attr] = tbl[:]
                        while len(tbl) <= lvl:
                            tbl.append(tbl[-1])
                        tbl[lvl] = val
                if gid in VERIFY:
                    opt = VERIFY[gid](cls, mod, lvl, spec, args.limit * 10)
                else:
                    opt, _explored = V.optimum(cls, mod, lvl, args.limit)
                hostile = gid in (getattr(R.HB, "GAMES", {}) if R.HB else {})
                bad = R.structural_violations(gid, mod, lvl, opt) if (opt and hostile) else []
            finally:
                levels[:] = saved
                for attr, val in saved_tables.items():
                    getattr(mod, attr)[:] = val
            if opt is None or bad:
                continue
            if best is None or abs(opt - target) < abs(best[1] - target):
                best = (spec, opt)
            if best[1] == target:
                break
        if best is not None:
            used.add(repr(best[0]["rows"]))
        results.append(best)
        print(f"   уровень {lvl+1}: цель {target:3} -> {best[1] if best else 'нет решения'}"
              f"   ({tries} попыток)", flush=True)
    return name, [r for r in results if r is not None], PROOF_BFS


def run_coord(gid, args):
    cls, md, mod = V.load(gid)
    mod.GRID, mod.CELL = G, R.NEW_CELL
    name = "LEVELS"
    levels = mod.LEVELS
    curve = R.curve_for(gid)
    rng = random.Random(args.seed)
    print(f"\n{gid}: было {md['baseline_actions']}")
    results, used = [], set()
    for lvl, target in enumerate(curve):
        best = None
        deadline = time.time() + args.seconds
        tries = 0
        while time.time() < deadline and tries < args.tries:
            tries += 1
            out = COORD[gid](rng, target)
            if out is None:
                continue
            spec, cost = out[0], out[1]
            extra = out[2] if len(out) > 2 else None
            if repr(spec) in used:
                continue
            if best is not None and abs(cost - target) >= abs(best[1] - target):
                continue
            # the planted answer must clear the level on the real engine, and
            # the untouched level must NOT be solved already
            saved = levels[:]
            try:
                while len(levels) <= lvl:
                    levels.append(levels[-1])
                levels[lvl] = spec
                if gid == "lz01":
                    orients0 = {(c, r): o for o, c, r in spec["mirrors"]}
                    if all(t in set(mod.trace_beam(spec, orients0)) for t in spec["targets"]):
                        continue
                    plan = [click(mod, c, r) for _o, c, r in spec["mirrors"]]
                elif gid == "mr01":
                    if mod.solved(spec, list(spec["mirrors"])):
                        continue
                    plan = mr01_plan(mod, spec, extra)
                else:
                    if mod.connected(spec, dict(spec["tiles"])):
                        continue
                    plan = pi01_plan(mod, spec)
                    if plan is None or len(plan) != cost:
                        continue
                if not replay(cls, mod, lvl, plan):
                    print(f"   уровень {lvl+1}: заложенное решение НЕ прошло на движке -- пропуск")
                    continue
                if args.exhaustive:
                    checker = {"lz01": lz01_min_clicks, "pi01": pi01_min_clicks,
                               "mr01": mr01_min_actions}[gid]
                    proof = checker(mod, spec)
                    if proof != cost:
                        print(f"   уровень {lvl+1}: ПЕРЕБОР даёт {proof}, построение {cost} -- пропуск")
                        continue
            finally:
                levels[:] = saved
            best = (spec, cost)
            if cost == target:
                break
        if best is not None:
            used.add(repr(best[0]))
        results.append(best)
        print(f"   уровень {lvl+1}: цель {target:3} -> {best[1] if best else 'нет решения'}"
              f"   ({tries} попыток)", flush=True)
    proof = {"lz01": PROOF_LZ, "pi01": PROOF_PI, "mr01": PROOF_MR}[gid]
    return name, [r for r in results if r is not None], proof


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--game", action="append", required=True)
    ap.add_argument("--tries", type=int, default=400)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--limit", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--exhaustive", action="store_true",
                    help="lz01/pi01: also prove each level's optimum by brute force (small levels)")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    for gid in args.game:
        if gid in ASCII:
            name, ok, proof = run_ascii(gid, args)
        elif gid in COORD:
            name, ok, proof = run_coord(gid, args)
        else:
            print(f"{gid}: нет конструктора")
            continue
        curve = R.curve_for(gid)
        print(f"   получено уровней: {len(ok)}/{len(curve)}")
        if args.write:
            if len(ok) < 6:
                print(f"   НЕ ПИШУ: уровней {len(ok)}, у публичных минимум шесть")
                continue
            write_levels(gid, name, ok, proof)
            print(f"   записано: {[o for _, o in ok]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

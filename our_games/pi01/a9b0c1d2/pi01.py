# Original game for the atlas testbed (30.08.2026) -- NOT an ARC Prize game.
# "Pipes" (pi01) -- rank-8 mechanic from the Gemini round-11 pool plan:
# topological connectivity (distance-to-goal = number of misaligned
# joints, not any spatial metric).
#
# Rules: the ONLY control is ACTION6 -- clicking a pipe tile rotates it
# 90 degrees clockwise. Fixed source (S) and sink (K) sit on the border;
# the level completes when an unbroken pipe path connects them (facing
# open ports on adjacent tiles).

from arcengine import (
    ARCBaseGame,
    BlockingMode,
    Camera,
    GameAction,
    Level,
    Sprite,
)

CELL = 3
GRID = 21


def _click_cell(data):
    """Frame pixel -> board cell, for any GRID/CELL the camera can show."""
    scale = max(1, 64 // (GRID * CELL))
    off = (64 - GRID * CELL * scale) // 2
    return ((int(data.get("x", 0)) - off) // (CELL * scale),
            (int(data.get("y", 0)) - off) // (CELL * scale))


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

WALL = 9
SOURCE = 2
SINK = 4
PIPE = 8
PIPE_LIT = 3

# ports bitmask: N=1, E=2, S=4, W=8. Rotate CW: N->E->S->W->N.
# Each level: source=(c,r,port_out), sink=(c,r,port_in) on the border ring;
# tiles={(c,r): ports} rotatable pipe tiles in the interior.
LEVELS = [
    # 03.09: built by scripts/plant_levels.py -- each level is derived from
    # a PLANTED solution on the full 64x64 board (GRID 21, CELL 3),
    # along a real public game's difficulty curve. A random layout could
    # not carry this mechanic (measured, scripts/probe_regen_fitness.py).
    # Baseline == planted rotations: the trace never touches itself, so it is the only route and every tile has one working orientation.
    dict(
         source=(0, 15, 2),
         sink=(20, 16, 8),
         tiles={(1, 15): 6, (1, 16): 3, (2, 16): 10, (3, 16): 10, (4, 16): 12, (4, 17): 3, (5, 17): 5, (6, 17): 10, (7, 17): 12, (7, 16): 10, (7, 15): 3, (8, 15): 5, (9, 15): 12, (9, 14): 6, (10, 14): 5, (11, 14): 6, (11, 15): 9, (12, 15): 5, (13, 15): 6, (13, 16): 3, (14, 16): 5, (15, 16): 6, (15, 17): 3, (16, 17): 5, (17, 17): 5, (18, 17): 5, (19, 17): 6, (19, 16): 12}),
    dict(
         source=(0, 4, 2),
         sink=(20, 6, 8),
         tiles={(1, 4): 10, (2, 4): 12, (2, 3): 3, (3, 3): 12, (3, 2): 3, (4, 2): 5, (5, 2): 10, (6, 2): 5, (7, 2): 5, (8, 2): 5, (9, 2): 12, (9, 1): 3, (10, 1): 5, (11, 1): 5, (12, 1): 5, (13, 1): 12, (13, 2): 5, (13, 3): 12, (12, 3): 3, (12, 4): 10, (12, 5): 10, (12, 6): 12, (13, 6): 5, (14, 6): 5, (15, 6): 3, (15, 7): 9, (16, 7): 5, (17, 7): 5, (18, 7): 5, (19, 7): 3, (19, 6): 12}),
    dict(
         source=(0, 9, 2),
         sink=(20, 8, 8),
         tiles={(1, 9): 12, (1, 8): 3, (2, 8): 12, (2, 7): 3, (3, 7): 12, (3, 6): 10, (3, 5): 6, (2, 5): 9, (2, 4): 10, (2, 3): 10, (2, 2): 3, (3, 2): 5, (4, 2): 5, (5, 2): 5, (6, 2): 12, (6, 1): 3, (7, 1): 5, (8, 1): 10, (9, 1): 10, (10, 1): 3, (10, 2): 12, (11, 2): 3, (11, 3): 10, (11, 4): 5, (11, 5): 10, (11, 6): 10, (11, 7): 12, (12, 7): 6, (12, 8): 12, (13, 8): 5, (14, 8): 6, (14, 7): 3, (15, 7): 10, (16, 7): 3, (16, 8): 12, (17, 8): 3, (17, 9): 12, (18, 9): 5, (19, 9): 3, (19, 8): 12}),
    dict(
         source=(0, 9, 2),
         sink=(20, 11, 8),
         tiles={(1, 9): 5, (2, 9): 5, (3, 9): 5, (4, 9): 10, (5, 9): 12, (5, 10): 9, (6, 10): 5, (7, 10): 5, (8, 10): 12, (8, 9): 10, (8, 8): 3, (9, 8): 10, (10, 8): 9, (10, 7): 3, (11, 7): 12, (11, 6): 10, (11, 5): 3, (12, 5): 5, (13, 5): 5, (14, 5): 5, (15, 5): 12, (15, 6): 9, (16, 6): 12, (16, 7): 3, (17, 7): 6, (17, 8): 9, (18, 8): 12, (18, 9): 9, (19, 9): 3, (19, 10): 10, (19, 11): 6}),
    dict(
         source=(0, 2, 2),
         sink=(20, 9, 8),
         tiles={(1, 2): 12, (1, 1): 3, (2, 1): 5, (3, 1): 5, (4, 1): 5, (5, 1): 3, (5, 2): 12, (6, 2): 5, (7, 2): 6, (7, 1): 3, (8, 1): 5, (9, 1): 3, (9, 2): 12, (10, 2): 5, (11, 2): 6, (11, 3): 12, (12, 3): 5, (13, 3): 12, (13, 2): 9, (14, 2): 5, (15, 2): 6, (15, 3): 12, (16, 3): 5, (17, 3): 5, (18, 3): 3, (18, 4): 10, (18, 5): 6, (17, 5): 5, (16, 5): 9, (16, 6): 10, (16, 7): 12, (17, 7): 3, (17, 8): 12, (18, 8): 3, (18, 9): 12, (19, 9): 5}),
    dict(
         source=(0, 13, 2),
         sink=(20, 10, 8),
         tiles={(1, 13): 5, (2, 13): 12, (2, 12): 10, (2, 11): 3, (3, 11): 5, (4, 11): 12, (4, 10): 3, (5, 10): 5, (6, 10): 6, (6, 11): 9, (7, 11): 5, (8, 11): 6, (8, 12): 9, (9, 12): 6, (9, 13): 10, (9, 14): 9, (10, 14): 10, (11, 14): 6, (11, 15): 3, (12, 15): 5, (13, 15): 10, (14, 15): 12, (14, 14): 10, (14, 13): 6, (13, 13): 5, (12, 13): 9, (12, 12): 12, (11, 12): 12, (11, 11): 10, (11, 10): 10, (11, 9): 9, (12, 9): 10, (13, 9): 5, (14, 9): 3, (14, 10): 9, (15, 10): 3, (15, 11): 12, (16, 11): 6, (16, 12): 12, (17, 12): 5, (18, 12): 6, (18, 11): 9, (19, 11): 6, (19, 10): 9}),
]


def rot_cw(ports):
    return ((ports << 1) | (ports >> 3)) & 0b1111


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _pipe_px(ports, lit=False):
    color = PIPE_LIT if lit else PIPE
    px = [[0] * CELL for _ in range(CELL)]
    _fill_inner(px, color)
    if ports & 1:
        _fill_edge(px, 'top', color)
    if ports & 2:
        _fill_edge(px, 'right', color)
    if ports & 4:
        _fill_edge(px, 'bottom', color)
    if ports & 8:
        _fill_edge(px, 'left', color)
    return px


OPP = {1: 4, 4: 1, 2: 8, 8: 2}
DELTA = {1: (0, -1), 2: (1, 0), 4: (0, 1), 8: (-1, 0)}


def connected(spec, ports_map):
    """True if source reaches sink through facing open ports."""
    sc, sr, sport = spec["source"]
    kc, kr, kport = spec["sink"]
    # start: the tile adjacent to the source, if it opens toward it
    dx, dy = DELTA[sport]
    cur = (sc + dx, sr + dy)
    if cur not in ports_map or not (ports_map[cur] & OPP[sport]):
        return False
    seen = {cur}
    stack = [cur]
    while stack:
        c, r = stack.pop()
        ports = ports_map[(c, r)]
        for p, (dx, dy) in DELTA.items():
            if not (ports & p):
                continue
            nb = (c + dx, r + dy)
            if nb == (kc, kr) and p == OPP[kport]:
                return True
            if nb in ports_map and (ports_map[nb] & OPP[p]) and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return False


def _build_level(index, spec):
    sprites = []
    for r in range(GRID):
        for c in range(GRID):
            if r in (0, GRID - 1) or c in (0, GRID - 1):
                if (c, r) not in ((spec["source"][0], spec["source"][1]),
                                  (spec["sink"][0], spec["sink"][1])):
                    sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=c * CELL, y=r * CELL,
                                          blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
    sc, sr, _ = spec["source"]
    kc, kr, _ = spec["sink"]
    sprites.append(Sprite(_rect(SOURCE), name="source", x=sc * CELL, y=sr * CELL,
                          layer=1, blocking=BlockingMode.PIXEL_PERFECT))
    sprites.append(Sprite(_rect(SINK), name="sink", x=kc * CELL, y=kr * CELL,
                          layer=1, blocking=BlockingMode.PIXEL_PERFECT))
    for i, ((c, r), ports) in enumerate(sorted(spec["tiles"].items())):
        sprites.append(Sprite(_pipe_px(ports), name=f"tile_{i}", x=c * CELL, y=r * CELL,
                              layer=2, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Pi01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="pi01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[6],
            seed=seed,
        )
        self._ports: dict = {}
        self._load()

    def _load(self):
        self._ports = dict(LEVELS[self.level_index]["tiles"])

    def on_set_level(self, level):
        self._load()

    def _atlas_reset_level_state(self):
        self._load()
        self._sync()

    def level_reset(self):
        super().level_reset()
        self._atlas_reset_level_state()

    def full_reset(self):
        super().full_reset()
        self._atlas_reset_level_state()

    def _sprite(self, name):
        found = self.current_level.get_sprites_by_name(name)
        return found[0] if found else None

    def _sync(self):
        spec = LEVELS[self.level_index]
        lit = connected(spec, self._ports)
        for i, ((c, r), _) in enumerate(sorted(spec["tiles"].items())):
            s = self._sprite(f"tile_{i}")
            if s is not None:
                px = _pipe_px(self._ports[(c, r)], lit=lit)
                for rr in range(CELL):
                    for cc in range(CELL):
                        s.pixels[rr][cc] = px[rr][cc]
        return lit

    def step(self) -> None:
        spec = LEVELS[self.level_index]
        action = self.action.id
        if action == GameAction.RESET:
            self._sync()
            self.complete_action()
            return
        if action == GameAction.ACTION6:
            x, y = _click_cell(self.action.data)
            if (x, y) in self._ports:
                self._ports[(x, y)] = rot_cw(self._ports[(x, y)])
        lit = self._sync()
        if lit:
            self.next_level()
        self.complete_action()

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

CELL = 4
GRID = 8

WALL = 9
SOURCE = 2
SINK = 4
PIPE = 8
PIPE_LIT = 3

# ports bitmask: N=1, E=2, S=4, W=8. Rotate CW: N->E->S->W->N.
# Each level: source=(c,r,port_out), sink=(c,r,port_in) on the border ring;
# tiles={(c,r): ports} rotatable pipe tiles in the interior.
LEVELS = [
    dict(source=(0, 5, 2), sink=(7, 5, 8),
         tiles={(1, 5): 10, (2, 5): 10, (3, 5): 5, (4, 5): 5, (5, 5): 5, (6, 5): 5}),
    dict(source=(0, 5, 2), sink=(7, 5, 8),
         tiles={(1, 5): 5, (2, 5): 10, (3, 5): 5, (4, 5): 10, (5, 5): 10, (6, 5): 5}),
    dict(source=(0, 4, 2), sink=(7, 2, 8),
         tiles={(1, 4): 10, (2, 4): 10, (3, 4): 5, (4, 4): 10, (5, 4): 12, (5, 3): 10, (5, 2): 12, (6, 2): 5}),
    dict(source=(0, 5, 2), sink=(7, 3, 8),
         tiles={(1, 5): 6, (1, 4): 10, (1, 3): 3, (2, 3): 10, (3, 3): 5, (4, 3): 10, (5, 3): 5, (6, 3): 5}),
    dict(source=(0, 2, 2), sink=(7, 5, 8),
         tiles={(1, 2): 5, (2, 2): 9, (2, 3): 9, (3, 3): 12, (3, 4): 10, (3, 5): 6, (4, 5): 10, (5, 5): 10, (6, 5): 5}),
]


def rot_cw(ports):
    return ((ports << 1) | (ports >> 3)) & 0b1111


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _pipe_px(ports, lit=False):
    color = PIPE_LIT if lit else PIPE
    px = [[0] * CELL for _ in range(CELL)]
    px[1][1] = px[1][2] = px[2][1] = px[2][2] = color
    if ports & 1:
        px[0][1] = px[0][2] = color
    if ports & 2:
        px[1][3] = px[2][3] = color
    if ports & 4:
        px[3][1] = px[3][2] = color
    if ports & 8:
        px[1][0] = px[2][0] = color
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
            x = int(self.action.data.get("x", 0)) // 8
            y = int(self.action.data.get("y", 0)) // 8
            if (x, y) in self._ports:
                self._ports[(x, y)] = rot_cw(self._ports[(x, y)])
        lit = self._sync()
        if lit:
            self.next_level()
        self.complete_action()

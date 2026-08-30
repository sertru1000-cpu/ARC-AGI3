# Original game for the atlas testbed (30.08.2026) -- NOT an ARC Prize game.
# "Laser Rotate" (lz01) -- rank-5 mechanic from the Gemini round-11 pool
# plan: the PURE select-then-act trainer. No walking at all -- the ONLY
# control is ACTION6: clicking a mirror rotates it 90 degrees.
#
# Rules: a fixed wall emitter fires a beam every frame; '/' and chr(92)
# mirrors reflect it; clicking a mirror toggles its orientation. The
# level completes the moment the beam threads ALL target rings at once.

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
EMITTER = 4
MIRROR_C = 8
TARGET_C = 2
TARGET_LIT = 3
BEAM = 4

# emitter: (col,row,(dx,dy)); mirrors: list of (orient, col, row) with
# orient 0='/' 1=backslash; targets: list of (col,row); walls: interior.
LEVELS = [
    dict(emitter=(0, 2, (1, 0)), walls=[],
         mirrors=[(0, 1, 2)],
         targets=[(1, 4)]),
    dict(emitter=(7, 1, (-1, 0)), walls=[],
         mirrors=[(1, 5, 4), (0, 5, 1)],
         targets=[(4, 4)]),
    dict(emitter=(0, 3, (1, 0)), walls=[],
         mirrors=[(0, 2, 4), (0, 2, 3), (1, 5, 5)],
         targets=[(4, 4)]),
    dict(emitter=(7, 1, (-1, 0)), walls=[],
         mirrors=[(1, 5, 6), (1, 4, 1), (1, 4, 6)],
         targets=[(4, 3), (5, 5)]),
    dict(emitter=(6, 0, (0, 1)), walls=[],
         mirrors=[(0, 3, 5), (0, 4, 1), (0, 4, 2), (1, 6, 2)],
         targets=[(3, 1), (1, 1)]),
]


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _mirror_px(orient):
    px = [[0] * CELL for _ in range(CELL)]
    for i in range(CELL):
        j = (CELL - 1 - i) if orient == 0 else i
        px[i][j] = MIRROR_C
    return px


def _target_px(lit):
    color = TARGET_LIT if lit else TARGET_C
    return [[color if r in (0, CELL - 1) or c in (0, CELL - 1) else 0
             for c in range(CELL)] for r in range(CELL)]


def _beam_px():
    px = [[0] * CELL for _ in range(CELL)]
    for i in range(CELL):
        px[1][i] = px[2][i] = BEAM
    return px


def trace_beam(spec, orients):
    """Cells the beam crosses; orients: {(c,r): 0/1}."""
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
        if (c, r) in orients:
            if orients[(c, r)] == 0:   # '/'
                dx, dy = -dy, -dx
            else:
                dx, dy = dy, dx
        c, r = c + dx, r + dy
    return lit


def _build_level(index, spec):
    sprites = []
    walls = set(spec["walls"])
    ec, er, _ = spec["emitter"]
    for r in range(GRID):
        for c in range(GRID):
            if r in (0, GRID - 1) or c in (0, GRID - 1) or (c, r) in walls:
                if (c, r) != (ec, er):
                    sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=c * CELL, y=r * CELL,
                                          blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
    sprites.append(Sprite(_rect(EMITTER), name="emitter", x=ec * CELL, y=er * CELL,
                          layer=1, blocking=BlockingMode.PIXEL_PERFECT))
    for i, (c, r) in enumerate(spec["targets"]):
        sprites.append(Sprite(_target_px(False), name=f"target_{i}", x=c * CELL, y=r * CELL,
                              layer=2, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    for i, (o, c, r) in enumerate(spec["mirrors"]):
        sprites.append(Sprite(_mirror_px(o), name=f"mirror_{i}", x=c * CELL, y=r * CELL,
                              layer=3, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Lz01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="lz01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[6],
            seed=seed,
        )
        self._orients: dict[int, list] = {}
        self._beam_count = 0
        self._load()

    def _load(self):
        self._orients = {i: [o, c, r] for i, (o, c, r) in enumerate(LEVELS[self.level_index]["mirrors"])}

    def on_set_level(self, level):
        self._load()

    def _atlas_reset_level_state(self):
        self._beam_count = 0
        self._load()

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
        for i, (o, c, r) in self._orients.items():
            s = self._sprite(f"mirror_{i}")
            if s is not None:
                px = _mirror_px(o)
                for rr in range(CELL):
                    for cc in range(CELL):
                        s.pixels[rr][cc] = px[rr][cc]
        orient_map = {(c, r): o for o, c, r in self._orients.values()}
        lit = trace_beam(spec, orient_map)
        for i in range(self._beam_count):
            s = self._sprite(f"beam_{i}")
            if s is not None:
                self.current_level.remove_sprite(s)
        mirror_cells = set(orient_map)
        draw = [cell for cell in lit if cell not in mirror_cells]
        for i, (c, r) in enumerate(draw):
            self.current_level.add_sprite(
                Sprite(_beam_px(), name=f"beam_{i}", x=c * CELL, y=r * CELL, layer=0,
                       blocking=BlockingMode.NOT_BLOCKED, collidable=False))
        self._beam_count = len(draw)
        lit_set = set(lit)
        for i, t in enumerate(spec["targets"]):
            s = self._sprite(f"target_{i}")
            if s is not None:
                px = _target_px(t in lit_set)
                for rr in range(CELL):
                    for cc in range(CELL):
                        s.pixels[rr][cc] = px[rr][cc]
        return lit_set

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
            for i, (o, c, r) in self._orients.items():
                if (c, r) == (x, y):
                    self._orients[i][0] = 1 - o
                    break
        lit = self._sync()
        if all(t in lit for t in spec["targets"]):
            self.next_level()
        self.complete_action()

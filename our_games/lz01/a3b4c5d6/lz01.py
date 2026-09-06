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

CELL = 3
GRID = 21


def _click_cell(data):
    """Frame pixel -> board cell, for any GRID/CELL the camera can show."""
    scale = max(1, 64 // (GRID * CELL))
    off = (64 - GRID * CELL * scale) // 2
    return ((int(data.get("x", 0)) - off) // (CELL * scale),
            (int(data.get("y", 0)) - off) // (CELL * scale))

WALL = 9
EMITTER = 4
MIRROR_C = 8
TARGET_C = 2
TARGET_LIT = 3
BEAM = 4

# emitter: (col,row,(dx,dy)); mirrors: list of (orient, col, row) with
# orient 0='/' 1=backslash; targets: list of (col,row); walls: interior.
LEVELS = [
    # 03.09: built by scripts/plant_levels.py -- each level is derived from
    # a PLANTED solution on the full 64x64 board (GRID 21, CELL 3),
    # along a real public game's difficulty curve. A random layout could
    # not carry this mechanic (measured, scripts/probe_regen_fitness.py).
    # Baseline == mirrors: each mirror's wrong deflection leaves the board without meeting another mirror or the target, so every mirror must be flipped.
    dict(
         emitter=(0, 4, (1, 0)),
         walls=[],
         mirrors=[(0, 2, 4), (0, 2, 5), (0, 3, 5), (0, 3, 6), (0, 4, 6), (0, 4, 7), (0, 5, 7), (0, 5, 9), (0, 8, 9), (0, 8, 10), (0, 9, 10), (0, 9, 11), (0, 10, 11), (0, 10, 12), (0, 11, 12), (0, 11, 13), (0, 12, 13), (0, 12, 14)],
         targets=[(15, 14)]),
    dict(
         emitter=(0, 2, (1, 0)),
         walls=[],
         mirrors=[(0, 2, 2), (0, 2, 3), (0, 3, 3), (0, 3, 4), (0, 4, 4), (0, 4, 5), (0, 7, 5), (0, 7, 6), (0, 8, 6), (0, 8, 8), (0, 9, 8), (0, 9, 9), (0, 10, 9), (0, 10, 10), (0, 11, 10), (0, 11, 11), (0, 12, 11), (0, 12, 12), (0, 13, 12), (0, 13, 13), (0, 14, 13), (0, 14, 14), (0, 15, 14), (0, 15, 15), (0, 16, 15), (0, 16, 16), (0, 17, 16), (0, 17, 17)],
         targets=[(19, 17)]),
    dict(
         emitter=(0, 2, (1, 0)),
         walls=[],
         mirrors=[(0, 3, 2), (0, 3, 3), (0, 4, 3), (0, 4, 6), (0, 5, 6), (0, 5, 7), (0, 6, 7), (0, 6, 8), (0, 7, 8), (0, 7, 9), (0, 8, 9), (0, 8, 10), (0, 9, 10), (0, 9, 11), (0, 10, 11), (0, 10, 12), (0, 13, 12), (0, 13, 13)],
         targets=[(14, 13)]),
    dict(
         emitter=(0, 3, (1, 0)),
         walls=[],
         mirrors=[(0, 2, 3), (0, 2, 4), (0, 3, 4), (0, 3, 5), (0, 4, 5), (0, 4, 6), (0, 5, 6), (0, 5, 7), (0, 6, 7), (0, 6, 8), (0, 7, 8), (0, 7, 9), (0, 8, 9), (0, 8, 12), (0, 9, 12), (0, 9, 13), (0, 10, 13), (0, 10, 14), (0, 11, 14)],
         targets=[(11, 16)]),
    dict(
         emitter=(0, 2, (1, 0)),
         walls=[],
         mirrors=[(0, 2, 2), (0, 2, 3), (0, 3, 3), (0, 3, 4), (0, 4, 4), (0, 4, 5), (0, 5, 5), (0, 5, 6), (0, 6, 6), (0, 6, 7), (0, 7, 7), (0, 7, 9), (0, 8, 9), (0, 8, 10), (0, 9, 10), (0, 9, 11), (0, 10, 11), (0, 10, 12), (0, 11, 12), (0, 11, 13), (0, 13, 13), (0, 13, 14), (0, 14, 14), (0, 14, 15), (0, 15, 15), (0, 15, 16), (0, 16, 16), (0, 16, 17), (0, 17, 17), (0, 17, 18), (0, 18, 18)],
         targets=[(18, 19)]),
    dict(
         emitter=(0, 4, (1, 0)),
         walls=[],
         mirrors=[(0, 2, 4), (0, 2, 5), (0, 3, 5), (0, 3, 6), (0, 4, 6), (0, 4, 7), (0, 5, 7), (0, 5, 8), (0, 6, 8), (0, 6, 9), (0, 7, 9), (0, 7, 10), (0, 8, 10), (0, 8, 11), (0, 9, 11), (0, 9, 14), (0, 10, 14), (0, 10, 15), (0, 11, 15), (0, 11, 16), (0, 12, 16), (0, 12, 17), (0, 13, 17)],
         targets=[(13, 19)]),
    dict(
         emitter=(0, 1, (1, 0)),
         walls=[],
         mirrors=[(0, 2, 1), (0, 2, 2), (0, 3, 2), (0, 3, 3), (0, 4, 3), (0, 4, 4), (0, 5, 4), (0, 5, 5), (0, 6, 5), (0, 6, 6), (0, 7, 6), (0, 7, 7), (0, 8, 7), (0, 8, 8), (0, 9, 8), (0, 9, 9), (0, 10, 9), (0, 10, 10), (0, 11, 10), (0, 11, 11), (0, 12, 11), (0, 12, 12), (0, 13, 12), (0, 13, 13), (0, 14, 13), (0, 14, 14), (0, 15, 14), (0, 15, 15), (0, 16, 15), (0, 16, 16), (0, 17, 16), (0, 17, 17), (0, 18, 17), (0, 18, 18), (0, 19, 18)],
         targets=[(19, 19)]),
    dict(
         emitter=(0, 1, (1, 0)),
         walls=[],
         mirrors=[(0, 2, 1), (0, 2, 2), (0, 5, 2), (0, 5, 3), (0, 6, 3), (0, 6, 4), (0, 7, 4), (0, 7, 5), (0, 8, 5), (0, 8, 6), (0, 9, 6), (0, 9, 7), (0, 12, 7), (0, 12, 8), (0, 13, 8), (0, 13, 10), (0, 15, 10), (0, 15, 11)],
         targets=[(18, 11)]),
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
    # the beam is a band through the middle of the cell: the two centre rows
    # at CELL 4 (as the original drew it), the single centre row when CELL is
    # odd. The cellify rewrite left a call to a helper this file never had.
    px = [[0] * CELL for _ in range(CELL)]
    rows = [CELL // 2] if CELL % 2 else [CELL // 2 - 1, CELL // 2]
    for r in rows:
        for i in range(CELL):
            px[r][i] = BEAM
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
            x, y = _click_cell(self.action.data)
            for i, (o, c, r) in self._orients.items():
                if (c, r) == (x, y):
                    self._orients[i][0] = 1 - o
                    break
        lit = self._sync()
        if all(t in lit for t in spec["targets"]):
            self.next_level()
        self.complete_action()

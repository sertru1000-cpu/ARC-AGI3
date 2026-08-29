# Original game for the atlas testbed (29.08.2026) -- NOT an ARC Prize game.
# "Mirrors" (mr01) -- mechanic DICTATED BY THE USER (designer-of-record
# protocol, Gemini round 7): "На поле система зеркал, их можно двигать и
# поворачивать. Нужно таким образом направить луч в цель (или сразу
# несколько целей -- для более сложных уровней)."
#
# Translation decisions (implementation only): a fixed wall emitter fires a
# beam every frame; mirrors '/' and '\' reflect it 90 degrees; the beam is
# DRAWN live, so this is non-local causality with immediate feedback --
# rotate a mirror here, the light spot jumps there. MOUSE click selects a
# mirror, ACTION1-4 move it, ACTION5 rotates. No launch button: the level
# completes by itself the moment every target cell is lit simultaneously
# (targets are transparent to the beam, so one beam can thread several).

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
SEL = 12

# emitter: (col,row,dir) on the border; dir = (dx,dy) into the field.
# mirrors: list of (orient, col, row); orient 0='/' 1='\'.
# targets: list of (col,row).  walls: extra interior walls.
# solution: mirror placements found by the validator (baked by the search).
LEVELS = [
    dict(emitter=(0, 3, (1, 0)), walls=[],
         mirrors=[(0, 5, 5)], targets=[(3, 6)],
         solution=[(1, 3, 3)]),
    dict(emitter=(4, 0, (0, 1)), walls=[],
         mirrors=[(0, 1, 1), (1, 6, 6)], targets=[(1, 5)],
         solution=[(0, 1, 1), (0, 4, 5)]),
    dict(emitter=(0, 1, (1, 0)), walls=[(4, 3)],
         mirrors=[(0, 2, 6), (1, 5, 2)], targets=[(6, 5), (2, 5)],
         solution=[(1, 1, 5), (1, 1, 1)]),
    dict(emitter=(7, 4, (-1, 0)), walls=[(3, 2), (3, 5)],
         mirrors=[(0, 1, 1), (0, 6, 6), (1, 2, 3)], targets=[(1, 6), (5, 1)],
         solution=[(0, 1, 1), (1, 6, 1), (1, 6, 4)]),
    dict(emitter=(2, 7, (0, -1)), walls=[(4, 4), (2, 2)],
         mirrors=[(0, 1, 5), (1, 6, 1), (0, 5, 6)], targets=[(4, 6), (3, 6), (1, 4)],
         solution=[(0, 1, 3), (1, 1, 6), (1, 2, 3)]),
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


def trace_beam(spec, mirrors):
    """Cells the beam crosses + reflections; mirrors: {(c,r): orient}."""
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
        if (c, r) in mirrors:
            o = mirrors[(c, r)]
            if o == 0:   # '/'
                dx, dy = -dy, -dx
            else:        # '\'
                dx, dy = dy, dx
        c, r = c + dx, r + dy
    return lit


def solved(spec, mirror_list):
    mirrors = {(c, r): o for o, c, r in mirror_list}
    lit = set(trace_beam(spec, mirrors))
    return all(t in lit for t in spec["targets"])


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


class Mr01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="mr01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )
        self._mirrors = {}
        self._selected = None
        self._beam_count = 0
        self._load_mirrors()

    def on_set_level(self, level):
        self._selected = None
        self._load_mirrors()

    def _atlas_reset_level_state(self):
        self._selected = None
        self._beam_count = 0
        self._load_mirrors()

    def level_reset(self):
        super().level_reset()
        self._atlas_reset_level_state()

    def full_reset(self):
        super().full_reset()
        self._atlas_reset_level_state()

    def _load_mirrors(self):
        self._mirrors = {i: list(m) for i, m in enumerate(LEVELS[self.level_index]["mirrors"])}

    def _sprite(self, name):
        found = self.current_level.get_sprites_by_name(name)
        return found[0] if found else None

    def _sync(self):
        spec = LEVELS[self.level_index]
        for i, (o, c, r) in self._mirrors.items():
            s = self._sprite(f"mirror_{i}")
            if s is not None:
                s.set_position(c * CELL, r * CELL)
                px = _mirror_px(o)
                for rr in range(CELL):
                    for cc in range(CELL):
                        s.pixels[rr][cc] = px[rr][cc]
        # beam redraw
        lit = trace_beam(spec, {(c, r): o for o, c, r in self._mirrors.values()})
        mirror_cells = {(c, r) for o, c, r in self._mirrors.values()}
        for i in range(self._beam_count):
            s = self._sprite(f"beam_{i}")
            if s is not None:
                self.current_level.remove_sprite(s)
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
        # selection box
        sel = self._sprite("selbox")
        if self._selected is None:
            if sel is not None:
                self.current_level.remove_sprite(sel)
        else:
            _, c, r = self._mirrors[self._selected]
            box = [[SEL if rr in (0, CELL - 1) or cc in (0, CELL - 1) else 0
                    for cc in range(CELL)] for rr in range(CELL)]
            if sel is None:
                self.current_level.add_sprite(
                    Sprite(box, name="selbox", x=c * CELL, y=r * CELL, layer=4,
                           blocking=BlockingMode.NOT_BLOCKED, collidable=False))
            else:
                sel.set_position(c * CELL, r * CELL)

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
            hit = None
            for i, (o, c, r) in self._mirrors.items():
                if (c, r) == (x, y):
                    hit = i
            self._selected = hit
        elif self._selected is not None and action in (
                GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4):
            d = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1),
                 GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}[action]
            o, c, r = self._mirrors[self._selected]
            nc, nr = c + d[0], r + d[1]
            occupied = {(mc, mr) for j, (_, mc, mr) in self._mirrors.items() if j != self._selected}
            occupied |= set(spec["walls"]) | set(spec["targets"])
            if 0 < nc < GRID - 1 and 0 < nr < GRID - 1 and (nc, nr) not in occupied:
                self._mirrors[self._selected] = [o, nc, nr]
        elif self._selected is not None and action == GameAction.ACTION5:
            o, c, r = self._mirrors[self._selected]
            self._mirrors[self._selected] = [1 - o, c, r]
        self._sync()
        if solved(spec, [tuple(m) for m in self._mirrors.values()]):
            self.next_level()
        self.complete_action()

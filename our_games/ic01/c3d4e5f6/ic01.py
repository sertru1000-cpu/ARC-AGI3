# Original game for the atlas testbed (29.08.2026) -- NOT an ARC Prize game.
# "Icebergs" (ic01) -- mechanic DICTATED BY THE USER (designer-of-record
# protocol, Gemini round 7): "судно нужно провести к выходу из пролива, в
# котором хаотично курсируют айсберги. При столкновении айсберга с судном
# оно погибает."
#
# Translation decisions (implementation only): the engine is turn-based and
# must be deterministic, so "chaotic" = deterministic berg patterns that
# LOOK chaotic (different speeds, directions and phases, bouncing off the
# shores) -- a careful world-modeler can predict them, a careless player
# dies. Every ship action advances one tick; bergs then move; overlap =
# ship dies (game over; RESET restarts the level). ACTION5 = hold position
# for a tick (drift). Exit = the gap in the top shore.

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

SHORE = 9
SHIP = 3
BERG = 8          # icy blue
EXIT = 4          # exit gap marker (yellow water)
WATER_BG = 0

# Per level: shore walls ('#'), ship start 'P', exit 'E' (in the top row),
# and bergs: (col, row, dx, speed_period) -- berg moves dx each `period`
# ticks and bounces off shore cells.
LEVELS = [
    # L1: two slow bergs
    dict(rows=[
        "###E####",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        "#..P...#",
        "########",
    ], bergs=[(2, 2, 1, 2), (5, 4, -1, 2)]),
    # L2: three bergs, one fast
    dict(rows=[
        "####E###",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        "#.P....#",
        "########",
    ], bergs=[(1, 1, 1, 1), (6, 3, -1, 2), (3, 5, 1, 2)]),
    # L3: four bergs, mixed speeds and phases
    dict(rows=[
        "#E######",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        "#....P.#",
        "########",
    ], bergs=[(4, 1, -1, 1), (1, 2, 1, 2), (6, 4, -1, 1), (2, 5, 1, 2)]),
    # L4: five bergs + a mid-channel rock narrowing the fairway
    dict(rows=[
        "######E#",
        "#......#",
        "#......#",
        "#..##..#",
        "#......#",
        "#......#",
        "#P.....#",
        "########",
    ], bergs=[(5, 1, -1, 1), (1, 2, 1, 1), (5, 4, -1, 2), (2, 5, 1, 1), (6, 5, -1, 2)]),
    # L5: six bergs, two rocks, far exit
    dict(rows=[
        "#E######",
        "#......#",
        "#....#.#",
        "#......#",
        "#.#....#",
        "#......#",
        "#.....P#",
        "########",
    ], bergs=[(3, 1, 1, 1), (6, 2, -1, 2), (1, 3, 1, 1), (5, 3, -1, 2), (2, 5, -1, 1), (4, 6, 1, 2)]),
]


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _berg_px():
    px = [[0] * CELL for _ in range(CELL)]
    for r in range(CELL):
        for c in range(CELL):
            if r + c >= 1 and r + c <= 5:
                px[r][c] = BERG
    return px


def _build_level(index, spec):
    sprites = []
    ship = exit_cell = None
    for r, row in enumerate(spec["rows"]):
        for c, ch in enumerate(row):
            x, y = c * CELL, r * CELL
            if ch == "#":
                sprites.append(Sprite(_rect(SHORE), name=f"shore_{r}_{c}", x=x, y=y,
                                      blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
            elif ch == "E":
                exit_cell = (c, r)
                sprites.append(Sprite(_rect(EXIT), name="exit", x=x, y=y, layer=0,
                                      blocking=BlockingMode.NOT_BLOCKED, collidable=False))
            elif ch == "P":
                ship = (c, r)
    for i, (c, r, dx, period) in enumerate(spec["bergs"]):
        sprites.append(Sprite(_berg_px(), name=f"berg_{i}", x=c * CELL, y=r * CELL,
                              layer=2, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sc, sr = ship
    sprites.append(Sprite(_rect(SHIP), name="ship", x=sc * CELL, y=sr * CELL,
                          layer=3, blocking=BlockingMode.PIXEL_PERFECT))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


def berg_positions(spec, t):
    """Deterministic berg cells at tick t (bounce off shores)."""
    shore = set()
    for r, row in enumerate(spec["rows"]):
        for c, ch in enumerate(row):
            if ch == "#":
                shore.add((c, r))
    out = []
    for (c0, r, dx, period) in spec["bergs"]:
        steps = t // period
        c, d = c0, dx
        for _ in range(steps):
            nc = c + d
            if (nc, r) in shore or not (0 <= nc < GRID):
                d = -d
                nc = c + d
                if (nc, r) in shore:
                    nc = c
            c = nc
        out.append((c, r))
    return out


class Ic01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="ic01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )
        self._tick = 0

    def on_set_level(self, level):
        self._tick = 0

    def _sprite(self, name):
        found = self.current_level.get_sprites_by_name(name)
        return found[0] if found else None

    def step(self) -> None:
        action = self.action.id
        ship = self._sprite("ship")
        spec = LEVELS[self.level_index]
        moved = False
        if ship is not None and action in (GameAction.ACTION1, GameAction.ACTION2,
                                           GameAction.ACTION3, GameAction.ACTION4):
            d = {GameAction.ACTION1: (0, -CELL), GameAction.ACTION2: (0, CELL),
                 GameAction.ACTION3: (-CELL, 0), GameAction.ACTION4: (CELL, 0)}[action]
            self.try_move("ship", *d)
            moved = True
        elif action == GameAction.ACTION5:
            moved = True  # drift in place for one tick
        if not moved or ship is None:
            self.complete_action()
            return

        sx, sy = ship.x // CELL, ship.y // CELL
        # sailed straight into a berg?
        now = berg_positions(spec, self._tick)
        if (sx, sy) in now:
            self.lose()
            self.complete_action()
            return

        # reached the exit BEFORE bergs move -> level done
        exit_s = self._sprite("exit")
        if exit_s is not None and (sx, sy) == (exit_s.x // CELL, exit_s.y // CELL):
            self.next_level()
            self.complete_action()
            return

        # bergs advance one tick; a berg entering the ship's cell sinks it
        self._tick += 1
        nxt = berg_positions(spec, self._tick)
        for i, (c, r) in enumerate(nxt):
            b = self._sprite(f"berg_{i}")
            if b is not None:
                b.set_position(c * CELL, r * CELL)
        if (sx, sy) in nxt:
            self.lose()
        self.complete_action()

# Original game for the atlas testbed (30.08.2026) -- NOT an ARC Prize game.
# "Switches" (sw01) -- rank-4 mechanic from the Gemini round-11 pool plan:
# layered hypercube state space (position x global mode).
#
# Rules: ACTION1-4 move; ACTION5 pressed while standing on a lever toggles
# the GLOBAL mode. 'A'-walls are solid only in mode 0, 'B'-walls only in
# mode 1 (the other kind fades to a faint outline). Reach the exit pad.

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
AGENT = 3
EXIT = 4
LEVER = 6
A_COLOR = 8
B_COLOR = 11

# rows: '#'=wall, 'A'=solid in mode 0, 'B'=solid in mode 1, 'L'=lever,
# 'P'=agent, 'E'=exit.
LEVELS = [
    dict(rows=[
        "########",
        "#...PA.#",
        "#.....L#",
        "#......#",
        "#......#",
        "#....AA#",
        "#....AE#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#...AAB#",
        "#...A.L#",
        "#......#",
        "#......#",
        "#A.....#",
        "#EAB.P.#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#EA....#",
        "#A...A.#",
        "#BAB..A#",
        "#PB.A..#",
        "#...L..#",
        "#......#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#EA...B#",
        "#.A...A#",
        "#.AA...#",
        "#A....B#",
        "#.LB..P#",
        "#..B...#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#AA...P#",
        "#AB....#",
        "#A..B..#",
        "#.....B#",
        "#AA.BB.#",
        "#E.A..L#",
        "########",
    ]),
]


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _faint_px(color):
    px = [[0] * CELL for _ in range(CELL)]
    px[0][0] = px[0][CELL - 1] = px[CELL - 1][0] = px[CELL - 1][CELL - 1] = color
    return px


def _agent_px():
    px = [[0] * CELL for _ in range(CELL)]
    for r in (1, 2):
        for c in (1, 2):
            px[r][c] = AGENT
    px[0][1] = px[0][2] = AGENT
    return px


def _parse(spec):
    walls, a_walls, b_walls, levers = set(), set(), set(), set()
    start = exit_cell = None
    for r, row in enumerate(spec["rows"]):
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
    return walls, a_walls, b_walls, levers, start, exit_cell


def _build_level(index, spec):
    walls, a_walls, b_walls, levers, start, exit_cell = _parse(spec)
    sprites = []
    for (c, r) in walls:
        sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=c * CELL, y=r * CELL,
                              blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
    for (c, r) in a_walls:
        sprites.append(Sprite(_rect(A_COLOR), name=f"awall_{r}_{c}", x=c * CELL, y=r * CELL,
                              layer=1, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    for (c, r) in b_walls:
        sprites.append(Sprite(_faint_px(B_COLOR), name=f"bwall_{r}_{c}", x=c * CELL, y=r * CELL,
                              layer=1, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    for i, (c, r) in enumerate(sorted(levers)):
        sprites.append(Sprite(_faint_px(LEVER), name=f"lever_{i}", x=c * CELL, y=r * CELL,
                              layer=0, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_rect(EXIT), name="exit", x=exit_cell[0] * CELL, y=exit_cell[1] * CELL,
                          layer=0, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_agent_px(), name="agent", x=start[0] * CELL, y=start[1] * CELL,
                          layer=3, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Sw01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="sw01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )
        self._mode = 0

    def on_set_level(self, level):
        self._mode = 0

    def _atlas_reset_level_state(self):
        self._mode = 0
        self._sync_mode()

    def level_reset(self):
        super().level_reset()
        self._atlas_reset_level_state()

    def full_reset(self):
        super().full_reset()
        self._atlas_reset_level_state()

    def _sprite(self, name):
        found = self.current_level.get_sprites_by_name(name)
        return found[0] if found else None

    def _sync_mode(self):
        spec = LEVELS[self.level_index]
        walls, a_walls, b_walls, levers, start, exit_cell = _parse(spec)
        solid_a = self._mode == 0
        for (c, r) in a_walls:
            s = self._sprite(f"awall_{r}_{c}")
            if s is not None:
                px = _rect(A_COLOR) if solid_a else _faint_px(A_COLOR)
                for rr in range(CELL):
                    for cc in range(CELL):
                        s.pixels[rr][cc] = px[rr][cc]
        for (c, r) in b_walls:
            s = self._sprite(f"bwall_{r}_{c}")
            if s is not None:
                px = _rect(B_COLOR) if not solid_a else _faint_px(B_COLOR)
                for rr in range(CELL):
                    for cc in range(CELL):
                        s.pixels[rr][cc] = px[rr][cc]

    def step(self) -> None:
        spec = LEVELS[self.level_index]
        action = self.action.id
        if action == GameAction.RESET:
            self._sync_mode()
            self.complete_action()
            return
        walls, a_walls, b_walls, levers, start, exit_cell = _parse(spec)
        agent = self._sprite("agent")
        if agent is None:
            self.complete_action()
            return
        c, r = agent.x // CELL, agent.y // CELL
        solid = walls | (a_walls if self._mode == 0 else b_walls)

        if action == GameAction.ACTION5:
            if (c, r) in levers:
                self._mode = 1 - self._mode
                self._sync_mode()
        elif action in (GameAction.ACTION1, GameAction.ACTION2,
                        GameAction.ACTION3, GameAction.ACTION4):
            d = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1),
                 GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}[action]
            nc, nr = c + d[0], r + d[1]
            if (nc, nr) not in solid and 0 <= nc < GRID and 0 <= nr < GRID:
                agent.set_position(nc * CELL, nr * CELL)
                if (nc, nr) == exit_cell:
                    self.next_level()
        self.complete_action()

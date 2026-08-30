# Original game for the atlas testbed (30.08.2026) -- NOT an ARC Prize game.
# "Color Algebra" (cm01) -- rank-6 mechanic from the Gemini round-11 pool
# plan: tracking an internal agent ATTRIBUTE, not just coordinates.
#
# Rules: ACTION1-4 move. Stepping on a paint pad recolors the agent:
# red pad -> RED (or PURPLE if currently blue), blue pad -> BLUE (or
# PURPLE if currently red). Doors pass only the matching color: '1'=RED,
# '2'=BLUE, '3'=PURPLE. The agent's own pixels show its current color.
# Reach the exit pad.

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
EXIT = 4
NONE_C = 3
RED = 2
BLUE = 8
PURPLE = 6
PAD_R = 2
PAD_B = 8
DOOR_COLORS = {"1": RED, "2": BLUE, "3": PURPLE}

# rows: '#'=wall, 'P'=agent, 'E'=exit, 'r'/'b'=paint pads,
# '1'/'2'/'3'=doors (RED/BLUE/PURPLE only).
LEVELS = [
    dict(rows=[
        "########",
        "#....#E#",
        "#.....2#",
        "#.#r...#",
        "#.1#.b.#",
        "#.#...P#",
        "##..#..#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#....###",
        "#.#E...#",
        "#.....2#",
        "##.#1#.#",
        "#.##P..#",
        "#1r.b..#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#.#P..##",
        "#.###r.#",
        "#2.1..##",
        "#.E.#.##",
        "#.3....#",
        "#.b....#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#.#E#.P#",
        "#3...2.#",
        "#.#.#r.#",
        "##.1#..#",
        "#.##.#.#",
        "#.1.#.b#",
        "########",
    ]),
    dict(rows=[
        "########",
        "##....##",
        "##.....#",
        "#r##1.E#",
        "#...P3.#",
        "#b22.###",
        "####..1#",
        "########",
    ]),
]


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _pad_px(color):
    px = [[0] * CELL for _ in range(CELL)]
    for r in (1, 2):
        for c in range(CELL):
            px[r][c] = color
    return px


def _door_px(color):
    px = [[color] * CELL for _ in range(CELL)]
    px[1][1] = px[1][2] = px[2][1] = px[2][2] = 0
    return px


def _agent_px(color):
    px = [[0] * CELL for _ in range(CELL)]
    for r in (1, 2):
        for c in (1, 2):
            px[r][c] = color
    px[0][1] = px[0][2] = color
    return px


def mix(color, pad):
    if pad == "r":
        return PURPLE if color == BLUE else RED
    return PURPLE if color == RED else BLUE


def _parse(spec):
    walls, pads, doors = set(), {}, {}
    start = exit_cell = None
    for r, row in enumerate(spec["rows"]):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "E":
                exit_cell = (c, r)
            elif ch in ("r", "b"):
                pads[(c, r)] = ch
            elif ch in DOOR_COLORS:
                doors[(c, r)] = DOOR_COLORS[ch]
    return walls, pads, doors, start, exit_cell


def _build_level(index, spec):
    walls, pads, doors, start, exit_cell = _parse(spec)
    sprites = []
    for (c, r) in walls:
        sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=c * CELL, y=r * CELL,
                              blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
    for (c, r), ch in pads.items():
        sprites.append(Sprite(_pad_px(RED if ch == "r" else BLUE), name=f"pad_{r}_{c}",
                              x=c * CELL, y=r * CELL, layer=0,
                              blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    for (c, r), col in doors.items():
        sprites.append(Sprite(_door_px(col), name=f"door_{r}_{c}", x=c * CELL, y=r * CELL,
                              layer=1, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_rect(EXIT), name="exit", x=exit_cell[0] * CELL, y=exit_cell[1] * CELL,
                          layer=0, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_agent_px(NONE_C), name="agent", x=start[0] * CELL, y=start[1] * CELL,
                          layer=3, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Cm01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="cm01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )
        self._color = NONE_C

    def on_set_level(self, level):
        self._color = NONE_C

    def _atlas_reset_level_state(self):
        self._color = NONE_C
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
        agent = self._sprite("agent")
        if agent is not None:
            px = _agent_px(self._color)
            for rr in range(CELL):
                for cc in range(CELL):
                    agent.pixels[rr][cc] = px[rr][cc]

    def step(self) -> None:
        spec = LEVELS[self.level_index]
        action = self.action.id
        if action == GameAction.RESET:
            self._sync()
            self.complete_action()
            return
        walls, pads, doors, start, exit_cell = _parse(spec)
        agent = self._sprite("agent")
        if agent is None:
            self.complete_action()
            return
        c, r = agent.x // CELL, agent.y // CELL

        if action in (GameAction.ACTION1, GameAction.ACTION2,
                      GameAction.ACTION3, GameAction.ACTION4):
            d = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1),
                 GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}[action]
            nc, nr = c + d[0], r + d[1]
            blocked = (nc, nr) in walls or not (0 <= nc < GRID and 0 <= nr < GRID)
            if not blocked and (nc, nr) in doors and doors[(nc, nr)] != self._color:
                blocked = True
            if not blocked:
                agent.set_position(nc * CELL, nr * CELL)
                if (nc, nr) in pads:
                    self._color = mix(self._color, pads[(nc, nr)])
                    self._sync()
                if (nc, nr) == exit_cell:
                    self.next_level()
        self.complete_action()

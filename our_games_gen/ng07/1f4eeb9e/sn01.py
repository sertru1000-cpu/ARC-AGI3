# Original game for the atlas testbed (30.08.2026) -- NOT an ARC Prize game.
# "Snake Trail" (sn01) -- rank-10 mechanic from the Gemini round-11 pool
# plan: self-modifying geometry (your own recent path is the obstacle).
#
# Rules: ACTION1-4 move. The agent leaves a trail over its last 3 cells;
# stepping onto your own trail is blocked (the trail fades as you move
# on). Reach the exit pad.

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
TRAIL_LEN = 3

WALL = 9
AGENT = 3
EXIT = 4
TRAIL = 6

# rows: '#'=wall, 'P'=agent, 'E'=exit.
LEVELS = [
    dict(rows=[
        "########",
        "#.##.###",
        "#.#..#.#",
        "#.P..#.#",
        "##....##",
        "#.#.##.#",
        "#E..##.#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#.#.E.##",
        "##...###",
        "##..####",
        "#......#",
        "##.#..##",
        "####.P##",
        "########",
    ]),
    dict(rows=[
        "########",
        "#.#..###",
        "##.#...#",
        "#....#P#",
        "#.##.###",
        "#E##.###",
        "#.#..###",
        "########",
    ]),
    dict(rows=[
        "########",
        "###..E.#",
        "#....###",
        "###..###",
        "##...###",
        "##.##.##",
        "#P.##.##",
        "########",
    ]),
    dict(rows=[
        "########",
        "########",
        "##.###.#",
        "##.....#",
        "##E##.##",
        "###...##",
        "#P....##",
        "########",
    ]),
]


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _trail_px():
    px = [[0] * CELL for _ in range(CELL)]
    for r in (1, 2):
        for c in (1, 2):
            px[r][c] = TRAIL
    return px


def _agent_px():
    px = [[0] * CELL for _ in range(CELL)]
    for r in (1, 2):
        for c in (1, 2):
            px[r][c] = AGENT
    px[0][1] = px[0][2] = AGENT
    return px


def _parse(spec):
    walls = set()
    start = exit_cell = None
    for r, row in enumerate(spec["rows"]):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "E":
                exit_cell = (c, r)
    return walls, start, exit_cell


def _build_level(index, spec):
    walls, start, exit_cell = _parse(spec)
    sprites = []
    for (c, r) in walls:
        sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=c * CELL, y=r * CELL,
                              blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
    sprites.append(Sprite(_rect(EXIT), name="exit", x=exit_cell[0] * CELL, y=exit_cell[1] * CELL,
                          layer=0, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    for i in range(TRAIL_LEN):
        sprites.append(Sprite(_trail_px(), name=f"trail_{i}", x=start[0] * CELL, y=start[1] * CELL,
                              layer=1, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_agent_px(), name="agent", x=start[0] * CELL, y=start[1] * CELL,
                          layer=3, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Sn01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="sn01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )
        self._trail: list = []
        self._init_trail()

    def _init_trail(self):
        _, start, _ = _parse(LEVELS[self.level_index])
        self._trail = [start] * TRAIL_LEN

    def on_set_level(self, level):
        self._init_trail()

    def _atlas_reset_level_state(self):
        self._init_trail()
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
        for i, (c, r) in enumerate(self._trail):
            s = self._sprite(f"trail_{i}")
            if s is not None:
                s.set_position(c * CELL, r * CELL)

    def step(self) -> None:
        spec = LEVELS[self.level_index]
        action = self.action.id
        if action == GameAction.RESET:
            self._sync()
            self.complete_action()
            return
        walls, start, exit_cell = _parse(spec)
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
            if ((nc, nr) not in walls and (nc, nr) not in self._trail
                    and 0 <= nc < GRID and 0 <= nr < GRID):
                self._trail = self._trail[1:] + [(c, r)]
                agent.set_position(nc * CELL, nr * CELL)
                self._sync()
                if (nc, nr) == exit_cell:
                    self.next_level()
        self.complete_action()

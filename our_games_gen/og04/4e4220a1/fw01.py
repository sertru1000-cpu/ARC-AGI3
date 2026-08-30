# Original game for the atlas testbed (30.08.2026) -- NOT an ARC Prize game.
# "Fog of War" (fw01) -- rank-11 mechanic from the Gemini round-11 pool
# plan: economical information gathering.
#
# Rules: the maze is covered by fog tiles. ACTION1-4 move (walking into
# fog is allowed but blind -- walls still block invisibly). ACTION6
# clicks a cell to lift the fog in a 3x3 area around it (walls, exit and
# floor become visible). Reach the exit pad. Baseline = the informed
# optimum (shortest path); the fog itself punishes wandering, not the
# perfect play -- exactly like the hidden layout of a real ARC game.

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
FOG = 5

# rows: '#'=wall, 'P'=agent, 'E'=exit.
LEVELS = [
    dict(rows=[
        "########",
        "##.....#",
        "##P##..#",
        "#.#.E..#",
        "#...#..#",
        "##.#.#.#",
        "#...#..#",
        "########",
    ]),
    dict(rows=[
        "########",
        "###.P.##",
        "#.##..##",
        "#....#.#",
        "#..#.###",
        "#......#",
        "#.#.#.E#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#.P...##",
        "####...#",
        "#..#...#",
        "#.##...#",
        "###..#.#",
        "#.E.##.#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#.#.#..#",
        "#.#...E#",
        "#...#..#",
        "#.#.####",
        "#..###.#",
        "##P..###",
        "########",
    ]),
    dict(rows=[
        "########",
        "##...###",
        "###....#",
        "#...##.#",
        "#..###.#",
        "##E#.P.#",
        "#...##.#",
        "########",
    ]),
]


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


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
    # fog covers everything except the border ring and the start cell
    for r in range(1, GRID - 1):
        for c in range(1, GRID - 1):
            if (c, r) != start:
                sprites.append(Sprite(_rect(FOG), name=f"fog_{r}_{c}", x=c * CELL, y=r * CELL,
                                      layer=5, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_agent_px(), name="agent", x=start[0] * CELL, y=start[1] * CELL,
                          layer=6, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Fw01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="fw01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4, 6],
            seed=seed,
        )

    def _atlas_reset_level_state(self):
        pass  # fog sprites restored by the engine's clean-level reset

    def level_reset(self):
        super().level_reset()
        self._atlas_reset_level_state()

    def full_reset(self):
        super().full_reset()
        self._atlas_reset_level_state()

    def _sprite(self, name):
        found = self.current_level.get_sprites_by_name(name)
        return found[0] if found else None

    def _reveal(self, cx, cy):
        for r in range(cy - 1, cy + 2):
            for c in range(cx - 1, cx + 2):
                s = self._sprite(f"fog_{r}_{c}")
                if s is not None:
                    self.current_level.remove_sprite(s)

    def step(self) -> None:
        spec = LEVELS[self.level_index]
        action = self.action.id
        if action == GameAction.RESET:
            self.complete_action()
            return
        walls, start, exit_cell = _parse(spec)
        agent = self._sprite("agent")
        if agent is None:
            self.complete_action()
            return
        c, r = agent.x // CELL, agent.y // CELL

        if action == GameAction.ACTION6:
            x = int(self.action.data.get("x", 0)) // 8
            y = int(self.action.data.get("y", 0)) // 8
            self._reveal(x, y)
        elif action in (GameAction.ACTION1, GameAction.ACTION2,
                        GameAction.ACTION3, GameAction.ACTION4):
            d = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1),
                 GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}[action]
            nc, nr = c + d[0], r + d[1]
            if (nc, nr) not in walls and 0 <= nc < GRID and 0 <= nr < GRID:
                agent.set_position(nc * CELL, nr * CELL)
                # walking clears the fog on the cell you stand on
                s = self._sprite(f"fog_{nr}_{nc}")
                if s is not None:
                    self.current_level.remove_sprite(s)
                if (nc, nr) == exit_cell:
                    self.next_level()
        self.complete_action()

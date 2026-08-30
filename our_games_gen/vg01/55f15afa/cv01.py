# Original game for the atlas testbed (30.08.2026) -- NOT an ARC Prize game.
# "Conveyors" (cv01) -- rank-7 mechanic from the Gemini round-11 pool plan:
# one-way directed edges (external force vectors, no backtracking).
#
# Rules: ACTION1-4 move. Stepping onto an arrow tile forcibly slides the
# agent one cell in the arrow's direction, chaining across consecutive
# arrows. Arrows only work one way. Reach the exit pad.

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
ARROW = 11

ARROWS = {"^": (0, -1), "v": (0, 1), "<": (-1, 0), ">": (1, 0)}

# rows: '#'=wall, 'P'=agent, 'E'=exit, '^v<>'=conveyor arrows.
LEVELS = [
    dict(rows=[
        "########",
        "#......#",
        "#...<..#",
        "###....#",
        "#....P>#",
        "#...#<.#",
        "##E....#",
        "########",
    ]),
    dict(rows=[
        "########",
        "##...^.#",
        "#..E...#",
        "#.#....#",
        "##.##.v#",
        "#P>.#..#",
        "#.>....#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#.#.>..#",
        "#.<.#E.#",
        "#....#>#",
        "##...#.#",
        "#...P#<#",
        "#^.....#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#..>..E#",
        "#.##...#",
        "#..##.^#",
        "#..##..#",
        "##.vv^.#",
        "#.P.v#.#",
        "########",
    ]),
    dict(rows=[
        "########",
        "##v..vE#",
        "#.#.#..#",
        "#^#.<v##",
        "##^...P#",
        "#...#.^#",
        "#...#..#",
        "########",
    ]),
]


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _arrow_px(d):
    px = [[0] * CELL for _ in range(CELL)]
    dx, dy = d
    if dy == -1:
        px[0][1] = px[0][2] = ARROW
        px[1][0] = px[1][3] = ARROW
        px[2][1] = px[2][2] = ARROW
    elif dy == 1:
        px[3][1] = px[3][2] = ARROW
        px[2][0] = px[2][3] = ARROW
        px[1][1] = px[1][2] = ARROW
    elif dx == -1:
        px[1][0] = px[2][0] = ARROW
        px[0][1] = px[3][1] = ARROW
        px[1][2] = px[2][2] = ARROW
    else:
        px[1][3] = px[2][3] = ARROW
        px[0][2] = px[3][2] = ARROW
        px[1][1] = px[2][1] = ARROW
    return px


def _agent_px():
    px = [[0] * CELL for _ in range(CELL)]
    for r in (1, 2):
        for c in (1, 2):
            px[r][c] = AGENT
    px[0][1] = px[0][2] = AGENT
    return px


def _parse(spec):
    walls, arrows = set(), {}
    start = exit_cell = None
    for r, row in enumerate(spec["rows"]):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "E":
                exit_cell = (c, r)
            elif ch in ARROWS:
                arrows[(c, r)] = ARROWS[ch]
    return walls, arrows, start, exit_cell


def slide(pos, walls, arrows):
    """Chain conveyor pushes (max 8 to stay finite on generated loops)."""
    c, r = pos
    for _ in range(8):
        if (c, r) not in arrows:
            break
        dx, dy = arrows[(c, r)]
        nc, nr = c + dx, r + dy
        if (nc, nr) in walls or not (0 <= nc < GRID and 0 <= nr < GRID):
            break
        c, r = nc, nr
    return (c, r)


def _build_level(index, spec):
    walls, arrows, start, exit_cell = _parse(spec)
    sprites = []
    for (c, r) in walls:
        sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=c * CELL, y=r * CELL,
                              blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
    for (c, r), d in arrows.items():
        sprites.append(Sprite(_arrow_px(d), name=f"arrow_{r}_{c}", x=c * CELL, y=r * CELL,
                              layer=1, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_rect(EXIT), name="exit", x=exit_cell[0] * CELL, y=exit_cell[1] * CELL,
                          layer=0, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_agent_px(), name="agent", x=start[0] * CELL, y=start[1] * CELL,
                          layer=3, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Cv01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="cv01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )

    def _atlas_reset_level_state(self):
        pass

    def level_reset(self):
        super().level_reset()
        self._atlas_reset_level_state()

    def full_reset(self):
        super().full_reset()
        self._atlas_reset_level_state()

    def _sprite(self, name):
        found = self.current_level.get_sprites_by_name(name)
        return found[0] if found else None

    def step(self) -> None:
        spec = LEVELS[self.level_index]
        action = self.action.id
        if action == GameAction.RESET:
            self.complete_action()
            return
        walls, arrows, start, exit_cell = _parse(spec)
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
            if (nc, nr) not in walls and 0 <= nc < GRID and 0 <= nr < GRID:
                nc, nr = slide((nc, nr), walls, arrows)
                agent.set_position(nc * CELL, nr * CELL)
                if (nc, nr) == exit_cell:
                    self.next_level()
        self.complete_action()

# Original game for the atlas testbed (30.08.2026) -- NOT an ARC Prize game.
# "Portals" (pt01) -- rank-2 mechanic from the Gemini round-11 pool plan:
# non-Euclidean space, breaking the Manhattan-distance prior of h(s).
#
# Rules: ACTION1-4 move the agent. Stepping onto a portal cell instantly
# relocates the agent to the SAME-COLOR twin portal elsewhere on the map.
# Reach the exit pad to complete the level. Portals are two-way and
# reusable; later levels add decoy pairs that lead away from the exit.

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
PORTAL_COLORS = {"1": 6, "2": 8, "3": 11}

# rows: '#'=wall, 'P'=agent, 'E'=exit, digits '1'..'3' = portal pairs
# (each digit appears exactly twice).
LEVELS = [
    dict(rows=[
        "########",
        "#E...1.#",
        "#.....P#",
        "#.#1...#",
        "#....#.#",
        "#..##..#",
        "#......#",
        "########",
    ]),
    dict(rows=[
        "########",
        "##.....#",
        "#.P#.###",
        "#.#....#",
        "#...1..#",
        "#..#.E.#",
        "#...1###",
        "########",
    ]),
    dict(rows=[
        "########",
        "#E.#...#",
        "##2.#..#",
        "##...#.#",
        "#.#1..1#",
        "#.#P...#",
        "#.2#...#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#...#.##",
        "#.....E#",
        "##.#..##",
        "#.#....#",
        "#.#1.###",
        "#21#.2P#",
        "########",
    ]),
    dict(rows=[
        "########",
        "##P.##1#",
        "#.#13..#",
        "##...###",
        "#E.2#..#",
        "#..3..##",
        "#.2..#.#",
        "########",
    ]),
]


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _portal_px(color):
    px = [[0] * CELL for _ in range(CELL)]
    for r in range(CELL):
        for c in range(CELL):
            if (r in (0, CELL - 1) or c in (0, CELL - 1)) and (r + c) % 2 == 0:
                px[r][c] = color
    px[1][1] = px[2][2] = px[1][2] = px[2][1] = color
    return px


def _agent_px():
    px = [[0] * CELL for _ in range(CELL)]
    for r in (1, 2):
        for c in (1, 2):
            px[r][c] = AGENT
    px[0][1] = px[0][2] = AGENT
    return px


def _parse(spec):
    walls, portals = set(), {}
    start = exit_cell = None
    for r, row in enumerate(spec["rows"]):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "E":
                exit_cell = (c, r)
            elif ch in PORTAL_COLORS:
                portals.setdefault(ch, []).append((c, r))
    return walls, portals, start, exit_cell


def _build_level(index, spec):
    walls, portals, start, exit_cell = _parse(spec)
    sprites = []
    for (c, r) in walls:
        sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=c * CELL, y=r * CELL,
                              blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
    sprites.append(Sprite(_rect(EXIT), name="exit", x=exit_cell[0] * CELL, y=exit_cell[1] * CELL,
                          layer=0, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    for ch, cells in sorted(portals.items()):
        for j, (c, r) in enumerate(sorted(cells)):
            sprites.append(Sprite(_portal_px(PORTAL_COLORS[ch]), name=f"portal_{ch}_{j}",
                                  x=c * CELL, y=r * CELL, layer=1,
                                  blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_agent_px(), name="agent", x=start[0] * CELL, y=start[1] * CELL,
                          layer=3, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Pt01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="pt01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )

    def _atlas_reset_level_state(self):
        pass  # stateless beyond sprites; hook kept for harness parity

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
        walls, portals, start, exit_cell = _parse(spec)
        agent = self._sprite("agent")
        if agent is None:
            self.complete_action()
            return
        ac, ar = agent.x // CELL, agent.y // CELL

        if action in (GameAction.ACTION1, GameAction.ACTION2,
                      GameAction.ACTION3, GameAction.ACTION4):
            d = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1),
                 GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}[action]
            nc, nr = ac + d[0], ar + d[1]
            if (nc, nr) not in walls and 0 <= nc < GRID and 0 <= nr < GRID:
                # portal hop: landing on a portal relocates to its twin
                for ch, cells in portals.items():
                    if (nc, nr) in cells and len(cells) == 2:
                        twin = cells[0] if cells[1] == (nc, nr) else cells[1]
                        nc, nr = twin
                        break
                agent.set_position(nc * CELL, nr * CELL)
                if (nc, nr) == exit_cell:
                    self.next_level()
        self.complete_action()

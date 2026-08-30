# Original game for the atlas testbed (30.08.2026) -- NOT an ARC Prize game.
# "One-Way Trap" (tr01) -- HOSTILE-POOL game 3 (Gemini round 13 Q4):
# irreversible dead subtree WITHOUT immediate game_over. A tempting short
# route passes through a one-way gate ('>'); crossing it walls the gate
# shut behind the agent (visible) and the exit is unreachable from inside
# -- but the game keeps accepting moves, so nothing tells the player the
# level is lost. Only RESET recovers. The honest route is longer. Search
# hostility: BFS/A* pour node budget into the large, attractive, dead
# region; greedy distance-heuristics actively prefer it.

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
GATE = 6

# rows: '#'=wall, 'P'=agent, 'E'=exit, '>'=one-way gate (enter -> seals).
# The gate area leads to a dead pocket NEAR the exit (tempting geometry).
LEVELS = [
    dict(rows=[
        "########",
        "#P.>...#",
        "#.##...#",
        "#.##...#",
        "#.######",
        "#......#",
        "#.....E#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#P>....#",
        "#.#....#",
        "#.#....#",
        "#.######",
        "#......#",
        "######E#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#P.>...#",
        "#..#...#",
        "#..#...#",
        "#..#####",
        "#......#",
        "#####.E#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#..>...#",
        "#P.#.#.#",
        "#..#...#",
        "#..#.#.#",
        "#..#####",
        "#.....E#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#..>...#",
        "#P.#...#",
        "#..#...#",
        "#..#...#",
        "#..#...#",
        "#.E#####",
        "########",
    ]),
]

# the two string-hack normalizations below are gone with the redesign

def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _agent_px():
    px = [[0] * CELL for _ in range(CELL)]
    for r in (1, 2):
        for c in (1, 2):
            px[r][c] = AGENT
    px[0][1] = px[0][2] = AGENT
    return px


def _gate_px():
    return [[GATE if c >= CELL // 2 else 0 for c in range(CELL)] for _ in range(CELL)]


def _parse(spec):
    walls, gates = set(), set()
    start = exit_cell = None
    for r, row in enumerate(spec["rows"]):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "E":
                exit_cell = (c, r)
            elif ch == ">":
                gates.add((c, r))
    return walls, gates, start, exit_cell


def _build_level(index, spec):
    walls, gates, start, exit_cell = _parse(spec)
    sprites = []
    for (c, r) in walls:
        sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=c * CELL, y=r * CELL,
                              blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
    for (c, r) in gates:
        sprites.append(Sprite(_gate_px(), name=f"gate_{r}_{c}", x=c * CELL, y=r * CELL,
                              layer=1, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_rect(EXIT), name="exit", x=exit_cell[0] * CELL, y=exit_cell[1] * CELL,
                          layer=0, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_agent_px(), name="agent", x=start[0] * CELL, y=start[1] * CELL,
                          layer=3, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Tr01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="tr01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )
        self._sealed = set()

    def on_set_level(self, level):
        self._sealed = set()

    def _atlas_reset_level_state(self):
        self._sealed = set()
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
        spec = LEVELS[self.level_index]
        walls, gates, start, exit_cell = _parse(spec)
        for (c, r) in gates:
            s = self._sprite(f"gate_{r}_{c}")
            if s is None:
                continue
            px = _rect(WALL) if (c, r) in self._sealed else _gate_px()
            for rr in range(CELL):
                for cc in range(CELL):
                    s.pixels[rr][cc] = px[rr][cc]

    def step(self) -> None:
        spec = LEVELS[self.level_index]
        action = self.action.id
        if action == GameAction.RESET:
            self._sync()
            self.complete_action()
            return
        walls, gates, start, exit_cell = _parse(spec)
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
            blocked = (
                (nc, nr) in walls
                or (nc, nr) in self._sealed
                or not (0 <= nc < GRID and 0 <= nr < GRID)
            )
            if not blocked:
                agent.set_position(nc * CELL, nr * CELL)
                if (nc, nr) in gates and (nc, nr) not in self._sealed:
                    # crossing the one-way gate seals it behind the agent
                    self._sealed.add((nc, nr))
                    self._sync()
                if (nc, nr) == exit_cell:
                    self.next_level()
        self.complete_action()

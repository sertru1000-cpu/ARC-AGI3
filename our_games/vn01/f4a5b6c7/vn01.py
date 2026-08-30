# Original game for the atlas testbed (30.08.2026) -- NOT an ARC Prize game.
# "Visual Noise" (vn01) -- HOSTILE-POOL game 4 (Gemini round 13 Q4): a
# plain maze walk whose empty cells carry a decorative pattern that shifts
# DETERMINISTICALLY every action (phase = action count). Passability is
# untouched -- the noise is pure render. Hostility: every frame is unique,
# so frame-signature dedup dies (search frontier explodes), the entropy
# cull never sees repeats, and the noop-guard's board-sig keys never match.

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
NOISE_COLORS = [5, 7, 12, 13]

# rows: '#'=wall, 'P'=agent, 'E'=exit, '.'=floor (noise-decorated).
LEVELS = [
    dict(rows=[
        "########",
        "#P.....#",
        "#.####.#",
        "#......#",
        "#.##.###",
        "#..#...#",
        "##...E.#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#..#..E#",
        "#..#.###",
        "#P.#...#",
        "#..##..#",
        "#......#",
        "#..#...#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#E...#.#",
        "###..#.#",
        "#....#.#",
        "#.####.#",
        "#......#",
        "#####.P#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#...#..#",
        "#.P.#.E#",
        "#...#..#",
        "#.#.#.##",
        "#.#...##",
        "#.#####.".replace(".", "#", 0)[:8],
        "########",
    ]),
    dict(rows=[
        "########",
        "#P..#..#",
        "##..#..#",
        "#..##.##",
        "#..#...#",
        "#.##.#.#",
        "#....#E#",
        "########",
    ]),
]
LEVELS[3]["rows"] = [
    "########",
    "#...#..#",
    "#.P.#.E#",
    "#...#..#",
    "#.#.#.##",
    "#.#...##",
    "#.######",
    "########",
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


def _noise_px(c, r, t):
    color = NOISE_COLORS[(t + 3 * c + 5 * r) % len(NOISE_COLORS)]
    px = [[0] * CELL for _ in range(CELL)]
    px[(t + r) % CELL][(t + c) % CELL] = color
    px[(t + c) % CELL][(t + r + 2) % CELL] = color
    return px


def _parse(spec):
    walls = set()
    floors = []
    start = exit_cell = None
    for r, row in enumerate(spec["rows"]):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            else:
                if ch == "P":
                    start = (c, r)
                elif ch == "E":
                    exit_cell = (c, r)
                floors.append((c, r))
    return walls, floors, start, exit_cell


def _build_level(index, spec):
    walls, floors, start, exit_cell = _parse(spec)
    sprites = []
    for (c, r) in walls:
        sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=c * CELL, y=r * CELL,
                              blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
    for (c, r) in floors:
        if (c, r) == exit_cell:
            continue
        sprites.append(Sprite(_noise_px(c, r, 0), name=f"noise_{r}_{c}", x=c * CELL, y=r * CELL,
                              layer=0, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_rect(EXIT), name="exit", x=exit_cell[0] * CELL, y=exit_cell[1] * CELL,
                          layer=0, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_agent_px(), name="agent", x=start[0] * CELL, y=start[1] * CELL,
                          layer=3, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Vn01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="vn01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )
        self._t = 0

    def on_set_level(self, level):
        self._t = 0

    def _atlas_reset_level_state(self):
        self._t = 0
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
        walls, floors, start, exit_cell = _parse(spec)
        for (c, r) in floors:
            if (c, r) == exit_cell:
                continue
            s = self._sprite(f"noise_{r}_{c}")
            if s is None:
                continue
            px = _noise_px(c, r, self._t)
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
        walls, floors, start, exit_cell = _parse(spec)
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
                agent.set_position(nc * CELL, nr * CELL)
                if (nc, nr) == exit_cell:
                    self.next_level()
                    self.complete_action()
                    return
            self._t += 1
            self._sync()
        self.complete_action()

# Original game for the atlas testbed (30.08.2026) -- NOT an ARC Prize game.
# "Chaser" (ch01) -- HOSTILE-POOL game 2 (Gemini round 13 Q4): a pseudo-
# stochastic enemy. Every agent action the chaser takes one step chosen by
# a hash of (t, chaser_pos, agent_pos): biased toward the agent but with
# hash-jitter -- DETERMINISTIC given the action sequence (so baselines and
# replays stay exact), yet unpredictable-looking for the model and state-
# explosive for search (the chaser position multiplies the frontier).
# Touching the chaser = lose() (game over).

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
CHASER = 2

# rows: '#'=wall, 'P'=agent, 'E'=exit, 'C'=chaser start.
LEVELS = [
    dict(rows=[
        "########",
        "#C.....#",
        "#......#",
        "#......#",
        "#......#",
        "#......#",
        "#P....E#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#P...#.#",
        "#..#...#",
        "#..#.C.#",
        "#..#...#",
        "#..#...#",
        "#....E.#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#E.#...#",
        "#..#.C.#",
        "#..#...#",
        "#......#",
        "#...#..#",
        "#P..#..#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#P.....#",
        "#.####.#",
        "#.#C...#",
        "#.#.##.#",
        "#.#..#.#",
        "#....#E#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#..C..P#",
        "#.####.#",
        "#......#",
        "#.####.#",
        "#......#",
        "#E.####.".replace(".", "#", 0)[:8],
        "########",
    ]),
]
# fix level 5 last playable row properly (avoid clever-string accidents)
LEVELS[4]["rows"] = [
    "########",
    "#..C..P#",
    "#.####.#",
    "#......#",
    "#.####.#",
    "#......#",
    "#E...#.#",
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


def _chaser_px():
    px = [[0] * CELL for _ in range(CELL)]
    for r in range(CELL):
        px[r][r] = CHASER
        px[r][CELL - 1 - r] = CHASER
    return px


def _parse(spec):
    walls = set()
    start = exit_cell = chaser = None
    for r, row in enumerate(spec["rows"]):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "E":
                exit_cell = (c, r)
            elif ch == "C":
                chaser = (c, r)
    return walls, start, exit_cell, chaser


def _mix(*vals) -> int:
    h = 0x9E3779B9
    for v in vals:
        h ^= (v + 0x7F4A7C15) & 0xFFFFFFFF
        h = (h * 0x85EBCA6B) & 0xFFFFFFFF
        h ^= h >> 13
    return h & 0xFFFFFFFF


def chaser_step(t, chaser, agent, walls):
    """Deterministic pseudo-random pursuit: 3/4 of hash outcomes step along
    the axis with the larger gap toward the agent, 1/4 jitter sideways."""
    cx, cy = chaser
    ax, ay = agent
    dx = 0 if ax == cx else (1 if ax > cx else -1)
    dy = 0 if ay == cy else (1 if ay > cy else -1)
    h = _mix(t, cx, cy, ax, ay)
    prefer_x = abs(ax - cx) >= abs(ay - cy)
    roll = h & 3
    if roll == 3:  # jitter: try the OTHER axis (or stay if that is zero)
        cand = [(cx, cy + dy), (cx + dx, cy)] if prefer_x else [(cx + dx, cy), (cx, cy + dy)]
    else:
        cand = [(cx + dx, cy), (cx, cy + dy)] if prefer_x else [(cx, cy + dy), (cx + dx, cy)]
    for nx, ny in cand:
        if (nx, ny) != (cx, cy) and (nx, ny) not in walls and 0 <= nx < GRID and 0 <= ny < GRID:
            return nx, ny
    return cx, cy


def _build_level(index, spec):
    walls, start, exit_cell, chaser = _parse(spec)
    sprites = []
    for (c, r) in walls:
        sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=c * CELL, y=r * CELL,
                              blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
    sprites.append(Sprite(_rect(EXIT), name="exit", x=exit_cell[0] * CELL, y=exit_cell[1] * CELL,
                          layer=0, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_chaser_px(), name="chaser", x=chaser[0] * CELL, y=chaser[1] * CELL,
                          layer=2, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_agent_px(), name="agent", x=start[0] * CELL, y=start[1] * CELL,
                          layer=3, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Ch01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="ch01",
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
        walls, start, exit_cell, chaser_start = _parse(spec)
        agent = self._sprite("agent")
        chaser = self._sprite("chaser")
        if agent is None or chaser is None:
            self.complete_action()
            return
        c, r = agent.x // CELL, agent.y // CELL
        kc, kr = chaser.x // CELL, chaser.y // CELL

        if action in (GameAction.ACTION1, GameAction.ACTION2,
                      GameAction.ACTION3, GameAction.ACTION4):
            d = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1),
                 GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}[action]
            nc, nr = c + d[0], r + d[1]
            if (nc, nr) not in walls and 0 <= nc < GRID and 0 <= nr < GRID:
                c, r = nc, nr
                agent.set_position(c * CELL, r * CELL)
            if (c, r) == exit_cell:
                self.next_level()
                self.complete_action()
                return
            kc, kr = chaser_step(self._t, (kc, kr), (c, r), walls)
            chaser.set_position(kc * CELL, kr * CELL)
            self._t += 1
            if (kc, kr) == (c, r):
                self.lose()
        self.complete_action()

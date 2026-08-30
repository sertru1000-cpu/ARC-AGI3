# Original game for the atlas testbed (30.08.2026) -- NOT an ARC Prize game.
# "Wildfire" (wf01) -- rank-9 mechanic from the Gemini round-11 pool plan:
# deterministic cellular automaton, spatio-temporal planning under a
# shrinking free space.
#
# Rules: ACTION1-4 move, ACTION5 waits. Fire starts at the marked source
# cells and spreads one ring every 2 actions (deterministic). Stepping
# into fire (or fire reaching you) is GAME OVER. The exit pad is
# fireproof. Occupancy is a pure function of the action count -- a
# careful world-modeler can outrun it, a dawdler burns.

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
SPREAD_EVERY = 2

WALL = 9
AGENT = 3
EXIT = 4
FIRE = 2

# rows: '#'=wall, 'P'=agent, 'E'=exit (fireproof), 'F'=fire source.
LEVELS = [
    dict(rows=[
        "########",
        "#......#",
        "#...E.##",
        "#......#",
        "##..#..#",
        "#..#...#",
        "#FP....#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#..F#..#",
        "#....P##",
        "#..#...#",
        "#.##...#",
        "#...#..#",
        "#..E...#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#...#..#",
        "#..#...#",
        "###....#",
        "#P..#..#",
        "#..#...#",
        "#F...E.#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#..P..F#",
        "#.##...#",
        "###....#",
        "###...##",
        "#...#..#",
        "#...E..#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#.#..E.#",
        "#F.##..#",
        "#.####.#",
        "#P.....#",
        "#.....##",
        "#......#",
        "########",
    ]),
]


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _fire_px():
    px = [[FIRE] * CELL for _ in range(CELL)]
    px[0][1] = px[2][3] = px[3][0] = 0
    return px


def _agent_px():
    px = [[0] * CELL for _ in range(CELL)]
    for r in (1, 2):
        for c in (1, 2):
            px[r][c] = AGENT
    px[0][1] = px[0][2] = AGENT
    return px


def _parse(spec):
    walls, sources = set(), set()
    start = exit_cell = None
    for r, row in enumerate(spec["rows"]):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "F":
                sources.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "E":
                exit_cell = (c, r)
    return walls, sources, start, exit_cell


def burn_dist(spec):
    """BFS distance of every floor cell from the fire sources (walls and
    the fireproof exit block the spread)."""
    from collections import deque
    walls, sources, start, exit_cell = _parse(spec)
    dist = {s: 0 for s in sources}
    q = deque(sources)
    while q:
        c, r = q.popleft()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nb = (c + dx, r + dy)
            if nb in walls or nb in dist or nb == exit_cell:
                continue
            if not (0 <= nb[0] < GRID and 0 <= nb[1] < GRID):
                continue
            dist[nb] = dist[(c, r)] + 1
            q.append(nb)
    return dist


def fire_mask(spec, t):
    return {cell for cell, d in burn_dist(spec).items() if d * SPREAD_EVERY <= t}


def _build_level(index, spec):
    walls, sources, start, exit_cell = _parse(spec)
    sprites = []
    for (c, r) in walls:
        sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=c * CELL, y=r * CELL,
                              blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
    sprites.append(Sprite(_rect(EXIT), name="exit", x=exit_cell[0] * CELL, y=exit_cell[1] * CELL,
                          layer=0, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    for i, (c, r) in enumerate(sorted(sources)):
        sprites.append(Sprite(_fire_px(), name=f"fire_{i}", x=c * CELL, y=r * CELL,
                              layer=2, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_agent_px(), name="agent", x=start[0] * CELL, y=start[1] * CELL,
                          layer=3, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Wf01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="wf01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )
        self._t = 0
        self._fire_sprites = 0

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
        mask = sorted(fire_mask(spec, self._t))
        for i in range(self._fire_sprites):
            s = self._sprite(f"flame_{i}")
            if s is not None:
                self.current_level.remove_sprite(s)
        walls, sources, start, exit_cell = _parse(spec)
        extra = [cell for cell in mask if cell not in sources]
        for i, (c, r) in enumerate(extra):
            self.current_level.add_sprite(
                Sprite(_fire_px(), name=f"flame_{i}", x=c * CELL, y=r * CELL, layer=2,
                       blocking=BlockingMode.NOT_BLOCKED, collidable=False))
        self._fire_sprites = len(extra)
        return set(mask)

    def step(self) -> None:
        spec = LEVELS[self.level_index]
        action = self.action.id
        if action == GameAction.RESET:
            self._sync()
            self.complete_action()
            return
        walls, sources, start, exit_cell = _parse(spec)
        agent = self._sprite("agent")
        if agent is None:
            self.complete_action()
            return
        c, r = agent.x // CELL, agent.y // CELL

        moved = False
        if action == GameAction.ACTION5:
            moved = True
        elif action in (GameAction.ACTION1, GameAction.ACTION2,
                        GameAction.ACTION3, GameAction.ACTION4):
            d = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1),
                 GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}[action]
            nc, nr = c + d[0], r + d[1]
            if (nc, nr) not in walls and 0 <= nc < GRID and 0 <= nr < GRID:
                c, r = nc, nr
                agent.set_position(c * CELL, r * CELL)
            moved = True
        if not moved:
            self.complete_action()
            return
        if (c, r) == exit_cell:
            self.next_level()
            self.complete_action()
            return
        self._t += 1
        mask = self._sync()
        if (c, r) in mask:
            self.lose()
        self.complete_action()

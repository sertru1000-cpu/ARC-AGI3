# Original game for the atlas testbed (30.08.2026) -- NOT an ARC Prize game.
# "Gravity" (gv01) -- rank-3 mechanic from the Gemini round-11 pool plan:
# asymmetric graph (falling is one step, climbing back is many or
# impossible).
#
# Rules: ACTION3/4 move sideways, ACTION1 climbs UP only on a ladder,
# ACTION2 steps down. After EVERY action the agent falls until supported
# (standing on a wall/platform or holding a ladder). Reach the exit pad.

from arcengine import (
    ARCBaseGame,
    BlockingMode,
    Camera,
    GameAction,
    Level,
    Sprite,
)

CELL = 3
GRID = 21


# --- pixel helpers, added 03.09 by scripts/cellify_sprites.py ---------------
# These replace literal pixel indices so the sprite renders at any CELL. The
# shapes are exactly what the literals drew at CELL == 4, which
# scripts/frame_fingerprint.py verifies by hashing every frame before and
# after the rewrite.

def _fill_inner(px, color):
    """Everything except the one-pixel border."""
    for _r in range(1, CELL - 1):
        for _c in range(1, CELL - 1):
            px[_r][_c] = color
    return px


def _fill_edge(px, side, color):
    """The middle stretch of one edge -- the part that is not a corner."""
    span = range(1, CELL - 1) if CELL > 2 else range(CELL)
    for _i in span:
        if side == "top":
            px[0][_i] = color
        elif side == "bottom":
            px[CELL - 1][_i] = color
        elif side == "left":
            px[_i][0] = color
        else:
            px[_i][CELL - 1] = color
    return px


def _fill_row(px, col, color):
    """The middle rows of one column."""
    for _r in range(1, CELL - 1):
        px[_r][col] = color
    return px

WALL = 9
AGENT = 3
EXIT = 4
LADDER = 11

# rows: '#'=wall/platform, 'H'=ladder, 'P'=agent, 'E'=exit.
LEVELS = [
    # 03.09: built by scripts/plant_levels.py -- each level is derived from
    # a PLANTED solution on the full 64x64 board (GRID 21, CELL 3),
    # along a real public game's difficulty curve. A random layout could
    # not carry this mechanic (measured, scripts/probe_regen_fitness.py).
    # Every baseline below is the optimum found by BFS through the engine.
    dict(rows=[
        '#####################',
        '#...#...............#',
        '###.###.#############',
        '#...#...#...........#',
        '#.###.###.#####.###.#',
        '#.#.......#...#.#...#',
        '#.#####.###.#.###.#.#',
        '#.....#...#.#.....#.#',
        '#####.#####.#######.#',
        '#...#.....#.....#...#',
        '#.#######.#.###.#####',
        '#H...HP.#.#...#.....#',
        '#H###.###.#.#######.#',
        '#..H#.....#.#H...H#.#',
        '###H#######.#H###H#.#',
        '#.#H#.......#H#H..#.#',
        '#.#H#########H#H###.#',
        '#.#......H#H.H#H#...#',
        '#.#######H#H###H#.#.#',
        '#..........H#..E..#.#',
        '#####################',
    ]),
    dict(rows=[
        '#####################',
        '#.....#.........#...#',
        '#####.###.#####.###.#',
        '#...#.#...#...#.....#',
        '#.#.#.#.###.#.#####.#',
        '#.#...#...#.#.....#.#',
        '#.#####.#.#.#####.###',
        '#.#.....#.#.#...#...#',
        '#.#.#####.#.###.###.#',
        '#.#.#.....#...#.#...#',
        '#.#.#.#######.#.#.###',
        '#.#.#...........#...#',
        '#.#####.###########.#',
        '#.....#...#.....#...#',
        '#####.###.#.###.#.#.#',
        '#...#.#...#.EH#...#.#',
        '#.###.#######H#####.#',
        '#...#.......#H.H#...#',
        '#.#.#######.###H#.###',
        '#.#.......P....H#...#',
        '#####################',
    ]),
    dict(rows=[
        '#####################',
        '#.....#.....#.......#',
        '#####.#####.#.#.###.#',
        '#...#.....#...#...#.#',
        '#.#######.#######.###',
        '#.......#.......#...#',
        '#######.#######.#.#.#',
        '#H.H......#.....#.#.#',
        '#H#H#####.#.#######.#',
        '#H#H#...#.#.........#',
        '#H#H#.#.###########.#',
        '#H#H#.#...#.....#...#',
        '#H#H#.###.#.###.#.###',
        '#H#H#...#.#.#.#...#.#',
        '#H#H###.#.#.#.#####.#',
        '#H#E#...#.#.#.......#',
        '#H###.###.#.#.#####.#',
        '#H....#.#.#.#.#...#.#',
        '#H#####.#.#.#.#.#.#.#',
        '#HP.....#.....#.#...#',
        '#####################',
    ]),
    dict(rows=[
        '#####################',
        '#...#.#.....#.......#',
        '###.#.#.###.#.#.#####',
        '#...#.#...#...#.....#',
        '#.###.###.#########.#',
        '#...#...#...#...#.#.#',
        '###.#.#####.#.#.#.#.#',
        '#.#.#.........#...#.#',
        '#.#.#.###########.#.#',
        '#.#.#...#H.....H#.#.#',
        '#.#.#####H#####H###.#',
        '#.#.P..H#H#H....#...#',
        '#.#####H#H#H#####.#.#',
        '#.#...#..H#H#.....#.#',
        '#.#.#.#####H#.#####.#',
        '#.#.#...#.E.#.#.#...#',
        '#.#.#.#.#.###.#.#.###',
        '#.#.#.#.#.....#.#...#',
        '#.#.#.#########.###.#',
        '#...#...............#',
        '#####################',
    ]),
    dict(rows=[
        '#####################',
        '#.#......H.H#.......#',
        '#.#.#####H#H#.#####.#',
        '#.#.#.#H.H#H#.#...#.#',
        '#.#.#.#H###H#.###.#.#',
        '#.#...#HP.#H#...#...#',
        '#.###.###.#H###.###.#',
        '#.#.....#.#..H#...#.#',
        '#.#####.#.###H###.#.#',
        '#.....#.#...#..H#.#.#',
        '#####.#####.###H#.###',
        '#H.H#.......#H..#...#',
        '#H#H#########H###.#.#',
        '#H#H...H..#.#H#...#.#',
        '#H#####H#.#.#H#####.#',
        '#..H#.#H#...#..H#...#',
        '###H#.#H###.###H#.###',
        '#..E#.#H#.....#H#...#',
        '#.###.#H#######H###.#',
        '#.....#H............#',
        '#####################',
    ]),
    dict(rows=[
        '#####################',
        '#.#.....#.......#...#',
        '#.#.###.###.###.#.#.#',
        '#...#.#...#.#.#...#.#',
        '#####.###.#.#.#####.#',
        '#.......#.#.......#.#',
        '#.#######.###.#####.#',
        '#.#H...H#...#.#.....#',
        '#.#H###H###.#.#.#####',
        '#E..#H.H#...#.#.....#',
        '#####H###.###.#####.#',
        '#H...H#...#.....#.#.#',
        '#H#####.#######.#.#.#',
        '#H#H..P.#.....#...#.#',
        '#H#H#####.###.#####.#',
        '#H#H#.....#.#.....#.#',
        '#H#H#####.#.#####.#.#',
        '#H#H#...#.#.....#...#',
        '#H#H#.#.#.#.#######.#',
        '#H....#...#.........#',
        '#####################',
    ]),
]


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _ladder_px():
    px = [[0] * CELL for _ in range(CELL)]
    for r in range(CELL):
        px[r][0] = px[r][CELL - 1] = LADDER
    # две перекладины: середина ряда 1 и середина последнего ряда.
    # Написано явно, а не через _fill_edge: исходник был ОДНОЙ
    # четырёхзвенной цепочкой, и правило, подходящее только к её
    # хвосту, оставляло голову присваивающей список.
    for _c in range(1, CELL - 1):
        px[1][_c] = px[CELL - 1][_c] = LADDER
    return px


def _agent_px():
    px = [[0] * CELL for _ in range(CELL)]
    _fill_inner(px, AGENT)
    _fill_edge(px, 'top', AGENT)
    return px


def _parse(spec):
    walls, ladders = set(), set()
    start = exit_cell = None
    for r, row in enumerate(spec["rows"]):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "H":
                ladders.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "E":
                exit_cell = (c, r)
    return walls, ladders, start, exit_cell


def settle(pos, walls, ladders):
    """Fall until supported: on a ladder, or the cell below is a wall or a
    ladder (standing on its top)."""
    c, r = pos
    while True:
        if (c, r) in ladders:
            return (c, r)
        below = (c, r + 1)
        if below in walls or below in ladders or r + 1 >= GRID:
            return (c, r)
        r += 1


def _build_level(index, spec):
    walls, ladders, start, exit_cell = _parse(spec)
    sprites = []
    for (c, r) in walls:
        sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=c * CELL, y=r * CELL,
                              blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
    for (c, r) in ladders:
        sprites.append(Sprite(_ladder_px(), name=f"ladder_{r}_{c}", x=c * CELL, y=r * CELL,
                              layer=1, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_rect(EXIT), name="exit", x=exit_cell[0] * CELL, y=exit_cell[1] * CELL,
                          layer=0, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_agent_px(), name="agent", x=start[0] * CELL, y=start[1] * CELL,
                          layer=3, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Gv01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="gv01",
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
        walls, ladders, start, exit_cell = _parse(spec)
        agent = self._sprite("agent")
        if agent is None:
            self.complete_action()
            return
        c, r = agent.x // CELL, agent.y // CELL

        if action in (GameAction.ACTION1, GameAction.ACTION2,
                      GameAction.ACTION3, GameAction.ACTION4):
            nc, nr = c, r
            if action == GameAction.ACTION1:
                if (c, r) in ladders and (c, r - 1) not in walls and r - 1 >= 0:
                    nr = r - 1
            elif action == GameAction.ACTION2:
                if (c, r + 1) not in walls and r + 1 < GRID:
                    nr = r + 1
            elif action == GameAction.ACTION3:
                if (c - 1, r) not in walls and c - 1 >= 0:
                    nc = c - 1
            else:
                if (c + 1, r) not in walls and c + 1 < GRID:
                    nc = c + 1
            nc, nr = settle((nc, nr), walls, ladders)
            agent.set_position(nc * CELL, nr * CELL)
            if (nc, nr) == exit_cell:
                self.next_level()
        self.complete_action()

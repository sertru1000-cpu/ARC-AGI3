# Original game for the atlas testbed (29.08.2026) -- NOT an ARC Prize game.
# "Boom" (bx01) -- mechanic DICTATED BY THE USER (designer-of-record
# protocol, Gemini round 7): "на поле шарики, главный можно двигать и
# взрывать -- разлетающиеся осколки должны лопнуть все шарики вокруг".
#
# Translation decisions (implementation only, not mechanics): shards fly in
# 8 rays (orthogonals + diagonals) until a wall; detonation consumes the
# main ball -- if target balls survive, the next main ball from a limited
# stock spawns at the start cell; stock exhausted with targets left = game
# over (RESET restarts the level). Win a level = all targets popped.
# ACTION1..4 move, ACTION5 detonates.

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
MAIN = 3          # player-controlled main ball (green)
TARGET = 2        # target balls (red)
STOCK_COLOR = 3   # stock pips shown in the top-left HUD row
FLASH = 4         # shard flash color

# '#'=wall, '.'=floor, 'P'=main ball start, 'o'=target ball.
# data per level: stock = how many main balls you get.
LAYOUTS = [
    # L1: trivial -- stand between two targets, one blast (optimal: 1-2 moves + boom)
    dict(stock=1, rows=[
        "########",
        "#......#",
        "#..o...#",
        "#.P....#",
        "#..o...#",
        "#......#",
        "#......#",
        "########",
    ]),
    # L2: four targets on the same cross point
    dict(stock=1, rows=[
        "########",
        "#..o...#",
        "#......#",
        "#o...o.#",
        "#.P....#",
        "#......#",
        "#..o...#",
        "########",
    ]),
    # L3 (procedurally generated, seed 20260829, BFS-verified)
    dict(stock=1, rows=[
        "########",
        "#.#...P#",
        "#......#",
        "#...oo##",
        "#o....##",
        "#......#",
        "##o....#",
        "########",
    ]),
    # L4 (procedurally generated, seed 20260829, BFS-verified)
    dict(stock=2, rows=[
        "########",
        "#o.#..o#",
        "#.#.#P##",
        "#..#o..#",
        "#..#...#",
        "#.o..oo#",
        "#......#",
        "########",
    ]),
    # L5 (procedurally generated, seed 20260829, BFS-verified)
    dict(stock=2, rows=[
        "########",
        "#....Po#",
        "#...#..#",
        "#.o.#..#",
        "#..#...#",
        "#o.##o##",
        "#oooo#.#",
        "########",
    ]),
]

N_STOCK = [d["stock"] for d in LAYOUTS]


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _ball(color):
    px = [[0] * CELL for _ in range(CELL)]
    for r in range(CELL):
        for c in range(CELL):
            if (r in (1, 2)) or (c in (1, 2)):
                px[r][c] = color
    return px


def _build_level(index, spec):
    sprites = []
    start = None
    tcount = 0
    for r, row in enumerate(spec["rows"]):
        for c, ch in enumerate(row):
            x, y = c * CELL, r * CELL
            if ch == "#":
                sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=x, y=y,
                                      blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
            elif ch == "P":
                start = (c, r)
            elif ch == "o":
                sprites.append(Sprite(_ball(TARGET), name=f"target_{tcount}", x=x, y=y,
                                      layer=1, blocking=BlockingMode.PIXEL_PERFECT))
                tcount += 1
    for s in range(spec["stock"]):
        sprites.append(Sprite(_rect(STOCK_COLOR), name=f"stock_{s}",
                              x=(6 - s) * CELL, y=0, layer=2,
                              blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    pc, pr = start
    sprites.append(Sprite(_ball(MAIN), name="main", x=pc * CELL, y=pr * CELL,
                          layer=3, blocking=BlockingMode.PIXEL_PERFECT))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


START_CELLS = []
for spec in LAYOUTS:
    for r, row in enumerate(spec["rows"]):
        for c, ch in enumerate(row):
            if ch == "P":
                START_CELLS.append((c, r))

RAYS = [(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]


class Bx01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LAYOUTS)]
        super().__init__(
            game_id="bx01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )
        self._stock_left = N_STOCK[0]

    def on_set_level(self, level):
        self._stock_left = N_STOCK[self.level_index]

    def _sprite(self, name):
        found = self.current_level.get_sprites_by_name(name)
        return found[0] if found else None

    def _walls(self):
        return {(s.x // CELL, s.y // CELL)
                for s in self.current_level._sprites if s.name.startswith("wall_")}

    def _targets(self):
        return [s for s in self.current_level._sprites if s.name.startswith("target_")]

    def step(self) -> None:
        action = self.action.id
        main = self._sprite("main")
        if action in (GameAction.ACTION1, GameAction.ACTION2,
                      GameAction.ACTION3, GameAction.ACTION4):
            if main is not None:
                d = {GameAction.ACTION1: (0, -CELL), GameAction.ACTION2: (0, CELL),
                     GameAction.ACTION3: (-CELL, 0), GameAction.ACTION4: (CELL, 0)}[action]
                self.try_move("main", *d)  # walls AND target balls block
            self.complete_action()
            return
        if action != GameAction.ACTION5 or main is None:
            self.complete_action()
            return

        # DETONATE: shards fly along 8 rays until a wall; pop every target hit.
        walls = self._walls()
        cx, cy = main.x // CELL, main.y // CELL
        hit_cells = set()
        for dx, dy in RAYS:
            x, y = cx + dx, cy + dy
            while 0 <= x < GRID and 0 <= y < GRID and (x, y) not in walls:
                hit_cells.add((x, y))
                x += dx
                y += dy
        for t in list(self._targets()):
            if (t.x // CELL, t.y // CELL) in hit_cells:
                self.current_level.remove_sprite(t)
        self.current_level.remove_sprite(main)
        self._stock_left -= 1
        pip = self._sprite(f"stock_{self._stock_left}")
        if pip is not None:
            self.current_level.remove_sprite(pip)

        if not self._targets():
            self.next_level()
        elif self._stock_left > 0:
            sx, sy = START_CELLS[self.level_index]
            self.current_level.add_sprite(
                Sprite(_ball(MAIN), name="main", x=sx * CELL, y=sy * CELL,
                       layer=3, blocking=BlockingMode.PIXEL_PERFECT))
        else:
            self.lose()
        self.complete_action()

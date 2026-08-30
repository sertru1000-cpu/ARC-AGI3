# Original game for the atlas testbed (29.08.2026) -- NOT an ARC Prize game.
# "Floor Paint" (fl01) -- mechanic DICTATED BY THE USER (designer-of-record
# protocol, Gemini round 7): "наступил на клетку -- она окрашена, по
# окрашенной ходить нельзя. Надо закрасить весь пол целиком (посетить
# каждую клетку ровно один раз)."
#
# Translation decisions (implementation only): the cursor walks with
# ACTION1-4; the start cell is painted immediately; stepping paints the
# new cell forever -- painted cells and walls block. Painting the WHOLE
# floor completes the level. Getting boxed in (no unpainted neighbor
# while floor remains) = LOSE -- irreversibility is the point: no probe
# and no rollback saves you, the route must be planned before step one.
# Exact constructive baseline: floor cells minus one (every move paints
# exactly one new cell).

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
CURSOR = 3
PAINT = 8

# rows: '#'=wall, 'P'=cursor start (floor), '.'=floor to paint.
LEVELS = [
    dict(rows=[
        "########",
        "########",
        "########",
        "##..####",
        "##...###",
        "#P...###",
        "#...####",
        "########",
    ]),
    dict(rows=[
        "########",
        "##.P####",
        "#....###",
        "#....###",
        "#...####",
        "#...####",
        "########",
        "########",
    ]),
    dict(rows=[
        "########",
        "#.######",
        "#.######",
        "#....###",
        "#.....##",
        "#.....##",
        "#P...###",
        "########",
    ]),
    dict(rows=[
        "########",
        "#......#",
        "#......#",
        "###.P..#",
        "##....##",
        "#....###",
        "########",
        "########",
    ]),
    dict(rows=[
        "########",
        "####.P.#",
        "#......#",
        "#......#",
        "##.....#",
        "##.##..#",
        "##.....#",
        "########",
    ]),
]


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _cursor_px():
    return [[CURSOR if r in (0, CELL - 1) or c in (0, CELL - 1) else 0
             for c in range(CELL)] for r in range(CELL)]


def _paint_px():
    px = [[PAINT] * CELL for _ in range(CELL)]
    px[0][0] = px[0][CELL - 1] = px[CELL - 1][0] = px[CELL - 1][CELL - 1] = 0
    return px


def _parse(spec):
    walls, floor = set(), set()
    start = None
    for r, row in enumerate(spec["rows"]):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "P":
                start = (c, r)
                floor.add((c, r))
            elif ch == ".":
                floor.add((c, r))
    return walls, floor, start


def _build_level(index, spec):
    walls, floor, start = _parse(spec)
    sprites = []
    for (c, r) in walls:
        sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=c * CELL, y=r * CELL,
                              blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
    sprites.append(Sprite(_cursor_px(), name="cursor", x=start[0] * CELL, y=start[1] * CELL,
                          layer=3, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Fl01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="fl01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )
        self._painted = set()
        self._init_painted()

    def _init_painted(self):
        _, _, start = _parse(LEVELS[self.level_index])
        self._painted = {start}

    def on_set_level(self, level):
        self._init_painted()

    def _atlas_reset_level_state(self):
        self._init_painted()

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
        walls, floor, start = _parse(spec)
        cursor = self._sprite("cursor")
        if cursor is None:
            self.complete_action()
            return
        cx, cy = cursor.x // CELL, cursor.y // CELL

        if action in (GameAction.ACTION1, GameAction.ACTION2,
                      GameAction.ACTION3, GameAction.ACTION4):
            d = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1),
                 GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}[action]
            nc, nr = cx + d[0], cy + d[1]
            if (nc, nr) in floor and (nc, nr) not in self._painted:
                # leave paint behind on the cell we vacate
                self.current_level.add_sprite(
                    Sprite(_paint_px(), name=f"paint_{cy}_{cx}", x=cx * CELL, y=cy * CELL,
                           layer=1, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
                cursor.set_position(nc * CELL, nr * CELL)
                self._painted.add((nc, nr))
                if self._painted == floor:
                    self.next_level()
                    self.complete_action()
                    return
                # boxed in with floor remaining -> irreversibly lost
                stuck = all(
                    (nc + dx, nr + dy) not in floor or (nc + dx, nr + dy) in self._painted
                    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)))
                if stuck:
                    self.lose()
        self.complete_action()

# Original game for the atlas testbed (29.08.2026) -- NOT an ARC Prize game.
# "Pharmacy" (ph01) -- mechanic DICTATED BY THE USER (designer-of-record
# protocol, Gemini round 7): "Аптека: по рецепту ровно столько-то таблеток
# каждого цвета; лишняя -- рецепт аннулирован."
#
# Translation decisions (implementation only): the prescription is the top
# strip (one colored cell per required pill); the cursor walks with
# ACTION1-4 and ACTION5 GRABS the pill under it (walking over pills is
# safe -- taking is deliberate). Taking ANY pill beyond its required count
# VOIDS the prescription: everything collected returns to the board and
# the strip relights. Deliver by stepping onto the counter window with the
# exact set collected. The board always holds MORE pills than needed, and
# later levels scatter colors the prescription wants ZERO of.

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
COUNTER = 4
STRIP_DONE = 0

PILL_COLORS = {"R": 2, "B": 8, "Y": 4, "M": 6}

# Levels: rows with '#'=wall, 'P'=cursor start, 'D'=counter window,
# letters R/B/Y/M = pills of that color lying on the floor.
# need: required counts per color (exactly).
LEVELS = [
    dict(need={'M': 2}, rows=[
        "########",
        "#......#",
        "#......#",
        "#.....M#",
        "#....M.#",
        "#..M...#",
        "#..P...#",
        "######D#",
    ]),
    dict(need={'B': 1, 'M': 1}, rows=[
        "########",
        "#....P.#",
        "#......#",
        "#.M....#",
        "#B....M#",
        "#B.....#",
        "#B.....#",
        "#####D##",
    ]),
    dict(need={'M': 1, 'R': 1, 'B': 2}, rows=[
        "########",
        "#..R...#",
        "#....RB#",
        "#..BBP##",
        "#...#..#",
        "#...MM.#",
        "##.B#RM#",
        "#####D##",
    ]),
    dict(need={'B': 1, 'R': 2, 'M': 0}, rows=[
        "########",
        "#.#..P.#",
        "#R...R.#",
        "#M.###.#",
        "#...B..#",
        "#....B.#",
        "#.R..R.#",
        "#D######",
    ]),
    dict(need={'B': 2, 'Y': 3, 'R': 3, 'M': 0}, rows=[
        "########",
        "#....PR#",
        "#B..R.R#",
        "#Y#Y.#Y#",
        "#...#..#",
        "##RB#RY#",
        "#B.B.M.#",
        "######D#",
    ]),
]


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _pill_px(color):
    px = [[0] * CELL for _ in range(CELL)]
    for r in (1, 2):
        for c in (1, 2):
            px[r][c] = color
    px[1][0] = px[2][3] = color
    return px


def _cursor_px():
    return [[CURSOR if r in (0, CELL - 1) or c in (0, CELL - 1) else 0
             for c in range(CELL)] for r in range(CELL)]


def _parse(spec):
    walls, pills = set(), {}
    start = door = None
    for r, row in enumerate(spec["rows"]):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "D":
                door = (c, r)
            elif ch in PILL_COLORS:
                pills[(c, r)] = ch
    return walls, pills, start, door


def _build_level(index, spec):
    walls, pills, start, door = _parse(spec)
    sprites = []
    for (c, r) in walls:
        if (c, r) != door:
            sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=c * CELL, y=r * CELL,
                                  blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
    sprites.append(Sprite(_rect(COUNTER), name="counter", x=door[0] * CELL, y=door[1] * CELL,
                          layer=0, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    # prescription strip along the top wall
    slot = 1
    for color_ch in sorted(spec["need"]):
        for _ in range(spec["need"][color_ch]):
            sprites.append(Sprite(_rect(PILL_COLORS[color_ch]), name=f"strip_{slot}",
                                  x=slot * CELL, y=0, layer=2,
                                  blocking=BlockingMode.NOT_BLOCKED, collidable=False))
            slot += 1
    for i, ((c, r), ch) in enumerate(sorted(pills.items())):
        sprites.append(Sprite(_pill_px(PILL_COLORS[ch]), name=f"pill_{i}",
                              x=c * CELL, y=r * CELL, layer=1,
                              blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_cursor_px(), name="cursor", x=start[0] * CELL, y=start[1] * CELL,
                          layer=3, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Ph01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="ph01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )
        self._taken = {}
        self._gone = set()

    def on_set_level(self, level):
        self._taken = {}
        self._gone = set()

    def _atlas_reset_level_state(self):
        self._taken = {}
        self._gone = set()

    def level_reset(self):
        super().level_reset()
        self._atlas_reset_level_state()

    def full_reset(self):
        super().full_reset()
        self._atlas_reset_level_state()

    def _sprite(self, name):
        found = self.current_level.get_sprites_by_name(name)
        return found[0] if found else None

    def _void(self, spec):
        """Extra pill -> prescription VOID: all collected pills return."""
        self._taken = {}
        _, pills, _, _ = _parse(spec)
        for i, ((pc, pr), ch) in enumerate(sorted(pills.items())):
            if (pc, pr) in self._gone:
                self.current_level.add_sprite(
                    Sprite(_pill_px(PILL_COLORS[ch]), name=f"pill_{i}",
                           x=pc * CELL, y=pr * CELL, layer=1,
                           blocking=BlockingMode.NOT_BLOCKED, collidable=False))
        self._gone = set()
        slot = 1
        for color_ch in sorted(spec["need"]):
            for _ in range(spec["need"][color_ch]):
                s = self._sprite(f"strip_{slot}")
                if s is not None:
                    s.pixels[:] = PILL_COLORS[color_ch]
                slot += 1

    def step(self) -> None:
        spec = LEVELS[self.level_index]
        action = self.action.id
        if action == GameAction.RESET:
            self.complete_action()
            return
        walls, pills, start, door = _parse(spec)
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
            if (nc, nr) == door:
                # delivery attempt: exact counts -> level complete
                if all(self._taken.get(ch, 0) == n for ch, n in spec["need"].items()):
                    cursor.set_position(nc * CELL, nr * CELL)
                    self.next_level()
                    self.complete_action()
                    return
                # not exact: the window rejects you (stay in place)
            elif (nc, nr) not in walls and 0 <= nc < GRID and 0 <= nr < GRID:
                cursor.set_position(nc * CELL, nr * CELL)
        elif action == GameAction.ACTION5:
            if (cx, cy) in pills and (cx, cy) not in self._gone:
                ch = pills[(cx, cy)]
                need = spec["need"].get(ch, 0)
                if self._taken.get(ch, 0) + 1 > need:
                    self._void(spec)
                else:
                    self._taken[ch] = self._taken.get(ch, 0) + 1
                    self._gone.add((cx, cy))
                    for i, ((pc, pr), pch) in enumerate(sorted(pills.items())):
                        if (pc, pr) == (cx, cy):
                            s = self._sprite(f"pill_{i}")
                            if s is not None:
                                self.current_level.remove_sprite(s)
                    # dim the first still-lit strip slot of this color
                    slot = 1
                    done = False
                    for color_ch in sorted(spec["need"]):
                        for _ in range(spec["need"][color_ch]):
                            if not done and color_ch == ch:
                                s = self._sprite(f"strip_{slot}")
                                if s is not None and s.pixels[0][0] != STRIP_DONE:
                                    s.pixels[:] = STRIP_DONE
                                    done = True
                            slot += 1
        self.complete_action()

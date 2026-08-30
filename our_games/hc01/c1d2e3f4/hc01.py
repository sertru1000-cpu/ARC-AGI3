# Original game for the atlas testbed (30.08.2026) -- NOT an ARC Prize game.
# "Hidden Combo" (hc01) -- HOSTILE-POOL game 1 (Gemini round 13 Q4):
# hidden state that is NOT rendered. Stepping on buttons in a secret order
# opens the door; a wrong press silently resets the hidden progress. The
# board looks IDENTICAL at different internal progress -- frame-signature
# search dedup prunes the winning line, which is the designed hostility.
#
# Rules: ACTION1-4 move. Buttons are colored pads; the correct order is a
# fixed hidden permutation per level. Visiting the correct next button
# advances hidden progress (no visual change); visiting a wrong one resets
# progress to zero (no visual change either). When the full sequence is
# entered the door opens (visible) and the exit becomes reachable.

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
DOOR = 2
BUTTON_COLORS = {"a": 6, "b": 8, "c": 11}

# rows: '#'=wall, 'P'=agent, 'E'=exit, 'D'=door, 'a'/'b'/'c'=buttons.
# "order": the hidden combo (button letters in required order).
# Every exit sits in a wall-sealed pocket whose ONLY entrance is the door
# cell -- verified by scripts/hostile_baselines.py (no-combo BFS must fail).
LEVELS = [
    dict(rows=[
        "########",
        "#P.....#",
        "#......#",
        "#a....b#",
        "#......#",
        "#....#D#",
        "#....#E#",
        "########",
    ], order="ba"),
    dict(rows=[
        "########",
        "#a..#DE#",
        "#...#.##",
        "#.P....#",
        "#...#..#",
        "#b..#.c#",
        "#...#..#",
        "########",
    ], order="cab"),
    dict(rows=[
        "########",
        "#..b...#",
        "#......#",
        "#P.....#",
        "#......#",
        "#a..#.c#",
        "###DE###",
        "########",
    ], order="acb"),
    dict(rows=[
        "########",
        "#c...#a#",
        "#....#.#",
        "#..P...#",
        "#......#",
        "#b..#.##",
        "#...#DE#",
        "########",
    ], order="bca"),
    dict(rows=[
        "########",
        "#a.#..c#",
        "#..#...#",
        "#..#.P.#",
        "#......#",
        "#b..#.##",
        "#..##DE#",
        "########",
    ], order="cba"),
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


def _parse(spec):
    walls, buttons = set(), {}
    start = exit_cell = door = None
    for r, row in enumerate(spec["rows"]):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "E":
                exit_cell = (c, r)
            elif ch == "D":
                door = (c, r)
            elif ch in BUTTON_COLORS:
                buttons[(c, r)] = ch
    return walls, buttons, start, exit_cell, door


def _build_level(index, spec):
    walls, buttons, start, exit_cell, door = _parse(spec)
    sprites = []
    for (c, r) in walls:
        sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=c * CELL, y=r * CELL,
                              blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
    for (c, r), ch in buttons.items():
        sprites.append(Sprite(_rect(BUTTON_COLORS[ch]), name=f"btn_{ch}", x=c * CELL, y=r * CELL,
                              layer=0, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_rect(DOOR), name="door", x=door[0] * CELL, y=door[1] * CELL,
                          layer=1, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_rect(EXIT), name="exit", x=exit_cell[0] * CELL, y=exit_cell[1] * CELL,
                          layer=0, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_agent_px(), name="agent", x=start[0] * CELL, y=start[1] * CELL,
                          layer=3, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Hc01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="hc01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )
        self._progress = 0

    def on_set_level(self, level):
        self._progress = 0

    def _atlas_reset_level_state(self):
        self._progress = 0
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
        door = self._sprite("door")
        if door is None:
            return
        opened = self._progress >= len(spec["order"])
        px = _rect(0) if opened else _rect(DOOR)
        for rr in range(CELL):
            for cc in range(CELL):
                door.pixels[rr][cc] = px[rr][cc]

    def step(self) -> None:
        spec = LEVELS[self.level_index]
        action = self.action.id
        if action == GameAction.RESET:
            self._sync()
            self.complete_action()
            return
        walls, buttons, start, exit_cell, door = _parse(spec)
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
            order = spec["order"]
            opened = self._progress >= len(order)
            blocked = (nc, nr) in walls or not (0 <= nc < GRID and 0 <= nr < GRID)
            if not blocked and (nc, nr) == door and not opened:
                blocked = True
            if not blocked:
                agent.set_position(nc * CELL, nr * CELL)
                if (nc, nr) in buttons and not opened:
                    ch = buttons[(nc, nr)]
                    if self._progress < len(order) and ch == order[self._progress]:
                        self._progress += 1        # NO visual change -- hidden
                    elif ch != order[max(0, self._progress - 1)]:
                        # standing on / revisiting the PREVIOUS correct button
                        # is neutral; any other wrong button silently resets.
                        self._progress = 0
                    self._sync()                    # door opens only when done
                if (nc, nr) == exit_cell:
                    self.next_level()
        self.complete_action()

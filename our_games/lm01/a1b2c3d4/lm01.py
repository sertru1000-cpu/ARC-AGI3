# Original game for the atlas testbed (01.09.2026) -- NOT an ARC Prize game.
#
# "Long March" (lm01) -- the PURE LENGTH control of the long-level batch.
#
# Why it exists. Measured 01.09 on three calibration runs: the eleven public
# games we never clear have a median human baseline of 54 actions on level 1
# (range 21-78), while the seven we always clear sit at 22. Our own testbed
# had nothing above 20, so it could not reproduce the failure that costs us
# the most. This batch fills that gap.
#
# lm01 is deliberately the DUMBEST long game: one mechanic, no discovery, no
# hidden rule. A serpentine corridor; walk to the exit. If an agent fails
# here, length alone is the problem -- not mechanics, not deduction, not
# theory-building. Every other game in the batch adds a discovery burden on
# top, so lm01 is the control that separates "long" from "hard".
#
# Level 1 optimal is verified by BFS in scripts/test_atlas_long_games.py.

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

WALL = 9
FLOOR = 0
PLAYER = 1
EXIT = 3

# Each level: rows that are open corridors, and the column each corridor
# connects to the next one through. Everything else inside the border is wall.
# start/exit are (col, row).
LEVELS = [
    # 03.09: retuned to the PUBLIC difficulty curve, and moved to the same
    # 21x3 board as the generated batch. lm01 used to be nine flat levels of
    # 65 actions -- deliberately flat, as the pure-length control. Measured
    # over five runs, it cleared level 1 exactly once and nothing beyond, and
    # the reason was the same as for the rest of the batch: level 1 cost 65
    # actions where the public median is 30, so nobody ever got in.
    #
    # It stays the control -- one mechanic, no rule to discover, walk to the
    # exit -- but it now carries the SAME length profile as the games that do
    # have a rule to discover. That is what makes it a control: the only
    # difference left between lm01 and gk01 is the discovery burden, not the
    # walking.
    #
    # Every entry below was found by searching corridor count and exit column
    # against shortest_path_len(), not counted by hand. This file already
    # shipped one hand-counted baseline of 53 whose true value was 65.
    dict(corridors=[1, 3], turns=[19],
         start=(1, 1), exit=(9, 3)),
    dict(corridors=[1, 3, 5], turns=[1, 19],
         start=(19, 1), exit=(5, 5)),
    dict(corridors=[1, 3, 5], turns=[1, 19],
         start=(19, 1), exit=(8, 5)),
    dict(corridors=[1, 3, 5], turns=[19, 1],
         start=(1, 1), exit=(15, 5)),
    dict(corridors=[1, 3, 5, 7, 9], turns=[1, 19, 1, 19],
         start=(19, 1), exit=(3, 9)),
    dict(corridors=[1, 3, 5, 7, 9], turns=[19, 1, 19, 1],
         start=(1, 1), exit=(1, 9)),
    dict(corridors=[1, 3, 5, 7, 9], turns=[19, 1, 19, 1],
         start=(1, 1), exit=(7, 9)),
    dict(corridors=[1, 3, 5, 7, 9], turns=[1, 19, 1, 19],
         start=(19, 1), exit=(7, 9)),
    dict(corridors=[1, 3, 5, 7, 9, 11, 13, 15, 17], turns=[19, 1, 19, 1, 19, 1, 19, 1],
         start=(1, 1), exit=(4, 17)),
]


def open_cells(spec):
    """The set of walkable (col,row) cells for a level spec."""
    cells = set()
    rows = spec["corridors"]
    for r in rows:
        for c in range(1, GRID - 1):
            cells.add((c, r))
    for i, col in enumerate(spec["turns"]):
        r0, r1 = rows[i], rows[i + 1]
        for r in range(min(r0, r1), max(r0, r1) + 1):
            cells.add((col, r))
    for cell in spec.get("detour", []):
        cells.add(cell)
    return cells


def shortest_path_len(spec):
    """BFS from start to exit over open_cells -- the level's true baseline."""
    from collections import deque

    cells = open_cells(spec)
    start, goal = spec["start"], spec["exit"]
    if start not in cells or goal not in cells:
        return None
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        (c, r), dist = queue.popleft()
        if (c, r) == goal:
            return dist
        for dc, dr in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nxt = (c + dc, r + dr)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, dist + 1))
    return None


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _player_px():
    return [[PLAYER if 0 < i < CELL - 1 and 0 < j < CELL - 1 else 0
             for j in range(CELL)] for i in range(CELL)]


def _exit_px():
    return [[EXIT if i in (0, CELL - 1) or j in (0, CELL - 1) else 0
             for j in range(CELL)] for i in range(CELL)]


def _build_level(index, spec):
    cells = open_cells(spec)
    sprites = []
    for r in range(GRID):
        for c in range(GRID):
            if (c, r) not in cells:
                sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}",
                                      x=c * CELL, y=r * CELL,
                                      blocking=BlockingMode.PIXEL_PERFECT,
                                      tags=["sys_static"]))
    ec, er = spec["exit"]
    sprites.append(Sprite(_exit_px(), name="exit", x=ec * CELL, y=er * CELL,
                          layer=1, blocking=BlockingMode.NOT_BLOCKED,
                          collidable=False))
    sc, sr = spec["start"]
    sprites.append(Sprite(_player_px(), name="player", x=sc * CELL, y=sr * CELL,
                          layer=2, blocking=BlockingMode.NOT_BLOCKED,
                          collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Lm01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="lm01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )
        self._pos = None
        self._load()

    def _load(self):
        self._pos = list(LEVELS[self.level_index]["start"])

    def on_set_level(self, level):
        self._load()

    def level_reset(self):
        super().level_reset()
        self._load()

    def full_reset(self):
        super().full_reset()
        self._load()

    def _sprite(self, name):
        found = self.current_level.get_sprites_by_name(name)
        return found[0] if found else None

    def _sync(self):
        player = self._sprite("player")
        if player is not None:
            player.set_position(self._pos[0] * CELL, self._pos[1] * CELL)

    def step(self) -> None:
        spec = LEVELS[self.level_index]
        action = self.action.id
        if action == GameAction.RESET:
            self._sync()
            self.complete_action()
            return
        moves = {
            GameAction.ACTION1: (0, -1),
            GameAction.ACTION2: (0, 1),
            GameAction.ACTION3: (-1, 0),
            GameAction.ACTION4: (1, 0),
        }
        if action in moves:
            dc, dr = moves[action]
            nxt = (self._pos[0] + dc, self._pos[1] + dr)
            if nxt in open_cells(spec):
                self._pos = list(nxt)
        self._sync()
        if tuple(self._pos) == spec["exit"]:
            self.next_level()
        self.complete_action()

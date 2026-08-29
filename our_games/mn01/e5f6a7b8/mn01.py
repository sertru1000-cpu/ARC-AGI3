# Original game for the atlas testbed (29.08.2026) -- NOT an ARC Prize game.
# "Manipulator" (mn01) -- mechanic DICTATED BY THE USER (designer-of-record
# protocol, Gemini round 7): "нужно манипулятором собрать в заданной
# последовательности фигуры. Фигуры хаотично расположены на доске и после
# каждого хода хаотично сдвигаются. Может быть ситуация, когда нужная
# фигура недоступна, тогда просто выполняем действие -- пропустить ход."
#
# Translation decisions (implementation only): the gripper is a hollow
# frame cursor (ACTION1..4 move it); ACTION5 = GRAB -- if the next-needed
# shape is under the gripper it is collected, otherwise the press is just a
# skipped tick (the user's "пропустить ход"). EVERY action advances one
# tick and every shape takes one step of its own deterministic
# pseudo-random walk (seeded per level, bounces off walls, walks are
# independent -- shapes may overlap). The hint strip shows the collection
# ORDER as shape silhouettes. No fail state: wasted ticks are the cost.

import random

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
T_MAX = 400
WALK_SEED = 918273

WALL = 9
GRIP = 3
HINT_DONE = 0

# shape forms: name -> (color, 4x4 pixel stencil rows as strings)
FORMS = {
    "square": (4,  ["....", ".##.", ".##.", "...."]),
    "cross":  (6,  [".#..", "###.", ".#..", "...."]),
    "corner": (7,  ["....", ".##.", ".#..", "...."]),
    "bar":    (11, ["....", "####", "....", "...."]),
    "tee":    (12, ["....", "###.", ".#..", "...."]),
}

# per level: gripper start, walls (extra, beyond border), shapes in
# COLLECTION ORDER: (form, start_col, start_row, period) -- period = the
# shape steps its walk every `period` ticks.
LEVELS = [
    dict(grip=(4, 4), walls=[],
         shapes=[("square", 1, 1, 2), ("cross", 6, 5, 2)]),
    dict(grip=(1, 6), walls=[],
         shapes=[("square", 5, 1, 2), ("cross", 2, 2, 1), ("corner", 6, 6, 2)]),
    dict(grip=(4, 1), walls=[(3, 4), (4, 4)],
         shapes=[("cross", 1, 5, 1), ("bar", 6, 2, 2), ("square", 2, 1, 1), ("corner", 5, 6, 2)]),
    dict(grip=(1, 1), walls=[(2, 3), (5, 3), (3, 5)],
         shapes=[("bar", 6, 1, 1), ("tee", 1, 5, 1), ("cross", 4, 2, 2), ("square", 6, 6, 1)]),
    dict(grip=(6, 3), walls=[(2, 2), (2, 5), (5, 5)],
         shapes=[("corner", 1, 1, 1), ("square", 3, 6, 1), ("bar", 1, 6, 2),
                 ("cross", 6, 1, 1), ("tee", 3, 1, 2)]),
]


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _form_px(form):
    color, rows = FORMS[form]
    return [[color if ch == "#" else 0 for ch in row] for row in rows]


def _grip_px():
    return [[GRIP if r in (0, CELL - 1) or c in (0, CELL - 1) else 0
             for c in range(CELL)] for r in range(CELL)]


def build_schedules(level_index):
    """Deterministic pseudo-random walk per shape: positions[tick] for each
    shape, walls-bounded, independent of grabs (shapes may overlap)."""
    spec = LEVELS[level_index]
    walls = set(spec["walls"])
    walls |= {(c, r) for c in range(GRID) for r in range(GRID)
              if r in (0, GRID - 1) or c in (0, GRID - 1)}
    schedules = []
    for si, (form, c0, r0, period) in enumerate(spec["shapes"]):
        rng = random.Random(WALK_SEED + level_index * 100 + si)
        pos = (c0, r0)
        track = [pos]
        for t in range(1, T_MAX + 1):
            if t % period == 0:
                dx, dy = rng.choice([(0, -1), (0, 1), (-1, 0), (1, 0)])
                np_ = (pos[0] + dx, pos[1] + dy)
                if np_ not in walls:
                    pos = np_
            track.append(pos)
        schedules.append(track)
    return schedules


SCHEDULES = [build_schedules(i) for i in range(len(LEVELS))]


def _build_level(index, spec):
    sprites = []
    walls = set(spec["walls"])
    for r in range(GRID):
        for c in range(GRID):
            if r in (0, GRID - 1) or c in (0, GRID - 1) or (c, r) in walls:
                sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=c * CELL, y=r * CELL,
                                      blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
    n = len(spec["shapes"])
    for order, (form, c0, r0, period) in enumerate(spec["shapes"]):
        sprites.append(Sprite(_form_px(form), name=f"shape_{order}", x=c0 * CELL, y=r0 * CELL,
                              layer=1, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
        sprites.append(Sprite(_form_px(form), name=f"hint_{order}",
                              x=(order + 1) * CELL, y=0, layer=2,
                              blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    gc, gr = spec["grip"]
    sprites.append(Sprite(_grip_px(), name="grip", x=gc * CELL, y=gr * CELL, layer=3,
                          blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Mn01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="mn01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )
        self._tick = 0
        self._next = 0

    def on_set_level(self, level):
        self._tick = 0
        self._next = 0

    def _sprite(self, name):
        found = self.current_level.get_sprites_by_name(name)
        return found[0] if found else None


    def _atlas_reset_level_state(self):
        self._tick = 0
        self._next = 0

    def level_reset(self):
        super().level_reset()
        self._atlas_reset_level_state()

    def full_reset(self):
        super().full_reset()
        self._atlas_reset_level_state()

    def step(self) -> None:
        spec = LEVELS[self.level_index]
        sched = SCHEDULES[self.level_index]
        action = self.action.id
        grip = self._sprite("grip")
        if action == GameAction.RESET:
            self.complete_action()
            return
        walls = set(spec["walls"])

        if grip is not None and action in (GameAction.ACTION1, GameAction.ACTION2,
                                           GameAction.ACTION3, GameAction.ACTION4):
            d = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1),
                 GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}[action]
            gc, gr = grip.x // CELL + d[0], grip.y // CELL + d[1]
            if 0 < gc < GRID - 1 and 0 < gr < GRID - 1 and (gc, gr) not in walls:
                grip.set_position(gc * CELL, gr * CELL)
        elif action == GameAction.ACTION5 and grip is not None:
            gc, gr = grip.x // CELL, grip.y // CELL
            if (self._next < len(spec["shapes"])
                    and sched[self._next][self._tick] == (gc, gr)):
                # grab the next-needed shape
                s = self._sprite(f"shape_{self._next}")
                if s is not None:
                    self.current_level.remove_sprite(s)
                hint = self._sprite(f"hint_{self._next}")
                if hint is not None:
                    hint.pixels[:] = HINT_DONE
                self._next += 1
                if self._next == len(spec["shapes"]):
                    self.next_level()
                    self.complete_action()
                    return
            # else: the user's "skip a turn" -- the tick simply passes
        # advance the world one tick
        self._tick = min(self._tick + 1, T_MAX)
        for order in range(len(spec["shapes"])):
            if order < self._next:
                continue
            s = self._sprite(f"shape_{order}")
            if s is not None:
                c, r = sched[order][self._tick]
                s.set_position(c * CELL, r * CELL)
        self.complete_action()

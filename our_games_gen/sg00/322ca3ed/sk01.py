# Original game for the atlas testbed (30.08.2026) -- NOT an ARC Prize game.
# "Sokoban" (sk01) -- rank-1 mechanic from the Gemini round-11 pool plan:
# irreversible-DAG topology, deadlock foresight, deep planning.
#
# Rules: ACTION1-4 move the agent; walking into a box PUSHES it one cell
# if the cell beyond is free (no pulls, no double-pushes). A box shoved
# into a corner is stuck forever -- no lose state, but an unwinnable
# position: RESET is the only way out (irreversibility is the point).
# The level completes when every box sits on a target pad.

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
BOX = 7
BOX_DONE = 12
TARGET = 2

# rows: '#'=wall, 'P'=agent, 'B'=box, 'T'=target pad, '*'=box on target.
LEVELS = [
    dict(rows=[
        "########",
        "#......#",
        "#......#",
        "#....#P#",
        "#.B....#",
        "#.#...T#",
        "#......#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#......#",
        "#...#..#",
        "#.#T...#",
        "#..P...#",
        "##B.#..#",
        "#......#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#.T#...#",
        "#......#",
        "#..TB..#",
        "#..B##.#",
        "#......#",
        "#P.....#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#......#",
        "##.BT..#",
        "###....#",
        "#..#...#",
        "##..B..#",
        "#.TP...#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#..T...#",
        "##B..B.#",
        "#......#",
        "#..BT..#",
        "#..P...#",
        "#.##T#.#",
        "########",
    ]),
]


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _target_px():
    return [[TARGET if r in (0, CELL - 1) or c in (0, CELL - 1) else 0
             for c in range(CELL)] for r in range(CELL)]


def _agent_px():
    px = [[0] * CELL for _ in range(CELL)]
    for r in (1, 2):
        for c in (1, 2):
            px[r][c] = AGENT
    px[0][1] = px[0][2] = AGENT
    return px


def _parse(spec):
    walls, boxes, targets = set(), set(), set()
    start = None
    for r, row in enumerate(spec["rows"]):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "B":
                boxes.add((c, r))
            elif ch == "T":
                targets.add((c, r))
            elif ch == "*":
                boxes.add((c, r))
                targets.add((c, r))
    return walls, boxes, targets, start


def _build_level(index, spec):
    walls, boxes, targets, start = _parse(spec)
    sprites = []
    for (c, r) in walls:
        sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=c * CELL, y=r * CELL,
                              blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
    for i, (c, r) in enumerate(sorted(targets)):
        sprites.append(Sprite(_target_px(), name=f"target_{i}", x=c * CELL, y=r * CELL,
                              layer=0, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    for i, (c, r) in enumerate(sorted(boxes)):
        sprites.append(Sprite(_rect(BOX), name=f"box_{i}", x=c * CELL, y=r * CELL,
                              layer=2, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_agent_px(), name="agent", x=start[0] * CELL, y=start[1] * CELL,
                          layer=3, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Sk01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="sk01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )
        self._boxes: dict[int, tuple[int, int]] = {}
        self._load_boxes()

    def _load_boxes(self):
        _, boxes, _, _ = _parse(LEVELS[self.level_index])
        self._boxes = {i: br for i, br in enumerate(sorted(boxes))}

    def on_set_level(self, level):
        self._load_boxes()

    def _atlas_reset_level_state(self):
        self._load_boxes()

    def level_reset(self):
        super().level_reset()
        self._atlas_reset_level_state()

    def full_reset(self):
        super().full_reset()
        self._atlas_reset_level_state()

    def _sprite(self, name):
        found = self.current_level.get_sprites_by_name(name)
        return found[0] if found else None

    def _sync(self, targets):
        for i, (c, r) in self._boxes.items():
            s = self._sprite(f"box_{i}")
            if s is not None:
                s.set_position(c * CELL, r * CELL)
                s.pixels[:] = BOX_DONE if (c, r) in targets else BOX

    def step(self) -> None:
        spec = LEVELS[self.level_index]
        action = self.action.id
        if action == GameAction.RESET:
            self.complete_action()
            return
        walls, _, targets, _ = _parse(spec)
        agent = self._sprite("agent")
        if agent is None:
            self.complete_action()
            return
        ac, ar = agent.x // CELL, agent.y // CELL

        if action in (GameAction.ACTION1, GameAction.ACTION2,
                      GameAction.ACTION3, GameAction.ACTION4):
            d = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1),
                 GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}[action]
            nc, nr = ac + d[0], ar + d[1]
            box_cells = set(self._boxes.values())
            if (nc, nr) in walls:
                pass
            elif (nc, nr) in box_cells:
                bc, br = nc + d[0], nr + d[1]
                if (bc, br) not in walls and (bc, br) not in box_cells \
                        and 0 <= bc < GRID and 0 <= br < GRID:
                    for i, cell in self._boxes.items():
                        if cell == (nc, nr):
                            self._boxes[i] = (bc, br)
                            break
                    agent.set_position(nc * CELL, nr * CELL)
                    self._sync(targets)
                    if set(self._boxes.values()) >= targets:
                        self.next_level()
                        self.complete_action()
                        return
            elif 0 <= nc < GRID and 0 <= nr < GRID:
                agent.set_position(nc * CELL, nr * CELL)
        self.complete_action()

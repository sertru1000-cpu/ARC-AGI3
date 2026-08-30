# Original game for the atlas testbed (30.08.2026) -- NOT an ARC Prize game.
# "Clockgates" (cl01) -- rank-12 mechanic from the Gemini round-11 pool
# plan: temporal arithmetic -- passing through gates requires being in
# phase with their period.
#
# Rules: ACTION1-4 move, ACTION5 waits in place. Each gate has a period k
# (2 or 3) and is OPEN only on the action where the running action count
# hits a multiple of k. The gate's pixels cycle through its phase colors
# every action, so the rule is fully visible in the frame (no hidden
# randomness). Reach the exit pad; waiting for the beat is part of the
# optimal plan.

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
GATE_OPEN = 3
PHASE_COLORS = [8, 6, 11]

# rows: '#'=wall, 'P'=agent, 'E'=exit, '2'/'3'=gate with that period.
LEVELS = [
    dict(rows=[
        "########",
        "#P..#..#",
        "#2..#..#",
        "#.#..#.#",
        "#....#.#",
        "#E....##",
        "#.#...##",
        "########",
    ]),
    dict(rows=[
        "########",
        "#..#..##",
        "##.#E..#",
        "#..3..##",
        "#..#..##",
        "#....2P#",
        "#.....##",
        "########",
    ]),
    dict(rows=[
        "########",
        "#.E.3#.#",
        "#......#",
        "###3.###",
        "###.#P.#",
        "#.....##",
        "#...#..#",
        "########",
    ]),
    dict(rows=[
        "########",
        "#..##..#",
        "#P#.#..#",
        "#3..2.##",
        "#.#.#E##",
        "#.##...#",
        "#.....2#",
        "########",
    ]),
    dict(rows=[
        "########",
        "##..#..#",
        "##E.##.#",
        "##.....#",
        "#.##..##",
        "#.3#2.3#",
        "##P...##",
        "########",
    ]),
]


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _gate_px(period, t):
    phase = t % period
    if phase == period - 1:
        # opens on the NEXT action -- show the open color as a hollow ring
        color = GATE_OPEN
        return [[color if r in (0, CELL - 1) or c in (0, CELL - 1) else 0
                 for c in range(CELL)] for r in range(CELL)]
    return _rect(PHASE_COLORS[phase % len(PHASE_COLORS)])


def _agent_px():
    px = [[0] * CELL for _ in range(CELL)]
    for r in (1, 2):
        for c in (1, 2):
            px[r][c] = AGENT
    px[0][1] = px[0][2] = AGENT
    return px


def _parse(spec):
    walls, gates = set(), {}
    start = exit_cell = None
    for r, row in enumerate(spec["rows"]):
        for c, ch in enumerate(row):
            if ch == "#":
                walls.add((c, r))
            elif ch == "P":
                start = (c, r)
            elif ch == "E":
                exit_cell = (c, r)
            elif ch in ("2", "3"):
                gates[(c, r)] = int(ch)
    return walls, gates, start, exit_cell


def _build_level(index, spec):
    walls, gates, start, exit_cell = _parse(spec)
    sprites = []
    for (c, r) in walls:
        sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=c * CELL, y=r * CELL,
                              blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
    for (c, r), k in gates.items():
        sprites.append(Sprite(_gate_px(k, 0), name=f"gate_{r}_{c}", x=c * CELL, y=r * CELL,
                              layer=1, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_rect(EXIT), name="exit", x=exit_cell[0] * CELL, y=exit_cell[1] * CELL,
                          layer=0, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_agent_px(), name="agent", x=start[0] * CELL, y=start[1] * CELL,
                          layer=3, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


class Cl01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="cl01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )
        self._t = 0

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
        walls, gates, start, exit_cell = _parse(spec)
        for (c, r), k in gates.items():
            s = self._sprite(f"gate_{r}_{c}")
            if s is not None:
                px = _gate_px(k, self._t)
                for rr in range(CELL):
                    for cc in range(CELL):
                        s.pixels[rr][cc] = px[rr][cc]

    def step(self) -> None:
        spec = LEVELS[self.level_index]
        action = self.action.id
        if action == GameAction.RESET:
            self._sync()
            self.complete_action()
            return
        walls, gates, start, exit_cell = _parse(spec)
        agent = self._sprite("agent")
        if agent is None:
            self.complete_action()
            return
        c, r = agent.x // CELL, agent.y // CELL

        acted = False
        if action == GameAction.ACTION5:
            acted = True
        elif action in (GameAction.ACTION1, GameAction.ACTION2,
                        GameAction.ACTION3, GameAction.ACTION4):
            d = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1),
                 GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}[action]
            nc, nr = c + d[0], r + d[1]
            blocked = (nc, nr) in walls or not (0 <= nc < GRID and 0 <= nr < GRID)
            if not blocked and (nc, nr) in gates:
                # the move lands on action count t+1: open iff (t+1) % k == 0
                if (self._t + 1) % gates[(nc, nr)] != 0:
                    blocked = True
            if not blocked:
                agent.set_position(nc * CELL, nr * CELL)
                if (nc, nr) == exit_cell:
                    self.next_level()
                    self.complete_action()
                    return
            acted = True
        if acted:
            self._t += 1
            self._sync()
        self.complete_action()

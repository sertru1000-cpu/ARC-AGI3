# Original game for the atlas testbed (29.08.2026) -- NOT an ARC Prize game.
# "Rube" (rg01) -- mechanic DICTATED BY THE USER (designer-of-record
# protocol, Gemini round 7): "мячик на одном конце поля, на другом --
# ворота. Между ними -- много деталей -- шестерёнки, доски и так далее.
# Детали можно перемещать и вращать. Цель -- заранее установить все детали,
# чтобы по кнопке запуска шарик сам докатился до ворот."
#
# Translation decisions (implementation only): setup phase -> MOUSE click
# selects a piece (or the LAUNCH button in the top-left corner), ACTION1-4
# move the selected piece one cell, ACTION5 rotates it. On LAUNCH the ball
# falls tick by tick: boards '/' and '\' deflect it diagonally, gears carry
# it one cell sideways (rotation direction), reaching the goal gap in the
# bottom wall completes the level; getting stuck just returns the ball to
# its start (pieces stay - iterate, every action still costs). Extra decoy
# pieces exist on later levels (noise: not every piece is needed).

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
BALL = 2
GOAL = 4
BUTTON = 3
BOARD_C = 11      # boards (slash / backslash)
GEAR_C = 7        # gears
SEL = 12          # selection highlight

# pieces: (kind, orient, col, row) -- kind 'b' board / 'g' gear;
# board orient 0='/' 1='\', gear orient 0=CCW(left) 1=CW(right).
# Each level: walls border + bottom wall with goal gap 'G'; ball fixed.
LEVELS = [
    # L1: one board -- move & rotate it under the ball
    dict(ball=(2, 1), goal=(4, 7),
         pieces=[('g', 0, 5, 2)],                       # scrambled
         solution=[('g', 1, 2, 2)]),                    # known-good
    # L2: board + gear chain
    dict(ball=(1, 1), goal=(4, 7),
         pieces=[('b', 0, 6, 1), ('g', 0, 3, 5)],
         solution=[('b', 1, 1, 2), ('g', 1, 2, 5)]),
    # L3: two boards, one gear; decoy board not needed
    dict(ball=(6, 1), goal=(1, 7),
         pieces=[('b', 1, 2, 2), ('b', 0, 5, 5), ('g', 1, 3, 3), ('b', 1, 6, 6)],
         solution=[('b', 0, 2, 6), ('b', 0, 3, 4), ('g', 0, 5, 3), ('b', 0, 6, 2)]),
    # L4: zigzag with two gears
    dict(ball=(3, 1), goal=(6, 7),
         pieces=[('g', 0, 1, 1), ('b', 0, 1, 6), ('g', 0, 6, 3), ('b', 1, 4, 4)],
         solution=[('g', 1, 3, 2), ('b', 0, 2, 6), ('g', 0, 6, 3), ('b', 1, 5, 4)]),
    # L5: long chain, one decoy gear
    dict(ball=(1, 1), goal=(6, 7),
         pieces=[('b', 0, 2, 6), ('g', 0, 5, 1), ('b', 0, 6, 2), ('g', 1, 2, 3), ('g', 0, 4, 4)],
         solution=[('b', 1, 2, 6), ('g', 0, 3, 2), ('b', 1, 5, 5), ('g', 1, 1, 3), ('g', 1, 3, 4)]),
]


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _slash_px(orient):
    px = [[0] * CELL for _ in range(CELL)]
    for i in range(CELL):
        j = (CELL - 1 - i) if orient == 0 else i
        px[i][j] = BOARD_C
        if j + 1 < CELL:
            px[i][j + 1] = BOARD_C
    return px


def _gear_px(orient):
    px = [[GEAR_C if (r in (0, 3) or c in (0, 3)) and (r + c) % 2 == 0 else 0
           for c in range(CELL)] for r in range(CELL)]
    px[1][1] = px[2][2] = GEAR_C
    if orient == 1:
        px[1][2] = GEAR_C
    else:
        px[2][1] = GEAR_C
    return px


def _piece_px(kind, orient):
    return _slash_px(orient) if kind == 'b' else _gear_px(orient)


def _build_level(index, spec):
    sprites = []
    for r in range(GRID):
        for c in range(GRID):
            edge = r in (0, GRID - 1) or c in (0, GRID - 1)
            if edge and (c, r) != spec["goal"] and (c, r) != (0, 0):
                sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=c * CELL, y=r * CELL,
                                      blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
    gc, gr = spec["goal"]
    sprites.append(Sprite(_rect(GOAL), name="goal", x=gc * CELL, y=gr * CELL, layer=0,
                          blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    sprites.append(Sprite(_rect(BUTTON), name="launch", x=0, y=0, layer=2,
                          blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    bc, br = spec["ball"]
    sprites.append(Sprite(_rect(BALL), name="ball", x=bc * CELL, y=br * CELL, layer=3,
                          blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    for i, (kind, orient, c, r) in enumerate(spec["pieces"]):
        sprites.append(Sprite(_piece_px(kind, orient), name=f"piece_{i}", x=c * CELL, y=r * CELL,
                              layer=1, blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


def simulate(spec, placements):
    """Ball physics: True if the ball reaches the goal gap in the bottom
    wall. Boards shift the falling ball 1 column ('/'=left, chr(92)=right),
    gears shift it 2 columns (orient 0=left, 1=right); shifted cell must be
    free and inside the fairway. Max 64 ticks."""
    occ = {(c, r): (kind, orient) for kind, orient, c, r in placements}
    goal = spec["goal"]
    c, r = spec["ball"]
    for _ in range(64):
        below = (c, r + 1)
        if below == goal:
            return True
        if below[1] >= GRID - 1:
            return False
        piece = occ.get(below)
        if piece is None:
            if below[0] <= 0 or below[0] >= GRID - 1:
                return False
            c, r = below
            continue
        kind, orient = piece
        shift = (1 if kind == 'b' else 2) * (-1 if orient == 0 else 1)
        nxt = (below[0] + shift, below[1])
        if nxt == goal:
            return True
        if nxt[0] <= 0 or nxt[0] >= GRID - 1 or nxt[1] >= GRID - 1 or nxt in occ:
            return False
        c, r = nxt
    return False


class Rg01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, spec) for i, spec in enumerate(LEVELS)]
        super().__init__(
            game_id="rg01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )
        self._pieces = {}
        self._selected = None
        self._load_pieces()

    def on_set_level(self, level):
        self._selected = None
        self._load_pieces()

    def _load_pieces(self):
        self._pieces = {i: list(p) for i, p in enumerate(LEVELS[self.level_index]["pieces"])}

    def _sprite(self, name):
        found = self.current_level.get_sprites_by_name(name)
        return found[0] if found else None

    def _sync(self):
        for i, (kind, orient, c, r) in self._pieces.items():
            s = self._sprite(f"piece_{i}")
            if s is not None:
                s.set_position(c * CELL, r * CELL)
                s.pixels[:] = 0
                px = _piece_px(kind, orient)
                for rr in range(CELL):
                    for cc in range(CELL):
                        s.pixels[rr][cc] = px[rr][cc]
        sel = self._sprite("selbox")
        if self._selected is None:
            if sel is not None:
                self.current_level.remove_sprite(sel)
        else:
            _, _, c, r = self._pieces[self._selected]
            box = [[SEL if rr in (0, CELL - 1) or cc in (0, CELL - 1) else 0
                    for cc in range(CELL)] for rr in range(CELL)]
            if sel is None:
                self.current_level.add_sprite(
                    Sprite(box, name="selbox", x=c * CELL, y=r * CELL, layer=4,
                           blocking=BlockingMode.NOT_BLOCKED, collidable=False))
            else:
                sel.set_position(c * CELL, r * CELL)

    def step(self) -> None:
        spec = LEVELS[self.level_index]
        action = self.action.id
        if action == GameAction.ACTION6:
            x = int(self.action.data.get("x", 0)) // 8
            y = int(self.action.data.get("y", 0)) // 8
            if (x, y) == (0, 0):
                # LAUNCH
                if simulate(spec, [tuple(p) for p in self._pieces.values()]):
                    ball = self._sprite("ball")
                    gc, gr = spec["goal"]
                    if ball is not None:
                        ball.set_position(gc * CELL, gr * CELL)
                    self.next_level()
                else:
                    bc, br = spec["ball"]
                    ball = self._sprite("ball")
                    if ball is not None:
                        ball.set_position(bc * CELL, br * CELL)
                self.complete_action()
                return
            hit = None
            for i, (kind, orient, c, r) in self._pieces.items():
                if (c, r) == (x, y):
                    hit = i
            self._selected = hit
            self._sync()
            self.complete_action()
            return
        if self._selected is not None and action in (
                GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4):
            d = {GameAction.ACTION1: (0, -1), GameAction.ACTION2: (0, 1),
                 GameAction.ACTION3: (-1, 0), GameAction.ACTION4: (1, 0)}[action]
            kind, orient, c, r = self._pieces[self._selected]
            nc, nr = c + d[0], r + d[1]
            occupied = {(pc, pr) for j, (_, _, pc, pr) in self._pieces.items() if j != self._selected}
            if (0 < nc < GRID - 1 and 0 < nr < GRID - 1
                    and (nc, nr) not in occupied
                    and (nc, nr) != spec["ball"]):
                self._pieces[self._selected] = [kind, orient, nc, nr]
                self._sync()
            self.complete_action()
            return
        if self._selected is not None and action == GameAction.ACTION5:
            kind, orient, c, r = self._pieces[self._selected]
            self._pieces[self._selected] = [kind, 1 - orient, c, r]
            self._sync()
            self.complete_action()
            return
        self.complete_action()

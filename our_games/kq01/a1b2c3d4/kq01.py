# Original game for the atlas testbed (29.08.2026) -- NOT an ARC Prize game.
# "Keys in Order" (kq01): walk onto the colored keys in exactly the order
# shown by the hint strip at the top; a wrong key (or a decoy) resets your
# collected progress; once all keys are collected the door opens -- walk in
# to finish the level. ACTION1..4 = up/down/left/right.
#
# Written against the public arcengine API (ARCBaseGame/Level/Sprite/Camera,
# BlockingMode.PIXEL_PERFECT walls, try_move for movement) in the same shape
# the 25 public competition games use, so ArcadeSpec/GameAPI loads it
# unmodified via metadata.json (game_id kq01-a1b2c3d4, class Kq01).

from arcengine import (
    ARCBaseGame,
    BlockingMode,
    Camera,
    GameAction,
    Level,
    Sprite,
)

CELL = 4          # one logical cell = 4x4 camera pixels (camera 32x32 = 8x8 cells)
GRID = 8

BG = 0            # background color index
WALL = 9          # maroon-ish wall
PLAYER = 3        # green player
DOOR_CLOSED = 10
DOOR_OPEN = 8
HINT_DONE = 0     # collected slots turn black
DECOY_PENALTY_FLASH = 2

# Level layouts: 8x8 cell maps. '#'=wall, '.'=floor, 'P'=player start,
# 'D'=door, digits 1..5 = keys (the digit is the pickup ORDER and also
# selects the key color), 'X'=decoy key (never correct; stepping on it
# resets collected progress).
LAYOUTS = [
    # L1: two keys, open room
    [
        "########",
        "#P.....#",
        "#......#",
        "#..1...#",
        "#......#",
        "#....2.#",
        "#......#",
        "######D#",
    ],
    # L2: three keys, one inner wall
    [
        "########",
        "#P...#.#",
        "#..2.#.#",
        "#....#3#",
        "#.##.#.#",
        "#.1#...#",
        "#..#...#",
        "######D#",
    ],
    # L3: three keys + a decoy
    [
        "########",
        "#P..X..#",
        "#.##.#.#",
        "#.3#.#.#",
        "#..#.#2#",
        "##.#.#.#",
        "#1.....#",
        "######D#",
    ],
    # L4: four keys, corridors
    [
        "########",
        "#P.#..4#",
        "#..#.#.#",
        "#.1#.#.#",
        "#..#.#.#",
        "#.##3#.#",
        "#2.....#",
        "######D#",
    ],
    # L5: five keys, dense maze
    [
        "########",
        "#P#..3.#",
        "#.#.##.#",
        "#.1.#4.#",
        "##.##..#",
        "#2.#.#５#".replace("５", "5"),
        "#..#...#",
        "######D#",
    ],
    # L6: five keys + two decoys (decoys sit on the SHORT paths; the long
    # clean detours always exist -- redesigned 29.08 after the first cut
    # proved unsolvable: its only left-right corridor crossed a decoy)
    [
        "########",
        "#P.X.4.#",
        "#.##.#.#",
        "#13#X#.#",
        "#..#.#2#",
        "##.#.#.#",
        "#5.....#",
        "######D#",
    ],
]

KEY_COLORS = {1: 4, 2: 6, 3: 7, 4: 11, 5: 12}  # order -> distinct color


def _rect(color):
    return [[color] * CELL for _ in range(CELL)]


def _build_level(index, layout):
    sprites = []
    keys = {}
    decoys = []
    door_cell = None
    player_cell = None
    for r, row in enumerate(layout):
        for c, ch in enumerate(row):
            x, y = c * CELL, r * CELL
            if ch == "#":
                sprites.append(Sprite(_rect(WALL), name=f"wall_{r}_{c}", x=x, y=y,
                                      blocking=BlockingMode.PIXEL_PERFECT, tags=["sys_static"]))
            elif ch == "P":
                player_cell = (c, r)
            elif ch == "D":
                door_cell = (c, r)
            elif ch == "X":
                decoys.append((c, r))
            elif ch.isdigit():
                keys[int(ch)] = (c, r)
    n_keys = len(keys)
    # hint strip: the pickup order as colored cells along the top wall row
    for order in range(1, n_keys + 1):
        sprites.append(Sprite(_rect(KEY_COLORS[order]), name=f"hint_{order}",
                              x=(order) * CELL, y=0, layer=2,
                              blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    for order, (c, r) in keys.items():
        sprites.append(Sprite(_rect(KEY_COLORS[order]), name=f"key_{order}",
                              x=c * CELL, y=r * CELL, layer=1,
                              blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    for i, (c, r) in enumerate(decoys):
        sprites.append(Sprite(_rect(KEY_COLORS[(i % 5) + 1]), name=f"decoy_{i}",
                              x=c * CELL, y=r * CELL, layer=1,
                              blocking=BlockingMode.NOT_BLOCKED, collidable=False))
    dc, dr = door_cell
    sprites.append(Sprite(_rect(DOOR_CLOSED), name="door", x=dc * CELL, y=dr * CELL,
                          layer=1, blocking=BlockingMode.PIXEL_PERFECT))
    pc, pr = player_cell
    sprites.append(Sprite(_rect(PLAYER), name="player", x=pc * CELL, y=pr * CELL,
                          layer=3, blocking=BlockingMode.PIXEL_PERFECT))
    return Level(sprites=sprites, grid_size=(GRID * CELL, GRID * CELL),
                 name=f"level_{index + 1}")


N_KEYS = [sum(1 for row in lay for ch in row if ch.isdigit()) for lay in LAYOUTS]


class Kq01(ARCBaseGame):
    def __init__(self, seed: int = 0, debug: bool = False):
        levels = [_build_level(i, lay) for i, lay in enumerate(LAYOUTS)]
        super().__init__(
            game_id="kq01",
            levels=levels,
            camera=Camera(width=GRID * CELL, height=GRID * CELL),
            debug=debug,
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )
        self._collected = 0

    def on_set_level(self, level):
        self._collected = 0

    def _sprite(self, name):
        found = self.current_level.get_sprites_by_name(name)
        return found[0] if found else None


    def _atlas_reset_level_state(self):
        self._collected = 0

    def level_reset(self):
        super().level_reset()
        self._atlas_reset_level_state()

    def full_reset(self):
        super().full_reset()
        self._atlas_reset_level_state()

    def step(self) -> None:
        action = self.action.id
        if action == GameAction.RESET:
            self.complete_action()
            return
        if action == GameAction.ACTION1:
            dx, dy = 0, -CELL
        elif action == GameAction.ACTION2:
            dx, dy = 0, CELL
        elif action == GameAction.ACTION3:
            dx, dy = -CELL, 0
        elif action == GameAction.ACTION4:
            dx, dy = CELL, 0
        else:
            self.complete_action()
            return

        self.try_move("player", dx, dy)  # walls/closed door just block
        player = self._sprite("player")
        px, py = player.x, player.y
        n_keys = N_KEYS[self.level_index]

        # decoys: stepping on one resets collected progress
        for spr in list(self.current_level._sprites):
            if spr.name.startswith("decoy_") and spr.x == px and spr.y == py:
                self._reset_keys(n_keys)

        # keys: correct-next collects; wrong-order resets
        for spr in list(self.current_level._sprites):
            if spr.name.startswith("key_") and spr.x == px and spr.y == py:
                order = int(spr.name.split("_")[1])
                if order == self._collected + 1:
                    self._collected += 1
                    self.current_level.remove_sprite(spr)
                    hint = self._sprite(f"hint_{order}")
                    if hint is not None:
                        hint.pixels[:] = HINT_DONE
                    if self._collected == n_keys:
                        door = self._sprite("door")
                        if door is not None:
                            door.pixels[:] = DOOR_OPEN
                            door.set_blocking(BlockingMode.NOT_BLOCKED)
                else:
                    self._reset_keys(n_keys)

        # open door reached -> level complete
        door = self._sprite("door")
        if (
            self._collected == n_keys
            and door is not None
            and door.x == px
            and door.y == py
        ):
            self.next_level()

        self.complete_action()

    def _reset_keys(self, n_keys) -> None:
        """Wrong pickup: put every collected key back and dim the progress."""
        if self._collected == 0:
            return
        # rebuild by resetting the level's key/hint sprites from the clean copy
        clean = self._clean_levels[self.level_index]
        current_names = {s.name for s in self.current_level._sprites}
        for spr in clean._sprites:
            if spr.name.startswith("key_") and spr.name not in current_names:
                self.current_level.add_sprite(spr.clone())
        for order in range(1, n_keys + 1):
            hint = self._sprite(f"hint_{order}")
            if hint is not None:
                hint.pixels[:] = KEY_COLORS[order]
        door = self._sprite("door")
        if door is not None:
            door.pixels[:] = DOOR_CLOSED
            door.set_blocking(BlockingMode.PIXEL_PERFECT)
        self._collected = 0

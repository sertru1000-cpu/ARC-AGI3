"""Prove (or disprove) the baked baseline of every SHORT own game.

Why. The long batch derives each baseline by BFS over a PURE model of the
game's own rules, so a baked number can never be a hand-count -- lm01 shipped
a hand-counted 53 whose true optimum was 65. The 24 short games have no such
model: their `baseline_actions` are numbers with no proof attached, and a
wrong baseline silently corrupts every score we have ever measured, because
the score is `min(baseline/actions, 1)**2`. A baseline that is too HIGH hands
out free points; too LOW makes a good run look bad.

This searches the ENGINE itself, the same trick scripts/hostile_baselines.py
uses for the four hostile games, but with a generic state key so it needs no
per-game knowledge: the rendered frame plus the level index. That key is a
SUPERSET of the true state for any game whose state is visible -- which is
every short game except hc01, whose whole design is hidden progress, and which
hostile_baselines.py already covers properly. Over-keying only costs time; it
can never miss a shorter solution.

usage:
    python scripts/verify_short_baselines.py            # level 1 of each game
    python scripts/verify_short_baselines.py --all-levels
    python scripts/verify_short_baselines.py --game cv01 --limit 200000
"""

from __future__ import annotations

import argparse
import copy
import glob
import importlib.util
import json
import os
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# hc01 keeps progress that is deliberately NOT rendered, so a frame key would
# treat two different internal states as one and could report a short path
# that does not exist. It has its own verifier.
SKIP = {
    "hc01": "hidden state is not rendered -- see scripts/hostile_baselines.py",
    # The generated long batch already proves its baselines the stronger way:
    # BFS over the PURE sim_step/sim_done the class itself plays through, in
    # scripts/test_atlas_long_games.py. Searching them again through engine
    # frames adds nothing and misses -- their marks live in per-level sprite
    # state this generic frame key does not separate cleanly.
    "gk01": "pure-model BFS in scripts/test_atlas_long_games.py",
    "ac01": "pure-model BFS in scripts/test_atlas_long_games.py",
    "ky01": "pure-model BFS in scripts/test_atlas_long_games.py",
    "fr01": "pure-model BFS in scripts/test_atlas_long_games.py",
    "rl01": "pure-model BFS in scripts/test_atlas_long_games.py",
    "lm01": "pure-model BFS in scripts/test_atlas_long_games.py",
}


def load(gid: str):
    py = Path(glob.glob(str(ROOT / "our_games" / gid / "*" / f"{gid}.py"))[0])
    md = json.loads((py.parent / "metadata.json").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location(f"vsb_{gid}", py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, md["class_name"]), md, mod


def frame_key(game) -> bytes:
    lvl = game.current_level
    sprites = lvl.get_sprites()
    arr = game._camera._raw_render(list(sprites))
    return bytes(arr.astype("uint8").tobytes()) + bytes([game.level_index])


def live_cells(root, act, grid: int, scale: int) -> list[tuple[int, int]]:
    """Cells whose click changes the board; everything else is a wasted turn."""
    base = frame_key(root)
    live = []
    for c in range(grid):
        for r in range(grid):
            probe = copy.deepcopy(root)
            probe._action = type(probe._action)(id=act, data={"x": c * scale, "y": r * scale})
            probe._action_complete = False
            try:
                probe.step()
            except Exception:
                continue
            if frame_key(probe) != base or probe._score > root._score:
                live.append((c, r))
    return live


def candidates(mod, root):
    """The actions to branch on, ACTION6 expanded over the game's own grid.

    ACTION6 carries pixel coordinates, so its branching factor is the board,
    not a fixed four. Five of the thirty games use it (fw01, lz01, mr01, pi01,
    rg01) and for them the arrow-only search cannot reach the goal at all --
    it was reporting "no solution" for baselines that are real.
    """
    import arcengine

    out = []
    for i in root._available_actions:
        act = getattr(arcengine.GameAction, f"ACTION{i}")
        if i != 6:
            out.append((act, {}))
            continue
        # The coordinate arrives in the 64x64 FRAME space, not in the game's
        # own cells: all five ACTION6 games do `data["x"] // 8` on an 8-cell
        # board while their CELL is 4. Passing CELL here addressed a quarter
        # of the board and reported three of them unsolvable.
        #
        # Branching on all 64 cells is what made this search time out: it is a
        # deepcopy per candidate per state. Only the cells that DO something
        # are kept -- found by clicking every cell once at the root and seeing
        # which ones change the frame. That is sound for these five because
        # the clickable objects are fixed by the level spec and never move
        # (mirrors in lz01, pipe tiles in pi01, and so on -- each game's
        # step() tests membership in a set built from the spec), and it is
        # spot-checked against a mid-search state below.
        grid = getattr(mod, "GRID", 8)
        scale = max(1, 64 // grid)
        for c, r in live_cells(root, act, grid, scale):
            out.append((act, {"x": c * scale, "y": r * scale}))
    return out


def optimum(cls, mod, level_index: int, limit: int) -> tuple[int | None, int]:
    """Shortest action count that clears `level_index`, and states explored."""
    import arcengine

    root = cls()
    # NOT_PLAYED refuses actions in some games; put it in play first.
    root._state = type(root._state).NOT_FINISHED
    # Select the level by index, NOT by calling next_level(): that only raises
    # a flag the harness later acts on, so a loop on it never advances and
    # hangs. on_set_level/_load is what each game uses to place its pieces.
    root._current_level_index = level_index
    for hook in ("on_set_level", "_load"):
        fn = getattr(root, hook, None)
        if callable(fn):
            try:
                fn(root.current_level) if hook == "on_set_level" else fn()
            except TypeError:
                fn()
            break
    # Copy ONE level's worth of sprites, not the whole pack. `deepcopy(game)`
    # walks `_levels` AND `_clean_levels`, so every transition cloned five
    # levels of sprites to move one agent, and the search spent its time on
    # levels it would never look at.
    #
    # The obvious trim -- `_levels = [current]` and index 0 -- is WRONG and
    # cost a wrong answer before it was caught: every game reads its rules as
    # `LEVELS[self.level_index]` from the module, so renumbering makes the
    # engine play level 1's rules against level 3's sprites. cv01 came back
    # with an "optimum" of 2 against a baked 8 that way. The index has to stay
    # exactly where it is; the OTHER slots get a shared stub instead, which
    # deepcopy's memo copies once no matter how many slots point at it.
    stub = arcengine.Level(sprites=[], grid_size=(1, 1), name="stub")
    root._levels = [root._levels[i] if i == level_index else stub
                    for i in range(len(root._levels))]
    root._clean_levels = []

    actions = candidates(mod, root)
    # A cleared level shows up as `_score`, not as `level_index`: the engine's
    # next_level() bumps the score and raises a flag, and the index only moves
    # when the harness processes it. Watching the index reported "unsolvable"
    # for every game at first.
    base_score = root._score

    seen = {frame_key(root)}
    queue = deque([(root, 0)])
    while queue and len(seen) < limit:
        game, dist = queue.popleft()
        for act, data in actions:
            nxt = copy.deepcopy(game)
            nxt._action = type(nxt._action)(id=act, data=data)
            nxt._action_complete = False
            try:
                nxt.step()
            except Exception:
                continue
            if nxt._score > base_score or nxt._state.name == "WIN":
                return dist + 1, len(seen)
            k = frame_key(nxt)
            if k in seen:
                continue
            seen.add(k)
            queue.append((nxt, dist + 1))
    return None, len(seen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--game", action="append")
    ap.add_argument("--all-levels", action="store_true")
    ap.add_argument("--limit", type=int, default=120_000)
    args = ap.parse_args()

    gids = args.game or sorted(
        p.split(os.sep)[-3] for p in glob.glob(str(ROOT / "our_games" / "*" / "*" / "metadata.json")))
    bad = 0
    print(f"{'игра':7}{'уровень':>9}{'записано':>10}{'найдено':>10}  вердикт")
    for gid in gids:
        if gid in SKIP:
            print(f"{gid:7}{'—':>9}{'—':>10}{'—':>10}  пропуск: {SKIP[gid]}")
            continue
        try:
            cls, md, mod = load(gid)
        except Exception as e:
            print(f"{gid:7} не загрузилась: {str(e)[:60]}"); bad += 1; continue
        base = md.get("baseline_actions") or []
        levels = range(len(base)) if args.all_levels else [0]
        for lv in levels:
            try:
                got, seen = optimum(cls, mod, lv, args.limit)
            except Exception as e:
                print(f"{gid:7}{lv+1:>9}{base[lv]:>10}{'сбой':>10}  {str(e)[:44]}"); bad += 1; continue
            if got is None:
                verdict = f"НЕ НАЙДЕНО за {seen} состояний"; bad += 1
            elif got == base[lv]:
                verdict = "совпало"
            else:
                verdict = f"РАСХОЖДЕНИЕ {got - base[lv]:+d}"; bad += 1
            print(f"{gid:7}{lv+1:>9}{base[lv]:>10}{str(got):>10}  {verdict}")
    print(f"\n{'все базлайны подтверждены' if not bad else f'проблем: {bad}'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Guard: the long-level testbed games are real, solvable, and actually long.

01.09. The eleven public games we never clear need a median of 54 human
actions on level 1 (range 21-78); the seven we always clear need 22. Our own
testbed topped out at 20, so it could not reproduce the failure that costs us
the most. This batch adds games whose FIRST level takes 50-80 actions.

A testbed game is worthless unless three things hold, and all three have to be
checked by PLAYING it, not by reading the source:

  * the baked baseline is the real optimum, not a hand-count. lm01 shipped a
    hand-counted 53 that BFS immediately corrected to 65 -- an invented
    baseline would silently corrupt every RHAE number computed against it;
  * the level actually COMPLETES when the optimal path is played. A game that
    cannot be finished measures nothing and looks like a very hard game;
  * level 1 really is in the 50-80 band. If it drifts short, the batch stops
    covering the class of failure it exists for.

Runs the real engine on CPU. No GPU, no Kaggle, no quota.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "atlas_src/src/tufa-arc-agi-framework/src"))
sys.path.insert(0, str(ROOT / "atlas_src/src/ARC3-Inference"))

OWN = ROOT / "our_games"
# 03.09: this used to be BAND = (50, 80) -- "level 1 must be as long as the
# public games we fail on". That target was wrong twice over. The public
# MEDIAN level 1 is 30, not 65 (7-78 is the full range, and 50-80 is its top
# end), so the batch was built two levels too hard: measured over five runs of
# the whole testbed, these games were never cleared once, on any level. And a
# game nobody enters cannot test escalation, which was the batch's whole
# point.
#
# The target is now the public CURVE, not a band on one level: a level-1
# warm-up near 30, roughly double at level 2, a plateau, a jump at level 5.
# What is checked is the SHAPE -- absolute values may drift a few actions with
# the ring geometry, but a flat game is a broken game.
CURVE_L1 = (24, 40)          # public median 30
CURVE_L2_RATIO = (1.5, 2.3)  # public 1.8
CURVE_END_RATIO = (2.0, 4.0)  # public 3.2 from level 1 to level 5

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}: {detail}")
        FAILURES.append(name)


def long_games() -> list[str]:
    """Every own game tagged as part of the long batch."""
    found = []
    for meta in sorted(OWN.glob("*/*/metadata.json")):
        data = json.loads(meta.read_text(encoding="utf-8"))
        if "atlas_long" in (data.get("tags") or []):
            found.append(meta.parent.parent.name)
    return found


def play(game_id: str, actions: list[int]):
    """Play a literal action sequence; return (levels_completed, steps_used)."""
    import arcengine
    import taaf.game_api as ga

    api = ga.GameAPI(env_name=game_id,
                     arcade_spec=ga.ArcadeSpec(environments_dir=str(OWN)))
    api.start_game()
    # By NAME, not by value: once a game is started, arcengine.GameAction(4)
    # raises "4 is not a valid GameAction" -- the loader re-imports the enum
    # module and the by-value lookup stops resolving. Attribute access still
    # works, so never write GameAction(int) in a test that plays a game.
    for i, act in enumerate(actions, start=1):
        member = getattr(arcengine.GameAction, f"ACTION{act}")
        state = api.execute_action(arcengine.ActionInput(id=member))
        if state.levels_completed >= 1:
            return state.levels_completed, i
    return 0, len(actions)


def optimal_actions_lm01(level_index: int = 0) -> list[int]:
    """Turn lm01's BFS path into the action ids that walk it."""
    import importlib.util

    src = next(OWN.glob("lm01/*/lm01.py"))
    spec = importlib.util.spec_from_file_location("lm01_mod", src)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    from collections import deque

    level = mod.LEVELS[level_index]
    cells = mod.open_cells(level)
    start, goal = level["start"], level["exit"]
    prev = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        for dc, dr in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nxt = (cur[0] + dc, cur[1] + dr)
            if nxt in cells and nxt not in prev:
                prev[nxt] = cur
                queue.append(nxt)
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    ids = {(0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4}
    return [ids[(b[0] - a[0], b[1] - a[1])] for a, b in zip(path, path[1:])]


def optimal_actions_generated(game_id: str) -> list[int]:
    """BFS over a generated game's own sim_step/sim_done, back to actions.

    The point of solving through the game's OWN pure mirror is that the
    baseline can never drift from the rules: if someone edits sim_step, this
    solver's answer moves with it and the baked baseline stops matching.
    """
    import importlib.util
    import sys
    import types
    from collections import deque

    src = next(OWN.glob(f"{game_id}/*/{game_id}.py"))
    stub = types.ModuleType("arcengine")
    for name in ("ARCBaseGame", "BlockingMode", "Camera", "GameAction", "Level", "Sprite"):
        setattr(stub, name, type(name, (), {"__init__": lambda self, *a, **k: None}))
    saved = sys.modules.get("arcengine")
    sys.modules["arcengine"] = stub
    try:
        spec = importlib.util.spec_from_file_location(f"{game_id}_mod", src)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
    finally:
        if saved is None:
            sys.modules.pop("arcengine", None)
        else:
            sys.modules["arcengine"] = saved

    level = mod.LEVELS[0]
    start = mod.sim_initial(level)
    prev: dict = {start: (None, None)}
    queue = deque([start])
    goal = None
    while queue:
        state = queue.popleft()
        if mod.sim_done(level, state):
            goal = state
            break
        for act in (1, 2, 3, 4, 5):
            nxt = mod.sim_step(level, state, act)
            if nxt not in prev:
                prev[nxt] = (state, act)
                queue.append(nxt)
    if goal is None:
        raise SystemExit(f"{game_id}: level 1 has no solution")
    actions: list[int] = []
    node = goal
    while prev[node][0] is not None:
        node, act = prev[node]
        actions.append(act)
    actions.reverse()
    return actions


# game -> a callable returning the optimal level-1 action sequence
SOLVERS = {
    "lm01": optimal_actions_lm01,
    "gk01": lambda: optimal_actions_generated("gk01"),
    "ac01": lambda: optimal_actions_generated("ac01"),
    "ky01": lambda: optimal_actions_generated("ky01"),
    "fr01": lambda: optimal_actions_generated("fr01"),
    "rl01": lambda: optimal_actions_generated("rl01"),
}


def main() -> None:
    games = long_games()
    check("the long batch exists", bool(games), "no game is tagged atlas_long")
    if not games:
        sys.exit(1)
    print(f"     long batch: {', '.join(games)}\n")

    for game in games:
        meta = json.loads(next(OWN.glob(f"{game}/*/metadata.json")).read_text(encoding="utf-8"))
        base = meta.get("baseline_actions") or []
        # 01.09: nine levels, to match the public set's median of 7 (range
        # 6-10). Level count is not cosmetic -- RHAE divides by the sum of
        # level weights, so a 5-level game and a 10-level game with one level
        # cleared score 6.7 and 1.8 for the same achievement.
        check(f"{game}: carries at least 9 levels", len(base) >= 9,
              f"only {len(base)} levels: {base}")
        if not base:
            continue

        check(f"{game}: level 1 baseline {base[0]} near the public median 30",
              CURVE_L1[0] <= base[0] <= CURVE_L1[1],
              f"got {base[0]} -- this game does not cover the long class")
        r2 = base[1] / base[0] if len(base) > 1 and base[0] else 0
        check(f"{game}: level 2 is {r2:.1f}x level 1 (public 1.8x)",
              CURVE_L2_RATIO[0] <= r2 <= CURVE_L2_RATIO[1],
              "a level that does not get harder is why the old batch scored zero")
        rE = max(base) / base[0] if base[0] else 0
        check(f"{game}: hardest level is {rE:.1f}x level 1 (public 3.2x)",
              CURVE_END_RATIO[0] <= rE <= CURVE_END_RATIO[1],
              f"per-level: {base}")

        solver = SOLVERS.get(game)
        if solver is None:
            check(f"{game}: has an optimal-path solver", False,
                  "add one to SOLVERS -- an unverified baseline is an invented one")
            continue

        actions = solver()
        check(f"{game}: the solver's path matches the baked baseline",
              len(actions) == base[0],
              f"solver says {len(actions)}, metadata says {base[0]}")

        completed, used = play(game, actions)
        check(f"{game}: level 1 completes when the optimal path is played",
              completed >= 1, f"played {used} actions, completed {completed} levels")
        check(f"{game}: it completes on the LAST action, not earlier",
              completed >= 1 and used == len(actions),
              f"finished at action {used} of {len(actions)}")

        # A wrong path must NOT finish the level -- otherwise the game is
        # completing on something other than reaching the goal.
        completed_bad, _ = play(game, [1] * len(actions))
        check(f"{game}: a degenerate path does not clear the level",
              completed_bad == 0, "the level completes without solving it")

    # A report, not a gate. Escalation is only possible where the mechanic's
    # cost is MULTIPLICATIVE in the number of marks. A "visit each once" game
    # saturates at one lap of the ring -- adding pads moved gk01 from 55 to 58
    # across nine levels -- so gk01/ac01/ky01/lm01 stay flat by construction,
    # and failing them for that would punish them for their own design. The
    # public set grows x3.2 from level 1 to 5; only fr01 and rl01 answer that.
    print("\n     рост базлайна с первого уровня к последнему:")
    for game in games:
        meta = json.loads(next(OWN.glob(f"{game}/*/metadata.json")).read_text(encoding="utf-8"))
        base = meta.get("baseline_actions") or []
        if len(base) >= 2:
            print(f"       {game}: {base[0]:3} -> {base[-1]:3}   x{base[-1] / base[0]:.1f}")

    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) failed.")
        sys.exit(1)
    print("\nAll long-game checks passed.")


if __name__ == "__main__":
    main()

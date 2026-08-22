"""Unit test for C1: planning over the model's own verified theory (22.08).

Until now a world model was only ever a key to the action-batch gate. The
model tuned predict() until verify_theory reached 0.6, the gate opened, and
the verified theory was discarded -- while bfs_path planned over a STATIC grid
with a passability predicate, i.e. nothing to do with learned dynamics.

plan_with_theory closes that loop: breadth-first search over the states
predict() forecasts, costing zero engine actions. Actions are scored
(human/ours)^2, so a searched plan is worth far more than the same moves found
by trial and error.

The fixture is a tiny deterministic game -- a dot that moves one cell per
direction key -- so the "theory" is exact and the correct plan is known by
construction.

Run:  .venv/Scripts/python.exe scripts/test_plan_with_theory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

from agent.harness.sandbox import Sandbox

DOT = 3
SIZE = 8


def move(grid: np.ndarray, action: str, data=None) -> np.ndarray:
    """The real mechanic: the dot steps one cell, walls block."""
    g = grid.copy()
    (r,), (c,) = np.where(g == DOT)[0], np.where(g == DOT)[1]
    dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}.get(action, (0, 0))
    nr, nc = min(max(r + dr, 0), SIZE - 1), min(max(c + dc, 0), SIZE - 1)
    g[r, c] = 0
    g[nr, nc] = DOT
    return g


def make_sandbox(start_rc=(4, 4)) -> Sandbox:
    grid = np.zeros((SIZE, SIZE), dtype=np.int8)
    grid[start_rc] = DOT
    sb = Sandbox(env_step=lambda *a: None, budget_left=lambda: 100)

    class _Frame:
        def __init__(self, g):
            self.grid = g
            self.level = 0
            self.ascii = ""
    sb.current = _Frame(grid)
    sb.valid_actions = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE"]
    # Recorded history the verifier replays: a few real transitions.
    for act in ("UP", "RIGHT", "DOWN", "LEFT", "RIGHT", "RIGHT"):
        before = sb.current.grid
        after = move(before, act)
        sb.transition_log.append((before, act, None, after))
        sb.current = _Frame(after)
    sb.current = _Frame(grid)  # plan from the original position
    return sb


def at(rc):
    return lambda g: bool(g[rc] == DOT)


def check(cond: bool, msg: str) -> None:
    print(f"[{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def main() -> None:
    sb = make_sandbox()

    # --- a correct theory plans a correct, MINIMAL route -------------------
    res = sb._plan_with_theory(move, at((4, 7)), actions=["UP", "DOWN", "LEFT", "RIGHT"])
    check(res["plan"] == ["RIGHT", "RIGHT", "RIGHT"], f"exact theory finds the shortest plan: {res['plan']}")
    check(res["verified_accuracy"] == 1.0, "the theory verified at 1.0 before planning")
    check(res["depth"] == 3, "breadth-first gives the minimal depth, not just any route")

    res = sb._plan_with_theory(move, at((1, 4)), actions=["UP", "DOWN", "LEFT", "RIGHT"])
    check(res["plan"] == ["UP", "UP", "UP"], f"plans in another direction too: {res['plan']}")

    # --- the plan is executable by action() verbatim ----------------------
    check(all(isinstance(s, str) for s in res["plan"]), "plan items are action() specs")

    # --- already at the goal -> empty plan, not None ----------------------
    res = sb._plan_with_theory(move, at((4, 4)), actions=["UP", "DOWN"])
    check(res["plan"] == [], "already at the goal returns an empty plan")

    # --- a WRONG theory is refused, not silently planned over -------------
    def wrong(grid, action, data=None):
        return grid  # claims nothing ever changes

    res = sb._plan_with_theory(wrong, at((4, 7)), actions=["UP", "DOWN", "LEFT", "RIGHT"])
    check(res["plan"] is None, "an unverified theory yields no plan")
    check(res["verified_accuracy"] == 0.0, "its accuracy is reported")
    check("not good enough" in res["reason"], "and the refusal explains itself")

    # --- unreachable goal is reported as such, without crashing -----------
    res = sb._plan_with_theory(move, lambda g: bool((g == 9).any()),
                               actions=["UP", "DOWN", "LEFT", "RIGHT"], max_depth=3)
    check(res["plan"] is None, "an unreachable goal yields no plan")
    check("exhausted" in res["reason"] or "budget spent" in res["reason"],
          f"and says why: {res['reason'][:60]}")

    # --- budget caps hold -------------------------------------------------
    res = sb._plan_with_theory(move, lambda g: False,
                               actions=["UP", "DOWN", "LEFT", "RIGHT"],
                               max_depth=8, max_nodes=25)
    check(res["nodes_expanded"] <= 30, f"max_nodes is respected ({res['nodes_expanded']})")

    # --- a predict() that raises must not kill the search -----------------
    def flaky(grid, action, data=None):
        if action == "LEFT":
            raise ValueError("boom")
        return move(grid, action, data)

    res = sb._plan_with_theory(flaky, at((4, 7)),
                               actions=["UP", "DOWN", "LEFT", "RIGHT"], min_accuracy=0.5)
    check(res["plan"] == ["RIGHT", "RIGHT", "RIGHT"], "a raising branch is skipped, not fatal")
    check(res["predict_errors"] > 0, "and the errors are counted")

    # --- goal() raising is reported, not swallowed ------------------------
    def bad_goal(g):
        raise RuntimeError("nope")

    res = sb._plan_with_theory(move, bad_goal, actions=["UP"])
    check(res["plan"] is None and "goal() raised" in res["reason"],
          "a broken goal() is reported clearly")

    # --- planning must not become a back door through the action gate -----
    # verify_gate_open() also unlocks on effort (verify_attempts >= 3), so if
    # the internal check counted, three throwaway plan calls would open the
    # gate for long action() batches without any working theory.
    sb3 = make_sandbox()
    for _ in range(4):
        sb3._plan_with_theory(wrong, at((4, 7)), actions=["UP", "RIGHT"])
    check(sb3.verify_attempts == 0, f"planning does not spend gate attempts ({sb3.verify_attempts})")
    check(not sb3.verify_gate_open(), "a bad theory cannot open the gate by planning repeatedly")
    sb3._verify_theory(move)
    check(sb3.verify_gate_open(), "an accurate theory still opens the gate normally")

    # --- reachable from model code under the documented name --------------
    # The prompt tells the model to call plan_with_theory(...); if it is not
    # in the sandbox scope that instruction is a lie.
    sb2 = make_sandbox()
    res = sb2.run_code(
        "def predict(grid, action, data=None):\n"
        "    g = grid.copy()\n"
        "    rs, cs = np.where(g == 3)\n"
        "    r, c = int(rs[0]), int(cs[0])\n"
        "    d = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}\n"
        "    dr, dc = d.get(action, (0, 0))\n"
        "    nr = min(max(r + dr, 0), 7)\n"
        "    nc = min(max(c + dc, 0), 7)\n"
        "    g[r, c] = 0\n"
        "    g[nr, nc] = 3\n"
        "    return g\n"
        "\n"
        "res = plan_with_theory(predict, lambda g: bool(g[4, 7] == 3),\n"
        "                       actions=['UP', 'DOWN', 'LEFT', 'RIGHT'])\n"
        "print('PLAN', res['plan'])\n"
    )
    check(res.error is None, f"model code can call it (error: {res.error})")
    check("PLAN ['RIGHT', 'RIGHT', 'RIGHT']" in (res.output or ""),
          f"and gets the plan back: {(res.output or '').strip()[:60]}")

    print("\nplan_with_theory behaves as specified.")


if __name__ == "__main__":
    main()

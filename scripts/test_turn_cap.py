"""Unit test: per-turn action cap (21.08) -- a model loop of many short
action() calls inside ONE code block must stop at MY_AGENT_TURN_ACTION_CAP
real actions, even if the loop wraps action() in `except Exception`, and the
model gets a TurnActionCap error; the NEXT block starts with a fresh cap.

Run:  .venv/Scripts/python.exe scripts/test_turn_cap.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MY_AGENT_TURN_ACTION_CAP"] = "10"

from agent.harness.sandbox import FrameView, Sandbox  # noqa: E402


class FakeState:
    def __init__(self, name): self._name = name
    def __str__(self): return f"GameState.{self._name}"


class FakeFrame:
    def __init__(self, grid, level=0):
        self.frame = [grid.tolist()]; self.levels_completed = level
        self.state = FakeState("NOT_FINISHED"); self.available_actions = [1, 2, 3, 4]


def main() -> None:
    grid = np.zeros((4, 4), dtype=np.int8)
    calls = {"n": 0}

    def env_step(engine, payload):
        calls["n"] += 1
        g = grid.copy(); g[0, 0] = calls["n"] % 16  # board changes every step
        return FakeFrame(g)

    sb = Sandbox(env_step=env_step, budget_left=lambda: 800 - calls["n"])
    sb.current = FrameView(grid, step=0, level=0)

    # Greedy loop with error swallowing: 300 single-action calls.
    code = """
for i in range(300):
    try:
        action(['UP'])
    except Exception as e:
        print('swallowed', e)
print('loop done')
"""
    res = sb.run_code(code)
    assert calls["n"] == 10, f"expected exactly 10 engine calls, got {calls['n']}"
    assert res.actions_executed == 10, res.actions_executed
    assert res.error and "TurnActionCap" in res.error, res.error
    assert "swallowed" not in res.output, "model's except Exception caught the unwind!"
    assert "loop done" not in res.output
    print("cap hit at 10, unwind not swallowable, error explains it: ok")

    # Next block: fresh cap, normal batch works.
    sb.last_verify = {"accuracy": 1.0, "tested": 10}  # open the verify gate so a 5-batch is allowed
    res2 = sb.run_code("r = action(['UP']*5); print(r['board_changed'])")
    assert res2.error is None and res2.actions_executed == 5, (res2.error, res2.actions_executed)
    print("next turn resets the cap: ok")

    # Cap off.
    sb.turn_action_cap = 0
    res3 = sb.run_code("for i in range(30): action(['UP'])")
    assert res3.error is None and res3.actions_executed == 30
    print("cap=0 disables: ok")

    # Frames WITHOUT a grid (seen after GAME_OVER on tr87) must still count
    # toward the cap -- step_counter doesn't advance for them, engine calls do.
    sb.turn_action_cap = 10
    calls["n"] = 0

    class EmptyFrame:
        frame = []  # no grid at all
        levels_completed = 0
        state = FakeState("NOT_FINISHED")
        available_actions = [1, 2, 3, 4]

    sb.env_step = lambda engine, payload: (calls.__setitem__("n", calls["n"] + 1), EmptyFrame())[1]
    sb.last_verify = {"accuracy": 1.0, "tested": 10}
    res4 = sb.run_code("for i in range(300): action(['UP'])")
    assert calls["n"] == 10, f"gridless frames bypassed the cap: {calls['n']} engine calls"
    assert res4.error and "TurnActionCap" in res4.error
    print("gridless frames still hit the cap: ok")

    # GAME_OVER guard: after a GAME_OVER frame, non-RESET actions are refused
    # (no engine call), RESET is allowed.
    class OverFrame(EmptyFrame):
        state = FakeState("GAME_OVER")
    sb.env_step = lambda engine, payload: (calls.__setitem__("n", calls["n"] + 1), OverFrame())[1]
    calls["n"] = 0
    sb.run_code("action(['UP'])")
    assert calls["n"] == 1 and sb.game_over
    res5 = sb.run_code("for i in range(50): action(['UP'])")
    assert calls["n"] == 1, f"actions were sent while GAME_OVER: {calls['n']}"
    assert res5.error and "GAME_OVER" in res5.error, res5.error
    sb.env_step = lambda engine, payload: (calls.__setitem__("n", calls["n"] + 1), EmptyFrame())[1]
    sb.run_code("action(['RESET'])")
    assert calls["n"] == 2 and not sb.game_over
    print("GAME_OVER guard: non-RESET refused, RESET allowed: ok")
    print("TURN CAP TEST PASSED")


if __name__ == "__main__":
    main()

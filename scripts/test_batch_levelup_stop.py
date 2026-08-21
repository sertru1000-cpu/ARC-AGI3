"""Unit test: action() batches stop on level-up, not just game_over/win
(Duck parity, backlog finding 20.08 -- see sandbox.py comment near
`level_at_batch_start`). Without this, actions planned for the OLD layout
kept firing blind against the NEW level.

Run:  .venv/Scripts/python.exe scripts/test_batch_levelup_stop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.harness.sandbox import Sandbox, FrameView


class FakeState:
    def __init__(self, name: str):
        self._name = name

    def __str__(self) -> str:
        return f"GameState.{self._name}"


class FakeFrame:
    """Minimal stand-in for arcengine's FrameData."""

    def __init__(self, grid: np.ndarray, level: int, state: str = "NOT_FINISHED"):
        self.frame = [grid.tolist()]
        self.levels_completed = level
        self.state = FakeState(state)
        self.available_actions = []


def main() -> None:
    grid_lvl1 = np.zeros((4, 4), dtype=np.int8)
    grid_lvl1[0, 0] = 1
    grid_lvl2 = np.zeros((4, 4), dtype=np.int8)
    grid_lvl2[3, 3] = 2  # different layout after level-up

    # Level increments on the 3rd action; the batch asks for 6.
    call_count = {"n": 0}

    def env_step(engine_name, payload):
        call_count["n"] += 1
        if call_count["n"] < 3:
            return FakeFrame(grid_lvl1, level=0)
        if call_count["n"] == 3:
            return FakeFrame(grid_lvl2, level=1)  # level-up here
        # Should never be reached if the stop-on-levelup fix works.
        return FakeFrame(grid_lvl2, level=1)

    sb = Sandbox(env_step=env_step, budget_left=lambda: 100)
    sb.current = FrameView(grid_lvl1, step=0, level=0)

    result = sb._action(["UP", "UP", "UP", "UP", "UP", "UP"])

    print("engine calls made:", call_count["n"])
    print("result:", result)

    assert call_count["n"] == 3, (
        f"expected exactly 3 real engine actions (batch stops at the level-up), "
        f"got {call_count['n']}")
    assert result["level_completed"] is True, "result must flag level_completed"
    assert result["level"] == 1, f"result level should be the new level, got {result['level']}"

    print("\nBATCH STOPS ON LEVEL-UP: PASSED")


if __name__ == "__main__":
    main()

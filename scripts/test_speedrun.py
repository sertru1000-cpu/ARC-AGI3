"""Tests for the speedrun phase.

1. Unit: build_replay_plan distills last-attempt segments and drops no-ops.
2. Integration: record real actions on ls20, replay them on a fresh env and
   verify frame-by-frame determinism (the safety net the replay relies on).

Run:  .venv/Scripts/python.exe scripts/test_speedrun.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

from agent.harness.speedrun import ReplayStep, build_replay_plan, execute_replay  # noqa: E402


def g(v):  # tiny grid factory
    return np.full((2, 2), v, dtype=np.int8)


def test_plan() -> None:
    log = [
        {"id": 0, "data": {}, "grid": g(0), "level": 0, "state": "S"},   # seed reset
        {"id": 1, "data": {}, "grid": g(1), "level": 0, "state": "S"},   # wasted try
        {"id": 2, "data": {}, "grid": g(2), "level": 0, "state": "S"},   # wasted try
        {"id": 0, "data": {}, "grid": g(0), "level": 0, "state": "S"},   # level reset
        {"id": 3, "data": {}, "grid": g(3), "level": 0, "state": "S"},   # real move
        {"id": 3, "data": {}, "grid": g(3), "level": 0, "state": "S"},   # no-op (same grid)
        {"id": 4, "data": {}, "grid": g(4), "level": 1, "state": "S"},   # completes L1
        {"id": 5, "data": {}, "grid": g(5), "level": 1, "state": "S"},   # L2 progress
        {"id": 6, "data": {"x": 3, "y": 7}, "grid": g(6), "level": 2, "state": "W"},  # completes L2
        {"id": 1, "data": {}, "grid": g(7), "level": 2, "state": "S"},   # after-win noise
    ]
    plan = build_replay_plan(log)
    ids = [s.action_id for s in plan]
    assert ids == [3, 4, 5, 6], f"plan ids wrong: {ids}"
    assert plan[-1].data == {"x": 3, "y": 7}
    print(f"unit: plan distilled to {ids} from {len(log)} log entries — OK")


def test_determinism() -> None:
    logging.disable(logging.CRITICAL)
    import arc_agi
    from arc_agi import OperationMode
    from arcengine import GameAction

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)

    # Record a short trajectory on a fresh env.
    env1 = arc.make("ls20")
    env1.reset()
    seq = [1, 4, 4, 2, 3, 1, 4, 2]
    log = []
    for aid in seq:
        f = env1.step(GameAction.from_id(aid))
        log.append({"id": aid, "data": {}, "grid": np.asarray(f.frame[-1], dtype=np.int8),
                    "level": f.levels_completed, "state": str(f.state)})

    # Build a raw plan (keep everything incl. no-ops: we test determinism, not distillation).
    plan = [ReplayStep(e["id"], {}, e["grid"], e["level"]) for e in log]

    # Replay on a brand-new env instance.
    env2 = arc.make("ls20")
    env2.reset()
    counter = {"n": 0}

    def env_step(aid, data):
        counter["n"] += 1
        return env2.step(GameAction.from_id(aid))

    res = execute_replay(plan, env_step, budget_left=lambda: 999)
    assert res["status"] in ("done", "win"), f"replay failed: {res}"
    assert res["replayed"] == len(plan)
    print(f"integration: {len(plan)} actions replayed on fresh env, всё детерминированно — OK ({res})")


if __name__ == "__main__":
    test_plan()
    test_determinism()
    print("\nSPEEDRUN TESTS PASSED")

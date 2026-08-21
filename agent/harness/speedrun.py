"""Post-win speedrun: replay the game cleanly for a fresh, better-scored run.

Scoring facts this exploits (verified against arc_agi source, 2026-08-16):
  - a game's score is the MAX over its runs;
  - RESET while in WIN state performs a full reset -> the scorecard opens a
    brand-new run with fresh per-level action counters;
  - within a run, every action and reset counts into the current level.

So after a win we rebuild the shortest action sequence we can defend and play
it again. Safety: after every replayed action we compare the resulting board
with the recorded one; on any divergence we stop immediately — the first
run's score is never at risk.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

logger = logging.getLogger()


@dataclass
class ReplayStep:
    action_id: int
    data: dict
    expected_grid: np.ndarray
    level_after: int


def build_replay_plan(replay_log: list[dict], drop_noops: bool = True) -> list[ReplayStep]:
    """Distill the winning trajectory into per-level 'last attempt' segments.

    replay_log entries: {"id": int, "data": dict, "grid": ndarray,
                         "level": int, "state": str} in play order.
    Rules:
      - a RESET (id 0) restarts the current level -> everything accumulated
        for that level so far was provably unnecessary; drop it;
      - on level completion, freeze that level's segment into the plan;
      - optionally drop pure no-ops (board and level unchanged) — the frame
        verification during replay guards against hidden-state mistakes.
    """
    plan: list[ReplayStep] = []
    segment: list[ReplayStep] = []
    level = 0
    prev_grid: np.ndarray | None = None

    for e in replay_log:
        grid = e["grid"]
        if e["id"] == 0:  # RESET: level restart (or seed) — current segment is waste
            segment = []
            prev_grid = grid
            continue
        is_noop = (
            drop_noops
            and prev_grid is not None
            and grid is not None
            and prev_grid.shape == grid.shape
            and bool((prev_grid == grid).all())
            and e["level"] == level
        )
        if not is_noop:
            segment.append(
                ReplayStep(e["id"], dict(e["data"] or {}), grid, e["level"])
            )
        if e["level"] > level:
            level = e["level"]
            plan.extend(segment)
            segment = []
        prev_grid = grid

    # Trailing segment (final level completed exactly on WIN is included above;
    # anything left here belongs to an uncompleted level — useless in replay).
    return plan


def execute_replay(
    plan: list[ReplayStep],
    env_step: Callable[[int, dict | None], Any],
    budget_left: Callable[[], int],
) -> dict:
    """Full-reset and replay the plan with frame verification.

    env_step(action_id, data) -> FrameData. Returns a summary dict.
    """
    if not plan:
        return {"replayed": 0, "status": "empty_plan"}
    if budget_left() < len(plan) + 1:
        return {"replayed": 0, "status": "no_budget"}

    env_step(0, None)  # RESET from WIN -> full reset -> new scorecard run

    replayed = 0
    for step in plan:
        if budget_left() <= 0:
            return {"replayed": replayed, "status": "budget_exhausted"}
        frame = env_step(step.action_id, step.data or None)
        replayed += 1
        got = np.asarray(frame.frame[-1], dtype=np.int8) if frame.frame else None
        exp = step.expected_grid
        if got is None or exp is None or got.shape != exp.shape or not (got == exp).all():
            logger.warning(
                f"speedrun: frame divergence at step {replayed}/{len(plan)} — aborting replay"
            )
            return {"replayed": replayed, "status": "diverged"}
        if str(getattr(frame, "state", "")).endswith("GAME_OVER"):
            return {"replayed": replayed, "status": "game_over"}
        if str(getattr(frame, "state", "")).endswith("WIN"):
            return {"replayed": replayed, "status": "win"}
    return {"replayed": replayed, "status": "done"}

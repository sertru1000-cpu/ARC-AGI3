"""Exercise the atlas speedrun-after-win port in atlas_src's solver.py.

Constructs a _HarnessGameSession with fake game/solver/env objects (no real
Kaggle game, no GPU) and drives _atlas_parse_action_display,
_atlas_pruned_winning_actions, and _atlas_speedrun_after_win directly --
verifying: display-string parsing round-trips correctly, no-op actions get
pruned, a mocked env.step() sequence is replayed and a WIN is detected, and a
failure mid-replay is swallowed without raising (the original win must never
be put at risk by a broken replay attempt).
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "tufa-arc-agi-framework" / "src"))
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "ARC3-Inference"))

import arcengine  # noqa: E402
from inference.agent.runtime_state import Frame, HistoryEntry  # noqa: E402
from inference.framework import solver as solver_mod  # noqa: E402

_HarnessGameSession = solver_mod._HarnessGameSession


def _fail(name: str, detail: str) -> None:
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def _ok(name: str) -> None:
    print(f"ok   {name}")


def _frame(grid, level=1, step=0) -> Frame:
    return Frame(grid=tuple(tuple(row) for row in grid), step=step, level=level)


class _FakeSolver:
    max_actions_per_game = None


class _FakeGameRun:
    def __init__(self):
        self.state = "won"
        self.history: list[Any] = []


class _FakeGame:
    def __init__(self, env):
        self.env = env
        self.game_id = "test-game"
        self.game_run = _FakeGameRun()
        self.current_state = type("S", (), {"raw": type("R", (), {"state": arcengine.GameState.WIN})()})()


class _FakeResp:
    def __init__(self, *, frame=True, state=arcengine.GameState.NOT_FINISHED, full_reset=None):
        self.frame = frame
        self.state = state
        self.full_reset = full_reset


def _session(game) -> Any:
    return _HarnessGameSession(
        solver=_FakeSolver(),
        game=game,
        analyzer=None,
        game_index=0,
        pass_index=0,
        state_path=Path("unused_state.json"),
        transcript_path=Path("unused_transcript.txt"),
        analysis_html_relpath="unused.html",
        stop_event=threading.Event(),
        viewer_data_path=Path("unused_viewer.json"),
    )


def main() -> None:
    session = _session(_FakeGame(env=None))

    # 1. Parsing round-trips both plain and MOUSE display strings.
    if session._atlas_parse_action_display("UP") != ("ACTION1", {}):
        _fail("parse UP", str(session._atlas_parse_action_display("UP")))
    _ok("parses 'UP' -> ('ACTION1', {})")

    if session._atlas_parse_action_display("MOUSE(row=4, col=7)") != ("ACTION6", {"x": 7, "y": 4}):
        _fail("parse MOUSE", str(session._atlas_parse_action_display("MOUSE(row=4, col=7)")))
    _ok("parses 'MOUSE(row=4, col=7)' -> ('ACTION6', {'x': 7, 'y': 4})")

    if session._atlas_parse_action_display("RESET") is not None:
        _fail("parse RESET", "RESET must not be replayable as a scripted step")
    _ok("refuses to treat 'RESET' as a replayable action")

    # 2. Pruning drops no-op actions (no grid or level change).
    grid_a = [[0, 1], [2, 3]]
    grid_b = [[9, 1], [2, 3]]  # UP changes the grid
    session.history_entries = [
        HistoryEntry(action="", frame=_frame(grid_a)),
        HistoryEntry(action="UP", frame=_frame(grid_b)),       # real move -> kept
        HistoryEntry(action="DOWN", frame=_frame(grid_b)),     # no-op -> dropped
        HistoryEntry(action="LEFT", frame=_frame(grid_b, level=2)),  # level change -> kept
    ]
    pruned = session._atlas_pruned_winning_actions()
    if pruned != [("ACTION1", {}), ("ACTION3", {})]:
        _fail("pruning", str(pruned))
    _ok(f"prunes the no-op DOWN, keeps UP and the level-changing LEFT: {pruned}")

    # 3. End-to-end replay: RESET (full_reset) then two actions, second one wins.
    calls: list[tuple[str, dict]] = []

    def fake_step(action_id, data=None):
        calls.append((action_id.name, dict(data or {})))
        if action_id.name == "RESET":
            return _FakeResp(full_reset=True)
        if len(calls) == 2:  # first replayed action: still playing
            return _FakeResp(state=arcengine.GameState.NOT_FINISHED)
        return _FakeResp(state=arcengine.GameState.WIN)  # second: wins early

    env = type("Env", (), {"step": staticmethod(fake_step)})()
    session2 = _session(_FakeGame(env=env))
    session2.history_entries = [
        HistoryEntry(action="", frame=_frame(grid_a)),
        HistoryEntry(action="UP", frame=_frame(grid_b)),
        HistoryEntry(action="DOWN", frame=_frame(grid_b)),      # pruned
        HistoryEntry(action="LEFT", frame=_frame(grid_b, level=2)),
        HistoryEntry(action="RIGHT", frame=_frame(grid_b, level=3)),
    ]
    session2._atlas_speedrun_after_win()
    if [c[0] for c in calls] != ["RESET", "ACTION1", "ACTION3"]:
        _fail("replay sequence", str(calls))
    _ok(f"replays RESET + pruned actions, stops at WIN: {calls}")

    # 4. A mid-replay exception must be swallowed, never raised.
    def raising_step(action_id, data=None):
        if action_id.name == "RESET":
            return _FakeResp(full_reset=True)
        raise RuntimeError("engine hiccup")

    env3 = type("Env", (), {"step": staticmethod(raising_step)})()
    session3 = _session(_FakeGame(env=env3))
    session3.history_entries = session2.history_entries
    try:
        session3._atlas_speedrun_after_win()
    except Exception as exc:  # noqa: BLE001
        _fail("exception safety", f"speedrun raised instead of swallowing: {exc!r}")
    _ok("a replay exception is swallowed, not raised (original win stays safe)")

    # 5. No full_reset reported -> abort without attempting any replay steps.
    replay_calls: list[str] = []

    def no_full_reset_step(action_id, data=None):
        replay_calls.append(action_id.name)
        return _FakeResp(full_reset=False)

    env4 = type("Env", (), {"step": staticmethod(no_full_reset_step)})()
    session4 = _session(_FakeGame(env=env4))
    session4.history_entries = session2.history_entries
    session4._atlas_speedrun_after_win()
    if replay_calls != ["RESET"]:
        _fail("no full_reset abort", str(replay_calls))
    _ok("aborts immediately if RESET doesn't report full_reset (no wasted actions)")

    # 6. Not enough remaining action budget -> skip without touching env at all.
    class _TightSolver:
        max_actions_per_game = 3  # session2.action_count is 0 (fake game_run has no .history)

    def unexpected_step(action_id, data=None):
        _fail("budget guard", f"env.step() called despite insufficient budget: {action_id.name}")

    env5 = type("Env", (), {"step": staticmethod(unexpected_step)})()
    session5 = _session(_FakeGame(env=env5))
    session5.solver = _TightSolver()
    session5.history_entries = session2.history_entries  # needs reset + 2 replayed = 3 actions
    session5.solver.max_actions_per_game = 1  # only 1 left, need 3
    session5._atlas_speedrun_after_win()
    _ok("skips the replay entirely when the action budget can't cover it")

    print("\nAll atlas speedrun checks passed.")


if __name__ == "__main__":
    main()

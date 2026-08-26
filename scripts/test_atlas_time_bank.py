"""Exercise the atlas time-bank scheduler addition in atlas_src's solver.py.

25.08: every game got the SAME max_runtime_s_per_game today, uniformly,
regardless of how it's actually doing. RHAE's own per-run scoring weights
LATER levels more heavily, so a game that's already proven it can clear
levels plausibly deserves more time than one stuck at level 0 -- but a game
stuck at level 0 currently burns its whole allocation anyway, for zero score.

This adds: a session that's spent a large fraction of its budget with ZERO
level progress ends itself early and returns its unused time to a shared
bank; a session that reaches a level it hadn't been credited for yet may
draw from that bank to extend its OWN deadline, capped so one game can't
monopolize it.

v19 (25.08): the stall trigger originally ALSO required 6 consecutive real
actions with a literally unchanged grid, on top of the zero-progress
fraction gate. v18's real logs showed this combination never fires in
practice -- 0 deposits, 0 draws across a full 4h/25-game run, even though
11 of 25 games spent nearly their whole allocation stuck at level 0. Real
unproductive play still changes the board constantly (wrong clicks, etc.)
-- it just never advances the LEVEL. Dropped the literal-no-op requirement;
zero level progress past the fraction gate is sufficient evidence on its
own (raised 0.4->0.5 for a bit more margin against cutting off a
near-success, since this fires more readily now).

Also exercises the retry-storm backstop added the same day after v17's real
logs showed 3 of 25 games (cn04, lp85, re86) spending 15-30 CONSECUTIVE
MINUTES retrying the same analysis_step on repeated analyzer request
timeouts (the shared local LLM backend under concurrency=14 load), never
producing a single python tool call the whole time -- a failure mode neither
the force-act circuit breaker (tool_agent.py) nor the stall detector above
can see, since it happens before any tool call exists. Reuses the same
shared bank: enough CONSECUTIVE retryable analyzer failures ends the session
early too, same as a gameplay stall, just a different trigger.

Tests the pure decision functions directly (no Game/session needed), the
HarnessSolver bank's thread-safe accounting, and the full wiring through
_HarnessGameSession._atlas_check_time_bank()/_atlas_check_retry_storm() with
a fake game -- no real Kaggle game, no GPU, no arcengine network calls.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "tufa-arc-agi-framework" / "src"))
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "ARC3-Inference"))

from inference.agent.runtime_state import Frame, HistoryEntry  # noqa: E402
from inference.framework import solver as solver_mod  # noqa: E402

_HarnessGameSession = solver_mod._HarnessGameSession
_atlas_stall_decision = solver_mod._atlas_stall_decision
_atlas_extension_request = solver_mod._atlas_extension_request
_atlas_retry_storm_decision = solver_mod._atlas_retry_storm_decision


def _fail(name: str, detail: str) -> None:
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def _ok(name: str) -> None:
    print(f"ok   {name}")


def _frame(grid, level=1, step=0) -> Frame:
    return Frame(grid=tuple(tuple(row) for row in grid), step=step, level=level)


def _entries(specs: list[tuple[list, int]]) -> list[HistoryEntry]:
    return [HistoryEntry(action="X", frame=_frame(g, level=lvl)) for g, lvl in specs]


class _FakeState:
    def __init__(self, *, levels_completed: int, won: bool, grid):
        self.levels_completed = levels_completed
        self.won = won
        self.frame = type("F", (), {"data": grid})()


class _FakeGame:
    def __init__(self, *, levels_completed: int = 0, won: bool = False, number_of_levels: int = 8, grid=None):
        self.number_of_levels = number_of_levels
        self.current_state = _FakeState(levels_completed=levels_completed, won=won, grid=grid or [[0, 0], [0, 0]])
        self.game_run = type("R", (), {"state": "playing", "history": [], "solver_note": None})()


def _session(game, solver) -> Any:
    return _HarnessGameSession(
        solver=solver,
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
    # === Pure function checks -- no Game/session needed at all ===

    # 5. _atlas_stall_decision: refuses before the early-stop fraction of the cap.
    stop, deposit = _atlas_stall_decision(
        elapsed=100.0, cap=1000.0, start_level=1, current_level=1, already_deposited=False,
    )
    if stop:
        _fail("stall decision respects the fraction gate", f"100/1000=10% elapsed, must not fire yet, got {(stop, deposit)}")
    _ok("_atlas_stall_decision refuses to stop early, before the fraction gate is crossed")

    # 6. Past the fraction gate (now 0.5), no progress, not yet deposited ->
    #    stop + deposit remainder -- note NO "stalled"/history_entries input
    #    at all anymore: zero level progress past the fraction is sufficient
    #    on its own (v19 -- see module docstring for why the old literal
    #    no-op requirement was dropped).
    stop, deposit = _atlas_stall_decision(
        elapsed=500.0, cap=1000.0, start_level=1, current_level=1, already_deposited=False,
    )
    if not stop or deposit != 500.0:
        _fail("stall decision fires past the gate", f"expected (True, 500.0), got {(stop, deposit)}")
    _ok(f"_atlas_stall_decision fires past the gate and deposits the unused remainder: {(stop, deposit)}")

    # 7. Any progress (current_level > start_level) suppresses the stall path entirely.
    stop, deposit = _atlas_stall_decision(
        elapsed=999.0, cap=1000.0, start_level=1, current_level=2, already_deposited=False,
    )
    if stop:
        _fail("progress suppresses stall", "a game that progressed must never be cut as stalled")
    _ok("_atlas_stall_decision never fires for a game that has progressed past its starting level")

    # 8. Already deposited once -> never fires again (one deposit per session).
    stop, deposit = _atlas_stall_decision(
        elapsed=999.0, cap=1000.0, start_level=1, current_level=1, already_deposited=True,
    )
    if stop:
        _fail("one deposit per session", "must not deposit twice for the same session")
    _ok("_atlas_stall_decision refuses to fire again once this session has already deposited")

    # 9. _atlas_extension_request: no request for a level already credited.
    req = _atlas_extension_request(cap=1000.0, current_extra=0.0, current_level=2, last_credited_level=2)
    if req != 0.0:
        _fail("no re-request for the same level", f"expected 0.0, got {req}")
    _ok("_atlas_extension_request asks for nothing when the level was already credited")

    # 10. A genuinely new level requests min(room, cap * draw_fraction).
    req = _atlas_extension_request(cap=1000.0, current_extra=0.0, current_level=2, last_credited_level=1)
    if req != 500.0:  # cap * _ATLAS_TIME_BANK_DRAW_FRACTION (0.5)
        _fail("requests the draw fraction", f"expected 500.0, got {req}")
    _ok(f"_atlas_extension_request asks for cap*draw_fraction on a new level: {req}")

    # 11. Once already extended up to the max multiplier, no more room to ask for.
    req = _atlas_extension_request(cap=1000.0, current_extra=1000.0, current_level=3, last_credited_level=2)
    if req != 0.0:
        _fail("caps total extension", f"expected 0.0 once at the max multiplier, got {req}")
    _ok("_atlas_extension_request asks for nothing once already at the max-extension cap")

    # === HarnessSolver bank accounting (thread-safe deposit/draw) ===

    solver = solver_mod.HarnessSolver()
    solver.max_runtime_s_per_game = 1000.0
    solver.atlas_deposit_time(300.0)
    if solver._atlas_time_bank_s != 300.0:
        _fail("deposit", str(solver._atlas_time_bank_s))
    granted = solver.atlas_draw_time(120.0)
    if granted != 120.0 or solver._atlas_time_bank_s != 180.0:
        _fail("draw within balance", f"granted={granted}, balance={solver._atlas_time_bank_s}")
    _ok("HarnessSolver.atlas_deposit_time/atlas_draw_time account correctly within balance")

    granted = solver.atlas_draw_time(500.0)  # only 180 left
    if granted != 180.0 or solver._atlas_time_bank_s != 0.0:
        _fail("draw capped at balance", f"expected to drain to 0, got granted={granted}, balance={solver._atlas_time_bank_s}")
    _ok("a draw request larger than the balance is capped at what's actually available")

    granted = solver.atlas_draw_time(1.0)
    if granted != 0.0:
        _fail("empty bank yields nothing", str(granted))
    _ok("drawing from an empty bank grants 0, does not go negative")

    # === Full wiring: _HarnessGameSession._atlas_check_time_bank() ===

    bank_solver = solver_mod.HarnessSolver()
    bank_solver.max_runtime_s_per_game = 1000.0

    # 12. The v18-observed real pattern: a session that's ACTIVELY changing
    #     the board every action (never a literal no-op) but never advances
    #     the level, well past the fraction gate, still stops itself and
    #     deposits its remainder -- this is exactly what the old
    #     _atlas_recent_stalled requirement missed (0 deposits/draws across
    #     v18's whole 4h/25-game run despite 11/25 games matching this
    #     pattern). history_entries below has a DIFFERENT grid every entry
    #     to prove the new decision no longer cares.
    stuck_game = _FakeGame(levels_completed=0, grid=[[0, 0], [0, 0]])
    stuck = _session(stuck_game, bank_solver)
    stuck.started_at = solver_mod.time.monotonic() - 600.0  # 60% of the 1000s cap elapsed
    stuck.history_entries = _entries([([[i]], 1) for i in range(7)])  # grid changes every entry
    if not stuck._atlas_check_time_bank():
        _fail("busy-but-unproductive session stops", "expected True (stop now) even with an actively-changing board")
    if bank_solver._atlas_time_bank_s <= 0:
        _fail("stalled session deposits", f"expected a positive deposit, bank={bank_solver._atlas_time_bank_s}")
    if not stuck._atlas_stall_deposited:
        _fail("stall flag set", "expected _atlas_stall_deposited=True after stopping")
    _ok(f"a busy-but-unproductive session (board changes every action, level never does) still stops "
        f"itself and deposits its remainder: {bank_solver._atlas_time_bank_s:.0f}s")

    # 12b. A brand-new session's FIRST poll must not treat its starting level
    #      as "new progress" -- fixed alongside the v19 stall-trigger change
    #      (the old default sentinel of -1 for _atlas_last_credited_level
    #      made level 1 look like a fresh level on every session's very
    #      first check; harmless while the bank was always empty at start,
    #      but not what "just reached a level" is supposed to mean).
    fresh_game = _FakeGame(levels_completed=0, grid=[[0, 0], [0, 0]])
    fresh = _session(fresh_game, bank_solver)
    fresh.started_at = solver_mod.time.monotonic()
    fresh._atlas_check_time_bank()
    if fresh._atlas_last_credited_level != 1:
        _fail("starting level pre-credited", f"expected the starting level (1) to be pre-credited, got {fresh._atlas_last_credited_level}")
    if fresh._atlas_extra_time_s != 0.0:
        _fail("no draw on the starting level", f"expected 0 extra time on a fresh session, got {fresh._atlas_extra_time_s}")
    _ok("a fresh session's own starting level is pre-credited, not treated as newly-reached progress")

    # 13. A progressing session (just reached level 2, previously credited at
    #     level 1) draws from the bank the stalled session just funded, and
    #     its effective runtime cap grows accordingly.
    leader_game = _FakeGame(levels_completed=1, grid=[[1, 1], [1, 1]])  # level_number = 2
    leader = _session(leader_game, bank_solver)
    leader.started_at = solver_mod.time.monotonic()
    leader._atlas_start_level = 1
    leader._atlas_last_credited_level = 1
    before_bank = bank_solver._atlas_time_bank_s
    stopped = leader._atlas_check_time_bank()
    if stopped:
        _fail("progressing session never stops itself", "a session making progress must not be told to stop")
    if leader._atlas_extra_time_s <= 0:
        _fail("progressing session draws extra time", f"expected extra_time_s > 0, got {leader._atlas_extra_time_s}")
    if bank_solver._atlas_time_bank_s >= before_bank:
        _fail("bank drained by the draw", f"before={before_bank}, after={bank_solver._atlas_time_bank_s}")
    effective_cap = leader._atlas_effective_runtime_cap()
    if effective_cap <= bank_solver.max_runtime_s_per_game:
        _fail("effective cap extended", f"expected > base {bank_solver.max_runtime_s_per_game}, got {effective_cap}")
    _ok(f"a session that just reached a new level draws from the bank and extends its own effective cap to {effective_cap:.0f}s")

    # 14. Disabling the feature makes _atlas_check_time_bank an unconditional no-op.
    solver_mod._ATLAS_TIME_BANK_ENABLED = False
    try:
        disabled_game = _FakeGame(levels_completed=0, grid=[[0, 0], [0, 0]])
        disabled = _session(disabled_game, bank_solver)
        disabled.started_at = solver_mod.time.monotonic() - 900.0
        disabled.history_entries = _entries([([[0, 0]], 1)] * 7)
        if disabled._atlas_check_time_bank():
            _fail("disabled flag is respected", "must be a no-op when _ATLAS_TIME_BANK_ENABLED is False")
    finally:
        solver_mod._ATLAS_TIME_BANK_ENABLED = True
    _ok("_ATLAS_TIME_BANK_ENABLED=False makes _atlas_check_time_bank a strict no-op")

    # === Retry-storm backstop: pure decision function ===

    # 15. Fewer than the threshold -> never fires.
    if _atlas_retry_storm_decision(consecutive_failures=4, threshold=5, already_deposited=False):
        _fail("retry storm needs the threshold", "4 consecutive failures with threshold 5 must not fire")
    _ok("_atlas_retry_storm_decision stays False below the consecutive-failure threshold")

    # 16. Exactly the threshold -> fires.
    if not _atlas_retry_storm_decision(consecutive_failures=5, threshold=5, already_deposited=False):
        _fail("retry storm fires at the threshold", "5 consecutive failures with threshold 5 must fire")
    _ok("_atlas_retry_storm_decision fires once consecutive failures reach the threshold")

    # 17. Already deposited once -> never fires again.
    if _atlas_retry_storm_decision(consecutive_failures=99, threshold=5, already_deposited=True):
        _fail("one retry-storm deposit per session", "must not deposit twice for the same session")
    _ok("_atlas_retry_storm_decision refuses to fire again once this session has already deposited")

    # === Retry-storm backstop: full wiring through _atlas_check_retry_storm() ===

    # 18. A session with enough consecutive analyzer failures ends itself and
    #     deposits its unused remainder -- exactly like a gameplay stall, but
    #     triggered by a completely different signal (no game state involved
    #     at all, since this failure happens before any tool call exists).
    retry_bank_solver = solver_mod.HarnessSolver()
    retry_bank_solver.max_runtime_s_per_game = 1000.0
    flaky_game = _FakeGame(levels_completed=0, grid=[[0, 0], [0, 0]])
    flaky = _session(flaky_game, retry_bank_solver)
    flaky.started_at = solver_mod.time.monotonic() - 300.0  # 300s elapsed of the 1000s cap
    flaky._atlas_consecutive_retry_failures = 5
    if not flaky._atlas_check_retry_storm():
        _fail("retry storm stops the session", "expected True (stop now) at 5 consecutive failures")
    if retry_bank_solver._atlas_time_bank_s <= 0:
        _fail("retry storm deposits", f"expected a positive deposit, bank={retry_bank_solver._atlas_time_bank_s}")
    if not flaky._atlas_retry_storm_deposited:
        _fail("retry storm flag set", "expected _atlas_retry_storm_deposited=True after stopping")
    if flaky_game.game_run.solver_note is None:
        _fail("retry storm leaves a note", "expected game_run.solver_note to explain the early stop")
    _ok(f"a session hitting the retry-storm threshold stops and deposits its remainder: "
        f"{retry_bank_solver._atlas_time_bank_s:.0f}s, note={flaky_game.game_run.solver_note!r}")

    # 19. Below the threshold, _atlas_check_retry_storm is a no-op (does not
    #     stop, does not touch the bank).
    calm_game = _FakeGame(levels_completed=0, grid=[[0, 0], [0, 0]])
    calm = _session(calm_game, retry_bank_solver)
    calm.started_at = solver_mod.time.monotonic() - 300.0
    calm._atlas_consecutive_retry_failures = 2
    bank_before = retry_bank_solver._atlas_time_bank_s
    if calm._atlas_check_retry_storm():
        _fail("retry storm respects the threshold", "2 consecutive failures must not stop the session")
    if retry_bank_solver._atlas_time_bank_s != bank_before:
        _fail("no deposit below the threshold", "bank must be untouched when the storm check doesn't fire")
    _ok("_atlas_check_retry_storm is a no-op below the consecutive-failure threshold")

    # 20. Disabling the feature makes _atlas_check_retry_storm a strict no-op too.
    solver_mod._ATLAS_TIME_BANK_ENABLED = False
    try:
        disabled_flaky_game = _FakeGame(levels_completed=0, grid=[[0, 0], [0, 0]])
        disabled_flaky = _session(disabled_flaky_game, retry_bank_solver)
        disabled_flaky.started_at = solver_mod.time.monotonic() - 300.0
        disabled_flaky._atlas_consecutive_retry_failures = 99
        if disabled_flaky._atlas_check_retry_storm():
            _fail("disabled flag respected (retry storm)", "must be a no-op when _ATLAS_TIME_BANK_ENABLED is False")
    finally:
        solver_mod._ATLAS_TIME_BANK_ENABLED = True
    _ok("_ATLAS_TIME_BANK_ENABLED=False makes _atlas_check_retry_storm a strict no-op too")

    print("\nAll atlas time-bank checks passed.")


if __name__ == "__main__":
    main()

"""Exercise dynamic budget re-planning in atlas_src's solver.py (backlog 27).

The problem it solves: the per-game cap is fitted ONCE, before the run, as
budget / ceil(n_games / concurrency). A stalled game ends early and deposits
its remainder into the shared time bank -- but with draws disabled (the V39
battle config, where 2 waves x 4 h exactly fill the 8 h budget) nobody ever
takes that time back. Measured on every Phase A window we have (25 min,
50 min, 2.2 h): 23 of 25 games self-terminate at 50% of their cap, so about
half the wall budget is returned and then evaporates.

The fix re-plans from an ABSOLUTE deadline on every game completion and only
ever raises the per-game window, so survivors inherit the freed hours.

Two safety properties are the whole point of this file, because the code
path that arms this only ever executes in the real-submission branch -- the
exact class of change that produced the 0.06 regression when it was shipped
on hand-traced arithmetic alone (see docs/plan_top10_by_3009.md item 24 and
the 'untestable code before real submission' rule):

  (1) NO session's cap can ever reach past the absolute deadline, whatever
      the floor, the ceiling, the game count or the completion order;
  (2) the shared deadline only ever moves FORWARD, so no running game is cut
      shorter than the budget it was already promised.

Tests the pure decision function, the solver-level planner (with a stubbed
clock, so a simulated 8-hour run takes milliseconds), the wiring through
_HarnessGameSession._atlas_effective_runtime_cap(), and the arming block as
it really ships -- exec'd out of build_atlas_notebook.ATLAS_CELL, not
reimplemented here where it could drift.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "tufa-arc-agi-framework" / "src"))
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "ARC3-Inference"))

from inference.framework import solver as solver_mod  # noqa: E402

_HarnessGameSession = solver_mod._HarnessGameSession
_window = solver_mod._atlas_dynamic_window_decision


def _fail(name: str, detail: str) -> None:
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def _ok(name: str) -> None:
    print(f"ok   {name}")


class _Clock:
    """Stand-in for the `time` module: only monotonic() is ever used by the
    code under test, and driving it by hand turns an 8-hour run into a loop."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


class _FakeState:
    def __init__(self) -> None:
        self.levels_completed = 0
        self.won = False
        self.frame = type("F", (), {"data": [[0, 0], [0, 0]]})()


class _FakeGame:
    def __init__(self) -> None:
        self.number_of_levels = 8
        self.current_state = _FakeState()
        self.game_run = type("R", (), {"state": "playing", "history": [], "solver_note": None})()


def _session(solver: Any, started_at: float) -> Any:
    session = _HarnessGameSession(
        solver=solver,
        game=_FakeGame(),
        analyzer=None,
        game_index=0,
        pass_index=0,
        state_path=Path("unused_state.json"),
        transcript_path=Path("unused_transcript.txt"),
        analysis_html_relpath="unused.html",
        stop_event=threading.Event(),
        viewer_data_path=Path("unused_viewer.json"),
    )
    session.started_at = started_at
    return session


@contextlib.contextmanager
def _armed(clock: _Clock, *, budget: float, ceiling: float, floor: float):
    """Turn the feature on with a stubbed clock, and put everything back
    afterwards -- the module defaults must stay OFF for other tests."""
    saved = (
        solver_mod.time,
        solver_mod._ATLAS_DYNAMIC_BUDGET_ENABLED,
        solver_mod._ATLAS_DYNAMIC_BUDGET_TOTAL_S,
        solver_mod._ATLAS_DYNAMIC_BUDGET_CEILING_S,
        solver_mod._ATLAS_DYNAMIC_BUDGET_MIN_S,
    )
    solver_mod.time = clock
    solver_mod._ATLAS_DYNAMIC_BUDGET_ENABLED = True
    solver_mod._ATLAS_DYNAMIC_BUDGET_TOTAL_S = budget
    solver_mod._ATLAS_DYNAMIC_BUDGET_CEILING_S = ceiling
    solver_mod._ATLAS_DYNAMIC_BUDGET_MIN_S = floor
    try:
        yield
    finally:
        (
            solver_mod.time,
            solver_mod._ATLAS_DYNAMIC_BUDGET_ENABLED,
            solver_mod._ATLAS_DYNAMIC_BUDGET_TOTAL_S,
            solver_mod._ATLAS_DYNAMIC_BUDGET_CEILING_S,
            solver_mod._ATLAS_DYNAMIC_BUDGET_MIN_S,
        ) = saved


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def main() -> None:
    # === 1. Pure function: reproduces the static wave arithmetic at t=0 ====
    # V39's real battle shape: 55 games, concurrency 28, 8 h budget.
    # 28 running, 27 queued -> one queued wave plus the batch in flight.
    w = _window(elapsed=0.0, budget_s=28800.0, pending=27, running=28,
                concurrency=28, ceiling_s=14400.0, min_s=1800.0)
    if abs(w - 14400.0) > 1e-6:
        _fail("matches the static fit at t=0", f"expected 14400, got {w}")
    _ok(f"window at run start equals the static per-game cap: {w:.0f}s (2 slices of the 8 h budget)")

    # === 2. Nothing queued -> the whole remaining budget goes to survivors =
    w = _window(elapsed=7200.0, budget_s=28800.0, pending=0, running=5,
                concurrency=28, ceiling_s=0.0, min_s=0.0)
    if abs(w - 21600.0) > 1e-6:
        _fail("empty queue hands everything to survivors", f"expected 21600, got {w}")
    _ok(f"with an empty queue the survivors get the entire remainder: {w:.0f}s")

    # === 3. The ceiling still binds ========================================
    w = _window(elapsed=0.0, budget_s=28800.0, pending=0, running=3,
                concurrency=28, ceiling_s=14400.0, min_s=0.0)
    if abs(w - 14400.0) > 1e-6:
        _fail("ceiling binds", f"expected 14400, got {w}")
    _ok("the configured ceiling still caps a single game at half the budget")

    # === 4. The floor can never push past the remaining time ===============
    # A floor larger than what is left would be the obvious way to blow the
    # deadline -- the final clamp is what forbids it.
    w = _window(elapsed=28000.0, budget_s=28800.0, pending=10, running=28,
                concurrency=28, ceiling_s=14400.0, min_s=1800.0)
    if w > 800.0 + 1e-6:
        _fail("floor never outruns the remainder", f"only 800s left, got {w}")
    _ok(f"a 1800s floor is clamped to the {w:.0f}s actually left before the deadline")

    # === 5. Past the deadline -> zero, never negative ======================
    w = _window(elapsed=30000.0, budget_s=28800.0, pending=5, running=2,
                concurrency=28, ceiling_s=14400.0, min_s=1800.0)
    if w != 0.0:
        _fail("no window past the deadline", f"expected 0.0, got {w}")
    _ok("past the deadline the window is exactly zero, never negative")

    # === 6. Queue depth divides the remainder ==============================
    w = _window(elapsed=0.0, budget_s=28800.0, pending=84, running=28,
                concurrency=28, ceiling_s=0.0, min_s=0.0)
    if abs(w - 28800.0 / 4) > 1e-6:
        _fail("queue depth divides the budget", f"expected 7200, got {w}")
    _ok(f"112 games at concurrency 28 -> 4 slices of {w:.0f}s (the 110-game final's shape)")

    # === 7. Solver level: the deadline only ever moves forward =============
    clock = _Clock()
    with _armed(clock, budget=28800.0, ceiling=14400.0, floor=1800.0):
        solver = solver_mod.HarnessSolver(concurrency=28, max_runtime_s_per_game=14400.0)
        _quiet(solver.atlas_dyn_arm, 55)
        state = solver._atlas_dyn_state()
        seen: list[float] = [state["deadline"]]
        for step in range(20):
            clock.advance(600.0)
            _quiet(solver.atlas_dyn_note_start)
            _quiet(solver.atlas_dyn_note_finish)
            seen.append(state["deadline"])
        backwards = [(a, b) for a, b in zip(seen, seen[1:]) if b < a - 1e-9]
        if backwards:
            _fail("deadline is monotonic", f"moved backwards: {backwards[:2]}")
        hard = state["t0"] + state["budget_s"]
        if max(seen) > hard + 1e-6:
            _fail("deadline never passes T_end", f"max {max(seen)} > T_end {hard}")
    _ok("shared deadline moves forward only, and never past the absolute T_end")

    # === 8. Full simulated run: the two safety properties ==================
    # 55 games, concurrency 28, 8 h budget, 4 h ceiling. Most games stall out
    # at half their cap (the measured 23-of-25 pattern); a few run long.
    clock = _Clock()
    violations: list[str] = []
    extended: list[float] = []
    with _armed(clock, budget=28800.0, ceiling=14400.0, floor=1800.0):
        solver = solver_mod.HarnessSolver(concurrency=28, max_runtime_s_per_game=14400.0)
        _quiet(solver.atlas_dyn_arm, 55)
        state = solver._atlas_dyn_state()
        t0, budget = state["t0"], state["budget_s"]
        hard = t0 + budget

        live: list[tuple[Any, float]] = []          # (session, planned_end)
        queued = 55
        # first wave
        for _ in range(min(28, queued)):
            _quiet(solver.atlas_dyn_note_start)
            live.append((_session(solver, clock.now), 0.0))
            queued -= 1

        for tick in range(200):
            clock.advance(300.0)
            if clock.now >= hard:
                break
            survivors: list[tuple[Any, float]] = []
            for session, _ in live:
                cap = session._atlas_effective_runtime_cap()
                if cap is None:
                    continue
                end = session.started_at + cap
                if end > hard + 1e-6:
                    violations.append(
                        f"session started {session.started_at - t0:.0f}s in "
                        f"would run to {end - hard:.0f}s past T_end"
                    )
                if cap > 14400.0 + 1e-6:
                    extended.append(cap)
                # stall pattern: most games give up at half their cap
                stalls = (id(session) % 4) != 0
                elapsed = clock.now - session.started_at
                if (stalls and elapsed >= 7200.0) or elapsed >= cap:
                    _quiet(solver.atlas_dyn_note_finish)
                    if queued > 0:
                        _quiet(solver.atlas_dyn_note_start)
                        survivors.append((_session(solver, clock.now), 0.0))
                        queued -= 1
                else:
                    survivors.append((session, end))
            live = survivors
            if not live and queued == 0:
                break

    if violations:
        _fail("no session outlives the budget", violations[0])
    _ok(f"simulated 55-game run: no session's cap ever reached past T_end ({len(violations)} violations)")

    if not extended:
        _fail("survivors actually gain time", "no session was ever extended beyond the static 14400s cap")
    _ok(f"survivors really inherited the freed hours: {len(extended)} extensions, longest {max(extended):.0f}s (static cap was 14400s)")

    # === 9. Disabled by default -> byte-identical behaviour ================
    plain = solver_mod.HarnessSolver(concurrency=28, max_runtime_s_per_game=14400.0)
    if plain.atlas_dynamic_cap(0.0, 14400.0) is not None:
        _fail("off by default", "atlas_dynamic_cap must return None when the feature is off")
    off_session = _session(plain, solver_mod.time.monotonic())
    off_session._atlas_extra_time_s = 900.0
    if off_session._atlas_effective_runtime_cap() != 14400.0 + 900.0:
        _fail("off by default", "effective cap must be exactly base + time-bank extra when off")
    _ok("with the feature off the effective cap is exactly base + bank extra -- battle behaviour unchanged")

    # === 10. Armed but with no budget set -> still inert ===================
    clock = _Clock()
    with _armed(clock, budget=0.0, ceiling=14400.0, floor=1800.0):
        inert = solver_mod.HarnessSolver(concurrency=28, max_runtime_s_per_game=14400.0)
        _quiet(inert.atlas_dyn_arm, 55)
        if inert.atlas_dynamic_cap(clock.now, 14400.0) is not None:
            _fail("no budget -> inert", "must not re-plan without a total budget")
    _ok("armed without a total budget the planner stays inert instead of guessing one")

    # === 11. The REAL arming block, exec'd from the shipped cell ===========
    # Same technique as test_atlas_fit_game_cap.py: run the actual notebook
    # cell text, so this cannot drift from what really ships.
    spec = importlib.util.spec_from_file_location(
        "build_atlas_notebook", ROOT / "scripts" / "build_atlas_notebook.py"
    )
    builder = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(builder)

    # 31.08: substitute through the builder's own helper, never by hand. A
    # hand-written list drifts the moment a knob is added -- which is exactly
    # how this test broke when ATLAS_PASSES_BUILD appeared, and a cousin of the
    # bug that cost three submissions.
    base_env = {"ATLAS_CALIBRATION_CAP_S": "1500", "ATLAS_CONCURRENCY_BUILD": "28",
                "ATLAS_DRAWS_BUILD": "0", "ATLAS_GAME_CAP_CEILING_BUILD": "14400"}
    cell_on = builder.substitute_knobs(builder.ATLAS_CELL,
                                       {**base_env, "ATLAS_DYNAMIC_BUDGET_BUILD": "1"})
    cell_off = builder.substitute_knobs(builder.ATLAS_CELL,
                                        {**base_env, "ATLAS_DYNAMIC_BUDGET_BUILD": "0"})
    for name, text in (("on", cell_on), ("off", cell_off)):
        left = [ph for ph in builder.CELL_KNOBS if ph in text]
        if left:
            _fail("cell fully substituted", f"{name}: {left}")

    class _Solver:
        def __init__(self) -> None:
            self.analyzer_timeout = 900.0
            self.concurrency = 28
            self.max_actions_per_game = None
            self.max_runtime_s_per_game = 7920.0

    class _Bm:
        def __init__(self) -> None:
            self.solver = _Solver()
            self.n_passes = 1

    def _run(cell_text: str, *, true_submission: bool, n_games: int, tmp: Path):
        ns: dict[str, Any] = {
            "bm": _Bm(),
            "true_submission": true_submission,
            "WORKING_DIR": tmp,
            "ATLAS_PRISTINE": {
                "analyzer_timeout": 900.0, "concurrency": 28,
                "max_actions_per_game": None, "max_runtime_s_per_game": 7920.0,
                "n_passes": 1,
            },
        }
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cell_text, "<ATLAS_CELL>", "exec"), ns)
            if true_submission:
                ns["atlas_fit_game_cap"](n_games)
        return ns

    import tempfile

    saved_flags = (
        solver_mod._ATLAS_DYNAMIC_BUDGET_ENABLED,
        solver_mod._ATLAS_DYNAMIC_BUDGET_TOTAL_S,
        solver_mod._ATLAS_DYNAMIC_BUDGET_CEILING_S,
        solver_mod._ATLAS_DYNAMIC_BUDGET_MIN_S,
    )
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            # Phase A (true_submission False): atlas_fit_game_cap never runs,
            # so the planner must stay off -- arming it here would stretch a
            # 30-minute calibration into the full 8-hour budget.
            solver_mod._ATLAS_DYNAMIC_BUDGET_ENABLED = False
            _run(cell_on, true_submission=False, n_games=55, tmp=tmp)
            if solver_mod._ATLAS_DYNAMIC_BUDGET_ENABLED:
                _fail("Phase A stays static", "the calibration branch must never arm the planner")
            _ok("Phase A never arms the planner -- calibration keeps its short cap")

            # Real submission, flag on -> armed with the notebook's own budget.
            ns = _run(cell_on, true_submission=True, n_games=55, tmp=tmp)
            if not solver_mod._ATLAS_DYNAMIC_BUDGET_ENABLED:
                _fail("submission arms the planner", "flag on + submission branch must arm it")
            if solver_mod._ATLAS_DYNAMIC_BUDGET_TOTAL_S != ns["ATLAS_SUBMISSION_BUDGET_S"]:
                _fail("budget handed over", f"got {solver_mod._ATLAS_DYNAMIC_BUDGET_TOTAL_S}")
            if solver_mod._ATLAS_DYNAMIC_BUDGET_CEILING_S != ns["ATLAS_SUBMISSION_GAME_CAP_CEILING_S"]:
                _fail("ceiling handed over", f"got {solver_mod._ATLAS_DYNAMIC_BUDGET_CEILING_S}")
            if solver_mod._ATLAS_DYNAMIC_BUDGET_MIN_S != ns["ATLAS_MIN_GAME_CAP_S"]:
                _fail("floor handed over", f"got {solver_mod._ATLAS_DYNAMIC_BUDGET_MIN_S}")
            _ok(
                "the submission branch arms it with the notebook's own numbers: budget "
                f"{solver_mod._ATLAS_DYNAMIC_BUDGET_TOTAL_S:.0f}s, ceiling "
                f"{solver_mod._ATLAS_DYNAMIC_BUDGET_CEILING_S:.0f}s, floor "
                f"{solver_mod._ATLAS_DYNAMIC_BUDGET_MIN_S:.0f}s"
            )

            # Static per-game fit must be untouched by the new block.
            bm_cap = ns["bm"].solver.max_runtime_s_per_game
            expected = min(14400.0, ns["ATLAS_SUBMISSION_BUDGET_S"] / math.ceil(55 / 28))
            if abs(bm_cap - expected) > 1e-6:
                _fail("static fit unchanged", f"expected {expected}, got {bm_cap}")
            _ok(f"the static per-game fit is unchanged by the new block: {bm_cap:.0f}s")

            # Flag off -> submission branch leaves the planner alone.
            solver_mod._ATLAS_DYNAMIC_BUDGET_ENABLED = False
            _run(cell_off, true_submission=True, n_games=55, tmp=tmp)
            if solver_mod._ATLAS_DYNAMIC_BUDGET_ENABLED:
                _fail("build flag respected", "ATLAS_DYNAMIC_BUDGET=False must leave it off")
            _ok("with the build flag off the submission branch leaves the planner off")
    finally:
        (
            solver_mod._ATLAS_DYNAMIC_BUDGET_ENABLED,
            solver_mod._ATLAS_DYNAMIC_BUDGET_TOTAL_S,
            solver_mod._ATLAS_DYNAMIC_BUDGET_CEILING_S,
            solver_mod._ATLAS_DYNAMIC_BUDGET_MIN_S,
        ) = saved_flags

    # === 12. Pickle/deepcopy hygiene -- the planner state carries a Lock ===
    import copy
    import pickle

    clock = _Clock()
    with _armed(clock, budget=28800.0, ceiling=14400.0, floor=1800.0):
        pick = solver_mod.HarnessSolver(concurrency=28, max_runtime_s_per_game=14400.0)
        _quiet(pick.atlas_dyn_arm, 55)
        try:
            revived = pickle.loads(pickle.dumps(pick))
        except Exception as exc:  # pragma: no cover - the failure IS the message
            _fail("solver still pickles", f"{exc!r}")
        if getattr(revived, "_atlas_dyn", None) is not None:
            _fail("planner state not pickled", "the Lock-bearing state must be dropped")
        try:
            copy.deepcopy(pick)
        except Exception as exc:  # pragma: no cover
            _fail("solver still deep-copies", f"{exc!r}")
    _ok("solver still pickles and deep-copies with the planner armed (state dropped, rebuilt lazily)")

    print("\nAll atlas dynamic-budget checks passed.")


if __name__ == "__main__":
    main()

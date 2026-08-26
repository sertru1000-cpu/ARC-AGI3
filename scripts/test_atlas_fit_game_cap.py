"""Dry-run the real `atlas_fit_game_cap` formula without touching Kaggle.

24.08 lesson: this function only ever runs in the real-submission branch --
Phase A calibration never exercises it. A ceiling-formula "fix" was shipped
on hand-traced arithmetic alone, spent the day's one real submission slot,
and came back at 0.06 (worse than the no-harness baseline). Whatever the
true cause, the fix was reverted; this test exists so any FUTURE change to
this formula gets checked against synthetic n_games/concurrency combinations
BEFORE it ever reaches a real, quota-costing submission again.

Drives the ACTUAL embedded cell text from scripts/build_atlas_notebook.py
(imported as a module, ATLAS_CELL is a real top-level string there) via
exec(), with a fake `bm`/`WORKING_DIR`/`true_submission`, so this is the
literal code that ships in the notebook -- not a hand-copied reimplementation
that could silently drift from the real formula.

25.08: ATLAS_CELL's own `bm.solver.concurrency = ATLAS_CONCURRENCY` line
OVERWRITES whatever concurrency a `_Benchmark(concurrency, ...)` fixture was
built with -- the concurrency value that actually matters is the real
ATLAS_CONCURRENCY constant baked into the cell, not whatever this test
passes in. This was already true before 25.08's 14->10 change (lowered to
cut concurrent-request contention after the retry-storm bug -- see
solver.py), it just went unnoticed because the test's hardcoded "14"
happened to match. Pull ATLAS_CONCURRENCY from the exec'd namespace (same
pattern already used for the budget/floor/ceiling constants) so this test
tracks the real value instead of drifting silently again.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARC3_INFERENCE = ROOT / "atlas_src" / "src" / "ARC3-Inference"
sys.path.insert(0, str(ARC3_INFERENCE))
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "tufa-arc-agi-framework" / "src"))

spec = importlib.util.spec_from_file_location("build_atlas_notebook", ROOT / "scripts" / "build_atlas_notebook.py")
build_atlas_notebook = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(build_atlas_notebook)


def _fail(name: str, detail: str) -> None:
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def _ok(name: str) -> None:
    print(f"ok   {name}")


class _Solver:
    def __init__(self, concurrency: int, max_runtime_s_per_game):
        self.concurrency = concurrency
        self.max_runtime_s_per_game = max_runtime_s_per_game
        self.analyzer_timeout = None
        self.max_actions_per_game = None


class _Benchmark:
    def __init__(self, concurrency: int, bundle_default_cap: float):
        self.solver = _Solver(concurrency, bundle_default_cap)
        self.n_passes = 1


def _run_cell(bm, working_dir: Path):
    """exec() the REAL ATLAS_CELL text and return the resulting namespace."""
    namespace = {
        "bm": bm,
        "true_submission": True,  # the branch atlas_fit_game_cap is actually called from
        "WORKING_DIR": working_dir,
        "ATLAS_PRISTINE": {
            "analyzer_timeout": bm.solver.analyzer_timeout,
            "concurrency": bm.solver.concurrency,
            "max_actions_per_game": bm.solver.max_actions_per_game,
            "max_runtime_s_per_game": bm.solver.max_runtime_s_per_game,
            "n_passes": bm.n_passes,
        },
    }
    # `import inference.agent.tool_agent as _atlas_tool_agent` is a real import
    # inside ATLAS_CELL -- exercise it for real, same as the notebook does.
    # The cell prints its own "atlas: ..." diagnostics as a side effect;
    # swallow them here so this test's own ok/FAIL lines stay readable.
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(build_atlas_notebook.ATLAS_CELL, "<ATLAS_CELL>", "exec"), namespace)
    return namespace


def _fit(n_games: int, bundle_default_cap: float = 7920.0):
    """Call the real, freshly-exec'd atlas_fit_game_cap(n_games) and report the outcome.

    The `concurrency` passed to the throwaway _Benchmark fixture here is
    irrelevant -- ATLAS_CELL always overwrites bm.solver.concurrency with the
    real ATLAS_CONCURRENCY constant regardless of what this starts with."""
    with tempfile.TemporaryDirectory() as tmp:
        bm = _Benchmark(1, bundle_default_cap)
        namespace = _run_cell(bm, Path(tmp))
        with contextlib.redirect_stdout(io.StringIO()):
            namespace["atlas_fit_game_cap"](n_games)
        diagnostics_path = Path(tmp) / "atlas_submission_diagnostics.json"
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8")) if diagnostics_path.exists() else None
        return bm.solver.max_runtime_s_per_game, diagnostics


def main() -> None:
    # ATLAS_SUBMISSION_BUDGET_S/ATLAS_MIN_GAME_CAP_S/ATLAS_SUBMISSION_GAME_CAP_CEILING_S/
    # ATLAS_CONCURRENCY are defined INSIDE ATLAS_CELL, not as module attributes
    # -- pull them from a throwaway exec so the test's expected numbers track
    # the real constants instead of hand-copied magic numbers that could drift.
    probe_ns = _run_cell(_Benchmark(1, 7920.0), Path(tempfile.mkdtemp()))
    budget_s = probe_ns["ATLAS_SUBMISSION_BUDGET_S"]
    min_cap_s = probe_ns["ATLAS_MIN_GAME_CAP_S"]
    ceiling_s = probe_ns["ATLAS_SUBMISSION_GAME_CAP_CEILING_S"]
    concurrency = probe_ns["ATLAS_CONCURRENCY"]

    # The largest whole number of waves where budget/waves still clears the
    # ceiling -- i.e. how many waves the ceiling-pin tolerates before the
    # budget itself starts binding instead. Concurrency-independent (only
    # budget_s/ceiling_s matter); n_boundary below is what turns it into an
    # actual n_games figure for the CURRENT concurrency.
    max_waves_for_pin = math.floor(budget_s / ceiling_s)
    n_boundary = max_waves_for_pin * concurrency  # largest n_games where the pin still holds

    # 1. 25 games -> the KNOWN, still-present limitation: affordable
    #    (budget/waves) comfortably exceeds the ceiling, so the
    #    ceiling-clamped formula pins the cap at the ceiling, not the budget.
    waves = math.ceil(25 / concurrency)
    affordable = budget_s / waves
    expected = max(min_cap_s, min(ceiling_s, affordable))
    fitted, diag = _fit(25)
    if fitted != expected:
        _fail("25 games", f"expected {expected}, got {fitted}")
    if expected != ceiling_s:
        _fail("25 games (sanity)", "test's own assumption about the known ceiling-pin drifted")
    _ok(f"25 games / concurrency {concurrency} -> {waves} wave(s), cap stays pinned at the ceiling {fitted:.0f}s "
        f"(affordable would allow {affordable:.0f}s -- the known, still-present limitation)")

    # 1b. the ceiling-pin is NOT specific to 25 -- it holds for the WHOLE range
    #     up to n_boundary games (waves stays low enough that affordable
    #     stays >=ceiling_s). n_boundary is the last n_games where it holds.
    waves = math.ceil(n_boundary / concurrency)
    fitted, _ = _fit(n_boundary)
    if waves > max_waves_for_pin or fitted != ceiling_s:
        _fail("pin boundary", f"expected <= {max_waves_for_pin} waves / pinned at {ceiling_s}, got {waves} waves / {fitted}")
    _ok(f"{n_boundary} games / concurrency {concurrency} -> still {waves} wave(s), cap still pinned at {fitted:.0f}s "
        f"(the pin holds for the WHOLE 1..{n_boundary} range, not just 25)")

    # 1c. one game more is the first n_games where the budget finally binds below the ceiling.
    n_past_boundary = n_boundary + 1
    waves = math.ceil(n_past_boundary / concurrency)
    fitted, _ = _fit(n_past_boundary)
    if waves <= max_waves_for_pin or fitted >= ceiling_s:
        _fail("past the boundary", f"expected > {max_waves_for_pin} waves / below {ceiling_s}, got {waves} waves / {fitted}")
    _ok(f"{n_past_boundary} games / concurrency {concurrency} -> {waves} waves, budget finally binds: {fitted:.0f}s "
        "(one game past the boundary is enough to flip it)")

    # 2. enough games to finally exceed the ceiling: budget/waves < ceiling_s.
    n_big = max(60, n_boundary * 2)
    waves = math.ceil(n_big / concurrency)
    affordable = budget_s / waves
    expected = max(min_cap_s, min(ceiling_s, affordable))
    fitted, _ = _fit(n_big)
    if fitted != expected or fitted >= ceiling_s:
        _fail(f"{n_big} games", f"expected {expected} (< ceiling), got {fitted}")
    _ok(f"{n_big} games / concurrency {concurrency} -> {waves} wave(s), budget finally binds below the ceiling: {fitted:.0f}s")

    # 3. pathological case: the floor must hold even when affordable collapses.
    n_huge = 2000
    fitted, _ = _fit(n_huge)
    if fitted != min_cap_s:
        _fail("floor holds", f"expected the {min_cap_s:.0f}s floor, got {fitted}")
    _ok(f"{n_huge} games / concurrency {concurrency} -> the {min_cap_s:.0f}s floor holds, never goes lower")

    # 4. the fitted cap can never exceed the total submission budget, whatever n_games is.
    for n in (1, 5, 25, n_big, 500):
        fitted, _ = _fit(n)
        if fitted > budget_s:
            _fail("never exceeds budget", f"n_games={n} produced {fitted}s > budget {budget_s}s")
    _ok("fitted cap never exceeds the total submission budget, across a range of n_games")

    # 5. it actually mutates bm.solver.max_runtime_s_per_game (not just computes
    #    and discards) -- use n_big, where the fitted value is known to
    #    differ from the starting 7920s bundle default (n=25 would not catch
    #    a no-op mutation here, since it now pins at the ceiling instead).
    bm = _Benchmark(1, 7920.0)
    namespace = _run_cell(bm, Path(tempfile.mkdtemp()))
    before = bm.solver.max_runtime_s_per_game
    with contextlib.redirect_stdout(io.StringIO()):
        namespace["atlas_fit_game_cap"](n_big)
    after = bm.solver.max_runtime_s_per_game
    expected_after = min(ceiling_s, budget_s / math.ceil(n_big / concurrency))
    if after == before or after != expected_after:
        _fail("mutates solver state", f"before={before}, after={after}, expected {expected_after}")
    _ok(f"atlas_fit_game_cap actually sets bm.solver.max_runtime_s_per_game ({before} -> {after})")

    # 6. the diagnostics json records the real n_games/waves/cap -- the one
    #    thing we'd want to recover from an actual submission if it survives.
    fitted, diagnostics = _fit(n_past_boundary)
    if diagnostics is None:
        _fail("diagnostics written", "atlas_submission_diagnostics.json was not written")
    if diagnostics["n_games"] != n_past_boundary or diagnostics["concurrency"] != concurrency:
        _fail("diagnostics content", f"got {diagnostics!r}")
    if diagnostics["waves"] != math.ceil(n_past_boundary / concurrency):
        _fail("diagnostics waves", f"got {diagnostics!r}")
    if diagnostics["per_game_cap_after_s"] != fitted:
        _fail("diagnostics cap matches mutation", f"got {diagnostics!r}")
    _ok(f"diagnostics json records n_games/waves/cap correctly: {diagnostics}")

    print("\nAll atlas_fit_game_cap dry-run checks passed (no Kaggle, no GPU, no quota spent).")


if __name__ == "__main__":
    main()

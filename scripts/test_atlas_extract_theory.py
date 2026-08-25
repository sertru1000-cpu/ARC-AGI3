"""Exercise the extract= abstraction added to verify_theory/plan_with_theory/execute_plan.

24.08 finding: three real transcripts (r11l, sc25, g50t) showed verify_theory's
pixel-perfect whole-board match causing total action paralysis when the
board carries decoration/rendering the model correctly judges irrelevant to
the mechanic but can never predict exactly. Separately, our own historical
teacher data (Gemini's lp85/ft09 wins) showed the winning pattern is the
opposite of pixel-perfect: reduce the board to a small discrete state FIRST
(object/tile positions), then verify/plan/execute over THAT.

extract(grid) -> state lets the model do exactly this: verify_theory,
plan_with_theory, and execute_plan all accept it and, when given, compare
predict()'s output against extract()'s output instead of the raw grid.
Without extract=, behavior is byte-for-byte unchanged (existing tests in
test_atlas_plan_with_theory.py / test_atlas_execute_plan.py still cover
that path).

Drives the REAL sandboxed subprocess with a synthetic "moving marker on a
noisy board" game: a player marker moves predictably, but a decoration cell
changes every real step in a way a genuine theory of the MECHANIC would
never bother to model -- exactly the shape of the real failure.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARC3_INFERENCE = ROOT / "atlas_src" / "src" / "ARC3-Inference"
sys.path.insert(0, str(ARC3_INFERENCE))

spec = importlib.util.spec_from_file_location(
    "python_tool_sandbox", ARC3_INFERENCE / "inference" / "agent" / "python_tool_sandbox.py"
)
python_tool_sandbox = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(python_tool_sandbox)
run_sandboxed_python = python_tool_sandbox.run_sandboxed_python


def _fail(name: str, detail: str) -> None:
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def _ok(name: str) -> None:
    print(f"ok   {name}")


def _grid(player_col: int, dec_val: int):
    row = [0, 0, 0, 0, 0, 0]
    row[player_col] = 9
    row[5] = dec_val
    return [row]


def _frame_payload(grid, step, level=1):
    return {"ascii": "", "step": step, "level": level, "shape": [len(grid), len(grid[0])], "grid": grid}


def _initial_state(frames, valid_actions=("RIGHT",)):
    history = [{"action": "RIGHT", "frame": _frame_payload(g, step=i)} for i, g in enumerate(frames)]
    return {
        "current_frame": _frame_payload(frames[-1], step=len(frames) - 1),
        "history": history,
        "valid_actions": list(valid_actions),
        "last_action_result": {},
    }


EXTRACT_CODE = (
    "def extract(grid):\n"
    "    row = grid[0]\n"
    "    return {'player_col': row.index(9)}\n"
)
PREDICT_STATE_CODE = (
    "def predict_state(state, action):\n"
    "    return {'player_col': state['player_col'] + 1}\n"
)
# A "correct about the mechanic, wrong about decoration it never modeled"
# whole-grid predict -- exactly the real-world shape (HUD/animation noise).
PREDICT_GRID_CODE = (
    "def predict_grid(grid, action):\n"
    "    row = list(grid[0])\n"
    "    col = row.index(9)\n"
    "    row[col] = 0\n"
    "    row[col + 1] = 9\n"
    "    return [row]\n"
)


def main() -> None:
    frames = [_grid(0, 10), _grid(1, 11), _grid(2, 12)]

    # 1. verify_theory: pixel-perfect predict scores 0.0 (decoration always
    #    wrong) even though it gets the actual mechanic (player movement)
    #    exactly right every time.
    code = PREDICT_GRID_CODE + "result = verify_theory(predict_grid)\n"
    result = run_sandboxed_python(
        code=code, timeout_seconds=10, initial_state=_initial_state(frames),
        action_handler=lambda actions: _fail("verify pixel", f"unexpected action() call: {actions}"),
    )
    if result.get("error"):
        _fail("verify_theory pixel-perfect", result["error"])
    payload = result["result"]
    if payload["accuracy"] != 0.0:
        _fail("verify_theory pixel-perfect", f"expected accuracy 0.0 (decoration mismatch), got {payload!r}")
    _ok("pixel-perfect verify_theory scores 0.0 on a predict() that is right about the mechanic "
        "but never modeled an irrelevant decoration cell")

    # 2. verify_theory(extract=...): the SAME underlying knowledge, expressed
    #    over the abstracted state, scores 1.0 -- the decoration never enters
    #    the comparison at all.
    code = EXTRACT_CODE + PREDICT_STATE_CODE + "result = verify_theory(predict_state, extract=extract)\n"
    result = run_sandboxed_python(
        code=code, timeout_seconds=10, initial_state=_initial_state(frames),
        action_handler=lambda actions: _fail("verify extract", f"unexpected action() call: {actions}"),
    )
    if result.get("error"):
        _fail("verify_theory extract", result["error"])
    payload = result["result"]
    if payload["accuracy"] != 1.0:
        _fail("verify_theory extract", f"expected accuracy 1.0 abstracting away decoration, got {payload!r}")
    _ok("verify_theory(extract=extract) scores 1.0 on the same mechanic once decoration is abstracted away")

    # 2b. counterexamples report the abstracted states, not raw grids -- for
    #     a NON-dict state (plain int here); dict states are matched as a
    #     subset instead (see test_atlas_gemini_critique.py for that case).
    code = (
        "def extract_scalar(grid):\n    return grid[0].index(9)\n"
        + "def bad_predict(state, action):\n    return state\n"
        + "result = verify_theory(bad_predict, extract=extract_scalar)\n"
    )
    result = run_sandboxed_python(
        code=code, timeout_seconds=10, initial_state=_initial_state(frames),
        action_handler=lambda actions: _fail("verify extract wrong", f"unexpected action() call: {actions}"),
    )
    if result.get("error"):
        _fail("verify_theory extract wrong", result["error"])
    payload = result["result"]
    if payload["accuracy"] != 0.0 or not payload["counterexamples"]:
        _fail("verify_theory extract wrong", f"expected accuracy 0.0 with counterexamples, got {payload!r}")
    if "predicted_state" not in payload["counterexamples"][0] or "actual_state" not in payload["counterexamples"][0]:
        _fail("verify_theory extract wrong", f"expected abstracted-state counterexamples, got {payload!r}")
    _ok("verify_theory(extract=...) counterexamples report predicted_state/actual_state, not raw grids")

    # 3. plan_with_theory(extract=...): searches over the abstracted state
    #    space and finds a 2-step plan.
    code = (
        EXTRACT_CODE + PREDICT_STATE_CODE
        + "def goal(state):\n    return state['player_col'] == 4\n"
        + "result = plan_with_theory(predict_state, goal, actions=['RIGHT'], extract=extract, max_depth=3)\n"
    )
    result = run_sandboxed_python(
        code=code, timeout_seconds=10, initial_state=_initial_state(frames),
        action_handler=lambda actions: _fail("plan extract", f"unexpected action() call: {actions}"),
    )
    if result.get("error"):
        _fail("plan_with_theory extract", result["error"])
    payload = result["result"]
    if payload["plan"] != ["RIGHT", "RIGHT"] or payload["verified_accuracy"] != 1.0:
        _fail("plan_with_theory extract", f"expected a 2-step RIGHT plan at accuracy 1.0, got {payload!r}")
    _ok(f"plan_with_theory(extract=extract) finds the 2-step plan {payload['plan']} over the abstracted state")

    # 4. extract() raising is reported, not a crash.
    code = (
        "def extract(grid):\n    raise ValueError('boom')\n"
        + PREDICT_STATE_CODE
        + "result = verify_theory(predict_state, extract=extract)\n"
    )
    result = run_sandboxed_python(
        code=code, timeout_seconds=10, initial_state=_initial_state(frames),
        action_handler=lambda actions: _fail("verify extract raises", f"unexpected action() call: {actions}"),
    )
    if result.get("error"):
        _fail("verify_theory extract raises", result["error"])
    payload = result["result"]
    if payload["predict_errors"] != payload["transitions_tested"] or payload["accuracy"] != 0.0:
        _fail("verify_theory extract raises", f"expected every transition reported as an error, got {payload!r}")
    _ok("a raising extract() is reported as predict_errors on every transition, not a sandbox crash")

    # 5. execute_plan: the DEMONSTRATION this feature exists for. Same real
    #    environment (player moves right each step; a decoration cell also
    #    changes every step, unrelated to the mechanic). The pixel-perfect
    #    executor false-positives and stops after step 1 because it never
    #    modeled decoration; the extract-based executor runs the full plan.
    def _make_handler():
        live = {"col": 0, "dec": 10}

        def handler(actions):
            live["col"] += 1
            live["dec"] += 1
            grid = _grid(live["col"], live["dec"])
            return {
                "action_result": {
                    "executed": True, "board_changed": True, "level_completed": False,
                    "done": False, "game_over": False, "run_complete": False,
                },
                "state": {
                    "current_frame": _frame_payload(grid, step=99),
                    "history": [{"action": "RIGHT", "frame": _frame_payload(grid, step=99)}],
                    "valid_actions": ["RIGHT"],
                    "last_action_result": {},
                },
            }

        return handler

    code = PREDICT_GRID_CODE + "result = execute_plan(['RIGHT', 'RIGHT'], predict_grid)\n"
    result = run_sandboxed_python(
        code=code, timeout_seconds=10, initial_state=_initial_state(frames[:1]), action_handler=_make_handler(),
    )
    if result.get("error"):
        _fail("execute_plan pixel-perfect", result["error"])
    payload = result["result"]
    if payload.get("steps_executed") != 1 or not payload.get("stopped_early") or payload.get("stop_reason") != "predicted_state_mismatch":
        _fail("execute_plan pixel-perfect", f"expected a false-positive stop after step 1, got {payload!r}")
    _ok(f"WITHOUT extract, execute_plan false-positives and stops after 1 of 2 steps on decoration "
        f"it never modeled -- the exact real-world failure: {payload}")

    code = (
        EXTRACT_CODE + PREDICT_STATE_CODE
        + "result = execute_plan(['RIGHT', 'RIGHT'], predict_state, extract=extract)\n"
    )
    result = run_sandboxed_python(
        code=code, timeout_seconds=10, initial_state=_initial_state(frames[:1]), action_handler=_make_handler(),
    )
    if result.get("error"):
        _fail("execute_plan extract", result["error"])
    payload = result["result"]
    if payload.get("steps_executed") != 2 or payload.get("stopped_early") is not False:
        _fail("execute_plan extract", f"expected both steps to run cleanly, got {payload!r}")
    _ok(f"WITH extract=extract, execute_plan runs both steps cleanly -- decoration never enters the comparison: {payload}")

    print("\nAll atlas extract-theory checks passed.")


if __name__ == "__main__":
    main()

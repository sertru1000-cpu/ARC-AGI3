"""Exercise execute_plan(), the safe multi-step plan executor in atlas_src.

Backlog: plan_with_theory()'s res['note'] warned that a multi-step plan is
only as reliable as verify_theory's single-step checks, but three real
transcripts (ls20, sc25, sp80) showed the model never acted on that warning
-- it always fired the whole plan in one action() call, because the note
only becomes visible AFTER that call already ran in the same script. A
prompt-level nudge can react to the incident on the NEXT turn (see
test_atlas_plan_nudge.py), but cannot prevent the incident itself.

execute_plan() is real prevention: it runs a plan_with_theory() plan one
REAL step at a time, comparing each real outcome to what predict() forecast
for that exact step, and stops itself the moment they diverge -- bounding
the wasted real actions to the first bad step instead of the whole plan.

Drives the REAL sandboxed subprocess with a controllable fake mechanic: a
sliding-block game where predict() always forecasts a rightward shift, but
the REAL mechanic stops shifting partway through in one scenario -- exactly
the "mechanic saturates mid-plan" failure mode found live.
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


def _frame_payload(grid, step, level=1):
    return {"ascii": "", "step": step, "level": level, "shape": [len(grid), len(grid[0])], "grid": grid}


def _shift_right(grid):
    return [row[-1:] + row[:-1] for row in grid]


BASE = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def _initial_state(current_grid, history_grids, valid_actions=("RIGHT",)):
    history = [
        {"action": "RIGHT", "frame": _frame_payload(g, step=i)}
        for i, g in enumerate(history_grids)
    ]
    return {
        "current_frame": _frame_payload(current_grid, step=len(history_grids) - 1),
        "history": history,
        "valid_actions": list(valid_actions),
        "last_action_result": {},
    }


PREDICT_CODE = (
    "def predict(grid, action):\n"
    "    return [row[-1:] + row[:-1] for row in grid]\n"
)


def main() -> None:
    grids = [BASE, _shift_right(BASE)]

    # 1. A plan that behaves exactly as predicted end to end: executes fully,
    #    stopped_early=False, no divergence ever detected.
    real = {"grid": [list(row) for row in grids[-1]]}

    def _handler_always_shifts(actions):
        real["grid"] = _shift_right(real["grid"])
        return {
            "action_result": {
                "executed": True, "board_changed": True, "level_completed": False,
                "done": False, "game_over": False, "run_complete": False,
            },
            "state": {
                "current_frame": _frame_payload(real["grid"], step=99),
                "history": [{"action": "RIGHT", "frame": _frame_payload(real["grid"], step=99)}],
                "valid_actions": ["RIGHT"],
                "last_action_result": {"board_changed": True},
            },
        }

    code = (
        PREDICT_CODE
        + "res = plan_with_theory(predict, lambda g: False, actions=['RIGHT'], max_depth=1)\n"
        + "result = execute_plan(['RIGHT', 'RIGHT', 'RIGHT'], predict)\n"
    )
    result = run_sandboxed_python(
        code=code, timeout_seconds=10,
        initial_state=_initial_state(grids[-1], grids), action_handler=_handler_always_shifts,
    )
    if result.get("error"):
        _fail("full success", result["error"])
    payload = result["result"]
    if payload != {"steps_executed": 3, "stopped_early": False, "stop_reason": None,
                    "last_action_result": {"board_changed": True, "level_completed": False,
                                            "done": False, "game_over": False, "run_complete": False,
                                            "executed": True}}:
        _fail("full success", f"unexpected payload: {payload!r}")
    _ok("executes a plan fully when every real step matches predict()'s forecast")

    # 2. The mechanic saturates after step 1 (real transitions stop shifting),
    #    but predict() still forecasts a shift -- execute_plan must stop right
    #    after the mismatching step, not spend the rest of the plan blind.
    real2 = {"grid": [list(row) for row in grids[-1]], "calls": 0}

    def _handler_saturates(actions):
        real2["calls"] += 1
        if real2["calls"] == 1:
            real2["grid"] = _shift_right(real2["grid"])  # step 1: matches predict()
        # step 2+: mechanic stopped moving -- grid unchanged, predict() is now wrong
        return {
            "action_result": {
                "executed": True, "board_changed": real2["calls"] == 1, "level_completed": False,
                "done": False, "game_over": False, "run_complete": False,
            },
            "state": {
                "current_frame": _frame_payload(real2["grid"], step=99),
                "history": [{"action": "RIGHT", "frame": _frame_payload(real2["grid"], step=99)}],
                "valid_actions": ["RIGHT"],
                "last_action_result": {},
            },
        }

    code = (
        PREDICT_CODE
        + "result = execute_plan(['RIGHT', 'RIGHT', 'RIGHT'], predict)\n"
    )
    result = run_sandboxed_python(
        code=code, timeout_seconds=10,
        initial_state=_initial_state(grids[-1], grids), action_handler=_handler_saturates,
    )
    if result.get("error"):
        _fail("stops on mismatch", result["error"])
    payload = result["result"]
    if payload.get("steps_executed") != 2 or not payload.get("stopped_early") or payload.get("stop_reason") != "predicted_state_mismatch":
        _fail("stops on mismatch", f"expected 2 steps executed then a mismatch stop, got {payload!r}")
    if real2["calls"] != 2:
        _fail("stops on mismatch (real call count)", f"expected exactly 2 real actions spent, action_handler saw {real2['calls']}")
    _ok(f"stops itself after 2 of 3 steps once the real outcome diverges from predict() -- saved 1 real action: {payload}")

    # 3. A terminal result (level_completed) mid-plan is a graceful stop, not
    #    a failure -- stopped_early stays False.
    real3 = {"grid": [list(row) for row in grids[-1]], "calls": 0}

    def _handler_wins_on_step1(actions):
        real3["calls"] += 1
        real3["grid"] = _shift_right(real3["grid"])
        return {
            "action_result": {
                "executed": True, "board_changed": True, "level_completed": True,
                "done": False, "game_over": False, "run_complete": False,
            },
            "state": {
                "current_frame": _frame_payload(real3["grid"], step=99),
                "history": [{"action": "RIGHT", "frame": _frame_payload(real3["grid"], step=99)}],
                "valid_actions": ["RIGHT"],
                "last_action_result": {},
            },
        }

    code = (
        PREDICT_CODE
        + "result = execute_plan(['RIGHT', 'RIGHT', 'RIGHT'], predict)\n"
    )
    result = run_sandboxed_python(
        code=code, timeout_seconds=10,
        initial_state=_initial_state(grids[-1], grids), action_handler=_handler_wins_on_step1,
    )
    if result.get("error"):
        _fail("terminal stop", result["error"])
    payload = result["result"]
    if payload.get("steps_executed") != 1 or payload.get("stopped_early") is not False or real3["calls"] != 1:
        _fail("terminal stop", f"expected a graceful 1-step stop, got {payload!r}, real calls={real3['calls']}")
    _ok("stops gracefully (stopped_early=False) the instant level_completed fires, without spending more actions")

    # 4. An empty plan (already at goal) is a no-op: 0 steps, no crash.
    code = PREDICT_CODE + "result = execute_plan([], predict)\n"
    result = run_sandboxed_python(
        code=code, timeout_seconds=10,
        initial_state=_initial_state(grids[-1], grids),
        action_handler=lambda actions: _fail("empty plan", f"unexpected action() call: {actions}"),
    )
    if result.get("error"):
        _fail("empty plan", result["error"])
    payload = result["result"]
    if payload != {"steps_executed": 0, "stopped_early": False, "stop_reason": None, "last_action_result": None}:
        _fail("empty plan", f"unexpected payload: {payload!r}")
    _ok("an empty plan is a safe no-op -- zero real actions spent")

    print("\nAll atlas execute_plan checks passed.")


if __name__ == "__main__":
    main()

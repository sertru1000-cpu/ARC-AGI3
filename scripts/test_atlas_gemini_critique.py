"""Exercise the 4 fixes made in response to gemini-3.7-flash's critique (24.08)
of the new extract= design (asked manually by the user via AI Studio, using
scripts/ask_gemini_extract_review.py's prompt as the basis).

gemini-3.7-flash actually won lp85/ft09 (old harness, 21.08) by extracting a
full dict of positions but only ever CARING about a subset of it -- its own
lp85 code tracked `changed = [s for s in slots if state_before[s] != state_after[s]]`,
never validating the untouched slots. Our extract= (added earlier today)
still forced whole-dict equality, which is the SAME pixel-perfect bug one
level up. Four fixes, in the order Gemini ranked them:

1. plan_with_theory(force=True): bypasses the min_accuracy refusal -- the
   actual escape hatch for r11l-style noise that can never reach 0.6 under
   strict equality.
2. Dict states are matched as a SUBSET: predict() only has to get the keys
   it actually predicts right; an empty dict never counts as a match (the
   trivial-pass risk Gemini's own proposal did not flag).
3. execute_plan(goal=...): checked BEFORE a mismatch abort, so a plan that
   already reached its goal isn't thrown out over a cosmetic divergence.
4. verify_theory(transitions=...): test against a hand-picked sublist
   instead of full (possibly noisy) history.

Drives the REAL sandboxed subprocess, not a reimplementation.
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


# extract() returns a dict with BOTH the player AND a noisy decoration key,
# exactly the shape Gemini's critique describes (extract() "accidentally
# captures" more than the theory cares about).
EXTRACT_BOTH_CODE = (
    "def extract(grid):\n"
    "    row = grid[0]\n"
    "    return {'player_col': row.index(9), 'decoration': row[5]}\n"
)
# predict() only ever commits to player_col -- it does not know or care
# what decoration will be, unlike EXTRACT_BOTH_CODE which reports it anyway.
PARTIAL_PREDICT_CODE = (
    "def partial_predict(state, action):\n"
    "    return {'player_col': state['player_col'] + 1}\n"
)
EMPTY_PREDICT_CODE = "def empty_predict(state, action):\n    return {}\n"


def main() -> None:
    frames = [_grid(0, 10), _grid(1, 11), _grid(2, 12)]

    # 1. Subset match: a partial dict predict() (only player_col) verifies at
    #    1.0 even though extract() also reports a decoration key that changes
    #    unpredictably every transition and predict() never mentions it.
    code = EXTRACT_BOTH_CODE + PARTIAL_PREDICT_CODE + "result = verify_theory(partial_predict, extract=extract)\n"
    result = run_sandboxed_python(
        code=code, timeout_seconds=10, initial_state=_initial_state(frames),
        action_handler=lambda a: _fail("subset match", f"unexpected action(): {a}"),
    )
    if result.get("error"):
        _fail("subset match", result["error"])
    payload = result["result"]
    if payload["accuracy"] != 1.0:
        _fail("subset match", f"expected accuracy 1.0 (decoration key never checked), got {payload!r}")
    _ok("verify_theory(extract=...) matches a dict as a SUBSET -- a predict() that only "
        "commits to player_col scores 1.0 even though extract() also reports a noisy key it never mentions")

    # 1b. An empty dict never counts as a match, even trivially -- must not
    #     let a predict() that predicts nothing score 1.0 by vacuous truth.
    code = EXTRACT_BOTH_CODE + EMPTY_PREDICT_CODE + "result = verify_theory(empty_predict, extract=extract)\n"
    result = run_sandboxed_python(
        code=code, timeout_seconds=10, initial_state=_initial_state(frames),
        action_handler=lambda a: _fail("empty dict", f"unexpected action(): {a}"),
    )
    if result.get("error"):
        _fail("empty dict guard", result["error"])
    payload = result["result"]
    if payload["accuracy"] != 0.0 or payload["predict_errors"] != payload["transitions_tested"]:
        _fail("empty dict guard", f"expected every transition reported as an error, got {payload!r}")
    _ok("an empty predicted dict is reported as an error on every transition, never a vacuous match")

    # 1c. A genuinely wrong partial key is still caught -- subset matching
    #     isn't a free pass, just a narrower comparison.
    code = (
        EXTRACT_BOTH_CODE
        + "def wrong_predict(state, action):\n    return {'player_col': state['player_col']}\n"
        + "result = verify_theory(wrong_predict, extract=extract)\n"
    )
    result = run_sandboxed_python(
        code=code, timeout_seconds=10, initial_state=_initial_state(frames),
        action_handler=lambda a: _fail("wrong subset", f"unexpected action(): {a}"),
    )
    if result.get("error"):
        _fail("wrong subset key", result["error"])
    payload = result["result"]
    if payload["accuracy"] != 0.0 or "mismatched_keys (predicted,actual)" not in payload["counterexamples"][0]:
        _fail("wrong subset key", f"expected accuracy 0.0 with a mismatched_keys counterexample, got {payload!r}")
    _ok("a genuinely wrong predicted key is still caught -- subset matching narrows scope, doesn't waive correctness")

    # 2. plan_with_theory(force=True) bypasses the min_accuracy refusal.
    code = (
        "def bad_predict(grid, action):\n    return grid\n"
        "def goal(grid):\n    return grid[0][0] == 9\n"
        "result = plan_with_theory(bad_predict, goal, force=True, max_depth=1)\n"
    )
    result = run_sandboxed_python(
        code=code, timeout_seconds=10, initial_state=_initial_state(frames),
        action_handler=lambda a: _fail("force bypass", f"unexpected action(): {a}"),
    )
    if result.get("error"):
        _fail("force bypass", result["error"])
    payload = result["result"]
    if payload.get("reason") is not None and payload.get("reason", "").startswith("theory not good enough"):
        _fail("force bypass", f"force=True should have skipped the accuracy refusal, got {payload!r}")
    _ok(f"plan_with_theory(force=True) bypasses the min_accuracy refusal on an unverified/wrong theory: {payload.get('reason')}")

    # 2b. Without force, the SAME bad theory is still refused (default unchanged).
    code = (
        "def bad_predict(grid, action):\n    return grid\n"
        "def goal(grid):\n    return grid[0][0] == 9\n"
        "result = plan_with_theory(bad_predict, goal, max_depth=1)\n"
    )
    result = run_sandboxed_python(
        code=code, timeout_seconds=10, initial_state=_initial_state(frames),
        action_handler=lambda a: _fail("force default", f"unexpected action(): {a}"),
    )
    if result.get("error"):
        _fail("force default", result["error"])
    payload = result["result"]
    if payload["plan"] is not None or "not good enough" not in payload["reason"]:
        _fail("force default", f"expected the usual refusal when force is omitted, got {payload!r}")
    _ok("force=False (the default) still refuses to plan on an unverified/wrong theory -- unchanged")

    # 3. verify_theory(transitions=...) tests a hand-picked sublist instead
    #    of the noisy full history: a wrong-everywhere-except-index-1 predict()
    #    scores 0.0 against full history but 1.0 against just that transition.
    #    Decoration-free grids here (constant dec) -- isolates the custom
    #    transitions feature from the subset-matching one tested above.
    clean_frames = [_grid(0, 0), _grid(1, 0), _grid(2, 0)]
    code = (
        "def spotty_predict(grid, action):\n"
        "    row = list(grid[0])\n"
        "    col = row.index(9)\n"
        "    if col == 1:  # only correct starting from player_col==1\n"
        "        row[col] = 0\n"
        "        row[col + 1] = 9\n"
        "        return [row]\n"
        "    return grid\n"
        "full = verify_theory(spotty_predict)\n"
        "clean = verify_theory(spotty_predict, transitions=[t for t in transitions if t.before_frame is not None][1:2])\n"
        "result = {'full': full['accuracy'], 'clean': clean['accuracy']}\n"
    )
    result = run_sandboxed_python(
        code=code, timeout_seconds=10, initial_state=_initial_state(clean_frames),
        action_handler=lambda a: _fail("custom transitions", f"unexpected action(): {a}"),
    )
    if result.get("error"):
        _fail("custom transitions", result["error"])
    payload = result["result"]
    if payload["full"] != 0.5 or payload["clean"] != 1.0:
        _fail("custom transitions", f"expected full=0.5 (1 of 2 transitions correct), clean=1.0, got {payload!r}")
    _ok(f"verify_theory(transitions=transitions[1:2]) tests only the hand-picked slice: {payload}")

    # 4. execute_plan(goal=...): a cosmetic mismatch on step 2 does not cost
    #    the win if goal() is already satisfied after step 1.
    def _make_handler():
        live = {"col": 0, "dec": 10}

        def handler(actions):
            live["col"] += 1
            live["dec"] += 1  # decoration always drifts, unrelated to the goal
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

    # predict_state is wrong about decoration (always predicts +1, but the
    # goal only cares about player_col reaching 1) -- would normally abort
    # on the FIRST step's mismatch; goal() being satisfied after step 1 must
    # save it.
    code = (
        EXTRACT_BOTH_CODE
        + "def predict_state(state, action):\n"
        + "    return {'player_col': state['player_col'] + 1, 'decoration': state['decoration'] + 999}\n"
        + "def goal(state):\n    return state['player_col'] == 1\n"
        + "result = execute_plan(['RIGHT', 'RIGHT'], predict_state, extract=extract, goal=goal)\n"
    )
    result = run_sandboxed_python(
        code=code, timeout_seconds=10, initial_state=_initial_state(frames[:1]), action_handler=_make_handler(),
    )
    if result.get("error"):
        _fail("execute_plan goal check", result["error"])
    payload = result["result"]
    if payload.get("steps_executed") != 1 or payload.get("stop_reason") != "goal_reached" or payload.get("stopped_early") is not False:
        _fail("execute_plan goal check", f"expected a graceful goal_reached stop after step 1, got {payload!r}")
    _ok(f"execute_plan(goal=goal) stops gracefully with stop_reason='goal_reached' the instant the "
        f"goal is met, even though decoration would otherwise have mismatched: {payload}")

    # 4b. Without goal=, the SAME scenario still aborts on the mismatch (default unchanged).
    code = (
        EXTRACT_BOTH_CODE
        + "def predict_state(state, action):\n"
        + "    return {'player_col': state['player_col'] + 1, 'decoration': state['decoration'] + 999}\n"
        + "result = execute_plan(['RIGHT', 'RIGHT'], predict_state, extract=extract)\n"
    )
    result = run_sandboxed_python(
        code=code, timeout_seconds=10, initial_state=_initial_state(frames[:1]), action_handler=_make_handler(),
    )
    if result.get("error"):
        _fail("execute_plan no goal", result["error"])
    payload = result["result"]
    if payload.get("stop_reason") != "predicted_state_mismatch" or not payload.get("stopped_early"):
        _fail("execute_plan no goal", f"expected the usual mismatch abort when goal is omitted, got {payload!r}")
    _ok("without goal=, the same decoration mismatch still aborts the plan -- unchanged default")

    print("\nAll gemini-critique fixes verified.")


if __name__ == "__main__":
    main()

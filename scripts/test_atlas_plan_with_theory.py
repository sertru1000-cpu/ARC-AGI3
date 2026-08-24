"""Exercise verify_theory/plan_with_theory ported into atlas_src's sandbox.

Loads inference/agent/python_tool_sandbox.py directly by file path (bypassing
inference.agent's __init__, which pulls in tool_agent.py's heavier deps we
don't need here) and drives the real sandboxed subprocess with a synthetic
sliding-block game: the model's `code` defines predict(), verifies it against
a fake transition history, then plans a path to a goal -- exactly the
contract the ported functions promise.
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


def _frame_payload(grid, step, level=1):
    return {"ascii": "", "step": step, "level": level, "shape": [len(grid), len(grid[0])], "grid": grid}


def _shift_right(grid):
    return [row[-1:] + row[:-1] for row in grid]


def _fail(name, detail):
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def _ok(name):
    print(f"ok   {name}")


def _run(code, *, history_grids, valid_actions):
    history = [
        {"action": "RIGHT", "frame": _frame_payload(g, step=i)}
        for i, g in enumerate(history_grids)
    ]
    state = {
        "current_frame": _frame_payload(history_grids[-1], step=len(history_grids) - 1),
        "history": history,
        "valid_actions": valid_actions,
        "last_action_result": {},
    }
    return run_sandboxed_python(
        code=code,
        timeout_seconds=10,
        initial_state=state,
        action_handler=lambda actions: _fail("action_handler", f"unexpected action() call: {actions}"),
    )


def main() -> None:
    base = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    grids = [base]
    for _ in range(6):
        grids.append(_shift_right(grids[-1]))

    # 1. grid is exposed publicly on FrameView.
    result = _run("result = current_frame.grid", history_grids=grids, valid_actions=["RIGHT", "UP"])
    if result.get("error"):
        _fail("grid exposed", result["error"])
    if result.get("result") != grids[-1]:
        _fail("grid exposed", f"got {result.get('result')!r}")
    _ok("FrameView.grid is publicly readable")

    # 2. verify_theory: a correct predict() scores accuracy 1.0.
    code = (
        "def predict(grid, action):\n"
        "    return [row[-1:] + row[:-1] for row in grid]\n"
        "result = verify_theory(predict)\n"
    )
    result = _run(code, history_grids=grids, valid_actions=["RIGHT", "UP"])
    if result.get("error"):
        _fail("verify_theory correct", result["error"])
    acc = result["result"]["accuracy"]
    if acc != 1.0:
        _fail("verify_theory correct", f"expected accuracy 1.0, got {result['result']!r}")
    _ok(f"verify_theory scores a correct predict() at accuracy {acc}")

    # 3. verify_theory: a wrong predict() scores accuracy 0.0 and reports a counterexample.
    code = (
        "def predict(grid, action):\n"
        "    return grid\n"  # no-op theory, wrong for every transition
        "result = verify_theory(predict)\n"
    )
    result = _run(code, history_grids=grids, valid_actions=["RIGHT", "UP"])
    if result.get("error"):
        _fail("verify_theory wrong", result["error"])
    payload = result["result"]
    if payload["accuracy"] != 0.0 or not payload["counterexamples"]:
        _fail("verify_theory wrong", f"expected accuracy 0.0 with counterexamples, got {payload!r}")
    _ok("verify_theory catches a wrong predict() with a counterexample")

    # 4. plan_with_theory: refuses to plan below min_accuracy.
    code = (
        "def bad_predict(grid, action):\n"
        "    return grid\n"
        "def goal(grid):\n"
        "    return grid[0][0] == 9\n"
        "result = plan_with_theory(bad_predict, goal)\n"
    )
    result = _run(code, history_grids=grids, valid_actions=["RIGHT", "UP"])
    if result.get("error"):
        _fail("plan_with_theory refusal", result["error"])
    payload = result["result"]
    if payload["plan"] is not None or "not good enough" not in payload["reason"]:
        _fail("plan_with_theory refusal", f"expected a refusal, got {payload!r}")
    _ok("plan_with_theory refuses to plan on an unverified/wrong theory")

    # 5. plan_with_theory: finds a real plan with a correct predict() (goal reachable in 2 RIGHTs).
    target = _shift_right(_shift_right(grids[-1]))
    code = (
        "def predict(grid, action):\n"
        "    return [row[-1:] + row[:-1] for row in grid]\n"
        f"def goal(grid):\n"
        f"    return grid == {target!r}\n"
        "result = plan_with_theory(predict, goal, actions=['RIGHT'], max_depth=3)\n"
    )
    result = _run(code, history_grids=grids, valid_actions=["RIGHT", "UP"])
    if result.get("error"):
        _fail("plan_with_theory success", result["error"])
    payload = result["result"]
    if payload["plan"] != ["RIGHT", "RIGHT"]:
        _fail("plan_with_theory success", f"expected a 2-step RIGHT plan, got {payload!r}")
    _ok(f"plan_with_theory finds the 2-step plan {payload['plan']} (accuracy {payload['verified_accuracy']})")
    if not payload.get("note") or "2 predicted steps" not in payload["note"]:
        _fail("multi-step note", f"expected a note warning about chained predictions, got {payload!r}")
    _ok("a >1-step plan carries a note warning verify_theory only checked single transitions")

    # 5b. plan_with_theory: a 1-step plan carries no extrapolation note (nothing chained).
    one_step_target = _shift_right(grids[-1])
    code = (
        "def predict(grid, action):\n"
        "    return [row[-1:] + row[:-1] for row in grid]\n"
        f"def goal(grid):\n"
        f"    return grid == {one_step_target!r}\n"
        "result = plan_with_theory(predict, goal, actions=['RIGHT'], max_depth=3)\n"
    )
    result = _run(code, history_grids=grids, valid_actions=["RIGHT", "UP"])
    if result.get("error"):
        _fail("plan_with_theory 1-step", result["error"])
    payload = result["result"]
    if payload["plan"] != ["RIGHT"] or payload.get("note") is not None:
        _fail("plan_with_theory 1-step", f"expected a 1-step plan with note=None, got {payload!r}")
    _ok("a 1-step plan carries no extrapolation note -- nothing was chained")

    # 6. plan_with_theory: MOUSE spec is fed to predict() in display-string form.
    # Needs at least one MOUSE transition in history, or verify_theory's
    # MOUSE-filtered check has nothing to test and refuses before predict()
    # is ever called for planning -- that refusal is itself correct behavior,
    # just not what this check is isolating.
    mouse_history = [
        {"action": "RIGHT", "frame": _frame_payload(g, step=i)}
        for i, g in enumerate(grids)
    ] + [{"action": "MOUSE(row=1, col=1)", "frame": _frame_payload(_shift_right(grids[-1]), step=len(grids))}]
    state = {
        "current_frame": mouse_history[-1]["frame"],
        "history": mouse_history,
        "valid_actions": ["RIGHT", "UP", "MOUSE"],
        "last_action_result": {},
    }
    code = (
        "def predict(grid, action):\n"
        "    seen.append(action)\n"
        "    return [row[-1:] + row[:-1] for row in grid]\n"
        "seen = []\n"
        "def goal(grid):\n"
        "    return False\n"  # never reached -- we only care what predict() saw
        "plan_with_theory(predict, goal, actions=[{'action': 'MOUSE', 'row': 4, 'col': 7}], max_depth=1)\n"
        "result = seen\n"
    )
    result = run_sandboxed_python(
        code=code,
        timeout_seconds=10,
        initial_state=state,
        action_handler=lambda actions: _fail("action_handler", f"unexpected action() call: {actions}"),
    )
    if result.get("error"):
        _fail("MOUSE display format", result["error"])
    if "MOUSE(row=4, col=7)" not in (result.get("result") or []):
        _fail("MOUSE display format", f"expected predict() to see 'MOUSE(row=4, col=7)', got {result.get('result')!r}")
    _ok("plan_with_theory formats a MOUSE spec into 'MOUSE(row=4, col=7)' for predict()")

    print("\nAll atlas plan_with_theory checks passed.")


if __name__ == "__main__":
    main()

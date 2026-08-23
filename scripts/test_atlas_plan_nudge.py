"""Exercise the atlas plan-with-theory/verify-theory nudge in tool_agent.py.

The prompt alone gets a documented tool used in ~0.2% of turns (measured on
our own harness's C0 mechanism). This ports the fix that actually worked
there: the harness tracks whether the model has verified a theory and
whether it has planned recently, and injects a reminder into the NEXT
turn's prompt until the model acts -- not a static one-time mention.

Drives ToolAgent._run_python_tool with real code strings against a real
sandboxed subprocess (writing an actual runtime-state file, like the real
harness does), then inspects _build_user_prompt's output for the checkpoint
text. No network calls -- ToolAgent() with no args never reaches the LLM
in this test, only the local sandbox and the prompt builder are exercised.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "tufa-arc-agi-framework" / "src"))
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "ARC3-Inference"))

from inference.agent.tool_agent import ToolAgent  # noqa: E402


def _fail(name: str, detail: str) -> None:
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def _ok(name: str) -> None:
    print(f"ok   {name}")


def _write_state(path: Path, grid, step: int, level: int = 1, history=None) -> None:
    payload = {
        "current_frame": {"grid": grid, "step": step, "level": level},
        "history": history or [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _shift_right(grid):
    return [row[-1:] + row[:-1] for row in grid]


def main() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="atlas_nudge_test_"))
    state_path = tmp_dir / "run_state.json"

    grid = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    history = [{"action": "RIGHT", "frame": {"grid": g, "step": i, "level": 1}}
               for i, g in enumerate([grid, _shift_right(grid)])]
    _write_state(state_path, _shift_right(grid), step=1, history=history)

    agent = ToolAgent(model="test-model")

    # 1. Fresh session, few python calls: neither checkpoint fires yet.
    prompt = agent._build_user_prompt(1, valid_actions=["UP", "RIGHT"])
    if "[atlas checkpoint]" in prompt:
        _fail("quiet at start", "checkpoint fired before enough calls had happened")
    _ok("silent on a fresh session with few python calls")

    # 2. Enough python-tool calls without ever verifying -> theory checkpoint.
    for _ in range(4):
        agent._run_python_tool(state_path, {"code": "result = 1\n"})
    prompt = agent._build_user_prompt(5, valid_actions=["UP", "RIGHT"])
    if "verify_theory(predict)" not in prompt or "[atlas checkpoint]" not in prompt:
        _fail("theory nag", f"expected the theory checkpoint, prompt tail: {prompt[-400:]!r}")
    _ok("nags to verify a theory once enough python calls passed without one")

    # 3. A real verify_theory( call with wrong predict() -> accuracy 0.0, still nags to verify.
    agent._run_python_tool(
        state_path,
        {"code": "def predict(grid, action):\n    return grid\nresult = verify_theory(predict)\n"},
    )
    if agent._atlas_last_verified_accuracy != 0.0:
        _fail("accuracy captured (wrong theory)", str(agent._atlas_last_verified_accuracy))
    prompt = agent._build_user_prompt(6, valid_actions=["UP", "RIGHT"])
    if "verify_theory(predict)" not in prompt:
        _fail("still nagging below 0.6", prompt[-400:])
    _ok(f"captured accuracy={agent._atlas_last_verified_accuracy} and keeps nagging to verify below 0.6")

    # 4. A correct predict() verifies at 1.0. It has never planned before
    #    (sentinel last-plan-index is far in the past), so the plan
    #    checkpoint should fire on the very next prompt -- no need to wait,
    #    a freshly verified theory with zero plans made so far is exactly
    #    the situation the nudge exists for. Matches our own harness's
    #    identical sentinel-based PLAN_NAG_EVERY logic.
    agent._run_python_tool(
        state_path,
        {
            "code": (
                "def predict(grid, action):\n"
                "    return [row[-1:] + row[:-1] for row in grid]\n"
                "result = verify_theory(predict)\n"
            )
        },
    )
    if agent._atlas_last_verified_accuracy != 1.0:
        _fail("accuracy captured (correct theory)", str(agent._atlas_last_verified_accuracy))
    _ok(f"captured accuracy={agent._atlas_last_verified_accuracy} for a correct predict()")

    prompt = agent._build_user_prompt(7, valid_actions=["UP", "RIGHT"])
    if "plan_with_theory(predict, goal)" not in prompt or "accuracy 1.00" not in prompt:
        _fail("plan nag on first verify", f"expected the plan checkpoint, prompt tail: {prompt[-400:]!r}")
    _ok("nags to plan (quoting the verified accuracy) right after the first successful verify")

    # 5. Calling plan_with_theory resets the cooldown -> checkpoint goes quiet again.
    agent._run_python_tool(
        state_path,
        {
            "code": (
                "def predict(grid, action):\n"
                "    return [row[-1:] + row[:-1] for row in grid]\n"
                "def goal(grid):\n"
                "    return False\n"
                "result = plan_with_theory(predict, goal)\n"
            )
        },
    )
    prompt = agent._build_user_prompt(11, valid_actions=["UP", "RIGHT"])
    if "[atlas checkpoint]" in prompt:
        _fail("cooldown reset", f"expected silence right after planning, got: {prompt[-400:]!r}")
    _ok("goes quiet again immediately after a real plan_with_theory( call")

    print("\nAll atlas plan-nudge checks passed.")


if __name__ == "__main__":
    main()

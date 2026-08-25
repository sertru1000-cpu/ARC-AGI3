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

    # 2. Enough python-tool calls without ever verifying -- theory checkpoint
    #    is DISABLED as of 24.08 (found live on r11l/v12: it can read as a
    #    hard gate against acting at all when verified_accuracy>=0.6 is hard
    #    to reach, causing total paralysis -- 1 real action in 4.4h). Must
    #    stay silent here, not nag.
    for _ in range(4):
        agent._run_python_tool(state_path, {"code": "result = 1\n"})
    prompt = agent._build_user_prompt(5, valid_actions=["UP", "RIGHT"])
    if "[atlas checkpoint]" in prompt:
        _fail("theory checkpoint disabled", f"expected silence (disabled), got: {prompt[-400:]!r}")
    _ok("stays silent even after enough calls without verifying -- theory checkpoint is disabled")

    # 3. A real verify_theory( call with wrong predict() -> accuracy 0.0 is
    #    still captured (the tool itself still works), but still no nag.
    agent._run_python_tool(
        state_path,
        {"code": "def predict(grid, action):\n    return grid\nresult = verify_theory(predict)\n"},
    )
    if agent._atlas_last_verified_accuracy != 0.0:
        _fail("accuracy captured (wrong theory)", str(agent._atlas_last_verified_accuracy))
    prompt = agent._build_user_prompt(6, valid_actions=["UP", "RIGHT"])
    if "[atlas checkpoint]" in prompt:
        _fail("still disabled below 0.6", prompt[-400:])
    _ok(f"captured accuracy={agent._atlas_last_verified_accuracy} but still does not nag (disabled)")

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

    # 6. Note-enforcement: a >1-step plan fired via a SINGLE action() call in
    #    the same script must queue a one-shot checkpoint for the NEXT turn.
    #    Needs a fake step_env -- action() is unavailable without one.
    def _fake_step_env(payload):
        return {
            "executed": True,
            "level": 1,
            "score": 0,
            "reward": 0.0,
            "board_changed": True,
            "done": False,
            "level_completed": False,
            "game_over": False,
            "run_complete": False,
            "valid_actions": ["UP", "RIGHT", "DOWN", "LEFT"],
        }

    agent2 = ToolAgent(model="test-model")
    agent2._step_env_callback = _fake_step_env
    agent2._current_valid_actions = ["UP", "RIGHT", "DOWN", "LEFT"]
    # current_frame in state_path is _shift_right(grid) (1 shift). RIGHT has
    # period 3 on this 3-column row, so the raw `grid` constant (0 shifts) is
    # reachable in exactly 2 more RIGHT actions from there -- a genuine
    # 2-step plan, not a degenerate 0- or 1-step one.
    target = grid
    agent2._run_python_tool(
        state_path,
        {
            "code": (
                "def predict(grid, action):\n"
                "    return [row[-1:] + row[:-1] for row in grid]\n"
                f"def goal(grid):\n"
                f"    return grid == {target!r}\n"
                "res = plan_with_theory(predict, goal, actions=['RIGHT'], max_depth=3)\n"
                "if res.get('plan'):\n"
                "    action(res['plan'])\n"
                "result = res\n"
            )
        },
    )
    if agent2._atlas_note_incident is None:
        _fail("note incident captured", "expected a queued note-enforcement incident after a 3-step plan fired at once")
    if "2 steps" not in agent2._atlas_note_incident:
        _fail("note incident step count", agent2._atlas_note_incident)
    _ok(f"captured the incident when a multi-step plan fired via one action() call: {agent2._atlas_note_incident[:70]}...")

    prompt = agent2._build_user_prompt(1, valid_actions=["UP", "RIGHT"])
    if "composed-rollout risk" not in prompt or "[atlas checkpoint]" not in prompt:
        _fail("note enforcement checkpoint injected", f"expected it in the next prompt, tail: {prompt[-500:]!r}")
    if agent2._atlas_note_incident is not None:
        _fail("incident cleared after injection", "expected it to reset to None once shown")
    _ok("injects the note-enforcement checkpoint into the very next turn, then clears the incident")

    prompt = agent2._build_user_prompt(2, valid_actions=["UP", "RIGHT"])
    if "composed-rollout risk" in prompt:
        _fail("one-shot, not recurring", f"expected silence on the turn after, got: {prompt[-400:]!r}")
    _ok("stays quiet on the following turn -- one-shot, not a recurring nag")

    # 7. A 1-step plan (no res['note']) fired via action() must NOT trigger it.
    agent3 = ToolAgent(model="test-model")
    agent3._step_env_callback = _fake_step_env
    agent3._current_valid_actions = ["UP", "RIGHT", "DOWN", "LEFT"]
    one_step_target = _shift_right(grid)
    agent3._run_python_tool(
        state_path,
        {
            "code": (
                "def predict(grid, action):\n"
                "    return [row[-1:] + row[:-1] for row in grid]\n"
                f"def goal(grid):\n"
                f"    return grid == {one_step_target!r}\n"
                "res = plan_with_theory(predict, goal, actions=['RIGHT'], max_depth=3)\n"
                "if res.get('plan'):\n"
                "    action(res['plan'])\n"
                "result = res\n"
            )
        },
    )
    if agent3._atlas_note_incident is not None:
        _fail("1-step plan stays quiet", f"a 1-step plan has no note; should not trigger: {agent3._atlas_note_incident}")
    _ok("a 1-step plan fired via action() does not trigger the note-enforcement checkpoint")

    print("\nAll atlas plan-nudge checks passed.")


if __name__ == "__main__":
    main()

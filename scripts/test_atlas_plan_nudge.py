"""Exercise the atlas plan-with-theory/verify-theory nudge in tool_agent.py.

The prompt alone gets a documented tool used in ~0.2% of turns (measured on
our own harness's C0 mechanism). This ports the fix that actually worked
there: the harness tracks whether the model has verified a theory and
whether it has planned recently, and injects a reminder into the NEXT
turn's prompt until the model acts -- not a static one-time mention.

25.08: also exercises the force-act circuit breaker added after v15 showed
disabling the theory checkpoint outright (the 24.08 fix for r11l's total
paralysis) had its own cost -- 0 verify_theory/plan_with_theory/execute_plan
calls across 612 real python-tool calls, since PLAN_CHECKPOINT can only fire
after a theory is already verified. The checkpoint is back on with softened
wording, backstopped by a hard, wording-independent override that fires once
too many python calls pass with zero real action() calls.

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

from inference.agent.tool_agent import (  # noqa: E402
    ToolAgent,
    _ATLAS_GOAL_RECONSIDER_AFTER_CALLS,
    _ATLAS_MEMO_NUDGE_AFTER_CALLS,
)


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

    # Fake step_env so `action()` is available on the MAIN agent -- needed
    # for step 3c below (a real action resetting the force-act counter).
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

    agent._step_env_callback = _fake_step_env
    agent._current_valid_actions = ["UP", "RIGHT", "DOWN", "LEFT"]

    # 1. Fresh session, few python calls: neither checkpoint fires yet.
    prompt = agent._build_user_prompt(1, valid_actions=["UP", "RIGHT"])
    if "[atlas checkpoint]" in prompt:
        _fail("quiet at start", "checkpoint fired before enough calls had happened")
    _ok("silent on a fresh session with few python calls")

    # 1b. Write to memo once, early, so the rest of THIS agent's session
    #    never triggers the new (26.08) memo checkpoint -- keeps every
    #    downstream assertion below about the ORIGINAL checkpoints isolated
    #    from it, the same way 5b/5c interleave real actions to isolate
    #    goal-reconsider from force-act. The memo checkpoint itself gets its
    #    own dedicated coverage in step 8 below.
    agent._run_python_tool(state_path, {"code": "action(['RIGHT'])\nmemo['seen'] = True\nresult = 1\n"})
    if not agent._atlas_memo_ever_written:
        _fail("memo write detected", "expected _atlas_memo_ever_written=True after writing to memo")

    # 2. Enough python-tool calls without ever verifying -- theory checkpoint
    #    was disabled 24.08-25.08 (found live on r11l/v12: it can read as a
    #    hard gate against acting at all when verified_accuracy>=0.6 is hard
    #    to reach, causing total paralysis -- 1 real action in 4.4h), but
    #    disabling it outright turned out to have its own cost: v15 showed
    #    verify_theory/plan_with_theory/execute_plan dropping to 0 calls
    #    across 612 real python-tool calls with NEITHER checkpoint able to
    #    fire. Re-enabled 25.08 with softened wording (no more "do not skip
    #    this turn"/"wastes actions" framing) plus the structural circuit
    #    breaker in steps 3b/3c below.
    for _ in range(4):
        agent._run_python_tool(state_path, {"code": "result = 1\n"})
    prompt = agent._build_user_prompt(5, valid_actions=["UP", "RIGHT"])
    if "[atlas checkpoint]" not in prompt or "THIS turn, write predict" not in prompt:
        _fail("theory checkpoint fires", f"expected the theory checkpoint, got: {prompt[-400:]!r}")
    if "wastes actions that are scored quadratically" in prompt:
        _fail("theory checkpoint softened", "old hard-gate wording must be gone -- it's what caused r11l's paralysis")
    if "suggestion, not a requirement" in prompt:
        _fail("theory checkpoint re-strengthened", "25.08 v16 found this framing drops adoption to 0/662 real calls -- must be gone")
    _ok("theory checkpoint fires with re-strengthened (but non-hard-gate) wording")

    # 3. A real verify_theory( call with wrong predict() -> accuracy 0.0 is
    #    captured, and the checkpoint keeps nagging (still below 0.6).
    agent._run_python_tool(
        state_path,
        {"code": "def predict(grid, action):\n    return grid\nresult = verify_theory(predict)\n"},
    )
    if agent._atlas_last_verified_accuracy != 0.0:
        _fail("accuracy captured (wrong theory)", str(agent._atlas_last_verified_accuracy))
    prompt = agent._build_user_prompt(6, valid_actions=["UP", "RIGHT"])
    if "[atlas checkpoint]" not in prompt:
        _fail("still nags below 0.6", prompt[-400:])
    _ok(f"captured accuracy={agent._atlas_last_verified_accuracy} and keeps nagging (still below 0.6)")

    # 3b. The force-act circuit breaker: after _ATLAS_FORCE_ACT_AFTER_CALLS
    #    (8) python calls in a row with zero real action() calls, it
    #    overrides the theory checkpoint outright -- the structural fix for
    #    r11l's paralysis, independent of whether the softened wording above
    #    is itself enough to prevent a model from over-literal-reading it.
    for _ in range(3):  # calls 6, 7, 8 -- none of them call action()
        agent._run_python_tool(state_path, {"code": "result = 1\n"})
    if agent._atlas_calls_since_real_action != 8:
        _fail("force-act counter", f"expected 8 calls since the last real action, got {agent._atlas_calls_since_real_action}")
    prompt = agent._build_user_prompt(9, valid_actions=["UP", "RIGHT"])
    if "8 `python` calls in a row" not in prompt:
        _fail("force-act override fires", f"expected the force-act override, got: {prompt[-500:]!r}")
    if "THIS turn, write predict" in prompt:
        _fail("force-act overrides theory checkpoint", "the theory checkpoint must not ALSO appear once force-act fires")
    _ok("force-act override fires after 8 calls with zero real actions, overriding the theory checkpoint")

    # 3c. A real action() call resets the counter immediately.
    agent._run_python_tool(state_path, {"code": "action(['RIGHT'])\nresult = 1\n"})
    if agent._atlas_calls_since_real_action != 0:
        _fail("force-act counter resets", f"expected 0 right after a real action() call, got {agent._atlas_calls_since_real_action}")
    _ok("a real action() call resets the force-act counter to 0")

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

    # 5b. Goal-reconsider checkpoint: many verify_theory( calls that never
    #     reach 0.6 accuracy should eventually stop nudging "refine
    #     predict()" and instead suggest the GOAL model might be wrong --
    #     found on dc22 (Gemini teacher data): 221 verify_theory calls
    #     cycling through 4 unrelated wrong theories, never once suggested
    #     to reconsider the goal instead of the mechanic. Interleaves a real
    #     action() call every other verify_theory attempt so
    #     _atlas_calls_since_real_action never crosses the force-act
    #     threshold -- isolating this from that check (see step 3b: a model
    #     that ALSO never acts at all should hit force-act first, which is
    #     correct and covered separately).
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

    goal_agent = ToolAgent(model="test-model")
    goal_agent._step_env_callback = _fake_step_env
    goal_agent._current_valid_actions = ["UP", "RIGHT", "DOWN", "LEFT"]
    for _ in range(_ATLAS_GOAL_RECONSIDER_AFTER_CALLS):
        goal_agent._run_python_tool(
            state_path,
            {"code": "def predict(grid, action):\n    return grid\nresult = verify_theory(predict)\n"},
        )
        goal_agent._run_python_tool(state_path, {"code": "action(['RIGHT'])\nresult = 1\n"})
    if goal_agent._atlas_verify_theory_call_count != _ATLAS_GOAL_RECONSIDER_AFTER_CALLS:
        _fail("verify_theory call count tracked", str(goal_agent._atlas_verify_theory_call_count))
    if goal_agent._atlas_calls_since_real_action >= _ATLAS_GOAL_RECONSIDER_AFTER_CALLS:
        _fail("force-act stays clear", "test setup should keep force-act's counter low -- check the interleaving")
    prompt = goal_agent._build_user_prompt(1, valid_actions=["UP", "RIGHT"])
    if "reconsider your GOAL model" not in prompt:
        _fail("goal-reconsider checkpoint fires", f"expected it after {_ATLAS_GOAL_RECONSIDER_AFTER_CALLS} failed verify_theory calls, got: {prompt[-500:]!r}")
    if "THIS turn, write predict" in prompt:
        _fail("goal-reconsider overrides theory checkpoint", "the plain theory checkpoint must not ALSO appear once this fires")
    _ok(f"goal-reconsider checkpoint fires after {_ATLAS_GOAL_RECONSIDER_AFTER_CALLS} failed verify_theory( calls, overriding the theory checkpoint")

    # 5c. One call short of the threshold -> the plain theory checkpoint still fires instead.
    short_agent = ToolAgent(model="test-model")
    short_agent._step_env_callback = _fake_step_env
    short_agent._current_valid_actions = ["UP", "RIGHT", "DOWN", "LEFT"]
    for _ in range(_ATLAS_GOAL_RECONSIDER_AFTER_CALLS - 1):
        short_agent._run_python_tool(
            state_path,
            {"code": "def predict(grid, action):\n    return grid\nresult = verify_theory(predict)\n"},
        )
        short_agent._run_python_tool(state_path, {"code": "action(['RIGHT'])\nresult = 1\n"})
    prompt = short_agent._build_user_prompt(1, valid_actions=["UP", "RIGHT"])
    if "reconsider your GOAL model" in prompt:
        _fail("goal-reconsider respects the threshold", f"must not fire one call short of the threshold, got: {prompt[-500:]!r}")
    if "THIS turn, write predict" not in prompt:
        _fail("theory checkpoint still fires below the goal-reconsider threshold", prompt[-500:])
    _ok("one call short of the threshold, the plain theory checkpoint fires instead of goal-reconsider")

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

    # 8. Memo checkpoint (26.08): fires once enough python calls pass with
    #    memo never written AND nothing higher-priority is active. Every
    #    call here calls action() (avoids force-act) and re-plans every time
    #    (keeps accuracy >= 0.6 so theory stays quiet, and the plan-checkpoint
    #    cooldown at 0 so it doesn't out-rank memo either) -- isolating the
    #    memo checkpoint specifically from the other four in the chain.
    memo_agent = ToolAgent(model="test-model")
    memo_agent._step_env_callback = _fake_step_env
    memo_agent._current_valid_actions = ["UP", "RIGHT", "DOWN", "LEFT"]
    memo_agent._run_python_tool(
        state_path,
        {
            "code": (
                "def predict(grid, action):\n"
                "    return [row[-1:] + row[:-1] for row in grid]\n"
                "result = verify_theory(predict)\n"
                "action(['RIGHT'])\n"
            )
        },
    )
    replan_code = (
        "def predict(grid, action):\n"
        "    return [row[-1:] + row[:-1] for row in grid]\n"
        "def goal(grid):\n"
        "    return False\n"
        "res = plan_with_theory(predict, goal)\n"
        "action(['RIGHT'])\n"
        "result = res\n"
    )
    for _ in range(_ATLAS_MEMO_NUDGE_AFTER_CALLS - 2):
        memo_agent._run_python_tool(state_path, {"code": replan_code})
    if memo_agent._atlas_python_call_index != _ATLAS_MEMO_NUDGE_AFTER_CALLS - 1:
        _fail("memo test setup call count", str(memo_agent._atlas_python_call_index))
    prompt = memo_agent._build_user_prompt(1, valid_actions=["UP", "RIGHT"])
    if "written nothing to `memo`" in prompt:
        _fail("memo checkpoint respects the threshold", f"must not fire one call short of the threshold, got: {prompt[-500:]!r}")
    _ok("one call short of the memo threshold, stays quiet")

    memo_agent._run_python_tool(state_path, {"code": replan_code})
    prompt = memo_agent._build_user_prompt(2, valid_actions=["UP", "RIGHT"])
    if "written nothing to `memo`" not in prompt:
        _fail("memo checkpoint fires", f"expected it at the threshold with memo never written, got: {prompt[-500:]!r}")
    _ok(f"memo checkpoint fires after {_ATLAS_MEMO_NUDGE_AFTER_CALLS} python calls with memo never written")

    # 8b. Writing to memo clears it immediately.
    memo_agent._run_python_tool(state_path, {"code": "action(['RIGHT'])\nmemo['anchor'] = {'row': 1, 'col': 2}\nresult = 1\n"})
    prompt = memo_agent._build_user_prompt(3, valid_actions=["UP", "RIGHT"])
    if "written nothing to `memo`" in prompt:
        _fail("memo checkpoint clears after a write", f"expected silence right after writing to memo, got: {prompt[-500:]!r}")
    _ok("goes quiet immediately after the model writes to memo")

    print("\nAll atlas plan-nudge checks passed.")


if __name__ == "__main__":
    main()

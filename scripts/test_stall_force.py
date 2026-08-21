"""Smoke test for the structural stall-break in LLMPolicy._forced_turn.

Scripts a MockLLM that only ever inspects the board (never calls action()),
reproducing the 20.08 stand failure mode (wa30: stall_turns -> 40+, 53
identical inspection-only replies, ignored NUDGE_NO_ACTION). Verifies the
harness stops asking after MY_AGENT_STALL_FORCE turns and takes a real
action itself, without spending an extra LLM call.

Run:  AGENT_BRAIN=llm .venv/Scripts/python.exe scripts/test_stall_force.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

os.environ["AGENT_BRAIN"] = "llm"
os.environ["MY_AGENT_STALL_FORCE"] = "4"
os.environ["MY_AGENT_MAX_TURNS"] = "8"

INSPECT_ONLY = """WORLD_MODEL:
controls: unknown yet
goal: unknown, exploring
plan: inspect board again

```python
print("Current Frame ASCII:")
print(current_frame.ascii)
```"""

# 6 identical inspection-only replies: turns 1-4 are real LLM calls (stall_turns
# climbs 1,2,3,4); turn 5 must be forced (LLM skipped); turn 6 resumes normal.
SCRIPT = [INSPECT_ONLY] * 6


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import arc_agi
    from arc_agi import OperationMode

    from agent.harness.llm import MockLLM
    import agent.harness.llm as llm_mod

    mock = MockLLM(SCRIPT)
    llm_mod.default_backend = lambda: mock

    import importlib.util

    spec = importlib.util.spec_from_file_location("user_agent_module", ROOT / "agent" / "my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("ls20")
    agent = module.MyAgent(
        card_id="mock-test", game_id="ls20", agent_name="mock.stall",
        ROOT_URL="http://localhost", record=False, arc_env=env, tags=["test"],
    )
    agent.main()

    policy = agent.policy
    print("\n===== RESULT =====")
    print("policy turns:", policy.turns if policy else 0)
    print("mock LLM calls:", mock.calls)
    print("actions taken (env):", agent.action_counter)
    print("final stall_turns:", policy.stall_turns if policy else None)

    assert policy is not None
    assert policy.turns >= 5, f"expected at least 5 policy turns, got {policy.turns}"
    forced_count = policy.turns - mock.calls
    assert forced_count >= 1, (
        f"expected >=1 forced turn (turns - LLM calls), got turns={policy.turns} calls={mock.calls}")
    # The scripted LLM NEVER calls action() itself, so any executed action
    # can only have come from a forced turn.
    assert agent.action_counter >= 1, "forced turn should have executed a real action"

    forced_msgs = [m for m in policy.messages if "harness note" in m.get("content", "")]
    assert forced_msgs, "expected a [harness note] message announcing the forced probe"
    print("forced-turn note:", forced_msgs[0]["content"][:200])
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()

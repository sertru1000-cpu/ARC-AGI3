"""Smoke test for the Duck-style tool-loop (backlog #11, 20.08; reworked
21.08 to native OpenAI-style tool-calling instead of a text ```python fence).

Verifies against a real game via MockLLM: multiple rounds of (LLM call ->
run_python tool call -> observe) collapse into ONE outer turn (self.turns +=
1 once, not once per round), the loop ends on a plain-text reply with no
tool call, a first-round no-call counts as a strike but a later-round
no-call doesn't, and WIN cuts the loop short even with rounds left.

Run:  AGENT_BRAIN=llm .venv/Scripts/python.exe scripts/test_tool_loop.py
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
os.environ["MY_AGENT_TOOL_LOOP_STEPS"] = "4"
os.environ["MY_AGENT_MAX_TURNS"] = "1"  # isolate: only the 3-round turn under test runs
os.environ["MY_AGENT_FLOOR_ACTIONS"] = "0"

ROUND1 = {
    "content": "WORLD_MODEL:\ncontrols: unknown\ngoal: unknown\nplan: probe one direction",
    "tool": "run_python",
    "arguments": {"code": "r = action(['UP'])\nprint('round1:', r.get('board_changed'))"},
}

ROUND2 = {
    "content": ("WORLD_MODEL:\ncontrols: UP moved something\ngoal: still unclear\n"
                "plan: probe another direction, reacting to round 1's result"),
    "tool": "run_python",
    "arguments": {"code": "r = action(['RIGHT'])\nprint('round2:', r.get('board_changed'))"},
}

ROUND3_FINAL = "Based on rounds 1-2 I'll pause here and re-plan next turn."

# Turn 1: 3 rounds (2 tool calls, 1 final plain-text) all within ONE outer
# turn. Turn 2: same script repeats (MockLLM loops the last entry, but we
# only need turn 1 to prove the mechanism).
SCRIPT = [ROUND1, ROUND2, ROUND3_FINAL]


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
        card_id="mock-test", game_id="ls20", agent_name="mock.toolloop",
        ROOT_URL="http://localhost", record=False, arc_env=env, tags=["test"],
    )

    policy_ref = {}
    orig_play_turn_setup = module.MyAgent.main
    agent.main()

    policy = agent.policy
    print("\n===== RESULT =====")
    print("policy.turns (outer turns):", policy.turns)
    print("mock LLM calls (rounds):", mock.calls)
    print("no_code_strikes:", policy.no_code_strikes)
    print("action_counter (real env actions):", agent.action_counter)

    assert policy is not None
    # 3 rounds happened (mock.calls) but they must have collapsed into far
    # fewer OUTER turns than raw calls -- proving the loop, not naive 1:1.
    assert mock.calls == 3, f"expected exactly the 3-round script to run, got {mock.calls} calls"
    assert policy.turns == 1, (
        f"expected exactly 1 outer turn (MAX_TURNS=1) despite 3 LLM calls -- "
        f"rounds should collapse into ONE turn, got policy.turns={policy.turns}")
    assert policy.no_code_strikes == 0, (
        f"round 3's no-code reply is a deliberate turn-end (round>1), should "
        f"NOT count as a strike, got {policy.no_code_strikes}")
    assert agent.action_counter >= 2, "both round 1 and round 2 should have executed real actions"

    print("\nTOOL-LOOP SMOKE TEST PASSED")


def main_win_cuts_loop_short() -> None:
    """A WIN on round 1 of 4 must stop the loop immediately, not burn the
    remaining rounds on a finished game."""
    from agent.harness.llm_policy import LLMPolicy
    from agent.harness.sandbox import Sandbox, FrameView
    import numpy as np

    class FakeState:
        def __init__(self, name):
            self._name = name

        def __str__(self):
            return f"GameState.{self._name}"

    class FakeFrame:
        def __init__(self, grid, level, state):
            self.frame = [grid.tolist()]
            self.levels_completed = level
            self.state = FakeState(state)
            self.available_actions = []

    grid = np.zeros((4, 4), dtype=np.int8)
    calls = {"n": 0}

    def env_step(engine_name, payload):
        calls["n"] += 1
        return FakeFrame(grid, level=1, state="WIN")

    sb = Sandbox(env_step=env_step, budget_left=lambda: 100)
    sb.current = FrameView(grid, step=0, level=0)

    class ScriptedBackend:
        name = "scripted"

        def __init__(self, code: str):
            self.code = code
            self.calls = 0

        def chat_tools(self, messages, tools, max_tokens=2048, temperature=0.6, tool_choice="auto"):
            self.calls += 1
            return {
                "content": "acting",
                "tool_calls": [{"id": "call-1", "name": "run_python", "arguments": {"code": self.code}}],
            }

    backend = ScriptedBackend("action(['UP'])")
    policy = LLMPolicy(backend=backend, sandbox=sb, game_id="synth", win_levels=1,
                       tool_loop_steps=4)
    policy.messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]

    result = policy.play_turn()
    print("WIN test -- backend calls:", backend.calls, "result:", result)
    assert backend.calls == 1, f"expected exactly 1 LLM call (WIN on round 1 of 4), got {backend.calls}"
    assert result["win"] is True

    print("WIN CUTS TOOL-LOOP SHORT: PASSED")


if __name__ == "__main__":
    main()
    main_win_cuts_loop_short()

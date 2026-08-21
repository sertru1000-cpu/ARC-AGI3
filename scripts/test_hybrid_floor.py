"""Smoke test for the hybrid floor (backlog item 2, 20.08).

Verifies: before the first LLM call, MY_AGENT_FLOOR_ACTIONS real actions get
spent via the heuristic explorer (same brain as AGENT_BRAIN=explorer) — a
scripted MockLLM that NEVER calls action() itself should still see real
env actions on the board, proving they came from the floor phase, not the
model. Also checks the floor spend counts against the real action budget
(not free) and that the model gets a note about it.

Run:  AGENT_BRAIN=llm .venv/Scripts/python.exe scripts/test_hybrid_floor.py
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
os.environ["MY_AGENT_FLOOR_ACTIONS"] = "10"
os.environ["MY_AGENT_MAX_TURNS"] = "3"
os.environ["MY_AGENT_MAX_ACTIONS"] = "50"

# The model never calls action() -- any real actions taken must be the floor's.
INSPECT_ONLY = """WORLD_MODEL:
controls: unknown
goal: unknown
plan: just look

```python
print("looking:", current_frame.ascii[:20])
```"""

SCRIPT = [INSPECT_ONLY] * 5


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
        card_id="mock-test", game_id="ls20", agent_name="mock.floor",
        ROOT_URL="http://localhost", record=False, arc_env=env, tags=["test"],
    )
    agent.main()

    print("\n===== RESULT =====")
    print("action_counter (real env actions):", agent.action_counter)
    print("mock LLM calls:", mock.calls)
    print("policy turns:", agent.policy.turns if agent.policy else 0)

    # The scripted LLM never calls action() -- every real action taken can
    # only have come from the floor phase (or the opening probe, which is a
    # handful of deterministic single presses, not board-changing loops).
    assert agent.action_counter >= 10, (
        f"expected the floor to spend ~10 real actions, got {agent.action_counter}")

    floor_notes = [
        m for m in agent.policy.messages
        if "heuristic opening burst" in m.get("content", "")
    ] if agent.policy else []
    assert floor_notes, "expected a [harness note] about the floor spend in the LLM context"
    print("floor note seen by model:", floor_notes[0]["content"][:200])

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()

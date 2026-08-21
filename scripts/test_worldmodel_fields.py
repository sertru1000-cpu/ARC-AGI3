"""Unit test for the 3 new WORLD_MODEL fields (backlog item 5, 20.08):
recent_findings, open_questions, cross_level_notes -- alongside the
original controls/prior/goal/plan.

Run:  AGENT_BRAIN=llm .venv/Scripts/python.exe scripts/test_worldmodel_fields.py
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
os.environ["MY_AGENT_MAX_TURNS"] = "2"
os.environ["MY_AGENT_FLOOR_ACTIONS"] = "0"  # isolate: no floor noise for this test

REPLY = """WORLD_MODEL:
controls: UP/DOWN move the avatar
prior: unknown yet - narrowing down
goal: reach the target
plan: probe directions
recent_findings: UP changed 4 cells near the avatar
open_questions: does color matter, or only shape
cross_level_notes: none yet

```python
print("noop")
```"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import arc_agi
    from arc_agi import OperationMode

    from agent.harness.llm import MockLLM
    import agent.harness.llm as llm_mod

    mock = MockLLM([REPLY])
    llm_mod.default_backend = lambda: mock

    import importlib.util

    spec = importlib.util.spec_from_file_location("user_agent_module", ROOT / "agent" / "my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("ls20")
    agent = module.MyAgent(
        card_id="mock-test", game_id="ls20", agent_name="mock.wm7",
        ROOT_URL="http://localhost", record=False, arc_env=env, tags=["test"],
    )
    agent.main()

    wm = agent.policy.sandbox.memo.get("world_model") if agent.policy else None
    print("\n===== RESULT =====")
    print("world_model:", wm)

    assert wm, "expected a world_model dict in memo"
    for field in ("controls", "prior", "goal", "plan",
                  "recent_findings", "open_questions", "cross_level_notes"):
        assert field in wm, f"missing field {field!r} in parsed world_model: {wm}"
    assert wm["open_questions"] == "does color matter, or only shape"
    assert wm["cross_level_notes"] == "none yet"

    print("\nALL 7 WORLD_MODEL FIELDS PARSED CORRECTLY")


if __name__ == "__main__":
    main()

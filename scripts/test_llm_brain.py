"""Smoke test for the LLM brain using a scripted MockLLM on a real game.

Verifies end-to-end: custom main loop -> policy turn -> code block parsing ->
sandbox exec -> real env stepping via action() -> feedback message assembly.

Run:  AGENT_BRAIN=llm .venv/Scripts/python.exe scripts/test_llm_brain.py
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

SCRIPT = [
    # Turn 1: inspect the board via segmentation + exercise toolbox & memo.
    """WORLD_MODEL:
controls: unknown yet, directions available
goal: unknown, exploring
plan: inspect board, then probe

```python
seg = current_frame.segmentation
print("level", current_frame.level, "shape", current_frame.shape)
print("background", seg["background"], "objects", len(seg["nodes"]))
objs = objects(current_frame.grid)
print("toolbox objects:", len(objs), "first cells:", objs[0]["cells"][:2])
p = bfs_path(current_frame.grid, (1, 1), (1, 3), passable=lambda c: True)
print("bfs sanity:", p)
memo["note"] = "turn1 done"
print("valid:", valid_actions)
```""",
    # Turn 2: probe all directions and diff.
    """Probing each direction to find the avatar.
```python
for a in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
    if a in valid_actions:
        r = action([a])
        print(a, "changed:", r["board_changed"], "cells:", r["changed_cells"], "hud:", r["hud_only_change"])
```""",
    # Turn 3: numpy diff between frames.
    """Checking what moved between the last two frames.
```python
if previous_frame is not None:
    d = np.argwhere(previous_frame.grid != current_frame.grid)
    print("changed cells:", len(d))
    if len(d):
        print("rows", int(d[:,0].min()), "-", int(d[:,0].max()),
              "cols", int(d[:,1].min()), "-", int(d[:,1].max()))
result = "world model updated"
```""",
    # Turn 4+: batch a few moves then stop emitting code (tests strike-out).
    """```python
r = action(['RIGHT', 'RIGHT', 'UP'])
print("after batch:", r["level"], r["budget_left"])
```""",
    "I believe I'm done analyzing.",  # no code -> strike 1
    "Nothing more to do.",  # strike 2
    "Stopping.",  # strike 3 -> loop must end
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import arc_agi
    from arc_agi import OperationMode

    from agent.harness.llm import MockLLM
    import agent.harness.llm as llm_mod

    # Force the mock backend regardless of env.
    llm_mod.default_backend = lambda: MockLLM(SCRIPT)

    import importlib.util

    spec = importlib.util.spec_from_file_location("user_agent_module", ROOT / "agent" / "my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("ls20")
    agent = module.MyAgent(
        card_id="mock-test", game_id="ls20", agent_name="mock.ls20",
        ROOT_URL="http://localhost", record=False, arc_env=env, tags=["test"],
    )
    agent.main()

    print("\n===== RESULT =====")
    print("actions taken:", agent.action_counter)
    print("frames:", len(agent.frames))
    print("final state:", agent.frames[-1].state)
    print("llm turns:", agent.policy.turns if agent.policy else 0)

    # Show the last feedback message the model would have seen.
    if agent.policy:
        for m in agent.policy.messages[-2:]:
            print(f"\n--- {m['role']} ---\n{m['content'][:600]}")

    assert agent.action_counter >= 8, "expected probes + batch to execute real actions"
    assert agent.policy and agent.policy.turns >= 6, "expected all scripted turns to run"
    wm = agent.policy.sandbox.memo.get("world_model")
    assert wm and "plan" in wm, f"WORLD_MODEL not parsed into memo: {wm}"
    assert agent.policy.sandbox.memo.get("note") == "turn1 done", "memo lost"
    print("world_model parsed:", wm)
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()

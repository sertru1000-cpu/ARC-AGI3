"""Exercise the atlas theory-force override checkpoint (27.08).

Found live on r11l (Kaggle v21): the soft theory checkpoint (ATLAS_THEORY_
CHECKPOINT, re-strengthened in v17) fired 5 times in that one game, and the
model explicitly derived the correct dynamics rule in its own reasoning
twice ("the head moves halfway toward the gray blob's center on each
click") -- but never once converted that rule into a predict()/
verify_theory() call. This is the THIRD distinct failure mode on this same
checkpoint (v12: total action paralysis; v16: soft wording silently
ignored, 0/662 real calls) -- diagnosed with Gemini as an external critic:
verify_theory requires the model to author original code encoding a rule it
already holds only in prose, which a soft nudge generalizes far worse
against than a hard directive to call an already-well-defined function
(contrast: rollback-ultimatum got first-try compliance on a totally
different game, wa30, in the very same run).

Design constraints, deliberately different from both ATLAS_FORCE_ACT_
OVERRIDE (which this is modeled on) and the v12 mistake it must not repeat:
- Gates on the ATTEMPT (verify_theory_call_count == 0), never on accuracy --
  an accuracy gate is exactly what made ">= 0.6" unreachable and caused
  v12's paralysis. Once the model calls verify_theory even once, at ANY
  accuracy, this override must go silent for the rest of the game.
- Does NOT forbid action() in the same turn -- nothing in the sandbox
  prevents predict()+verify_theory()+action() living in one python snippet,
  so there is no real "theory OR play" trade-off being forced here.
- Checked BEFORE the soft theory nag in the priority chain (its threshold,
  8, is 2x the soft nag's 4, so if checked after, the soft nag's condition
  would already be satisfied and this branch would be unreachable).

Drives ToolAgent._run_python_tool/_build_user_prompt directly against a
real sandboxed subprocess, same pattern as test_atlas_plan_nudge.py.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "tufa-arc-agi-framework" / "src"))
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "ARC3-Inference"))

from inference.agent.tool_agent import (  # noqa: E402
    ToolAgent,
    _ATLAS_THEORY_FORCE_AFTER_CALLS,
    _ATLAS_THEORY_NAG_AFTER_CALLS,
)
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state  # noqa: E402


def _fail(name: str, detail: str) -> None:
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def _ok(name: str) -> None:
    print(f"ok   {name}")


def main() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="atlas_theory_force_test_"))
    state_path = tmp_dir / "run_state.json"

    # A real recorded transition (grid -> grid shifted right by one column
    # via MOUSE) so verify_theory has something to score in step 3 below --
    # a wrong predict() (identity) then correctly scores 0.0, not None/0-of-0.
    grid = ((0, 1, 2),)
    shifted = ((2, 0, 1),)
    write_runtime_state(
        state_path,
        current_frame=Frame(grid=shifted, step=1, level=1),
        history=[
            HistoryEntry(action="MOUSE(row=0, col=0)", frame=Frame(grid=grid, step=0, level=1)),
            HistoryEntry(action="MOUSE(row=0, col=0)", frame=Frame(grid=shifted, step=1, level=1)),
        ],
    )

    agent = ToolAgent(model="test-model")

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
            "valid_actions": ["MOUSE"],
        }

    agent._step_env_callback = _fake_step_env
    agent._current_valid_actions = ["MOUSE"]

    # Satisfy explore-first immediately (one real MOUSE click, board_changed
    # True per _fake_step_env) so it never intercepts ahead of theory-force
    # for the rest of this test -- this test is isolated to the theory-force
    # checkpoint specifically, same isolation pattern as the other
    # checkpoint tests use for their own target mechanism.
    agent._run_python_tool(state_path, {"code": "action([{'action': 'MOUSE', 'row': 1, 'col': 1}])\n"})

    # 1. Below threshold: the soft nag fires (once its own lower threshold
    #    is crossed), but the force override must not appear yet.
    for _ in range(_ATLAS_THEORY_NAG_AFTER_CALLS - 1):  # call index reaches _ATLAS_THEORY_NAG_AFTER_CALLS
        agent._run_python_tool(state_path, {"code": "result = 1\n"})
    prompt = agent._build_user_prompt(0, valid_actions=["MOUSE"])
    if "ZERO verify_theory() calls" in prompt:
        _fail("force override silent below its own threshold", prompt[-500:])
    if "THIS turn, write predict" not in prompt:
        _fail("soft nag fires at its own (lower) threshold", prompt[-500:])
    _ok(f"soft theory nag fires at call {_ATLAS_THEORY_NAG_AFTER_CALLS}, force override stays silent")

    # 2. Cross the force-override threshold with verify_theory still never
    #    called: the override replaces the soft nag outright.
    while agent._atlas_python_call_index < _ATLAS_THEORY_FORCE_AFTER_CALLS:
        agent._run_python_tool(state_path, {"code": "result = 1\n"})
    if agent._atlas_verify_theory_call_count != 0:
        _fail("setup: verify_theory never called", str(agent._atlas_verify_theory_call_count))
    prompt = agent._build_user_prompt(0, valid_actions=["MOUSE"])
    if "ZERO verify_theory() calls" not in prompt:
        _fail("force override fires at threshold", f"expected it at call {_ATLAS_THEORY_FORCE_AFTER_CALLS}, got: {prompt[-500:]!r}")
    if "THIS turn, write predict" in prompt:
        _fail("force override replaces the soft nag", "both must not appear in the same prompt")
    if "SAME snippet" not in prompt:
        _fail("force override does not forbid action() in the same turn", prompt[-800:])
    if "0% accuracy result is a GOOD outcome" not in prompt:
        _fail("force override frames low accuracy as a good outcome, not a failure", prompt[-800:])
    _ok(
        f"theory-force override fires after {_ATLAS_THEORY_FORCE_AFTER_CALLS} python calls with zero "
        "verify_theory() calls, replacing the soft nag and permitting action() in the same turn"
    )

    # 3. A real verify_theory( call, even at 0.0 accuracy, silences the
    #    override for the REST of the game -- this is the critical safety
    #    property: the gate is on the ATTEMPT, never on accuracy, so it can
    #    never become the same unreachable-bar trap that caused v12's total
    #    paralysis (">= 0.6 accuracy" was the unsatisfiable condition there;
    #    here the bar is just "called once", which any predict() clears).
    agent._run_python_tool(
        state_path,
        {"code": "def predict(grid, action):\n    return grid\nresult = verify_theory(predict)\n"},
    )
    if agent._atlas_last_verified_accuracy != 0.0:
        _fail("accuracy captured (wrong theory)", str(agent._atlas_last_verified_accuracy))
    # Push well past the force-override threshold again to confirm it stays
    # quiet permanently, not just for one turn.
    for _ in range(5):
        agent._run_python_tool(state_path, {"code": "result = 1\n"})
    prompt = agent._build_user_prompt(0, valid_actions=["MOUSE"])
    if "ZERO verify_theory() calls" in prompt:
        _fail(
            "force override never returns once verify_theory has been called at all",
            "a 0.0-accuracy verify_theory() call must satisfy this gate permanently, exactly like a real "
            f"accuracy would -- got: {prompt[-500:]!r}",
        )
    _ok("a single verify_theory( call at ANY accuracy (even 0.0) silences the force override for good -- "
        "the gate is on the attempt, not on accuracy, so it cannot recreate v12's unreachable-bar trap")

    print("\nAll atlas theory-force override checks passed.")


if __name__ == "__main__":
    main()

"""Exercise the atlas explore-first checkpoint (27.08).

Found live on r11l: the human baseline solves this MOUSE-only game in 2
moves, but the model spent 45 minutes and tens of thousands of tokens
verifying a pixel-frequency theory without ever trying more than 2 real
actions. The static PYTHON_ADDENDUM already says "probe... then plan", but
that alone didn't stop the model reaching for verify_theory()'s statistical
rigor before it had tried every control the game currently offers --
confirmed directly by the user's own play-testing ("я, как человек, всегда
начинаю с проб, что какие рычаги правления делают, как реагируют фигуры").

Design, per explicit user refinement during the same conversation:
- Not just "was this action KIND called at all" (a single unlucky click into
  empty water would wrongly count as "explored MOUSE") -- a kind only
  resolves once it visibly changed the board (board_changed=True) at least
  once, or has been tried _ATLAS_EXPLORE_MAX_ATTEMPTS_PER_KIND times with no
  effect (accepted as "seems inert here" -- the same class of unsatisfiable-
  gate trap that caused r11l's OWN v12 total-paralysis incident, this time
  avoided on purpose).
- Not just level 1 -- resets on every level-up, since a new level can
  introduce new mechanics or make a previously-irrelevant control matter.
- Checked BEFORE goal-reconsider/theory/extract-suggestion in the priority
  chain -- exploring the control surface is more foundational than any of
  the theories built on top of it.

Drives ToolAgent._run_python_tool/_build_user_prompt directly against a
real sandboxed subprocess, same pattern as test_atlas_rollback.py.
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
    _ATLAS_EXPLORE_MAX_ATTEMPTS_PER_KIND,
    _ATLAS_EXPLORE_NUDGE_AFTER_CALLS,
)
from inference.agent.runtime_state import Frame, normalize_grid, write_runtime_state  # noqa: E402


def _fail(name: str, detail: str) -> None:
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def _ok(name: str) -> None:
    print(f"ok   {name}")


def _write_state(path: Path, grid, step: int, level: int = 1) -> None:
    write_runtime_state(path, current_frame=Frame(grid=normalize_grid(grid), step=step, level=level), history=[])


class ScriptedEnv:
    """A step_env_callback whose board_changed/level outcomes are scripted
    per action kind, so tests can precisely control what counts as a real
    discovery vs. an inert probe -- real games can't be scripted this
    cleanly, but the checkpoint's bookkeeping only cares about
    executed/board_changed/level, which this reproduces faithfully.
    """

    def __init__(self, state_path: Path, changes_board: set[str], valid_actions: list[str]):
        self.state_path = state_path
        self.changes_board = changes_board
        self.valid_actions = valid_actions
        self.counter = 0
        self.level = 1
        self.next_level_completed = False

    def __call__(self, payload):
        self.counter += 1
        action = payload["actions"][0]
        kind = str(action.get("action"))
        board_changed = kind in self.changes_board
        if self.next_level_completed:
            self.level += 1
            self.next_level_completed = False
            level_completed = True
        else:
            level_completed = False
        _write_state(self.state_path, [[self.counter]], step=self.counter, level=self.level)
        return {
            "executed": True,
            "action_num": self.counter,
            "level": self.level,
            "score": 0,
            "reward": 0.0,
            "board_changed": board_changed,
            "done": False,
            "level_completed": level_completed,
            "game_over": False,
            "run_complete": False,
            "valid_actions": self.valid_actions,
        }


def main() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="atlas_explore_first_test_"))

    # 1. Fires once enough calls have passed with an untried control, BEFORE
    #    goal-reconsider/theory would otherwise have a chance to fire.
    state_path = tmp_dir / "run_state.json"
    _write_state(state_path, [[0]], step=0, level=1)
    env = ScriptedEnv(state_path, changes_board={"UP", "RIGHT"}, valid_actions=["UP", "RIGHT"])
    agent = ToolAgent(model="test-model")
    agent._step_env_callback = env
    agent._current_valid_actions = ["UP", "RIGHT"]

    for _ in range(_ATLAS_EXPLORE_NUDGE_AFTER_CALLS):
        agent._run_python_tool(state_path, {"code": "action(['UP'])\n"})
    prompt = agent._build_user_prompt(0, valid_actions=["UP", "RIGHT"])
    if "[atlas checkpoint]" not in prompt or "RIGHT" not in prompt:
        _fail("explore-first fires for an untried control", prompt[-500:])
    if "This game currently offers" not in prompt:
        _fail("explore-first uses its own wording, not a different checkpoint", prompt[-500:])
    _ok(f"explore-first checkpoint fires after {_ATLAS_EXPLORE_NUDGE_AFTER_CALLS} calls, naming the untried control")

    # 2. A single unlucky call that does NOT change the board does not count
    #    as "explored" -- the checkpoint stays active for that same kind.
    inert_state_path = tmp_dir / "inert_run_state.json"
    _write_state(inert_state_path, [[0]], step=0, level=1)
    inert_env = ScriptedEnv(inert_state_path, changes_board=set(), valid_actions=["MOUSE"])
    inert_agent = ToolAgent(model="test-model")
    inert_agent._step_env_callback = inert_env
    inert_agent._current_valid_actions = ["MOUSE"]

    for _ in range(_ATLAS_EXPLORE_NUDGE_AFTER_CALLS):
        inert_agent._run_python_tool(inert_state_path, {"code": "action([{'action':'MOUSE','row':1,'col':1}])\n"})
    prompt = inert_agent._build_user_prompt(0, valid_actions=["MOUSE"])
    if "[atlas checkpoint]" not in prompt or "MOUSE" not in prompt:
        _fail("one no-effect click does not resolve the control", prompt[-500:])
    _ok("a single click with no visible effect does NOT count as 'explored' -- the exact r11l-style gap this closes")

    # 3. Once a control DOES visibly change the board, it resolves and the
    #    checkpoint goes quiet (nothing else to explore here).
    inert_env.changes_board = {"MOUSE"}
    inert_agent._run_python_tool(inert_state_path, {"code": "action([{'action':'MOUSE','row':2,'col':2}])\n"})
    prompt = inert_agent._build_user_prompt(0, valid_actions=["MOUSE"])
    if "[atlas checkpoint]" in prompt and "This game currently offers" in prompt:
        _fail("resolves once the control visibly changes the board", prompt[-500:])
    _ok("a control that visibly changes the board resolves immediately -- checkpoint goes quiet")

    # 4. A control that NEVER changes the board resolves anyway after
    #    _ATLAS_EXPLORE_MAX_ATTEMPTS_PER_KIND attempts -- must not become an
    #    unsatisfiable gate (the same trap as the old verified_accuracy>=0.6
    #    one that caused r11l's OWN v12 total-paralysis incident).
    dead_state_path = tmp_dir / "dead_run_state.json"
    _write_state(dead_state_path, [[0]], step=0, level=1)
    dead_env = ScriptedEnv(dead_state_path, changes_board=set(), valid_actions=["ACTION7"])
    dead_agent = ToolAgent(model="test-model")
    dead_agent._step_env_callback = dead_env
    dead_agent._current_valid_actions = ["ACTION7"]

    for _ in range(_ATLAS_EXPLORE_MAX_ATTEMPTS_PER_KIND):
        dead_agent._run_python_tool(dead_state_path, {"code": "action(['ACTION7'])\n"})
    # Confirmed still active right up to the cap (attempts == cap - 1 before
    # this last check point), then must clear once the cap is reached.
    if dead_agent._atlas_action_kind_attempts.get("ACTION7", 0) != _ATLAS_EXPLORE_MAX_ATTEMPTS_PER_KIND:
        _fail(
            "attempt count tracked correctly",
            str(dead_agent._atlas_action_kind_attempts),
        )
    prompt = dead_agent._build_user_prompt(0, valid_actions=["ACTION7"])
    if "[atlas checkpoint]" in prompt and "This game currently offers" in prompt:
        _fail(
            f"a genuinely inert control resolves after {_ATLAS_EXPLORE_MAX_ATTEMPTS_PER_KIND} attempts, not forever",
            prompt[-500:],
        )
    _ok(
        f"a control with zero effect across {_ATLAS_EXPLORE_MAX_ATTEMPTS_PER_KIND} real attempts resolves anyway "
        "-- not an unsatisfiable gate"
    )

    # 5. Resets on level-up: a control resolved on level 1 is untried again
    #    on level 2, since the new level can introduce new mechanics.
    level_state_path = tmp_dir / "level_run_state.json"
    _write_state(level_state_path, [[0]], step=0, level=1)
    level_env = ScriptedEnv(level_state_path, changes_board={"UP"}, valid_actions=["UP"])
    level_agent = ToolAgent(model="test-model")
    level_agent._step_env_callback = level_env
    level_agent._current_valid_actions = ["UP"]

    level_agent._run_python_tool(level_state_path, {"code": "action(['UP'])\n"})
    prompt = level_agent._build_user_prompt(0, valid_actions=["UP"])
    if "[atlas checkpoint]" in prompt and "This game currently offers" in prompt:
        _fail("UP resolved on level 1 before testing the level-up reset", prompt[-500:])

    level_env.next_level_completed = True
    level_env.changes_board = set()  # UP no longer does anything on level 2 (yet)
    level_agent._run_python_tool(level_state_path, {"code": "action(['UP'])\n"})
    prompt = level_agent._build_user_prompt(0, valid_actions=["UP"])
    if "[atlas checkpoint]" not in prompt or "This game currently offers" not in prompt:
        _fail(
            "explore-first re-fires for UP on a NEW level even though it resolved on the previous level",
            prompt[-500:],
        )
    _ok("resolved status resets on level-up -- a new level gets its own fresh exploration requirement")

    print("\nAll atlas explore-first checkpoint checks passed.")


if __name__ == "__main__":
    main()

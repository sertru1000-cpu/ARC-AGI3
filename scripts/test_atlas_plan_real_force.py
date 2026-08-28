"""Exercise the plan_real principle-force checkpoint + harness auto-run (27.08 late).

The plan_real RunPod adoption run showed the familiar ~0% voluntary uptake
(0 calls in its first hour despite static docs and the theory-checkpoint
pointer) -- the exact pattern every un-nudged tool in this project has
shown (memo 0%, verify_theory 0.2%). Same two-layer playbook as the
rollback ultimatum:

1. Principle-force checkpoint: once the game is stalled
   (_ATLAS_PLAN_REAL_STALL_AFTER_ACTIONS real actions without level
   progress, BELOW the rollback trigger's threshold so search fires before
   give-up) and plan_real was never used this level, the model's next call
   MUST call plan_real.
2. Harness auto-run: after _ATLAS_PLAN_REAL_AUTO_FORCE_AFTER ignored
   showings, the harness runs the search ITSELF and EXECUTES a found plan
   (possible precisely because plan_real needs no model-authored code for
   non-MOUSE games). MOUSE-only games never auto-run (the harness cannot
   pick candidate clicks) -- their nag caps at _ATLAS_PLAN_REAL_NAG_CAP
   showings instead.

Drives the real ToolAgent + sandbox with the same WalkEnv mini-game as
test_atlas_real_search.py.
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
    _ATLAS_PLAN_REAL_AUTO_FORCE_AFTER,
    _ATLAS_PLAN_REAL_NAG_CAP,
    _ATLAS_PLAN_REAL_STALL_AFTER_ACTIONS,
    _ATLAS_ROLLBACK_STALL_AFTER_CALLS,
)
from inference.agent.runtime_state import Frame, write_runtime_state  # noqa: E402


def _fail(name: str, detail: str) -> None:
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def _ok(name: str) -> None:
    print(f"ok   {name}")


class WalkEnv:
    """Same 1-D walk as test_atlas_real_search.py: UP advances, DOWN
    retreats (floor 0), ACTION7 is a game-over trap, level completes at GOAL.
    LEFT is valid but a guaranteed no-op -- the 'stall fuel' the
    model burns without progress."""

    VALID = ("UP", "DOWN", "LEFT")

    def __init__(self, state_path: Path, goal: int):
        self.state_path = state_path
        self.goal = goal
        self.pos = 0
        self.level = 1
        self.step = 0
        self.game_over = False
        self._write()

    def _write(self) -> None:
        write_runtime_state(
            self.state_path,
            current_frame=Frame(grid=((self.pos, self.step % 7),), step=self.step, level=self.level),
            history=[],
        )

    def snapshot(self):
        return {"pos": self.pos, "level": self.level, "step": self.step, "game_over": self.game_over}

    def restore(self, snap):
        if not isinstance(snap, dict):
            return False
        self.pos, self.level, self.step, self.game_over = (
            snap["pos"], snap["level"], snap["step"], snap["game_over"],
        )
        self._write()
        return True

    def __call__(self, payload):
        specs = list(payload.get("actions") or [])
        executed = 0
        board_changed = False
        level_completed = False
        stop_reason = None
        for spec in specs:
            kind = str(spec.get("action") if isinstance(spec, dict) else spec).strip().upper()
            if kind not in self.VALID:
                if executed == 0:
                    return {"executed": False, "error": f"{kind} is not valid right now.", "executed_count": 0}
                stop_reason = "invalid_action"
                break
            before = self.pos
            self.step += 1
            executed += 1
            if kind == "ACTION7":
                self.game_over = True
                self._write()
                stop_reason = "game_over"
                break
            if kind == "UP":
                self.pos += 1
            elif kind == "DOWN":
                self.pos = max(0, self.pos - 1)
            if self.pos != before:
                board_changed = True
            if self.pos == self.goal:
                self.level += 1
                level_completed = True
                self._write()
                # Each level starts back at the floor -- 3 UPs complete
                # EVERY level, so multi-level scenarios work.
                self.pos = 0
                stop_reason = "level_completed"
                break
            self._write()
        return {
            "executed": executed > 0,
            "executed_count": executed,
            "requested_count": len(specs),
            "board_changed": board_changed,
            "level": self.level,
            "score": self.level - 1,
            "level_completed": level_completed,
            "game_over": self.game_over,
            "run_complete": False,
            "done": False,
            "stop_reason": stop_reason,
            "valid_actions": list(self.VALID),
        }


def _make_agent(state_path: Path, env: WalkEnv, valid=None) -> ToolAgent:
    agent = ToolAgent(model="test-model")
    # The stall fuel below is a deliberately repeated no-op action; the hard
    # noop guard would (correctly) block those repeats in real play, which
    # is orthogonal to what this test isolates -- disable it here.
    agent._hard_noop_guard_enabled = False
    agent._step_env_callback = env
    agent._checkpoint_env_callback = env.snapshot
    agent._restore_env_callback = env.restore
    agent._current_valid_actions = list(valid or WalkEnv.VALID)
    return agent


def _stall(agent: ToolAgent, state_path: Path, n: int) -> None:
    """Burn n real no-op actions (LEFT) -- real actions, zero progress.
    Interleaves nothing else; force-act never fires (every call acts)."""
    for _ in range(n):
        agent._run_python_tool(state_path, {"code": "action(['LEFT'])\n"})


def main() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="atlas_plan_real_force_test_"))

    # 1. Stall to threshold -> the principle-force checkpoint fires (and
    #    stays below the rollback trigger's own threshold).
    state_path = tmp_dir / "run_state.json"
    env = WalkEnv(state_path, goal=3)
    agent = _make_agent(state_path, env)
    agent._run_python_tool(state_path, {"code": "action(['UP'])\naction(['DOWN'])\n"})  # explore-first fuel
    _stall(agent, state_path, _ATLAS_PLAN_REAL_STALL_AFTER_ACTIONS)
    if agent._atlas_actions_since_level_progress < _ATLAS_PLAN_REAL_STALL_AFTER_ACTIONS:
        _fail("stall counter reached threshold", str(agent._atlas_actions_since_level_progress))
    if agent._atlas_actions_since_level_progress >= _ATLAS_ROLLBACK_STALL_AFTER_CALLS:
        _fail("test setup stays below the rollback trigger", str(agent._atlas_actions_since_level_progress))
    prompt = agent._build_user_prompt(0, valid_actions=list(WalkEnv.VALID))
    if "MUST call plan_real(" not in prompt:
        _fail("plan_real force checkpoint fires at stall", prompt[-600:])
    if "the harness will run the search itself" not in prompt:
        _fail("first showings carry the auto-run escalation warning", prompt[-600:])
    _ok(f"principle-force fires after {_ATLAS_PLAN_REAL_STALL_AFTER_ACTIONS} stalled actions, "
        "before the rollback trigger, with the escalation warning")

    # 2. A real plan_real( call silences it for the level.
    used_agent_prompt_before = agent._atlas_plan_real_force_streak
    agent._run_python_tool(state_path, {"code": "result = plan_real(max_depth=2)\n"})
    if not agent._atlas_plan_real_used_this_level:
        _fail("plan_real usage tracked", "flag not set after a real plan_real( call")
    prompt = agent._build_user_prompt(0, valid_actions=list(WalkEnv.VALID))
    if "MUST call plan_real(" in prompt:
        _fail("checkpoint silent after a real plan_real( call", prompt[-400:])
    _ok(f"a real plan_real( call silences the checkpoint (streak was {used_agent_prompt_before})")

    # 3. Ignored showings -> harness auto-runs, finds the 3-UP plan, and
    #    EXECUTES it: the real game advances a level without the model.
    env2_path = tmp_dir / "run2_state.json"
    env2 = WalkEnv(env2_path, goal=3)
    agent2 = _make_agent(env2_path, env2)
    agent2._run_python_tool(env2_path, {"code": "action(['UP'])\naction(['DOWN'])\n"})
    _stall(agent2, env2_path, _ATLAS_PLAN_REAL_STALL_AFTER_ACTIONS)
    for _ in range(_ATLAS_PLAN_REAL_AUTO_FORCE_AFTER + 1):
        agent2._build_user_prompt(0, valid_actions=list(WalkEnv.VALID))
    if not agent2._atlas_pending_auto_plan_real:
        _fail("auto-run scheduled after ignored showings", f"streak={agent2._atlas_plan_real_force_streak}")
    level_before = env2.level
    agent2._run_python_tool(env2_path, {"code": "result = 1\n"})
    if env2.level != level_before + 1:
        _fail("harness auto-ran plan_real and EXECUTED the found plan", f"level={env2.level} pos={env2.pos}")
    prompt = agent2._build_user_prompt(0, valid_actions=list(WalkEnv.VALID))
    if "the harness ran the search itself" not in prompt or "EXECUTED it" not in prompt:
        _fail("auto-run result note injected once", prompt[-600:])
    prompt = agent2._build_user_prompt(0, valid_actions=list(WalkEnv.VALID))
    if "the harness ran the search itself" in prompt:
        _fail("auto-run note is one-shot", "note appeared twice")
    _ok("ignored showings -> harness runs the search itself, executes the found plan "
        f"(level {level_before} -> {env2.level}), and reports it once")

    # 4. MOUSE-only game (28.08 unified semantics): clicks-variant text on
    #    the showings, and -- now that the harness can auto-derive click
    #    candidates from segmentation -- the SAME auto-run escalation as
    #    every other game after the ignored-showings threshold.
    env3_path = tmp_dir / "run3_state.json"
    env3 = WalkEnv(env3_path, goal=9)
    agent3 = _make_agent(env3_path, env3, valid=["MOUSE"])
    agent3._atlas_actions_since_level_progress = _ATLAS_PLAN_REAL_STALL_AFTER_ACTIONS
    agent3._atlas_checkpoint_available = True
    shown = 0
    for _ in range(_ATLAS_PLAN_REAL_AUTO_FORCE_AFTER):
        prompt = agent3._build_user_prompt(0, valid_actions=["MOUSE"])
        if "MUST call plan_real(" not in prompt:
            _fail("MOUSE variant shows the force checkpoint", prompt[-500:])
        shown += 1
        if "'action': 'MOUSE', 'row': r, 'col': c" not in prompt:
            _fail("MOUSE variant asks for candidate clicks", prompt[-500:])
        if "auto-derived clicks" not in prompt:
            _fail("MOUSE variant warns about the auto-derived-clicks fallback", prompt[-500:])
    prompt = agent3._build_user_prompt(0, valid_actions=["MOUSE"])
    if not agent3._atlas_pending_auto_plan_real:
        _fail(
            "MOUSE-only game NOW schedules an auto-run after ignored showings (28.08 unification)",
            f"streak={agent3._atlas_plan_real_force_streak}",
        )
    _ok(f"MOUSE-only game: clicks-variant wording with auto-derived fallback warning on {shown} "
        "showings, then the same auto-run escalation as any other game")

    # 5. Level-up resets the whole runway (usage flag, streak, auto-done).
    agent2._atlas_plan_real_used_this_level = True
    agent2._run_python_tool(env2_path, {"code": "action(['UP'])\naction(['UP'])\naction(['UP'])\n"})
    if env2.level != level_before + 2:
        _fail("setup: second level completed for the reset check", f"level={env2.level}")
    if (
        agent2._atlas_plan_real_used_this_level
        or agent2._atlas_plan_real_force_streak
        or agent2._atlas_plan_real_auto_done_this_level
    ):
        _fail(
            "level-up resets the plan_real runway",
            f"used={agent2._atlas_plan_real_used_this_level} streak={agent2._atlas_plan_real_force_streak} "
            f"auto_done={agent2._atlas_plan_real_auto_done_this_level}",
        )
    _ok("level-up resets usage flag, streak, and the once-per-level auto-run allowance")

    print("\nAll atlas plan_real principle-force checks passed.")


if __name__ == "__main__":
    main()

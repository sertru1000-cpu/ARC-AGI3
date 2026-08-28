"""Exercise the L2 auto-replay of solved levels (28.08, Gemini round 5).

The run records the action sequence that completed each level; when the
game falls back to an earlier level (an engine RESET after game over, or a
deliberate full restart), the harness batch-replays the recorded solutions
instead of letting the model re-derive them turn by turn -- and stops
honestly on the first divergence.

Scripted multi-level walk env: level N completes at pos == GOALS[N-1]
(pos resets to 0 on each level entry), RESET restarts the whole game from
level 1. Same fake-callback pattern as test_atlas_real_search.py.
"""

from __future__ import annotations

import json
import os

# Proactive plan_real would auto-solve the scripted levels before the
# scenarios run -- keep it off; this suite tests the REPLAY lever.
os.environ["ATLAS_PLAN_REAL_PROACTIVE"] = "0"
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "tufa-arc-agi-framework" / "src"))
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "ARC3-Inference"))

from inference.agent.tool_agent import ToolAgent  # noqa: E402
from inference.agent.runtime_state import Frame, write_runtime_state  # noqa: E402


def _fail(name: str, detail: str) -> None:
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def _ok(name: str) -> None:
    print(f"ok   {name}")


class MultiLevelEnv:
    """UP advances pos; level N completes at pos == goals[N-1] (pos then
    resets to 0); RESET restarts the whole game from level 1."""

    VALID = ("UP", "DOWN", "RESET")

    def __init__(self, state_path: Path, goals: list[int]):
        self.state_path = state_path
        self.goals = list(goals)
        self.pos = 0
        self.level = 1
        self.step = 0
        self.game_over = False
        self._write()

    def _write(self) -> None:
        write_runtime_state(
            self.state_path,
            current_frame=Frame(grid=((self.pos, self.level),), step=self.step, level=self.level),
            history=[],
        )

    def snapshot(self):
        return {"pos": self.pos, "level": self.level, "step": self.step, "game_over": self.game_over}

    def restore(self, snap, probe=False):
        self.pos = snap["pos"]
        self.level = snap["level"]
        self.step = snap["step"]
        self.game_over = snap["game_over"]
        if not probe:
            self._write()
        return True

    def __call__(self, payload):
        specs = list(payload.get("actions") or [])
        executed = 0
        level_completed = False
        stop_reason = None
        board_changed = False
        for spec in specs:
            kind = str(spec.get("action") if isinstance(spec, dict) else spec).strip().upper()
            if kind not in self.VALID:
                if executed == 0:
                    return {"executed": False, "error": f"{kind} is not valid right now.", "executed_count": 0}
                stop_reason = "invalid_action"
                break
            self.step += 1
            executed += 1
            board_changed = True
            if kind == "RESET":
                self.pos = 0
                self.level = 1
                self.game_over = False
                self._write()
                stop_reason = "reset"
                break
            if kind == "UP":
                self.pos += 1
            elif kind == "DOWN":
                self.pos = max(0, self.pos - 1)
            goal = self.goals[self.level - 1] if self.level - 1 < len(self.goals) else 10**9
            if self.pos == goal:
                self.level += 1
                self.pos = 0
                level_completed = True
                self._write()
                stop_reason = "level_completed"
                break
            self._write()
        return {
            "executed": executed > 0,
            "executed_count": executed,
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


def _make_agent(state_path: Path, env: MultiLevelEnv) -> ToolAgent:
    agent = ToolAgent(model="test-model")
    agent._step_env_callback = env
    agent._checkpoint_env_callback = env.snapshot
    agent._restore_env_callback = env.restore
    agent._current_valid_actions = list(MultiLevelEnv.VALID)
    agent._hard_noop_guard_enabled = False
    agent._noop_guard = None
    return agent


def main() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="atlas_level_replay_test_"))

    # 1. Solve levels 1 (2xUP) and 2 (3xUP) by hand, RESET -> the harness
    #    replays both solutions and the game lands back on level 3.
    state_path = tmp_dir / "run_state.json"
    env = MultiLevelEnv(state_path, goals=[2, 3, 4])
    agent = _make_agent(state_path, env)
    agent._run_python_tool(state_path, {"code": "action(['UP', 'UP'])\n"})
    if env.level != 2:
        _fail("setup: level 1 solved", f"level={env.level}")
    agent._run_python_tool(state_path, {"code": "action(['UP', 'UP', 'UP'])\n"})
    if env.level != 3:
        _fail("setup: level 2 solved", f"level={env.level}")
    if agent._atlas_level_solutions.get(1) != [{"action": "UP"}, {"action": "UP"}]:
        _fail("level 1 solution recorded", str(agent._atlas_level_solutions))
    if agent._atlas_level_solutions.get(2) != [{"action": "UP"}] * 3:
        _fail("level 2 solution recorded", str(agent._atlas_level_solutions))
    agent._run_python_tool(state_path, {"code": "action('RESET')\n"})
    if env.level != 3 or env.pos != 0:
        _fail("RESET triggers auto-replay of both solved levels back to level 3",
              f"level={env.level} pos={env.pos}")
    prompt = agent._build_user_prompt(0, valid_actions=list(MultiLevelEnv.VALID))
    if "auto-replayed your OWN previously successful solutions" not in prompt:
        _fail("auto-replay note lands in the next prompt", prompt[-600:])
    if "1, 2" not in prompt:
        _fail("the note names the replayed levels", prompt[-400:])
    _ok("RESET after two solved levels -> harness replays both recorded solutions, game back "
        "on level 3, one-shot note injected")

    # 2. Divergence: tamper the recorded level-1 solution so the replay
    #    cannot complete the level -- the replay stops honestly and says so.
    env2_path = tmp_dir / "run2_state.json"
    env2 = MultiLevelEnv(env2_path, goals=[2, 3])
    agent2 = _make_agent(env2_path, env2)
    agent2._run_python_tool(env2_path, {"code": "action(['UP', 'UP'])\n"})
    if env2.level != 2:
        _fail("setup: level 1 solved (scenario 2)", f"level={env2.level}")
    agent2._atlas_level_solutions[1] = [{"action": "DOWN"}]  # wrong on purpose
    agent2._run_python_tool(env2_path, {"code": "action('RESET')\n"})
    if env2.level != 1:
        _fail("diverged replay leaves the game where the replay stopped", f"level={env2.level}")
    prompt = agent2._build_user_prompt(0, valid_actions=list(MultiLevelEnv.VALID))
    if "did not complete on replay" not in prompt:
        _fail("divergence is reported honestly in the note", prompt[-500:])
    _ok("a tampered/diverging solution stops the replay at level 1 and the note says so")

    # 3. A VOLUNTARY rollback to a level anchor never trips the replay
    #    (rollback updates the level pointer itself; replay is for RESETs).
    env3_path = tmp_dir / "run3_state.json"
    env3 = MultiLevelEnv(env3_path, goals=[2, 3])
    agent3 = _make_agent(env3_path, env3)
    agent3._run_python_tool(env3_path, {"code": "action(['UP', 'UP'])\n"})
    if env3.level != 2 or "sys_level_2" not in agent3._atlas_checkpoints:
        _fail("setup: level 2 anchor exists", str(list(agent3._atlas_checkpoints)))
    agent3._run_python_tool(
        env3_path, {"code": "rollback('sys_level_2', lesson_learned='fresh approach to level 2')\n"}
    )
    if agent3._atlas_auto_replay_note is not None:
        _fail("voluntary rollback does not trigger auto-replay", str(agent3._atlas_auto_replay_note))
    if env3.level != 2:
        _fail("rollback landed on the level-2 anchor", f"level={env3.level}")
    _ok("a voluntary rollback to an anchor never trips the auto-replay (it is for engine RESETs)")

    print("\nAll atlas level-replay (L2) checks passed.")


if __name__ == "__main__":
    main()

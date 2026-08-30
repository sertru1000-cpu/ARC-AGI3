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
os.environ["ATLAS_MECHANIC_HANDOFF"] = "0"
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "tufa-arc-agi-framework" / "src"))
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "ARC3-Inference"))

from inference.agent import tool_agent  # noqa: E402
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
            # round-8 valve: the real probe fast path returns the resulting
            # grid; the handoff diagnostic is only injected when cells
            # actually changed vs level entry (and no game_over).
            "grid": [[self.pos, self.level]],
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

    # 4. D4 mechanic handoff (29.08, Gemini round 6): after solving level 1,
    #    the harness probes that exact solution on level 2. Case A: the
    #    solution does NOT solve level 2 -> diagnostic note, game untouched.
    import inference.agent.tool_agent as ta_module
    ta_module._ATLAS_MECHANIC_HANDOFF = True
    try:
        h1_path = tmp_dir / "handoff1_state.json"
        h1_env = MultiLevelEnv(h1_path, goals=[2, 9])
        h1_agent = _make_agent(h1_path, h1_env)
        h1_agent._run_python_tool(h1_path, {"code": "action(['UP', 'UP'])\n"})
        if h1_env.level != 2 or h1_agent._atlas_pending_mechanic_handoff != 1:
            _fail("setup: level 1 solved, handoff pending", f"level={h1_env.level}")
        h1_agent._run_python_tool(h1_path, {"code": "result = 1\n"})
        if h1_env.level != 2 or h1_env.pos != 0:
            _fail("a failed handoff probe leaves the real game untouched", f"pos={h1_env.pos}")
        prompt = h1_agent._build_user_prompt(0, valid_actions=list(MultiLevelEnv.VALID))
        if "[HARNESS DIAGNOSTIC]" not in prompt or "level NOT completed" not in prompt:
            _fail("handoff diagnostic note lands in the prompt", prompt[-600:])
        _ok("mechanic handoff: level-1 solution probed on level 2, escalation reported, game untouched")

        # 4b. Round-8 safety valve: a probe that ends in game_over rolls the
        #     env back AND injects NO diagnostic (a game_over/stop_reason
        #     note measurably confused the model on the B8 testbed run).
        h1b_path = tmp_dir / "handoff1b_state.json"
        h1b_env = MultiLevelEnv(h1b_path, goals=[2, 9])
        h1b_agent = _make_agent(h1b_path, h1b_env)
        h1b_agent._run_python_tool(h1b_path, {"code": "action(['UP', 'UP'])\n"})
        if h1b_env.level != 2 or h1b_agent._atlas_pending_mechanic_handoff != 1:
            _fail("setup: level 1 solved, handoff pending (case game_over)", f"level={h1b_env.level}")
        original_call = MultiLevelEnv.__call__

        def _lethal_call(self, payload):
            out = original_call(self, payload)
            if payload.get("probe"):
                self.game_over = True
                out["game_over"] = True
            return out

        MultiLevelEnv.__call__ = _lethal_call
        try:
            h1b_agent._run_python_tool(h1b_path, {"code": "result = 1\n"})
        finally:
            MultiLevelEnv.__call__ = original_call
        if h1b_env.level != 2 or h1b_env.pos != 0 or h1b_env.game_over:
            _fail("game_over probe rolled back cleanly", f"pos={h1b_env.pos} go={h1b_env.game_over}")
        prompt = h1b_agent._build_user_prompt(0, valid_actions=list(MultiLevelEnv.VALID))
        if "[HARNESS DIAGNOSTIC]" in prompt and "DANGEROUS" in prompt:
            _fail("game_over handoff diagnostic is SUPPRESSED", prompt[-600:])
        _ok("mechanic handoff valve: game_over probe -> rollback, no diagnostic injected")

        # 5. Case B: the level-1 solution ALSO solves level 2 -> the harness
        #    executes it for real, zero model turns.
        h2_path = tmp_dir / "handoff2_state.json"
        h2_env = MultiLevelEnv(h2_path, goals=[2, 2, 9])
        h2_agent = _make_agent(h2_path, h2_env)
        h2_agent._run_python_tool(h2_path, {"code": "action(['UP', 'UP'])\n"})
        if h2_env.level != 2:
            _fail("setup: level 1 solved (case B)", f"level={h2_env.level}")
        h2_agent._run_python_tool(h2_path, {"code": "result = 1\n"})
        if h2_env.level != 3:
            _fail("handoff auto-executes a solution that also solves level 2", f"level={h2_env.level}")
        prompt = h2_agent._build_user_prompt(0, valid_actions=list(MultiLevelEnv.VALID))
        if "ALSO SOLVES" not in prompt:
            _fail("auto-solve handoff note lands in the prompt", prompt[-500:])
        _ok("mechanic handoff: a transferable solution is executed for real -- level 3 with zero model turns")
    finally:
        ta_module._ATLAS_MECHANIC_HANDOFF = False

    # 6. Hail Mary (29.08): nearly-dead game on level 2 -> one last-gasp
    #    deep search runs and its plan is executed for real.
    hm_path = tmp_dir / "hailmary_state.json"
    hm_env = MultiLevelEnv(hm_path, goals=[2, 3])
    hm_agent = _make_agent(hm_path, hm_env)
    hm_agent._run_python_tool(hm_path, {"code": "action(['UP', 'UP'])\n"})
    if hm_env.level != 2:
        _fail("setup: on level 2 for the hail mary", f"level={hm_env.level}")
    hm_agent._atlas_time_remaining_callback = lambda: 120.0
    hm_agent._run_python_tool(hm_path, {"code": "result = 1\n"})
    if not hm_agent._atlas_hail_mary_done:
        _fail("hail mary fires when the clock is nearly dead on level 2+", "flag not set")
    if hm_env.level != 3:
        _fail("hail-mary search solves level 2 and executes the plan", f"level={hm_env.level}")
    prompt = hm_agent._build_user_prompt(0, valid_actions=list(MultiLevelEnv.VALID))
    if "last-gasp" not in prompt:
        _fail("hail-mary note lands in the prompt", prompt[-500:])
    hm_agent._run_python_tool(hm_path, {"code": "result = 2\n"})
    _ok("hail mary: <10 min left on level 2 -> deep search runs once, plan executed (level 3), note injected")

    # 7. Draft-speedrun (30.08, Gemini round 10b Q3): a sloppy L1 win (with
    #    a provable UP/DOWN loop in the trace) triggers a voluntary full
    #    restart + loop-compressed clean replay; an optimal draft does not.
    #    30.08 evening: default flipped OFF (dead before WIN under
    #    ONLY_RESET_LEVELS -- mock-LLM stress finding); the mechanism is
    #    still tested here in isolation, forced ON for this scenario only.
    _saved_draft_speedrun = tool_agent._ATLAS_DRAFT_SPEEDRUN
    tool_agent._ATLAS_DRAFT_SPEEDRUN = True
    sp_path = tmp_dir / "speedrun_state.json"
    sp_env = MultiLevelEnv(sp_path, goals=[2, 9])
    sp_agent = _make_agent(sp_path, sp_env)
    for code in ("action('UP')\n", "action('DOWN')\n", "action('UP')\n", "action('UP')\n"):
        sp_agent._run_python_tool(sp_path, {"code": code})
    tool_agent._ATLAS_DRAFT_SPEEDRUN = _saved_draft_speedrun
    if sp_env.level != 2:
        _fail("setup: sloppy 4-action draft still wins level 1", f"level={sp_env.level}")
    if sp_agent._atlas_speedrun_done_upto != 1:
        _fail("draft-speedrun fired after the sloppy win", f"done_upto={sp_agent._atlas_speedrun_done_upto}")
    comp = sp_agent._atlas_level_solutions.get(1) or []
    if len(comp) != 2:
        _fail("solution loop-compressed 4 -> 2 (UP/DOWN cycle cut)", f"len={len(comp)} {comp}")
    prompt = sp_agent._build_user_prompt(0, valid_actions=list(MultiLevelEnv.VALID))
    if "autopilot" not in prompt:
        _fail("speedrun replay note lands in the prompt", prompt[-500:])
    _ok("draft-speedrun: sloppy win -> voluntary restart, 4->2 compression, clean replay back to level 2")

    print("\nAll atlas level-replay (L2) checks passed.")


if __name__ == "__main__":
    main()

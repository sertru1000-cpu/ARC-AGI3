"""Exercise try_actions()/plan_real() -- speculative execution on the REAL
engine (27.08, the direction change away from checkpoint nudges).

The insight this feature is built on: the OFFLINE engine snapshot/restore
built for rollback() doubles as a PERFECT world model. game_run (whose
actions_per_level/levels_completed feed _compute_final_score) is part of
the snapshot, so actions executed inside a probe and then rewound never
reach the recorded score, while the engine scorecard is a monotonic max()
that keeps any level a probe happens to complete. That removes the need
for a model-authored predict() for planning ENTIRELY -- the exact
authorship friction the theory-checkpoint chain (v12/v16/v21/v22 sagas)
has been fighting all along: instead of forcing the model to encode a
theory as code and verify it, the model (or the harness BFS) just tries
sequences for real and rewinds.

Drives the real sandbox subprocess and the real ToolAgent._handle_checkpoint
dispatch through a scripted mini-game (a 1-D walk: UP advances, DOWN
retreats with a floor at 0, ACTION7 is an instant game-over trap, and the
level completes at position GOAL). Covers: per-sequence probe outcomes,
BFS finding the exact shortest plan with dedup and trap pruning, honest
exhaustion reporting, the real run being left untouched afterwards, and
graceful ONLINE-mode unavailability.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "tufa-arc-agi-framework" / "src"))
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "ARC3-Inference"))

from inference.agent.tool_agent import (  # noqa: E402
    ToolAgent,
    _ATLAS_PROBE_THEORY_GRACE_CALLS,
    _ATLAS_THEORY_NAG_AFTER_CALLS,
)
from inference.agent.runtime_state import Frame, load_runtime_state, write_runtime_state  # noqa: E402


def _fail(name: str, detail: str) -> None:
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def _ok(name: str) -> None:
    print(f"ok   {name}")


class WalkEnv:
    """1-D walk with real-step_env-shaped batch semantics. UP: pos+1,
    DOWN: pos-1 (floor 0, a no-op there -- exercises BFS dedup), ACTION7:
    instant game_over (exercises trap pruning). Level completes at GOAL.
    """

    VALID = ("UP", "DOWN", "ACTION7")

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
            current_frame=Frame(grid=((self.pos,),), step=self.step, level=self.level),
            history=[],
        )

    def snapshot(self):
        return {"pos": self.pos, "level": self.level, "step": self.step, "game_over": self.game_over}

    def restore(self, snap):
        if not isinstance(snap, dict):
            return False
        self.pos = snap["pos"]
        self.level = snap["level"]
        self.step = snap["step"]
        self.game_over = snap["game_over"]
        self._write()
        return True

    def __call__(self, payload):
        specs = list(payload.get("actions") or [])
        executed = 0
        board_changed = False
        level_completed = False
        stop_reason = None
        requested = [str(s.get("action") if isinstance(s, dict) else s) for s in specs]
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
                stop_reason = "level_completed"
                break
            self._write()
        return {
            "executed": executed > 0,
            "executed_count": executed,
            "requested_count": len(specs),
            "requested_actions": requested,
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


def _make_agent(state_path: Path, env: WalkEnv) -> ToolAgent:
    agent = ToolAgent(model="test-model")
    agent._step_env_callback = env
    agent._checkpoint_env_callback = env.snapshot
    agent._restore_env_callback = env.restore
    agent._current_valid_actions = list(WalkEnv.VALID)
    return agent


def main() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="atlas_real_search_test_"))

    # 1. try_actions: three speculative sequences, each judged from the same
    #    starting state, and the real game left EXACTLY where it was.
    state_path = tmp_dir / "run_state.json"
    env = WalkEnv(state_path, goal=3)
    agent = _make_agent(state_path, env)

    dispatch = agent._run_python_tool(
        state_path,
        {"code": "result = try_actions([['UP'], ['DOWN'], ['UP', 'UP', 'UP']])\n"},
    )
    payload = json.loads(dispatch.content)
    if payload.get("error"):
        _fail("try_actions runs", str(payload))
    results = (payload.get("result") or {}).get("results") or []
    if len(results) != 3:
        _fail("one outcome per sequence", str(results))
    if not results[0]["board_changed"] or results[0]["cells_changed_vs_start"] != 1:
        _fail("UP visibly moves the walker", str(results[0]))
    if results[1]["board_changed"]:
        _fail("DOWN at the floor is reported as a no-op", str(results[1]))
    if not results[2]["level_completed"] or results[2]["level_after"] != 2:
        _fail("UP,UP,UP completes the level speculatively", str(results[2]))
    if env.pos != 0 or env.level != 1 or env.game_over:
        _fail("real game untouched after try_actions", f"pos={env.pos} level={env.level}")
    frame, _ = load_runtime_state(state_path)
    if frame.grid != ((0,),):
        _fail("runtime state file restored after try_actions", str(frame.grid))
    _ok("try_actions judges each sequence from the same start and rewinds the real game to exactly where it was")

    # 2. plan_real: BFS with default candidates finds the exact shortest
    #    plan, pruning the ACTION7 trap and deduping the DOWN no-op.
    dispatch = agent._run_python_tool(state_path, {"code": "result = plan_real(max_depth=5)\n"})
    payload = json.loads(dispatch.content)
    if payload.get("error"):
        _fail("plan_real runs", str(payload))
    result = payload.get("result") or {}
    plan = result.get("plan")
    if not plan or [str(s.get("action")).upper() for s in plan] != ["UP", "UP", "UP"]:
        _fail("plan_real finds the exact shortest plan", str(result))
    if result.get("reason") != "level_completed":
        _fail("plan_real reports why it stopped", str(result))
    if env.pos != 0 or env.level != 1 or env.game_over:
        _fail("real game untouched after plan_real", f"pos={env.pos} level={env.level} go={env.game_over}")
    _ok("plan_real BFS finds the exact shortest level-completing plan on the real engine, then rewinds")

    # 3. The found plan replays for real -- and the game actually advances.
    agent._run_python_tool(state_path, {"code": "action(['UP', 'UP', 'UP'])\n"})
    if env.level != 2 or env.pos != 3:
        _fail("replaying the found plan for real advances the game", f"pos={env.pos} level={env.level}")
    _ok("replaying the found plan with action(...) advances the REAL game -- speculation converted to score")

    # 4. Honest exhaustion: an unreachable goal within max_depth reports
    #    state_space_exhausted (every reachable state genuinely tried),
    #    not a fake success and not an error.
    far_state_path = tmp_dir / "far_run_state.json"
    far_env = WalkEnv(far_state_path, goal=9)
    far_agent = _make_agent(far_state_path, far_env)
    dispatch = far_agent._run_python_tool(far_state_path, {"code": "result = plan_real(max_depth=3, rollouts=False)\n"})
    payload = json.loads(dispatch.content)
    result = payload.get("result") or {}
    if result.get("plan") is not None:
        _fail("no fake plan for an unreachable goal", str(result))
    if result.get("reason") != "state_space_exhausted":
        _fail("exhaustion reported honestly", str(result))
    if int(result.get("unique_states_reached") or 0) != 3:
        _fail("dedup makes the search linear here (3 unique states at depth 3)", str(result))
    if int(result.get("rollouts") or 0) != 0:
        _fail("rollouts=False really disables the rollout phase", str(result))
    if far_env.pos != 0 or far_env.game_over:
        _fail("real game untouched after an exhausted search", f"pos={far_env.pos}")
    _ok("an unreachable goal reports state_space_exhausted (3 unique states tried at depth 3, rollouts off) and rewinds cleanly")

    # 4b. v2 Monte-Carlo rollouts: the same deep goal IS found once
    #     rollouts are allowed -- here the only valid action is UP, so a
    #     depth-24 random rollout deterministically walks into the goal at
    #     depth 9, far beyond the systematic max_depth of 3.
    deep_state_path = tmp_dir / "deep_run_state.json"
    deep_env = WalkEnv(deep_state_path, goal=9)
    deep_env.VALID = ("UP",)
    deep_agent = _make_agent(deep_state_path, deep_env)
    deep_agent._current_valid_actions = ["UP"]
    dispatch = deep_agent._run_python_tool(deep_state_path, {"code": "result = plan_real(max_depth=3)\n"})
    payload = json.loads(dispatch.content)
    result = payload.get("result") or {}
    plan = result.get("plan") or []
    if result.get("found_by") != "rollout":
        _fail("deep goal found by the rollout phase", str(result))
    if len(plan) != 9 or {str(s.get("action")).upper() for s in plan} != {"UP"}:
        _fail("rollout plan is exactly the 9 UPs that complete the level", str(result))
    if deep_env.pos != 0 or deep_env.level != 1:
        _fail("real game untouched after a rollout-found plan", f"pos={deep_env.pos} level={deep_env.level}")
    _ok("Monte-Carlo rollout finds the depth-9 solution the depth-3 frontier cannot, and rewinds cleanly")

    # 5. ONLINE mode (no snapshot/restore wired): graceful unavailability,
    #    same contract as save_checkpoint/rollback.
    online_state_path = tmp_dir / "online_run_state.json"
    online_env = WalkEnv(online_state_path, goal=3)
    online_agent = ToolAgent(model="test-model")
    online_agent._step_env_callback = online_env
    online_agent._current_valid_actions = list(WalkEnv.VALID)
    dispatch = online_agent._run_python_tool(online_state_path, {"code": "plan_real()\n"})
    payload = json.loads(dispatch.content)
    if "not available" not in str(payload.get("error", "")):
        _fail("ONLINE mode graceful unavailability", str(payload))
    _ok("no snapshot/restore wired (ONLINE mode) -> plan_real fails gracefully, same contract as rollback")

    # 6. Probe/checkpoint integration (found live 27.08 on the first
    #    OFFLINE pod run: explore-first fired 153x in 37 min because probes
    #    were invisible to it): a UNIFORM probe with a visible effect
    #    credits that control kind for explore-first; a uniform inert probe
    #    counts toward the same 3-attempt inert resolution.
    probe_state_path = tmp_dir / "probe_credit_state.json"
    probe_env = WalkEnv(probe_state_path, goal=5)
    probe_agent = _make_agent(probe_state_path, probe_env)
    probe_agent._current_valid_actions = ["UP", "DOWN"]
    probe_agent._run_python_tool(
        probe_state_path, {"code": "result = try_actions([['UP', 'UP'], ['UP', 'DOWN']])\n"}
    )
    if "UP" not in probe_agent._atlas_action_kinds_resolved:
        _fail(
            "a uniform probe with a visible effect credits its kind for explore-first",
            str(probe_agent._atlas_action_kinds_resolved),
        )
    if "DOWN" in probe_agent._atlas_action_kinds_resolved:
        _fail(
            "a MIXED probe sequence credits nothing (effect not attributable)",
            str(probe_agent._atlas_action_kinds_resolved),
        )
    for _ in range(3):
        probe_agent._run_python_tool(probe_state_path, {"code": "result = try_actions([['DOWN']])\n"})
    if "DOWN" not in probe_agent._atlas_action_kinds_resolved:
        _fail(
            "3 uniform inert probes resolve the kind as inert (same rule as real actions)",
            str(probe_agent._atlas_action_kind_attempts),
        )
    _ok("probes feed explore-first: uniform+effect credits the kind, mixed credits nothing, "
        "3 inert uniform probes resolve as inert")

    # 7. Theory-nag grace after probes: a successfully executed probe
    #    keeps the soft theory nag AND theory-force quiet for
    #    _ATLAS_PROBE_THEORY_GRACE_CALLS calls; once the grace elapses
    #    with no further probes, the nag resumes.
    prompt = probe_agent._build_user_prompt(0, valid_actions=["UP", "DOWN"])
    if "THIS turn, write predict" in prompt or "ZERO verify_theory() calls" in prompt:
        _fail("theory nags stay quiet right after a probe", prompt[-400:])
    for _ in range(_ATLAS_PROBE_THEORY_GRACE_CALLS):
        probe_agent._run_python_tool(probe_state_path, {"code": "result = 1\n"})
        probe_agent._atlas_calls_since_real_action = 0
    if probe_agent._atlas_python_call_index < _ATLAS_THEORY_NAG_AFTER_CALLS:
        _fail("test setup: past the soft theory threshold", str(probe_agent._atlas_python_call_index))
    prompt = probe_agent._build_user_prompt(0, valid_actions=["UP", "DOWN"])
    if "THIS turn, write predict" not in prompt and "ZERO verify_theory() calls" not in prompt:
        _fail("theory nag resumes once the probe grace elapses", prompt[-500:])
    _ok(f"a real probe opens a {_ATLAS_PROBE_THEORY_GRACE_CALLS}-call grace window for theory nags, "
        "after which they resume")

    print("\nAll atlas real-search (try_actions/plan_real) checks passed.")


if __name__ == "__main__":
    main()

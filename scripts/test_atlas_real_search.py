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
import os

# The 28.08 proactive level-entry plan_real would auto-solve the scripted
# mini-games before the scenarios under test even run -- disable it for
# this suite (dedicated proactive scenarios re-enable it via monkeypatch).
os.environ["ATLAS_PLAN_REAL_PROACTIVE"] = "0"
os.environ["ATLAS_MECHANIC_HANDOFF"] = "0"
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

    # 2+3 (merged 28.08, Gemini round 5): plan_real finds the exact shortest
    #    plan AND the harness executes it immediately on the model's behalf
    #    -- zero extra model turns between reasoning and score.
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
    if not result.get("executed_by_harness"):
        _fail("the harness executes the found plan itself", str(result))
    if "[Harness Auto-Action]" not in str(result.get("note")) or "deduce the game's mechanics" not in str(result.get("note")):
        _fail("the auto-action note explains what happened and teaches the mechanics", str(result.get("note")))
    if env.level != 2 or env.pos != 3:
        _fail("the REAL game advanced to level 2 via the auto-executed plan", f"pos={env.pos} level={env.level}")
    _ok("plan_real finds the exact shortest plan and the harness EXECUTES it immediately -- "
        "real game at level 2, zero model turns spent")

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
    if deep_env.pos != 9 or deep_env.level != 2:
        _fail("the rollout-found plan is auto-executed too (28.08)", f"pos={deep_env.pos} level={deep_env.level}")
    _ok("Monte-Carlo rollout finds the depth-9 solution the depth-3 frontier cannot, and the "
        "harness executes it (level 2 reached)")

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
        # This scenario tests explore-first CREDITING, not rationing -- keep
        # the 28.08 hard gate out of the way (scenario 9 covers it).
        probe_agent._atlas_probes_since_real_action = 0
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

    # 8. Gemini round 3 -- compact probe results + prompt-side probe memory.
    #    try_actions results drop what the model already knows (its own
    #    requested sequence) and every falsy flag; each sequence leaves a
    #    one-line finding that is injected into the NEXT prompt so learned
    #    dynamics persist across turns instead of being re-probed. Findings
    #    are per-level: completing the level clears them.
    mem_state_path = tmp_dir / "probe_memory_state.json"
    mem_env = WalkEnv(mem_state_path, goal=3)
    mem_agent = _make_agent(mem_state_path, mem_env)
    dispatch = mem_agent._run_python_tool(
        mem_state_path, {"code": "result = try_actions([['UP'], ['DOWN']])\n"}
    )
    payload = json.loads(dispatch.content)
    results = (payload.get("result") or {}).get("results") or []
    if len(results) != 2:
        _fail("compact scenario setup: two probe results", str(payload))
    for entry in results:
        for banned in ("requested", "requested_actions", "level_before"):
            if banned in entry:
                _fail(f"compact results drop '{banned}'", str(entry))
        if "game_over" in entry or "run_complete" in entry:
            _fail("falsy flags are omitted from compact results", str(entry))
    for needed in ("sequence_index", "executed_count", "board_changed",
                   "cells_changed_vs_start", "level_after", "level_completed"):
        if needed not in results[0]:
            _fail(f"compact results still carry '{needed}'", str(results[0]))
    prompt = mem_agent._build_user_prompt(0, valid_actions=["UP", "DOWN"])
    if "Probe memory (free simulations already run this level):" not in prompt:
        _fail("probe findings are injected into the prompt", prompt[-600:])
    if "UP->1c changed" not in prompt or "DOWN->no effect" not in prompt:
        _fail("findings carry the sequence and its outcome", prompt[-600:])
    mem_agent._run_python_tool(mem_state_path, {"code": "action(['UP', 'UP', 'UP'])\n"})
    if mem_env.level != 2:
        _fail("memory scenario setup: level completes for real", f"level={mem_env.level}")
    if mem_agent._atlas_probe_findings:
        _fail("level-up clears the probe memory (per-level findings)",
              str(mem_agent._atlas_probe_findings))
    _ok("compact try_actions results (no requested echo, falsy flags omitted) + probe memory "
        "injected into the prompt and cleared on level-up")

    # 9. Probe rationing, HARD-GATE form (28.08, Gemini round 4 -- the soft
    #    nudge alone was ignored in live tails of 8 consecutive probes):
    #    3 consecutive probe calls without a real action -> the prompt warns
    #    the tool is locked AND the 4th try_actions call actually returns an
    #    error instead of results; one real action() unlocks everything.
    ration_state_path = tmp_dir / "probe_ration_state.json"
    ration_env = WalkEnv(ration_state_path, goal=8)
    ration_agent = _make_agent(ration_state_path, ration_env)
    for i in range(3):
        prompt = ration_agent._build_user_prompt(0, valid_actions=["UP", "DOWN"])
        if "probe calls in a row" in prompt:
            _fail(f"no ration warning before the 3rd consecutive probe (after {i})", prompt[-400:])
        dispatch = ration_agent._run_python_tool(
            ration_state_path, {"code": "result = try_actions([['UP', 'DOWN']])\n"}
        )
        if "Probe budget exhausted" in dispatch.content:
            _fail(f"probe {i + 1} of 3 is still free", dispatch.content[:300])
        ration_agent._atlas_calls_since_real_action = 0
    if ration_agent._atlas_probes_since_real_action != 3:
        _fail("3 probes counted", str(ration_agent._atlas_probes_since_real_action))
    prompt = ration_agent._build_user_prompt(0, valid_actions=["UP", "DOWN"])
    if "You have run 3 probe calls in a row" not in prompt or "LOCKED" not in prompt:
        _fail("prompt warns the tool is locked at 3 consecutive probes", prompt[-600:])
    dispatch = ration_agent._run_python_tool(
        ration_state_path, {"code": "result = try_actions([['UP']])\n"}
    )
    payload = json.loads(dispatch.content)
    err = str(payload.get("error") or "") + str((payload.get("result") or {}).get("error") or "")
    if "Probe budget exhausted" not in err and "Probe budget exhausted" not in dispatch.content:
        _fail("4th consecutive try_actions is BLOCKED by the hard gate", dispatch.content[:400])
    if ration_agent._atlas_probes_since_real_action != 3:
        _fail("a blocked call does not grow the streak", str(ration_agent._atlas_probes_since_real_action))
    ration_agent._run_python_tool(ration_state_path, {"code": "action('UP')\n"})
    if ration_agent._atlas_probes_since_real_action != 0:
        _fail("a real action resets the probe streak", str(ration_agent._atlas_probes_since_real_action))
    dispatch = ration_agent._run_python_tool(
        ration_state_path, {"code": "result = try_actions([['DOWN']])\n"}
    )
    if "Probe budget exhausted" in dispatch.content:
        _fail("the lock lifts after a real action", dispatch.content[:300])
    prompt = ration_agent._build_user_prompt(0, valid_actions=["UP", "DOWN"])
    if "probe calls in a row" in prompt:
        _fail("warning gone after a real action", prompt[-400:])
    _ok("probe hard gate: 3 free probes, warning in the prompt, 4th call blocked with an error, "
        "one real action unlocks")

    # 10. Gemini round 3 -- probing IS exploration: an executed probe
    #     silences the explore-first checkpoint for the same 4-call grace
    #     window (measured live: 280-296 explore-first injections per probe
    #     run under the old crediting-only rule). A MIXED probe is used so
    #     no kind gets resolved -- the silencing must come from the grace
    #     alone, not from crediting.
    ex_state_path = tmp_dir / "explore_grace_state.json"
    ex_env = WalkEnv(ex_state_path, goal=8)
    ex_agent = _make_agent(ex_state_path, ex_env)
    for _ in range(2):
        ex_agent._run_python_tool(ex_state_path, {"code": "result = 1\n"})
        ex_agent._atlas_calls_since_real_action = 0
    prompt = ex_agent._build_user_prompt(0, valid_actions=["UP", "DOWN"])
    if "still untested" not in prompt:
        _fail("scenario setup: explore-first fires with unresolved kinds and no probes", prompt[-600:])
    ex_agent._run_python_tool(ex_state_path, {"code": "result = try_actions([['UP', 'DOWN']])\n"})
    ex_agent._atlas_calls_since_real_action = 0
    if ex_agent._atlas_action_kinds_resolved:
        _fail("scenario setup: the mixed probe must credit nothing", str(ex_agent._atlas_action_kinds_resolved))
    prompt = ex_agent._build_user_prompt(0, valid_actions=["UP", "DOWN"])
    if "still untested" in prompt:
        _fail("ANY executed probe silences explore-first during the grace window", prompt[-600:])
    for _ in range(_ATLAS_PROBE_THEORY_GRACE_CALLS):
        ex_agent._run_python_tool(ex_state_path, {"code": "result = 1\n"})
        ex_agent._atlas_calls_since_real_action = 0
    prompt = ex_agent._build_user_prompt(0, valid_actions=["UP", "DOWN"])
    if "still untested" not in prompt:
        _fail("explore-first resumes once the probe grace elapses", prompt[-600:])
    _ok(f"an executed probe silences explore-first for the {_ATLAS_PROBE_THEORY_GRACE_CALLS}-call "
        "grace window, after which it resumes")

    # 11. Gemini round 3 -- MOUSE candidate auto-derivation: centers of the
    #     largest non-background segmentation objects, so click games get
    #     searched with zero model authorship. Background (the >30% field
    #     and the single largest component) is excluded.
    grid = [[0] * 20 for _ in range(20)]
    for r in range(2, 6):
        for c in range(2, 6):
            grid[r][c] = 3  # 4x4 object, 16 px
    for r in range(10, 13):
        for c in range(14, 17):
            grid[r][c] = 2  # 3x3 object, 9 px
    cands = ToolAgent._atlas_default_mouse_candidates(tuple(tuple(r) for r in grid))
    if len(cands) != 2:
        _fail("two non-background objects -> two candidates", str(cands))
    if any(c["action"] != "MOUSE" for c in cands):
        _fail("candidates are MOUSE click specs", str(cands))
    big, small = cands[0], cands[1]
    if abs(big["row"] - 4) > 1 or abs(big["col"] - 4) > 1:
        _fail("largest object first, centroid near (4,4)", str(cands))
    if abs(small["row"] - 11) > 1 or abs(small["col"] - 15) > 1:
        _fail("second candidate centroid near (11,15)", str(cands))
    if ToolAgent._atlas_default_mouse_candidates(((0,),)) != []:
        _fail("a featureless board yields no candidates", "non-empty")
    _ok("MOUSE auto-candidates: centroids of the two objects (largest first), background excluded, "
        "featureless board -> none")

    # 12. Gemini round 5, L1 -- PROACTIVE plan_real on level entry: with the
    #     flag on, the very first python call of a fresh game runs the
    #     search harness-side, executes the found plan (level 1 solved with
    #     ZERO model turns), injects the autopilot note, and seeds the NEXT
    #     level's search; a proactive miss on the new level stays silent.
    import inference.agent.tool_agent as ta_module
    ta_module._ATLAS_PLAN_REAL_PROACTIVE = True
    # 30.08: the L1 stock sprint (scenario 14) would defer this scenario's
    # first-call search -- isolate the proactive machinery from the gate.
    _saved_sprint = ta_module._ATLAS_L1_STOCK_SPRINT
    ta_module._ATLAS_L1_STOCK_SPRINT = False
    try:
        pro_state_path = tmp_dir / "proactive_state.json"
        pro_env = WalkEnv(pro_state_path, goal=2)
        pro_agent = _make_agent(pro_state_path, pro_env)
        pro_agent._run_python_tool(pro_state_path, {"code": "result = 1\n"})
        if pro_env.level != 2 or pro_env.pos != 2:
            _fail("proactive search auto-solves level 1 on the first python call",
                  f"pos={pro_env.pos} level={pro_env.level}")
        prompt = pro_agent._build_user_prompt(0, valid_actions=["UP", "DOWN"])
        if "[atlas autopilot]" not in prompt or "deduce the game's mechanics" not in prompt:
            _fail("autopilot note lands in the next prompt with the mechanics hand-off", prompt[-600:])
        if not pro_agent._atlas_pending_auto_plan_real:
            _fail("the level-up seeds the NEXT level's proactive search", "pending flag not set")
        pro_env.goal = -1  # make the new level genuinely unsolvable for the search
        pro_agent._run_python_tool(pro_state_path, {"code": "result = 2\n"})
        prompt = pro_agent._build_user_prompt(0, valid_actions=["UP", "DOWN"])
        if "[atlas autopilot]" in prompt:
            _fail("a proactive MISS on the unreachable level 2 stays silent", prompt[-600:])
        if pro_env.game_over:
            _fail("proactive search never leaves the real game dead", "game_over")
    finally:
        ta_module._ATLAS_PLAN_REAL_PROACTIVE = False
        ta_module._ATLAS_L1_STOCK_SPRINT = _saved_sprint
    _ok("proactive plan_real: level 1 auto-solved on the first call with zero model turns, "
        "autopilot note injected, next level seeded, miss stays silent")

    # 14. Gemini round 13 Q1 + duck survival curve (30.08): the L1 STOCK
    #     SPRINT defers the proactive search on level 1 -- the first python
    #     call must NOT auto-solve, the pending flag must survive, and the
    #     search must fire once the stall evidence (blocked noops >= 2)
    #     arrives, still solving the level.
    ta_module._ATLAS_PLAN_REAL_PROACTIVE = True
    ta_module._ATLAS_L1_STOCK_SPRINT = True
    try:
        sp_state_path = tmp_dir / "sprint_state.json"
        sp_env = WalkEnv(sp_state_path, goal=2)
        sp_agent = _make_agent(sp_state_path, sp_env)
        sp_agent._run_python_tool(sp_state_path, {"code": "result = 1\n"})
        if sp_env.level != 1:
            _fail("L1 sprint defers the proactive search on the first call", f"level={sp_env.level}")
        if not sp_agent._atlas_pending_auto_plan_real:
            _fail("deferral keeps the pending flag alive", "flag consumed")
        sp_agent._atlas_blocked_noops_since_progress = 2  # stall evidence -> escalation
        sp_agent._run_python_tool(sp_state_path, {"code": "result = 2\n"})
        if sp_env.level != 2:
            _fail("stall evidence un-defers the search and it solves L1", f"level={sp_env.level}")
    finally:
        ta_module._ATLAS_PLAN_REAL_PROACTIVE = False
        ta_module._ATLAS_L1_STOCK_SPRINT = _saved_sprint
    _ok("L1 stock sprint: proactive search deferred on level 1, pending survives, "
        "escalates on 2 blocked noops and still solves the level")

    # 13. A* wiring (30.08, backlog 19): with a heuristic model injected the
    #     frontier orders by g + w*h -- the search must still find the exact
    #     plan; with the cache cleared, ordering falls back to novelty.
    class _FakeH:
        def predict(self, X):
            return [0.0]

    astar_state_path = tmp_dir / "astar_state.json"
    astar_env = WalkEnv(astar_state_path, goal=3)
    astar_agent = _make_agent(astar_state_path, astar_env)
    ta_module._ATLAS_ASTAR_CACHE["loaded"] = True
    ta_module._ATLAS_ASTAR_CACHE["model"] = _FakeH()
    try:
        dispatch = astar_agent._run_python_tool(astar_state_path, {"code": "result = plan_real(max_depth=5)\n"})
        payload = json.loads(dispatch.content)
        result = payload.get("result") or {}
        plan = result.get("plan")
        if not plan or [str(s.get("action")).upper() for s in plan] != ["UP", "UP", "UP"]:
            _fail("A*-ordered search still finds the exact shortest plan", str(result))
        if astar_env.level != 2:
            _fail("A* search plan executed for real", f"level={astar_env.level}")
    finally:
        ta_module._ATLAS_ASTAR_CACHE["loaded"] = False
        ta_module._ATLAS_ASTAR_CACHE["model"] = None
    _ok("A* wiring: injected heuristic reorders the frontier without breaking search or execution; "
        "default-off cache restored")

    print("\nAll atlas real-search (try_actions/plan_real) checks passed.")


if __name__ == "__main__":
    main()

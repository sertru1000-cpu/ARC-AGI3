"""Exercise the atlas save_checkpoint/rollback feature (Gemini idea #1,
refined with host-enforced auto-anchors and a two-step coercion ultimatum)
and the extract-suggestion checkpoint (idea #2).

Background: the project's own C0 finding is that a tool merely documented in
the static prompt gets used in ~0.2%-0% of turns (memo: 0/81 real actions
across 3 games, live). A voluntary save_checkpoint()/rollback() would suffer
the exact same fate, so this feature is host-enforced: the harness
auto-creates checkpoints at game start (sys_start) and every level-up
(sys_level_N), detects two stall/loop signals itself (Trigger A: many real
actions since the last level progress; Trigger B: the board looped back to a
state seen _ATLAS_ROLLBACK_LOOP_WINDOW actions ago), and if the resulting
ultimatum is ignored for _ATLAS_ROLLBACK_AUTO_FORCE_AFTER turns in a row, the
harness performs the rollback itself.

Two levels, same split as test_atlas_memo.py: the raw run_sandboxed_python
round-trip (save_checkpoint/rollback wire protocol against the real
subprocess), and the real ToolAgent._run_python_tool/_build_user_prompt path
(auto-anchors, both triggers, the ultimatum escalation, and the auto-force
backstop) via a fake env that actually persists to the runtime-state file
like the real solver.py does.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARC3_INFERENCE = ROOT / "atlas_src" / "src" / "ARC3-Inference"
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "tufa-arc-agi-framework" / "src"))
sys.path.insert(0, str(ARC3_INFERENCE))

spec = importlib.util.spec_from_file_location(
    "python_tool_sandbox", ARC3_INFERENCE / "inference" / "agent" / "python_tool_sandbox.py"
)
python_tool_sandbox = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(python_tool_sandbox)
run_sandboxed_python = python_tool_sandbox.run_sandboxed_python

from inference.agent.tool_agent import (  # noqa: E402
    ToolAgent,
    _ATLAS_EXTRACT_NUDGE_AFTER_CALLS,
    _ATLAS_ROLLBACK_AUTO_FORCE_AFTER,
    _ATLAS_ROLLBACK_LOOP_WINDOW,
    _ATLAS_ROLLBACK_STALL_AFTER_CALLS,
)
from inference.agent.runtime_state import (  # noqa: E402
    Frame,
    frame_from_payload,
    frame_to_payload,
    history_entry_from_payload,
    history_entry_to_payload,
    load_runtime_state,
    normalize_grid,
    write_runtime_state,
)


def _fail(name: str, detail: str) -> None:
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def _ok(name: str) -> None:
    print(f"ok   {name}")


def _frame_payload(grid=None, step: int = 0, level: int = 1):
    return {"ascii": "", "step": step, "level": level, "shape": [1, 1], "grid": grid or [[0]]}


def _base_state() -> dict:
    return {
        "current_frame": _frame_payload(),
        "history": [],
        "valid_actions": ["UP"],
        "last_action_result": {},
    }


def _write_state(path: Path, grid, step: int, level: int = 1) -> None:
    write_runtime_state(path, current_frame=Frame(grid=normalize_grid(grid), step=step, level=level), history=[])


def _make_env_callbacks(state_path: Path):
    """Mirrors solver.py's real atlas_snapshot_env/atlas_restore_env at the
    granularity that matters for this feature: dump enough of the runtime
    state to fully reconstruct it, and write it straight back on restore.
    """

    def checkpoint_env():
        current_frame, history_entries = load_runtime_state(state_path)
        return {
            "current_frame": frame_to_payload(current_frame),
            "history": [history_entry_to_payload(e) for e in history_entries],
        }

    def restore_env(snapshot):
        if not isinstance(snapshot, dict):
            return False
        frame = frame_from_payload(snapshot.get("current_frame"))
        history = [
            entry
            for raw in snapshot.get("history", [])
            for entry in [history_entry_from_payload(raw)]
            if entry is not None
        ]
        write_runtime_state(state_path, current_frame=frame, history=history)
        return True

    return checkpoint_env, restore_env


class FakeEnv:
    """A step_env_callback that actually persists to state_path, like the
    real harness does, so board_signature-based loop detection and
    snapshot/restore round-trips have real state to work with.
    """

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.level = 1
        self.counter = 0
        self.next_level_completed = False

    def __call__(self, payload):
        self.counter += 1
        if self.next_level_completed:
            self.level += 1
            self.next_level_completed = False
            level_completed = True
        else:
            level_completed = False
        # Distinct grid per call by default (no accidental Trigger B
        # matches) -- tests that want a loop overwrite this afterward.
        _write_state(self.state_path, [[self.counter]], step=self.counter, level=self.level)
        return {
            "executed": True,
            "action_num": self.counter,
            "level": self.level,
            "score": 0,
            "reward": 0.0,
            "board_changed": True,
            "done": False,
            "level_completed": level_completed,
            "game_over": False,
            "run_complete": False,
            "valid_actions": ["UP"],
        }


def main() -> None:
    # ---- Level 1: raw sandbox wire-protocol round-trip -------------------

    # 1. No checkpoint_handler wired (ONLINE mode) -> save_checkpoint raises,
    #    surfaced as a graceful error, not a crash.
    result = run_sandboxed_python(
        code="save_checkpoint('probe')\n",
        timeout_seconds=5,
        initial_state=_base_state(),
        action_handler=lambda actions: _fail("action", "unexpected action() call"),
    )
    if not result.get("error") or "not available" not in result["error"]:
        _fail("save_checkpoint without a handler fails gracefully", str(result))
    _ok("save_checkpoint raises a clean error when no checkpoint_handler is wired")

    # 2. A handler is wired: save request carries the right fields, returns
    #    a checkpoint_id.
    seen_requests = []

    def _handler_save_only(request):
        seen_requests.append(dict(request))
        return {"checkpoint_id": "cp1"}

    result = run_sandboxed_python(
        code="cid = save_checkpoint('before the risky move')\nresult = cid\n",
        timeout_seconds=5,
        initial_state=_base_state(),
        action_handler=lambda actions: _fail("action", "unexpected action() call"),
        checkpoint_handler=_handler_save_only,
    )
    if result.get("result") != "cp1":
        _fail("save_checkpoint returns the checkpoint_id", str(result))
    if len(seen_requests) != 1 or seen_requests[0].get("action") != "save" or seen_requests[0].get("label") != "before the risky move":
        _fail("save request carries action/label", str(seen_requests))
    _ok("save_checkpoint round-trips a checkpoint_id and sends action='save' + label")

    # 3. rollback(...) success path also refreshes preloaded state (state
    #    payload from the handler reaches current_frame in the SAME script).
    def _handler_rollback_ok(request):
        if request.get("action") == "rollback":
            return {"state": _base_state() | {"current_frame": _frame_payload(step=99)}}
        return {"checkpoint_id": "cpX"}

    result = run_sandboxed_python(
        code="rollback('cp1', 'that hypothesis was wrong')\nresult = current_frame.step\n",
        timeout_seconds=5,
        initial_state=_base_state(),
        action_handler=lambda actions: _fail("action", "unexpected action() call"),
        checkpoint_handler=_handler_rollback_ok,
    )
    if result.get("error") or result.get("result") != 99:
        _fail("rollback refreshes preloaded state", str(result))
    _ok("rollback(...) success reloads current_frame from the handler's state payload")

    # 4. rollback(...) failure (e.g. missing lesson_learned rejected by the
    #    host) surfaces as an error, not a silent no-op.
    def _handler_rollback_error(request):
        return {"error": "rollback(checkpoint_id, lesson_learned) requires a non-empty lesson_learned"}

    result = run_sandboxed_python(
        code="rollback('cp1', '')\n",
        timeout_seconds=5,
        initial_state=_base_state(),
        action_handler=lambda actions: _fail("action", "unexpected action() call"),
        checkpoint_handler=_handler_rollback_error,
    )
    if not result.get("error") or "lesson_learned" not in result["error"]:
        _fail("rollback error surfaces", str(result))
    _ok("a rejected rollback (e.g. empty lesson_learned) surfaces as a real error")

    # ---- Level 2: real ToolAgent path --------------------------------

    tmp_dir = Path(tempfile.mkdtemp(prefix="atlas_rollback_test_"))
    state_path = tmp_dir / "run_state.json"
    _write_state(state_path, [[9]], step=0, level=1)

    checkpoint_env, restore_env = _make_env_callbacks(state_path)
    env = FakeEnv(state_path)

    agent = ToolAgent(model="test-model")
    agent._checkpoint_env_callback = checkpoint_env
    agent._restore_env_callback = restore_env
    agent._step_env_callback = env
    agent._current_valid_actions = ["UP"]

    # 5. First-ever call auto-creates sys_start from the state on disk at
    #    that moment.
    agent._run_python_tool(state_path, {"code": "result = 1\n"})
    if "sys_start" not in agent._atlas_checkpoints:
        _fail("sys_start auto-anchor", str(agent._atlas_checkpoints))
    if agent._atlas_last_checkpoint_id != "sys_start":
        _fail("sys_start is the last checkpoint", agent._atlas_last_checkpoint_id)
    sys_start_grid = agent._atlas_checkpoints["sys_start"]["env_snapshot"]["current_frame"]["grid"]
    if sys_start_grid != [[9]]:
        _fail("sys_start captured the real game-start grid", str(sys_start_grid))
    _ok("sys_start auto-anchor created on the first call, snapshotting the real start-of-game state")

    # 6. Graceful unavailability: a fresh agent with no callbacks wired
    #    (ONLINE mode) gets a clean error, not a crash, and creates nothing.
    other_state_path = tmp_dir / "other_run_state.json"
    _write_state(other_state_path, [[1]], step=0, level=1)
    online_agent = ToolAgent(model="test-model")
    dispatch = online_agent._run_python_tool(other_state_path, {"code": "save_checkpoint('x')\n"})
    payload = json.loads(dispatch.content)
    if "not available" not in str(payload.get("error", "")):
        _fail("ONLINE mode graceful unavailability", str(payload))
    if online_agent._atlas_checkpoints:
        _fail("ONLINE mode creates no checkpoints", str(online_agent._atlas_checkpoints))
    _ok("no checkpoint_env/restore_env wired -> save_checkpoint fails gracefully, no auto-anchors created")

    # 7. Manual save_checkpoint + rollback actually reverts the real env AND
    #    the memo, and requires a non-empty lesson_learned.
    agent._atlas_memo = {"anchor": "first"}
    dispatch = agent._run_python_tool(state_path, {"code": "result = save_checkpoint('anchor set')\n"})
    payload = json.loads(dispatch.content)
    manual_cid = payload.get("result")
    if not manual_cid or manual_cid not in agent._atlas_checkpoints:
        _fail("manual save_checkpoint", str(payload))
    if agent._atlas_checkpoints[manual_cid]["memo"] != {"anchor": "first"}:
        _fail("manual checkpoint captured the memo at save time", str(agent._atlas_checkpoints[manual_cid]))

    agent._run_python_tool(state_path, {"code": "memo['anchor'] = 'second'\naction(['UP'])\n"})
    grid_before_rollback, _ = load_runtime_state(state_path)
    if agent._atlas_memo.get("anchor") != "second" or grid_before_rollback.grid == ((9,),):
        _fail("state actually advanced before rollback", str((agent._atlas_memo, grid_before_rollback)))

    dispatch = agent._run_python_tool(state_path, {"code": f"rollback({manual_cid!r}, '')\n"})
    payload = json.loads(dispatch.content)
    if "lesson_learned" not in str(payload.get("error", "")):
        _fail("rollback rejects empty lesson_learned", str(payload))

    dispatch = agent._run_python_tool(
        state_path, {"code": f"rollback({manual_cid!r}, 'the UP move was a dead end')\n"}
    )
    payload = json.loads(dispatch.content)
    if payload.get("error"):
        _fail("rollback with a real lesson succeeds", str(payload))
    restored_frame, _ = load_runtime_state(state_path)
    if restored_frame.grid == grid_before_rollback.grid or restored_frame.grid != ((9,),):
        _fail(
            "rollback actually reverts the real env state, not just bookkeeping",
            str((restored_frame.grid, grid_before_rollback.grid)),
        )
    if agent._atlas_memo != {"anchor": "first"}:
        _fail("rollback restores memo to save-time value", str(agent._atlas_memo))
    if agent._atlas_rollback_lesson != "the UP move was a dead end":
        _fail("rollback queues the lesson for next turn's prompt", agent._atlas_rollback_lesson)
    prompt = agent._build_user_prompt(0, valid_actions=["UP"])
    if "[rollback landed]" not in prompt or "the UP move was a dead end" not in prompt:
        _fail("rollback lesson injected into next prompt", prompt[-400:])
    if agent._atlas_rollback_lesson is not None:
        _fail("rollback lesson is one-shot", agent._atlas_rollback_lesson)
    _ok("manual save_checkpoint/rollback reverts real env state and memo, requires a real lesson_learned, and the lesson is injected once into the next prompt")

    # 8. Level-up creates a sys_level_N auto-anchor and resets stall tracking.
    env.next_level_completed = True
    agent._run_python_tool(state_path, {"code": "action(['UP'])\n"})
    if "sys_level_2" not in agent._atlas_checkpoints:
        _fail("sys_level_2 auto-anchor", str(agent._atlas_checkpoints))
    if agent._atlas_last_checkpoint_id != "sys_level_2" or agent._atlas_actions_since_level_progress != 0:
        _fail(
            "level-up bookkeeping",
            str((agent._atlas_last_checkpoint_id, agent._atlas_actions_since_level_progress)),
        )
    _ok("a level-up auto-creates sys_level_N and resets the stall counter")

    # 9. Trigger A (soft stall): _ATLAS_ROLLBACK_STALL_AFTER_CALLS real
    #    actions with zero level progress sets the rollback target.
    if agent._atlas_rollback_target_checkpoint is not None:
        _fail("precondition: no rollback pending before the stall test", agent._atlas_rollback_target_checkpoint)
    for i in range(_ATLAS_ROLLBACK_STALL_AFTER_CALLS):
        agent._run_python_tool(state_path, {"code": "action(['UP'])\n"})
    if agent._atlas_rollback_target_checkpoint != "sys_level_2":
        _fail(
            "Trigger A fires after the stall threshold",
            str((agent._atlas_actions_since_level_progress, agent._atlas_rollback_target_checkpoint)),
        )
    _ok(f"Trigger A (soft stall) fires after {_ATLAS_ROLLBACK_STALL_AFTER_CALLS} stalled actions, targeting the most recent anchor")

    # 10. Ultimatum escalation: the checkpoint fires for
    #     _ATLAS_ROLLBACK_AUTO_FORCE_AFTER turns, naming the actual target
    #     checkpoint, then the (N+1)th turn schedules the auto-force instead.
    for streak in range(1, _ATLAS_ROLLBACK_AUTO_FORCE_AFTER + 1):
        prompt = agent._build_user_prompt(0, valid_actions=["UP"])
        if "MUST call rollback('sys_level_2'" not in prompt:
            _fail(f"ultimatum shown at streak {streak}", prompt[-500:])
    if agent._atlas_pending_auto_rollback is not None:
        _fail("no auto-force scheduled yet", agent._atlas_pending_auto_rollback)
    prompt = agent._build_user_prompt(0, valid_actions=["UP"])
    if "will perform rollback('sys_level_2')" not in prompt:
        _fail("auto-force scheduled on the turn after the streak caps out", prompt[-500:])
    if agent._atlas_pending_auto_rollback != "sys_level_2":
        _fail("auto-force target recorded", agent._atlas_pending_auto_rollback)
    _ok(f"the ultimatum escalates for {_ATLAS_ROLLBACK_AUTO_FORCE_AFTER} turns, then the harness schedules an auto-force rollback")

    # 11. The scheduled auto-force actually performs the rollback at the top
    #     of the NEXT real python-tool call, injects a generic lesson, and
    #     clears all the pending/streak state.
    frame_before_auto, _ = load_runtime_state(state_path)
    agent._run_python_tool(state_path, {"code": "result = 1\n"})
    frame_after_auto, _ = load_runtime_state(state_path)
    if agent._atlas_pending_auto_rollback is not None or agent._atlas_rollback_target_checkpoint is not None:
        _fail("auto-force clears its own state", str((agent._atlas_pending_auto_rollback, agent._atlas_rollback_target_checkpoint)))
    if agent._atlas_rollback_ultimatum_streak != 0:
        _fail("streak resets after the auto-force fires", agent._atlas_rollback_ultimatum_streak)
    if not agent._atlas_rollback_lesson or "harness auto-rollback" not in agent._atlas_rollback_lesson:
        _fail("auto-force queues a generic lesson", agent._atlas_rollback_lesson)
    if frame_after_auto.step == frame_before_auto.step:
        _fail("auto-force actually changed the persisted state", str((frame_before_auto.step, frame_after_auto.step)))
    _ok("the harness auto-force backstop actually performs the rollback and queues a generic lesson, independent of the model")

    # 12. Trigger B (hard loop): board_signature repeats
    #     _ATLAS_ROLLBACK_LOOP_WINDOW actions ago -> fires even with plenty
    #     of level-progress headroom left.
    tmp_dir2 = Path(tempfile.mkdtemp(prefix="atlas_rollback_loop_test_"))
    loop_state_path = tmp_dir2 / "run_state.json"
    _write_state(loop_state_path, [[0]], step=0, level=1)
    loop_checkpoint_env, loop_restore_env = _make_env_callbacks(loop_state_path)

    class LoopEnv:
        def __init__(self, path):
            self.path = path
            self.counter = 0

        def __call__(self, payload):
            self.counter += 1
            _write_state(self.path, [[self.counter % 2]], step=self.counter, level=1)
            return {
                "executed": True,
                "action_num": self.counter,
                "level": 1,
                "score": 0,
                "reward": 0.0,
                "board_changed": True,
                "done": False,
                "level_completed": False,
                "game_over": False,
                "run_complete": False,
                "valid_actions": ["UP"],
            }

    loop_agent = ToolAgent(model="test-model")
    loop_agent._checkpoint_env_callback = loop_checkpoint_env
    loop_agent._restore_env_callback = loop_restore_env
    loop_agent._step_env_callback = LoopEnv(loop_state_path)
    loop_agent._current_valid_actions = ["UP"]

    for i in range(_ATLAS_ROLLBACK_LOOP_WINDOW + 1):
        loop_agent._run_python_tool(loop_state_path, {"code": "action(['UP'])\n"})
        if i < _ATLAS_ROLLBACK_LOOP_WINDOW and loop_agent._atlas_rollback_target_checkpoint is not None:
            _fail("Trigger B does not fire early", str(i))
    if loop_agent._atlas_rollback_target_checkpoint != "sys_start":
        _fail(
            f"Trigger B fires once the board repeats a state seen {_ATLAS_ROLLBACK_LOOP_WINDOW} actions ago",
            str(loop_agent._atlas_rollback_target_checkpoint),
        )
    _ok(f"Trigger B (hard loop) fires once the board repeats a state seen {_ATLAS_ROLLBACK_LOOP_WINDOW} actions ago")

    # ---- extract-suggestion checkpoint (idea #2) --------------------------

    extract_agent = ToolAgent(model="test-model")
    extract_state_path = tmp_dir / "extract_run_state.json"
    _write_state(extract_state_path, [[0]], step=0, level=1)

    def _wrong_predict_code():
        return "def predict(grid, action):\n    return grid\nresult = verify_theory(predict)\n"

    for _ in range(_ATLAS_EXTRACT_NUDGE_AFTER_CALLS):
        extract_agent._run_python_tool(extract_state_path, {"code": _wrong_predict_code()})
    prompt = extract_agent._build_user_prompt(0, valid_actions=["UP"])
    if "never once passed `extract=`" not in prompt:
        _fail(f"extract-suggestion checkpoint fires after {_ATLAS_EXTRACT_NUDGE_AFTER_CALLS} verify_theory calls", prompt[-500:])
    _ok(f"extract-suggestion checkpoint fires after {_ATLAS_EXTRACT_NUDGE_AFTER_CALLS} verify_theory( calls with no extract= used")

    extract_agent._run_python_tool(
        extract_state_path,
        {
            "code": (
                "def predict(state, action):\n    return state\n"
                "def extract(grid):\n    return {'x': 0}\n"
                "result = verify_theory(predict, extract=extract)\n"
            )
        },
    )
    prompt = extract_agent._build_user_prompt(0, valid_actions=["UP"])
    if "never once passed `extract=`" in prompt:
        _fail("extract-suggestion checkpoint clears once extract= is used", prompt[-500:])
    _ok("extract-suggestion checkpoint goes quiet immediately after extract= is actually used")

    print("\nAll atlas rollback/extract checkpoint checks passed.")


if __name__ == "__main__":
    main()

"""Exercise the atlas `memo` port -- persistent scratch memory across turns.

Backlog: "in-episode learning" needs SOME way for the model to accumulate
structured (not just prose) experience across turns. Our own harness has
had this forever (Sandbox.memo); Duck's sandbox is a fresh subprocess per
turn with nothing analogous. This threads a host-side dict (on ToolAgent,
mirroring _summarized_knowledge's persistence) into each turn's subprocess
and reads it back out -- carefully NOT wiping it when _refresh_state fires
mid-turn after an action() call, and NOT wiping it on a crash or timeout
that never sent a clean reply.

Two levels: the raw run_sandboxed_python round-trip (mid-turn survival
across an action() call, crash survival, non-JSON-safe graceful decay), and
the real ToolAgent._run_python_tool path (cross-turn persistence through
the actual host object, and reset on a new session).
"""

from __future__ import annotations

import importlib.util
import sys
import threading
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

from inference.agent.tool_agent import ToolAgent  # noqa: E402


def _fail(name: str, detail: str) -> None:
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def _ok(name: str) -> None:
    print(f"ok   {name}")


def _frame_payload(step: int = 0):
    return {"ascii": "", "step": step, "level": 1, "shape": [1, 1], "grid": [[0]]}


def _base_state(memo=None) -> dict:
    state = {
        "current_frame": _frame_payload(),
        "history": [],
        "valid_actions": ["UP"],
        "last_action_result": {},
    }
    if memo is not None:
        state["memo"] = memo
    return state


def main() -> None:
    # 1. No memo in initial state -> sandbox starts with an empty dict.
    result = run_sandboxed_python(
        code="result = memo\n",
        timeout_seconds=5,
        initial_state=_base_state(),
        action_handler=lambda actions: _fail("action", "unexpected action() call"),
    )
    if result.get("error") or result.get("result") != {}:
        _fail("empty memo by default", str(result))
    _ok("memo starts as an empty dict when none is supplied")

    # 2. A write is returned in the final payload's memo.
    result = run_sandboxed_python(
        code="memo['x'] = 1\n",
        timeout_seconds=5,
        initial_state=_base_state(),
        action_handler=lambda actions: _fail("action", "unexpected action() call"),
    )
    if result.get("memo") != {"x": 1}:
        _fail("write round-trips", str(result))
    _ok("a write to memo comes back in the sandbox result")

    # 3. Feeding a previous memo back in makes it available (simulates the
    #    host passing turn N's memo into turn N+1's fresh subprocess).
    result = run_sandboxed_python(
        code="memo['x'] += 1\nresult = memo['x']\n",
        timeout_seconds=5,
        initial_state=_base_state(memo={"x": 1}),
        action_handler=lambda actions: _fail("action", "unexpected action() call"),
    )
    if result.get("result") != 2:
        _fail("fed-back memo is usable", str(result))
    _ok("a memo fed into initial_state is readable and mutable")

    # 4. memo survives an action() call mid-turn -- _refresh_state fires
    #    after action() and must NOT touch memo, unlike current_frame/history.
    def _action_handler(actions):
        return {"action_result": {"executed": True}, "state": _base_state()}  # no "memo" key at all

    result = run_sandboxed_python(
        code=(
            "memo['before'] = True\n"
            "action(['UP'])\n"
            "memo['after'] = True\n"
        ),
        timeout_seconds=5,
        initial_state=_base_state(),
        action_handler=_action_handler,
    )
    if result.get("memo") != {"before": True, "after": True}:
        _fail("survives mid-turn action()", str(result))
    _ok("memo survives an action() call in the middle of the same turn")

    # 5. A crash after writing to memo still reports what was written.
    result = run_sandboxed_python(
        code="memo['before_crash'] = True\nraise RuntimeError('boom')\n",
        timeout_seconds=5,
        initial_state=_base_state(),
        action_handler=lambda actions: _fail("action", "unexpected action() call"),
    )
    if not result.get("error") or result.get("memo") != {"before_crash": True}:
        _fail("survives a crash", str(result))
    _ok("memo written before a crash is still reported back")

    # 6. Non-JSON-safe values decay to a string instead of breaking anything.
    #    (The sandbox disallows `class` -- no __build_class__ in its safe
    #    builtins -- so a function object is the simplest non-serializable
    #    value reachable in this restricted environment.)
    result = run_sandboxed_python(
        code="def f():\n    return 1\nmemo['bad'] = f\n",
        timeout_seconds=5,
        initial_state=_base_state(),
        action_handler=lambda actions: _fail("action", "unexpected action() call"),
    )
    if result.get("error") or not isinstance(result.get("memo", {}).get("bad"), str):
        _fail("non-JSON-safe value decays gracefully", str(result))
    _ok(f"a non-JSON-safe value silently stringifies instead of crashing: {result['memo']['bad']!r}")

    # 7. Real cross-turn persistence through ToolAgent itself.
    agent = ToolAgent(model="test-model")
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="atlas_memo_test_"))
    state_path = tmp_dir / "run_state.json"
    import json as _json
    state_path.write_text(_json.dumps({"current_frame": {"grid": [[0]], "step": 0, "level": 1}, "history": []}), encoding="utf-8")

    agent._run_python_tool(state_path, {"code": "memo['count'] = 1\n"})
    if agent._atlas_memo.get("count") != 1:
        _fail("cross-turn persistence, turn 1", str(agent._atlas_memo))
    agent._run_python_tool(state_path, {"code": "memo['count'] = memo.get('count', 0) + 1\n"})
    if agent._atlas_memo.get("count") != 2:
        _fail("cross-turn persistence, turn 2", str(agent._atlas_memo))
    _ok(f"memo accumulates across two real ToolAgent turns: {agent._atlas_memo}")

    # 8. A new session (different runtime dir) resets memo.
    other_state_path = tmp_dir.parent / f"{tmp_dir.name}_other" / "run_state.json"
    other_state_path.parent.mkdir(parents=True, exist_ok=True)
    other_state_path.write_text(state_path.read_text(encoding="utf-8"), encoding="utf-8")
    agent._run_python_tool(other_state_path, {"code": "result = memo\n"})
    if agent._atlas_memo != {}:
        _fail("new session resets memo", str(agent._atlas_memo))
    _ok("a new game session resets memo to empty")

    print("\nAll atlas memo checks passed.")


if __name__ == "__main__":
    main()

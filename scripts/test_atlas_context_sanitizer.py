"""Exercise the atlas context sanitizer (Gemini idea #3 -- "episodic context
cleanup").

Background: games run up to ~4h. The raw action/observation transcript
(`ToolAgent._history_messages`) accumulates false theories, analyzer-timeout
artifacts, and dead-end attempts that token-budget trimming only prunes
reactively (oldest-block-first, once the context gets too big), never
deliberately. Gemini's proposal: at a level-up or after N steps, fire a
SEPARATE fast LLM call that synthesizes memo + the running world/goal/action
model + current board state into a compact "state of the world", then
completely replace the raw history with that synthesis.

This is host-triggered, not model-voluntary -- same C0 lesson as
memo/save_checkpoint: a tool merely offered to the model gets used in
~0.2%-0% of turns.

Two levels: the trigger-detection bookkeeping (_atlas_note_context_sanitize_
progress, called from _handle_action after every real action, independent of
the rollback/checkpoint feature -- this one works in ONLINE mode too), and
the actual sanitize execution (_atlas_run_context_sanitizer, which makes a
real _chat_completion call -- stubbed here to avoid a network dependency,
same reasoning test_atlas_time_bank.py stubs its own external dependencies).
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
    _ATLAS_CONTEXT_SANITIZE_EVERY_CALLS,
    _ChatCompletionResult,
)
from inference.agent.runtime_state import Frame, write_runtime_state, normalize_grid  # noqa: E402


def _fail(name: str, detail: str) -> None:
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def _ok(name: str) -> None:
    print(f"ok   {name}")


def _write_state(path: Path, grid, step: int, level: int = 1) -> None:
    write_runtime_state(path, current_frame=Frame(grid=normalize_grid(grid), step=step, level=level), history=[])


class FakeEnv:
    """A step_env_callback that persists real state to disk, like the real
    harness does, and lets the test drive level completion explicitly.
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


def _stub_chat_completion(content: str):
    def _stub(messages, *, tools=None, request_timeout_seconds=None):
        return _ChatCompletionResult(message={"content": content})

    return _stub


def _raising_chat_completion(messages, *, tools=None, request_timeout_seconds=None):
    raise RuntimeError("simulated network failure")


def main() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="atlas_sanitizer_test_"))
    state_path = tmp_dir / "run_state.json"
    _write_state(state_path, [[0]], step=0, level=1)

    # 1. Step-count trigger: no checkpoint_env/restore_env wired at all
    #    (ONLINE mode) -- the sanitizer must still work, unlike rollback.
    agent = ToolAgent(model="test-model")
    env = FakeEnv(state_path)
    agent._step_env_callback = env
    agent._current_valid_actions = ["UP"]
    # Establishes the session first (_ensure_session resets memo/knowledge on
    # the first call for a new runtime dir) -- a code-only call with no
    # action() never touches the sanitize-progress counter, so it's a safe
    # no-op to call before setting up the state this test actually checks.
    agent._run_python_tool(state_path, {"code": "result = 1\n"})
    agent._summarized_knowledge["world_model"] = "the player is a red square"
    agent._atlas_memo = {"anchor": {"row": 3, "col": 4}}

    if agent._atlas_context_sanitize_pending:
        _fail("precondition", "sanitize should not be pending before any actions")
    for _ in range(_ATLAS_CONTEXT_SANITIZE_EVERY_CALLS - 1):
        agent._run_python_tool(state_path, {"code": "action(['UP'])\n"})
        if agent._atlas_context_sanitize_pending:
            _fail("no early trigger", f"fired before reaching {_ATLAS_CONTEXT_SANITIZE_EVERY_CALLS} calls")
    agent._run_python_tool(state_path, {"code": "action(['UP'])\n"})
    if not agent._atlas_context_sanitize_pending or agent._atlas_context_sanitize_reason != "step_count":
        _fail(
            "step-count trigger fires",
            str((agent._atlas_context_sanitize_pending, agent._atlas_context_sanitize_reason)),
        )
    captured = agent._atlas_context_sanitize_input or {}
    if captured.get("memo") != {"anchor": {"row": 3, "col": 4}}:
        _fail("trigger captures memo", str(captured))
    if not any("world_model" in line.lower() or "red square" in line for line in captured.get("knowledge_lines", [])):
        _fail("trigger captures world-model knowledge", str(captured.get("knowledge_lines")))
    _ok(f"step-count trigger fires after {_ATLAS_CONTEXT_SANITIZE_EVERY_CALLS} real actions, no checkpoint feature required (ONLINE-safe)")

    # 2. Level-up trigger fires immediately, independent of the step count,
    #    and -- the actual bug this design avoids -- captures the world
    #    model BEFORE the end-of-turn wipe that a real level transition
    #    causes (_update_summarized_knowledge_from_step_summary).
    level_agent = ToolAgent(model="test-model")
    level_state_path = tmp_dir / "level_run_state.json"
    _write_state(level_state_path, [[0]], step=0, level=1)
    level_env = FakeEnv(level_state_path)
    level_agent._step_env_callback = level_env
    level_agent._current_valid_actions = ["UP"]
    level_agent._run_python_tool(level_state_path, {"code": "action(['UP'])\n"})
    if level_agent._atlas_context_sanitize_pending:
        _fail("no premature level-up trigger", "should not fire on a normal action")
    level_agent._summarized_knowledge["world_model"] = "confirmed: UP moves the player up one cell"
    level_env.next_level_completed = True
    level_agent._run_python_tool(level_state_path, {"code": "action(['UP'])\n"})
    if not level_agent._atlas_context_sanitize_pending or level_agent._atlas_context_sanitize_reason != "level_up":
        _fail(
            "level-up trigger fires regardless of step count",
            str((level_agent._atlas_context_sanitize_pending, level_agent._atlas_context_sanitize_reason)),
        )
    level_captured = level_agent._atlas_context_sanitize_input or {}
    if not any("UP moves the player" in line for line in level_captured.get("knowledge_lines", [])):
        _fail("level-up trigger captures pre-transition knowledge", str(level_captured.get("knowledge_lines")))
    # Simulate the real end-of-turn wipe a level transition causes (see
    # _update_summarized_knowledge_from_step_summary) and confirm the ALREADY
    # captured input snapshot is unaffected -- this is the timing bug this
    # design specifically avoids (capturing at trigger time, not run time).
    level_agent._last_step_summary = {"level_transition": True}
    level_agent._update_summarized_knowledge_from_step_summary()
    if level_agent._summarized_knowledge.get("world_model"):
        _fail("wipe simulation sanity check", "expected the simulated wipe to actually clear world_model")
    if not any("UP moves the player" in line for line in (level_agent._atlas_context_sanitize_input or {}).get("knowledge_lines", [])):
        _fail(
            "frozen snapshot survives the post-transition wipe",
            str(level_agent._atlas_context_sanitize_input),
        )
    _ok("level-up trigger fires immediately and freezes world-model knowledge BEFORE the level-transition wipe clears it")

    # 3. Actual sanitize execution: replaces raw history with a synthesis
    #    from a (stubbed) separate LLM call.
    sanitize_agent = ToolAgent(model="test-model")
    sanitize_agent._history_messages = [
        {"role": "user", "content": "turn 1 raw prompt"},
        {"role": "assistant", "content": "turn 1 raw reasoning", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "turn 1 raw tool output"},
    ]
    sanitize_agent._atlas_memo = {"x": 1}
    sanitize_agent._summarized_knowledge["goal_model"] = "reach the green tile"
    sanitize_agent._atlas_context_sanitize_pending = True
    sanitize_agent._atlas_context_sanitize_reason = "step_count"
    sanitize_agent._atlas_context_sanitize_input = {
        "memo": {"x": 1},
        "knowledge_lines": ["- Goal model: reach the green tile"],
        "level": 1,
        "step": 20,
        "ascii": "..\n..",
    }
    sanitize_agent._chat_completion = _stub_chat_completion("SYNTHESIZED STATE OF THE WORLD")
    analyzer_log = tmp_dir / "sanitize_test_analyzer.txt"
    sanitize_agent._atlas_run_context_sanitizer(analyzer_log=analyzer_log)

    if sanitize_agent._atlas_context_sanitize_pending or sanitize_agent._atlas_context_sanitize_input is not None:
        _fail("sanitize clears its own pending state", str(sanitize_agent._atlas_context_sanitize_pending))
    if sanitize_agent._atlas_calls_since_sanitize != 0:
        _fail("sanitize resets the step counter", sanitize_agent._atlas_calls_since_sanitize)
    if len(sanitize_agent._history_messages) != 2:
        _fail("raw history is fully replaced", str(sanitize_agent._history_messages))
    if sanitize_agent._history_messages[0]["role"] != "user" or "SYNTHESIZED STATE OF THE WORLD" not in sanitize_agent._history_messages[0]["content"]:
        _fail("synthesis lands in the replacement history", str(sanitize_agent._history_messages))
    if sanitize_agent._history_messages[1]["role"] != "assistant":
        _fail("replacement history is a well-formed user/assistant pair", str(sanitize_agent._history_messages))
    if sanitize_agent._atlas_context_snapshot != "SYNTHESIZED STATE OF THE WORLD":
        _fail("snapshot stored for diagnostics", sanitize_agent._atlas_context_snapshot)
    if sanitize_agent._atlas_context_sanitize_count != 1:
        _fail("sanitize count incremented", sanitize_agent._atlas_context_sanitize_count)
    log_text = analyzer_log.read_text(encoding="utf-8")
    if "CONTEXT SANITIZER" not in log_text or "SYNTHESIZED STATE OF THE WORLD" not in log_text:
        _fail("sanitize logs a transcript section", log_text)
    _ok("sanitize replaces raw history with a synthesized user/assistant pair and logs it")

    # 4. Graceful failure: a network error leaves the existing history
    #    completely untouched (fail-open, same pattern as time-bank/
    #    retry-storm backstops elsewhere in this file).
    fail_agent = ToolAgent(model="test-model")
    original_history = [{"role": "user", "content": "keep me"}, {"role": "assistant", "content": "ack"}]
    fail_agent._history_messages = list(original_history)
    fail_agent._atlas_context_sanitize_pending = True
    fail_agent._atlas_context_sanitize_reason = "step_count"
    fail_agent._atlas_context_sanitize_input = {"memo": {}, "knowledge_lines": [], "level": 1, "step": 1, "ascii": ""}
    fail_agent._chat_completion = _raising_chat_completion
    fail_agent._atlas_run_context_sanitizer(analyzer_log=tmp_dir / "fail_test_analyzer.txt")
    if fail_agent._history_messages != original_history:
        _fail("a failed sanitize leaves history untouched", str(fail_agent._history_messages))
    if fail_agent._atlas_context_sanitize_pending:
        _fail("a failed sanitize still clears the pending flag (no infinite retry loop)", "still pending")
    _ok("a failed sanitizer request leaves the existing history untouched (fail-open)")

    # 5. Empty response content is treated the same as a failure.
    empty_agent = ToolAgent(model="test-model")
    empty_agent._history_messages = list(original_history)
    empty_agent._atlas_context_sanitize_pending = True
    empty_agent._atlas_context_sanitize_input = {"memo": {}, "knowledge_lines": [], "level": 1, "step": 1, "ascii": ""}
    empty_agent._chat_completion = _stub_chat_completion("")
    empty_agent._atlas_run_context_sanitizer(analyzer_log=tmp_dir / "empty_test_analyzer.txt")
    if empty_agent._history_messages != original_history:
        _fail("an empty synthesis leaves history untouched", str(empty_agent._history_messages))
    _ok("an empty synthesis response leaves the existing history untouched")

    # 6. No-op on empty history -- must not even call the (stubbed) LLM.
    def _must_not_be_called(messages, *, tools=None, request_timeout_seconds=None):
        _fail("no-op on empty history", "the LLM must not be called when there is no history to sanitize")

    noop_agent = ToolAgent(model="test-model")
    noop_agent._history_messages = []
    noop_agent._atlas_context_sanitize_pending = True
    noop_agent._atlas_context_sanitize_input = {"memo": {}, "knowledge_lines": [], "level": 1, "step": 1, "ascii": ""}
    noop_agent._chat_completion = _must_not_be_called
    noop_agent._atlas_run_context_sanitizer(analyzer_log=tmp_dir / "noop_test_analyzer.txt")
    if noop_agent._atlas_context_sanitize_pending:
        _fail("empty-history no-op still clears pending", "still pending")
    _ok("no LLM call is made when there is no history to sanitize yet")

    print("\nAll atlas context-sanitizer checks passed.")


if __name__ == "__main__":
    main()

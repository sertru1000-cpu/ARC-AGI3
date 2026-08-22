"""Unit test for A2 (no-code retry) and A3 (repeated-code guard), 22.08.

Both target the same scarce resource: turns. In Phase B the base gets roughly
40 turns per hidden game, and we measured two ways of throwing them away.

A2 -- a reply with no python block cost a whole turn AND a strike. 10% of the
27B's turns and 31% of the 35B's produced nothing; three of the 35B's twelve
games died on five consecutive no-code strikes. Now the policy asks once more
inside the same turn before spending either.

A3 -- byte-identical code on consecutive turns means the context is a fixed
point: the board barely moved, so the same prompt yields the same argmax reply
(student v6 emitted one block 38x in a row on ft09, and 118x on lp85 at
temperature 0). The guard refuses such a repeat and says so, which changes the
context. It must NOT fire when the repeat did something: stepping RIGHT turn
after turn is legitimate play.

Run:  .venv/Scripts/python.exe scripts/test_no_code_retry_and_repeat_guard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

from agent.harness.llm_policy import LLMPolicy
from agent.harness.sandbox import Sandbox


class ScriptedLLM:
    """Returns the queued replies in order; records how many calls happened."""

    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, max_tokens=2048, temperature=0.6) -> str:
        self.calls += 1
        return self.replies.pop(0) if self.replies else "(exhausted)"


def make_policy(replies: list[str], actions_per_run: int = 0) -> LLMPolicy:
    """Policy over a sandbox whose run_code always reports actions_per_run."""
    sb = Sandbox(env_step=lambda *a: None, budget_left=lambda: 100)

    class _Res:
        output = "Level: 0, nodes: 15"
        error = None
        actions_executed = actions_per_run
        interrupted = None
        win = False

    sb.run_code = lambda code: _Res()  # type: ignore[method-assign]
    sb.valid_actions = ["UP", "DOWN"]
    p = LLMPolicy(backend=ScriptedLLM(replies), sandbox=sb, game_id="synth", win_levels=1)
    p.messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    return p


def check(cond: bool, msg: str) -> None:
    print(f"[{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


CODE = "```python\nprint(current_frame.level)\n```"


def main() -> None:
    # --- A2: a code-less reply is retried once, inside the same turn ---------
    p = make_policy(["WORLD_MODEL:\ncontrols: none\n(no code here)", CODE])
    res = p.play_turn()
    check(p.backend.calls == 2, "A2: code-less reply triggers a second LLM call")
    check(p.no_code_retries == 1, "A2: the retry is counted")
    check(p.no_code_strikes == 0, "A2: a successful retry costs NO strike")
    check(res["error"] is None, "A2: the turn ends normally after the retry")

    # --- A2: two failures in a row still cost exactly one strike ------------
    p = make_policy(["no code at all", "still no code"])
    res = p.play_turn()
    check(p.backend.calls == 2, "A2: only ONE retry, not an unbounded loop")
    check(p.no_code_strikes == 1, "A2: a failed retry costs one strike, not two")
    check(res["error"] == "no_code", "A2: the turn is reported as no_code")

    # --- A2: an EMPTY reply gets its own corrective, not the generic one ----
    p = make_policy(["", CODE])
    p.play_turn()
    nudges = [m["content"] for m in p.messages if m["role"] == "user"]
    check(any("EMPTY" in n for n in nudges), "A2: empty reply gets the empty-reply nudge")

    # --- A2: the retry can be switched off ----------------------------------
    p = make_policy(["no code", CODE])
    p.retry_no_code = False
    p.play_turn()
    check(p.backend.calls == 1, "A2: MY_AGENT_NO_CODE_RETRY=0 restores the old path")

    # --- A3: identical code after a fruitless run is refused ----------------
    p = make_policy([CODE, CODE], actions_per_run=0)
    p.play_turn()
    ran_first = p.last_code is not None
    res = p.play_turn()
    check(ran_first, "A3: the first occurrence runs normally")
    check(p.repeats_blocked == 1, "A3: the identical repeat is blocked")
    check(res["actions"] == 0, "A3: a blocked repeat executes nothing")
    last_user = [m["content"] for m in p.messages if m["role"] == "user"][-1]
    check("byte-identical" in last_user, "A3: the model is told why it was refused")
    check("Level: 0, nodes: 15" in last_user, "A3: the earlier output is quoted back")

    # --- A3: a repeat that DID act is legitimate and must run ---------------
    p = make_policy([CODE, CODE], actions_per_run=3)
    p.play_turn()
    res = p.play_turn()
    check(p.repeats_blocked == 0, "A3: a productive repeat is NOT blocked")
    check(res["actions"] == 3, "A3: it executes normally")

    print("\nA2 and A3 behave as specified.")


if __name__ == "__main__":
    main()

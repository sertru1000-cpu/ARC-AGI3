"""Unit test for LLMPolicy._trim's token-budget trimming (backlog item 6, 20.08).

Old behavior: kept a fixed pair COUNT regardless of size -- a few verbose
turns could blow the context even with few pairs kept. New behavior: keep
as many recent pairs as fit a token budget, always keeping at least the
most recent pair.

Run:  .venv/Scripts/python.exe scripts/test_context_trim.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

from agent.harness.llm_policy import LLMPolicy
from agent.harness.sandbox import Sandbox


def make_policy(max_context_tokens: int) -> LLMPolicy:
    sb = Sandbox(env_step=lambda *a: None, budget_left=lambda: 100)
    p = LLMPolicy(backend=None, sandbox=sb, game_id="synth", win_levels=1,
                  max_context_tokens=max_context_tokens)
    p.messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "FIRST USER MESSAGE"},
    ]
    return p


def add_pair(p: LLMPolicy, assistant_text: str, user_text: str) -> None:
    p.messages.append({"role": "assistant", "content": assistant_text})
    p.messages.append({"role": "user", "content": user_text})
    p._trim()


def main() -> None:
    # 1) Many SMALL pairs: old code capped by a fixed count (8); new code
    #    should keep well more than 8 since they're all tiny relative to
    #    a generous token budget.
    p = make_policy(max_context_tokens=2000)
    for i in range(20):
        add_pair(p, f"a{i}", f"u{i}")
    kept_pairs = (len(p.messages) - 2) // 2
    print(f"small pairs: kept {kept_pairs}/20 pairs under budget 2000")
    assert kept_pairs > 8, f"expected more than the old fixed cap of 8, got {kept_pairs}"

    # 2) A handful of HUGE pairs: old code would still keep up to 8 of them
    #    (blowing the context); new code must cap by token budget instead.
    p2 = make_policy(max_context_tokens=2000)
    huge = "X" * 6000  # ~3000 approx-tokens per message alone
    for i in range(8):
        add_pair(p2, huge, huge)
    kept_pairs2 = (len(p2.messages) - 2) // 2
    print(f"huge pairs: kept {kept_pairs2}/8 pairs under budget 2000 (each pair ~6000 approx-tokens)")
    assert kept_pairs2 < 8, f"expected token budget to cap below the old fixed pair count, got {kept_pairs2}"
    assert kept_pairs2 >= 1, "must always keep at least the most recent pair"

    # 3) A single pair bigger than the whole budget must still be kept
    #    (never strand the model with zero feedback on its last move).
    p3 = make_policy(max_context_tokens=10)
    add_pair(p3, "Y" * 1000, "Y" * 1000)
    assert len(p3.messages) == 4, "single oversized pair must survive trimming"
    print("oversized single pair: kept anyway (no starvation)")

    # 4) Context note appears exactly when something was actually dropped.
    p4 = make_policy(max_context_tokens=50)
    for i in range(5):
        add_pair(p4, "z" * 100, "z" * 100)
    notes = [m for m in p4.messages if "[context note]" in m.get("content", "")]
    assert notes, "expected a [context note] once trimming actually drops something"
    print("context note present:", notes[0]["content"][:100])

    print("\nALL CONTEXT-TRIM CHECKS PASSED")


if __name__ == "__main__":
    main()

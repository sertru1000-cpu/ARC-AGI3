"""Guard: the context sanitizer recovers from a reasoning-only reply.

01.09. The sanitizer fires on every level-up and every N stalled actions, and
it was returning EMPTY 19 times out of 23 on calib_1 (3 of 3 on the V46 arm).
Cause: Qwen3 answers a synthesis request through the reasoning channel and
sends `content: ""`. The transcript renderer already handled that channel via
`_extract_reasoning_text`; the sanitizer did not, so the mechanism had been
silently dead for about a week while the log said so once per fire.

The naive fix -- "just use the reasoning text" -- is WRONG in a way that is
easy to miss: this mechanism exists to SHRINK the context. Pasting several
thousand tokens of raw deliberation back into the history does the opposite of
what it is for, and would make the very problem it was built to fix worse.

So the salvage is capped, takes the TAIL (a reasoning model puts its
conclusion last), and refuses anything too short to be a synthesis. These
tests pin all three properties plus the unchanged no-op path.

This runs against the pure helper -- no GPU, no Kaggle, no quota.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "atlas_src/src/ARC3-Inference/inference/agent/tool_agent.py"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}: {detail}")
        FAILURES.append(name)


def _load_helper():
    """Import just the helper, without dragging in the agent's dependencies."""
    src = AGENT.read_text(encoding="utf-8")
    start = src.index("_ATLAS_SALVAGE_MAX_CHARS = ")
    end = src.index("def _is_context_length_error")
    ns: dict = {}
    exec(compile(src[start:end], "<salvage helper>", "exec"), ns)
    return ns


def main() -> None:
    ns = _load_helper()
    salvage = ns["_atlas_salvage_synthesis"]
    cap = ns["_ATLAS_SALVAGE_MAX_CHARS"]
    floor = ns["_ATLAS_SALVAGE_MIN_CHARS"]

    # --- 1. the no-op path is unchanged -----------------------------------
    check("empty reasoning still yields nothing", salvage("") == "")
    check("whitespace-only reasoning yields nothing", salvage("   \n\t ") == "")
    check(f"a fragment under {floor} chars is refused",
          salvage("Player is the blue block.") == "")

    # --- 2. a normal synthesis passes through untouched --------------------
    good = ("Confirmed: the blue 3x3 is the player; grey blocks are walls.\n"
            "Goal: reach the yellow circle. ACTION6 selects, then ACTION1-4 move.\n"
            "Tried and failed: clicking empty water, pushing walls from the left.")
    check("a short synthesis survives verbatim", salvage(good) == good.strip())

    # --- 3. a long deliberation is capped and takes the TAIL --------------
    # This is the real case: thousands of tokens of thinking whose conclusion
    # sits at the end.
    tail = ("CONCLUSION: the mechanic is a rotating dial, not a camera. "
            "The counter at the top decrements per action. Goal is to align "
            "all four arms before it reaches zero.")
    long_reasoning = ("Let me think about this step by step. " * 400) + "\n" + tail
    out = salvage(long_reasoning)
    check(f"a long deliberation is capped at {cap} chars", len(out) <= cap,
          f"got {len(out)}")
    check("the salvage keeps the CONCLUSION at the tail", tail in out,
          f"tail missing from: ...{out[-160:]!r}")
    check("the salvage does not start mid-sentence",
          not out.startswith("hink") and not out.startswith("tep"),
          f"starts with {out[:40]!r}")

    # --- 4. the cap is what actually protects the context ------------------
    # 8000 output tokens is roughly 32000 characters; the whole point is that
    # none of that reaches the history.
    huge = "x" * 40000
    check("even a 40k-char deliberation cannot grow the context",
          len(salvage(huge)) <= cap, f"got {len(salvage(huge))}")

    # --- 5. the call site actually uses it --------------------------------
    src = AGENT.read_text(encoding="utf-8")
    site = src[src.index("def _atlas_run_context_sanitizer"):]
    site = site[:site.index("\n    def ", 10)]
    check("the sanitizer calls the salvage before giving up",
          "_atlas_salvage_synthesis(_extract_reasoning_text(" in site)
    check("the give-up branch is still reachable",
          site.count("keeping existing history untouched") >= 1)

    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) failed.")
        sys.exit(1)
    print("\nAll sanitizer-salvage checks passed.")


if __name__ == "__main__":
    main()

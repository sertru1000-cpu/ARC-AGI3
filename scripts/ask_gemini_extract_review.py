"""One-off: ask gemini-3.7-flash to critique verify_theory/plan_with_theory/execute_plan.

24.08: rather than us guessing at Gemini's perspective, ask the actual model
that produced our two confirmed old-harness full wins (lp85, ft09, both
gemini-3.7-flash, round4A3, 21.08 -- data/teacher/gemini-3.7-flash_round4A3_vision_20260821_071509/)
to critique OUR tool design, with real excerpts of its own past reasoning as
context. Reuses agent/harness/llm.py's OpenAICompatLLM (already the tested
AI-Studio client for this model) rather than a fresh HTTP client.

Prints the critique to stdout and saves it to docs/gemini_critique_extract_theory_2026-08-24.md.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ENV_PATH = ROOT / ".env"
for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
    if line.startswith("LLM_API_KEY_AISTUDIO="):
        os.environ["LLM_API_KEY"] = line.split("=", 1)[1].strip()
        break
else:
    raise SystemExit("LLM_API_KEY_AISTUDIO not found in .env")

os.environ["LLM_BASE_URL"] = "https://generativelanguage.googleapis.com/v1beta/openai"
os.environ["LLM_MODEL"] = "gemini-3.7-flash"
os.environ["LLM_AUTH"] = "key"
os.environ["LLM_TIMEOUT_S"] = "300"
os.environ["LLM_RETRIES"] = "4"

from agent.harness.llm import OpenAICompatLLM  # noqa: E402

SANDBOX_PATH = ROOT / "atlas_src" / "src" / "ARC3-Inference" / "inference" / "agent" / "python_tool_sandbox.py"
sandbox_lines = SANDBOX_PATH.read_text(encoding="utf-8").splitlines()
TOOLS_SOURCE = "\n".join(sandbox_lines[411:661])  # verify_theory .. execute_plan, 0-indexed


def _turn(path: Path, turn_index: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[turn_index]).get("reply", "")


LP85_DIR = ROOT / "data" / "teacher" / "gemini-3.7-flash_round4A3_vision_20260821_071509"
LP85_EXCERPT = "\n\n".join(_turn(LP85_DIR / "lp85-r0.jsonl", i) for i in (80, 83))
FT09_EXCERPT = _turn(LP85_DIR / "ft09-r0.jsonl", 55)

PROMPT = f"""You are gemini-3.7-flash. On 21.08.2026 you played ARC-AGI-3 games through an
older Python-tool harness we built (raw numpy grid + segmentation info, `action(...)`
to act, arbitrary Python code per turn -- no dedicated theory/planning helper functions
existed yet). Two of your runs are confirmed full wins: lp85 (8/8 levels, 546 actions,
85 turns) and ft09 (6/6 levels, 618 actions, 76 turns). Below are real excerpts of your
OWN code and reasoning from those two wins.

=== Your own code, lp85, turns 80-83 (final level: probing 8 buttons, extracting a
tile-position dict by hand via a `get_all_tiles(grid)` scan, then BFS over that
abstracted state space to find the winning move sequence) ===
{LP85_EXCERPT}

=== Your own code, ft09, turn 55 (reducing the board to a coarse tile grid and comparing
against a target pattern read from four "pinwheel" reference icons) ===
{FT09_EXCERPT}

In BOTH wins, the pattern is the same: you abstracted the raw pixel board down to a
small discrete state (tile/button positions, a coarse grid) BEFORE reasoning or
searching, and you did this by hand with ad hoc code, spending real game actions to
probe each mechanic empirically (no way to test a theory for free).

We have since forked a different, stronger open-source harness for the actual
competition (Kaggle ARC Prize 2026, ARC-AGI-3 track) that gives the model three
Python-callable tools: `verify_theory`, `plan_with_theory`, `execute_plan`. Their
current implementation (just extended today, 24.08) is pasted verbatim below. The key
design point: `verify_theory(predict)` replays every ALREADY-OBSERVED real transition
in history against a model-written `predict(grid, action) -> next_grid` function, for
FREE (zero real actions) -- but by default it requires predict() to reproduce the
entire next grid EXACTLY, cell for cell (pixel-perfect), not a percentage. Only once a
predict() clears 0.6 accuracy does `plan_with_theory` allow a BFS-style search over
predicted future grids, and `execute_plan` lets a found plan be run one real step at a
time with automatic abort on divergence.

```python
{TOOLS_SOURCE}
```

We found (via real Kaggle submission transcripts, 24.08) that this pixel-perfect
default caused a severe failure mode on at least 3 of 25 games: a board with
significant decorative/rendering motion unrelated to the actual mechanic (moving
diamonds + line rendering) made whole-grid pixel-perfect prediction essentially
impossible, and a harness-injected reminder telling the model "you must verify_theory
before acting further" was read as a hard gate -- one game (r11l) spent its ENTIRE
4.4-hour budget with only 1 real action taken, endlessly trying and failing to hit 0.6
pixel-perfect accuracy, never realizing it could just act without a perfect theory.

Today (24.08) we added an optional `extract(grid) -> state` parameter to all three
functions -- exactly matching what you did by hand in lp85/ft09: when passed,
`predict`/`goal`/mismatch-comparison all operate on `extract()`'s output (any small
JSON-safe state you define) instead of the raw grid. It's already implemented, tested,
and pasted above (see the `if extract is not None` branches).

Please critique this design from your OWN experience actually solving lp85/ft09 by
hand, without pulling punches:

1. Looking at how YOU actually built and used an abstraction (get_all_tiles, the
   pinwheel tile-grid) in those two wins -- what does our `extract=` design get wrong,
   or miss, compared to how you naturally worked? Be concrete: cite specific lines of
   our code or specific moments in your own transcript excerpts above.
2. In your own words, what's the single biggest weakness in `verify_theory` /
   `plan_with_theory` / `execute_plan` as they stand now (with or without `extract`)
   that would have made lp85/ft09 harder, or that would still risk a paralysis
   incident like r11l's?
3. Concrete, prioritized recommendations -- ranked by how much they would have helped
   YOU, specifically, on lp85/ft09 or on a game like r11l. Prefer small, targeted
   changes to the existing three functions over a full rewrite.

Structure your answer as: (1) design gaps found in extract=, (2) single biggest
remaining weakness, (3) ranked recommendations. Be direct and specific; skip generic
praise."""


def main() -> None:
    llm = OpenAICompatLLM()
    print(f"Querying {llm.name} ...", file=sys.stderr)
    reply = llm.chat([{"role": "user", "content": PROMPT}], max_tokens=4096, temperature=0.3)
    print(reply)
    out_path = ROOT / "docs" / "gemini_critique_extract_theory_2026-08-24.md"
    out_path.write_text(
        "# gemini-3.7-flash critique of verify_theory/plan_with_theory/execute_plan (extract=)\n\n"
        f"Asked 24.08.2026, model `{llm.model}` via AI Studio.\n\n---\n\n{reply}\n",
        encoding="utf-8",
    )
    print(f"\n[saved to {out_path.relative_to(ROOT)}]", file=sys.stderr)


if __name__ == "__main__":
    main()

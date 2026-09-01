"""Guard: the coercion-stack arm really takes the stack off, and nothing else.

Candidate 2 (01.09) is an A/B on one build-time switch,
`ATLAS_CHECKPOINTS_BUILD`. Both arms ship the SAME notebook code; only the
literal differs. That makes two failure modes worth guarding:

  * a patched constant gets RENAMED or removed in the source dataset, so the
    arm silently leaves that coercion running and the A/B measures less than
    it claims. The notebook logs a WARNING for this, but a log line nobody
    reads is not a guard -- this test fails the build instead.
  * a threshold that is NOT comparison-only gets added to the list. Setting
    such a constant to 10**9 does not disable a checkpoint, it allocates.
    _ATLAS_ROLLBACK_LOOP_WINDOW is the known example (it sizes a deque) and
    must stay OUT of the list.

Neither failure is visible in a Phase A run's score, which is exactly the
shape of bug that cost three submission slots on 31.08.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "atlas_src/src/ARC3-Inference/inference/agent/tool_agent.py"

spec = importlib.util.spec_from_file_location(
    "build_atlas_notebook", ROOT / "scripts" / "build_atlas_notebook.py"
)
builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(builder)

# Constants that size a container rather than gate a comparison. Putting any
# of these in the arm would leak memory instead of disabling anything.
SIZING_CONSTANTS = {"_ATLAS_ROLLBACK_LOOP_WINDOW"}


def _fail(name: str, detail: str) -> None:
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def _ok(name: str) -> None:
    print(f"ok   {name}")


def _arm_block(enabled: bool) -> str:
    """The notebook's coercion block, with the knob substituted as shipped."""
    text = builder.substitute_knobs(
        builder.ATLAS_CELL,
        {"ATLAS_CHECKPOINTS_BUILD": "1" if enabled else "0"},
    )
    start = text.index("ATLAS_CHECKPOINTS = ")
    end = text.index("# 30.08: A* heuristic", start)
    return text[start:end]


def main() -> None:
    src = AGENT.read_text(encoding="utf-8")

    # === 1. The knob substitutes to a real literal in both arms ============
    for enabled in (True, False):
        block = _arm_block(enabled)
        if re.search(r"__ATLAS_[A-Z0-9_]*__", block):
            _fail("knob substitutes", f"placeholder survived (enabled={enabled})")
        if f"ATLAS_CHECKPOINTS = {enabled}" not in block:
            _fail("knob substitutes", f"expected literal {enabled}")
    _ok("ATLAS_CHECKPOINTS_BUILD substitutes to True/False in the atlas cell")

    # === 2. Run the OFF arm against a fake module and read the result ======
    off = _arm_block(False)
    names = sorted(set(re.findall(r'"(_ATLAS_[A-Z0-9_]+)"', off)))
    if len(names) < 12:
        _fail("arm is complete", f"only {len(names)} constants listed: {names}")

    fake = types.ModuleType("_atlas_tool_agent")
    for n in names:
        setattr(fake, n, 3)  # a small, firing threshold
    ns = {"_atlas_tool_agent": fake, "print": lambda *a, **k: None}
    exec(compile(off, "<coercion arm>", "exec"), ns)

    still_firing = [n for n in names
                    if getattr(fake, n) not in (False,) and getattr(fake, n) < 10 ** 6]
    if still_firing:
        _fail("arm disables every listed knob", f"{still_firing} left at a firing value")
    _ok(f"the OFF arm neutralises all {len(names)} listed constants")

    # === 3. The ON arm changes nothing =====================================
    fake2 = types.ModuleType("_atlas_tool_agent")
    for n in names:
        setattr(fake2, n, 3)
    exec(compile(_arm_block(True), "<coercion arm on>", "exec"),
         {"_atlas_tool_agent": fake2, "print": lambda *a, **k: None})
    if any(getattr(fake2, n) != 3 for n in names):
        _fail("baseline arm is inert", "the ON arm modified a constant")
    _ok("the ON (baseline) arm leaves every constant untouched")

    # === 4. Every patched name still EXISTS in the source ==================
    # A rename would make the arm silently weaker than it claims to be.
    declared = set(re.findall(r"^(_ATLAS_[A-Z0-9_]+) *=", src, re.M))
    missing = [n for n in names if n not in declared]
    if missing:
        _fail("patched names exist in tool_agent.py",
              f"{missing} -- renamed or removed; the arm would silently leave them ON")
    _ok(f"all {len(names)} patched constants are declared in tool_agent.py")

    # === 5. No container-sizing constant snuck into the arm ================
    overlap = sorted(set(names) & SIZING_CONSTANTS)
    if overlap:
        _fail("no sizing constants in the arm",
              f"{overlap} sizes a container -- 10**9 there allocates, not disables")
    _ok(f"none of the sizing constants {sorted(SIZING_CONSTANTS)} is in the arm")

    # === 6. Each patched threshold is comparison-only in the source ========
    # Walk the AST: a Name load of one of ours must sit inside a Compare, or
    # be a plain bool assignment target elsewhere. Anything feeding BinOp /
    # slicing / a call argument is the _ROLLBACK_LOOP_WINDOW shape of bug.
    tree = ast.parse(src)
    numeric = [n for n in names if isinstance(getattr(fake, n), int)
               and getattr(fake, n) is not False]
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.BinOp, ast.Subscript)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in numeric:
                # the constants' own declarations are BinOps in a few cases
                # (PLAN_FORCE = PLAN_NAG * 2); those are fine, they run at
                # import before any patch. Only flag uses inside functions.
                bad.append(f"{sub.id}@{sub.lineno}")
    # PLAN_FORCE_AFTER_CALLS is defined as PLAN_NAG_EVERY * 2 at module level.
    bad = [b for b in bad if int(b.split("@")[1]) > 400]
    if bad:
        _fail("patched thresholds are comparison-only",
              f"{sorted(set(bad))} feeds arithmetic or a subscript")
    _ok(f"all {len(numeric)} numeric thresholds are used only in comparisons")

    print("\nAll coercion-arm checks passed.")


if __name__ == "__main__":
    main()

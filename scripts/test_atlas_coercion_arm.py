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


SOLVER = ROOT / "atlas_src/src/ARC3-Inference/inference/framework/solver.py"

# Constants that size a container in EITHER source file. Same trap as
# _ATLAS_ROLLBACK_LOOP_WINDOW: a huge value allocates instead of disabling.
SIZING_ALL = SIZING_CONSTANTS | {"_ATLAS_ACTION_EFFECT_HISTORY_WINDOW"}


def _cell(**env: str) -> str:
    """The atlas cell with the given build knobs substituted, as shipped."""
    return builder.substitute_knobs(builder.ATLAS_CELL, env)


def _arm_block(enabled: bool) -> str:
    """Just the coercion block -- it now ENDS where the ablation block starts.

    01.09: the ablation block was inserted between the coercion block and the
    A* section, so the old end-anchor silently swallowed it and this guard
    started reporting the ablation's constants as 'left firing'. Anchoring on
    the next block's own first line keeps the two independent.
    """
    text = _cell(ATLAS_CHECKPOINTS_BUILD="1" if enabled else "0",
                 ATLAS_STOCK_BUILD="0")
    start = text.index("ATLAS_CHECKPOINTS = ")
    return text[start:text.index("ATLAS_STOCK = ", start)]


def _stock_block(enabled: bool) -> str:
    """Just the maximal-ablation block."""
    text = _cell(ATLAS_CHECKPOINTS_BUILD="0",
                 ATLAS_STOCK_BUILD="1" if enabled else "0")
    start = text.index("ATLAS_STOCK = ")
    return text[start:text.index("# 30.08: A* heuristic", start)]


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
    declared = set(re.findall(r"^\s*(_ATLAS_[A-Z0-9_]+)\s*(?::[^=\n]+)? *=", src, re.M))
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

    # === 7. MAXIMAL ABLATION: names exist, nothing sizes, nothing survives ==
    # Same two failure modes as the coercion arm, one layer up: a renamed
    # constant would leave a whole layer silently running, and a container-
    # sizing constant would allocate instead of disabling.
    on = _stock_block(True)
    if "ATLAS_STOCK = True" not in on:
        _fail("stock knob substitutes", "expected the literal True")
    if "ATLAS_STOCK = False" not in _stock_block(False):
        _fail("stock knob substitutes", "expected the literal False")

    ab_names = sorted(set(re.findall(r'"(_ATLAS_[A-Z0-9_]+)"', on)))
    if len(ab_names) < 15:
        _fail("ablation is complete", f"only {len(ab_names)} constants listed")

    declared_all = declared | set(
        re.findall(r"^\s*(_ATLAS_[A-Z0-9_]+)\s*(?::[^=\n]+)? *=", SOLVER.read_text(encoding="utf-8"), re.M)
    )
    gone = [n for n in ab_names if n not in declared_all]
    if gone:
        _fail("ablated names exist in the sources",
              f"{gone} -- renamed or moved; a whole layer would silently stay ON")

    sized = sorted(set(ab_names) & SIZING_ALL)
    if sized:
        _fail("no container-sizing constant in the ablation",
              f"{sized} slices a list -- a huge value allocates, not disables")
    _ok(f"maximal ablation lists {len(ab_names)} constants, all declared, "
        f"none container-sizing")

    # Execute it against fakes and check every listed name was actually written.
    mods = {}
    for mod_name in ("_atlas_tool_agent", "_atlas_solver_mod"):
        m = types.ModuleType(mod_name)
        for n in ab_names:
            setattr(m, n, 3)  # a sentinel no ablation value can equal
        mods[mod_name] = m
    solver_fake = type("S", (), {"analyzer_timeout": 480.0})()
    ns = dict(mods)
    ns["bm"] = type("B", (), {"solver": solver_fake})()
    ns["print"] = lambda *a, **k: None
    exec(compile(on, "<stock ablation>", "exec"), ns)

    untouched = [f"{mod}.{n}" for mod, m in mods.items()
                 for n in ab_names if getattr(m, n) == 3
                 and n in re.findall(r'"(_ATLAS_[A-Z0-9_]+)"',
                                     on.split(f'"{mod}"')[1].split("},")[0])]
    if untouched:
        _fail("ablation writes every constant it lists", f"{untouched}")
    if solver_fake.analyzer_timeout != 900.0:
        _fail("ablation restores the bundle's analyzer timeout",
              f"got {solver_fake.analyzer_timeout}, expected 900.0")
    _ok("the ablation block runs clean against fakes and puts "
        "analyzer_timeout back to 900s")

    # === 7b. the heuristic is REALLY gone, not merely weighted zero =======
    # 02.09, paid for with a whole submission build. _priority() branches on
    # whether a model OBJECT exists; with a model loaded and weight 0 the
    # ordering becomes attempt_len (breadth-first), not the -change novelty
    # ordering a disabled heuristic gives. V47 shipped in that third regime
    # because the A* section sits BELOW the ablation block and handed the path
    # back. Two things have to hold now, and both are checked.
    if '"_ATLAS_ASTAR_CACHE"' not in on:
        _fail("the ablation empties the A* cache",
              "weight 0 alone leaves _priority on the heuristic branch")
    cache = getattr(mods["_atlas_tool_agent"], "_ATLAS_ASTAR_CACHE", None)
    if not (isinstance(cache, dict) and cache.get("loaded") and cache.get("model") is None):
        _fail("the A* cache says loaded-with-nothing", f"got {cache!r}")

    whole = _cell(ATLAS_CHECKPOINTS_BUILD="0", ATLAS_STOCK_BUILD="1")
    # bound the window by the section's own next statement, not by a byte
    # count -- a comment growing past 400 chars once hid the gate from this
    # very check and made a correct build look broken.
    astar_at = whole.index("_astar_hits = sorted(")
    gate = whole[astar_at:whole.index("if _astar_hits and hasattr", astar_at)]
    if "if ATLAS_STOCK:" not in gate or "_astar_hits = []" not in gate:
        _fail("the A* loader is gated on the ablation",
              "the section below the ablation block can hand the path back")
    _ok("the A* heuristic is fully off under ablation: cache emptied AND the "
        "loader gated, so block order no longer matters")

    # The OFF side must be completely inert.
    off_mods = {}
    for mod_name in ("_atlas_tool_agent", "_atlas_solver_mod"):
        m = types.ModuleType(mod_name)
        for n in ab_names:
            setattr(m, n, 3)
        off_mods[mod_name] = m
    solver_fake2 = type("S", (), {"analyzer_timeout": 480.0})()
    ns2 = dict(off_mods)
    ns2["bm"] = type("B", (), {"solver": solver_fake2})()
    ns2["print"] = lambda *a, **k: None
    exec(compile(_stock_block(False), "<stock ablation off>", "exec"), ns2)
    if any(getattr(m, n) != 3 for m in off_mods.values() for n in ab_names):
        _fail("ablation OFF is inert", "the OFF side modified a constant")
    if solver_fake2.analyzer_timeout != 480.0:
        _fail("ablation OFF is inert", "the OFF side changed analyzer_timeout")
    _ok("with ATLAS_STOCK_BUILD=0 the ablation block changes nothing")

    # === 8. PROMPT ARM: patched on the right module, and really stock =====
    # 02.09. `from .prompts import PYTHON_ADDENDUM` binds the name at import,
    # so patching the prompts module would be silently ineffective -- the same
    # shape as the A* loader that kept handing its path back in V47. The arm
    # must write to tool_agent, and the file it loads must not smuggle our own
    # tools back in.
    whole_p = _cell(ATLAS_CHECKPOINTS_BUILD="0", ATLAS_STOCK_BUILD="1",
                    ATLAS_STOCK_PROMPT_BUILD="1")
    arm_start = whole_p.index("ATLAS_STOCK_PROMPT = ")
    arm = whole_p[arm_start:whole_p.index('print("atlas: effective solver config:")', arm_start)]
    if "ATLAS_STOCK_PROMPT = True" not in arm:
        _fail("prompt knob substitutes", "expected the literal True")
    if "_atlas_tool_agent.PYTHON_ADDENDUM =" not in arm:
        _fail("prompt arm patches tool_agent",
              "patching the prompts module would be silently ineffective")
    if "raise RuntimeError" not in arm:
        _fail("a missing addendum file is fatal",
              "a silent fallback would keep the nine tools and look green")
    _ok("the prompt arm patches tool_agent and refuses to run without the file")

    STOCK_FILE = ROOT / "atlas_src/stock_python_addendum.txt"
    if not STOCK_FILE.exists():
        _fail("the stock addendum ships in the dataset", f"{STOCK_FILE} missing")
    stock_text = STOCK_FILE.read_text(encoding="utf-8")
    ours_in = [t for t in ("plan_real", "try_actions", "save_checkpoint",
                           "rollback", "execute_plan", "memo") if t in stock_text]
    if ours_in:
        _fail("the stock addendum carries none of our tools", f"{ours_in}")
    absent = [t for t in ("action(", "verify_theory(", "plan_with_theory(")
              if t not in stock_text]
    if absent:
        _fail("the stock addendum keeps the three real ones", f"{absent}")
    ours_len = len(AGENT.read_text(encoding="utf-8"))
    _ok(f"the stock addendum is {len(stock_text)} chars with three tools, "
        f"none of our six")

    print("\nAll coercion-arm, ablation and prompt checks passed.")


if __name__ == "__main__":
    main()

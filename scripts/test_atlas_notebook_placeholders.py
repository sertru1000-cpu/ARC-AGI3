"""Guard: no build-time placeholder may ever reach Kaggle, in any cell.

31.08, paid for with three submission slots. The builder substitutes its
knobs (concurrency, caps, draws) into the atlas cell -- but RUN_CELL_PATCH
carries `__ATLAS_DRAWS__` too, and it was spliced into the run cell RAW:

    _atlas_os.environ["ATLAS_TIME_BANK_DRAWS"] = "1" if __ATLAS_DRAWS__ else "0"

`__ATLAS_DRAWS__` is a perfectly valid Python identifier, so nothing about
the notebook looked broken. The line lives inside `if true_submission:`,
right after Kaggle's gateway hands over the hidden game list -- a branch
Phase A never executes. So every calibration run stayed green while V39 and
V40 both died on a NameError roughly half an hour into the scoring rerun,
three submissions in a row, reported only as Kaggle's generic "A system
error" with no retrievable log.

Two lessons encoded here:
  * a placeholder must never survive anywhere in the notebook, not just in
    the cell the builder happens to think about;
  * the submission-only branch needs a check that does not depend on Phase A
    ever running it (the same rule that already covers atlas_fit_game_cap).

This test builds the notebook exactly as `build()` does, into a temporary
directory, and refuses any surviving placeholder. It also compiles every
code cell and dry-runs the run cell's submission branch against fakes, so a
NameError of this shape fails here instead of on the leaderboard.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "tufa-arc-agi-framework" / "src"))
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "ARC3-Inference"))

spec = importlib.util.spec_from_file_location(
    "build_atlas_notebook", ROOT / "scripts" / "build_atlas_notebook.py"
)
builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(builder)

PLACEHOLDER_RE = re.compile(r"__ATLAS_[A-Z0-9_]*__")


def _fail(name: str, detail: str) -> None:
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def _ok(name: str) -> None:
    print(f"ok   {name}")


def main() -> None:
    # === 1. The shipped notebook on disk carries no placeholder ============
    nb_path = ROOT / "notebooks_atlas" / "submission.ipynb"
    if nb_path.exists():
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        hits = []
        for idx, cell in enumerate(nb["cells"]):
            source = "".join(cell["source"])
            for m in PLACEHOLDER_RE.findall(source):
                hits.append((idx, m))
        if hits:
            _fail("shipped notebook is placeholder-free",
                  f"{hits[:4]} -- this is what killed three submissions on 31.08")
        _ok(f"the notebook currently on disk carries no placeholder across {len(nb['cells'])} cells")

    # === 2. A freshly built notebook carries none either ===================
    # Build for real, into a temp dir, so the assertion covers build() itself
    # rather than whatever happened to be committed.
    saved_out_dir, saved_out_nb = builder.OUT_DIR, builder.OUT_NB
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        builder.OUT_DIR, builder.OUT_NB = tmp, tmp / "submission.ipynb"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                builder.build()
            fresh = json.loads((tmp / "submission.ipynb").read_text(encoding="utf-8"))
        finally:
            builder.OUT_DIR, builder.OUT_NB = saved_out_dir, saved_out_nb

    sources = ["".join(c["source"]) for c in fresh["cells"]]
    hits = [(i, m) for i, s in enumerate(sources) for m in PLACEHOLDER_RE.findall(s)]
    if hits:
        _fail("freshly built notebook is placeholder-free", f"{hits[:4]}")
    _ok(f"a freshly built notebook carries no placeholder across {len(sources)} cells")

    # === 3. Every code cell still compiles =================================
    for idx, (cell, source) in enumerate(zip(fresh["cells"], sources)):
        if cell.get("cell_type") != "code":
            continue
        try:
            # Notebook cells may use top-level await (the run cell does),
            # which plain compile() rejects -- allow it, as Jupyter does.
            compile(source, f"<cell {idx}>", "exec",
                    flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as exc:
            _fail("every cell compiles", f"cell {idx}: {exc}")
    _ok("every code cell of the built notebook compiles (top-level await allowed)")

    # === 4. The knobs really landed as literals ============================
    # A placeholder is valid Python, so "it compiles" proves nothing on its
    # own -- check the substituted values are actually there.
    joined = "\n".join(sources)
    for name in ("ATLAS_CONCURRENCY", "ATLAS_SUBMISSION_GAME_CAP_CEILING_S",
                 "ATLAS_CALIBRATION_CAP_S"):
        if not re.search(rf"^{name} = [-\d.]", joined, re.M):
            _fail("knobs are literals", f"{name} is not a literal assignment")
    if 'ATLAS_TIME_BANK_DRAWS"] = "1" if ' not in joined:
        _fail("draws knob present", "the run-cell draws line went missing")
    draws_line = next(l for l in joined.splitlines() if 'ATLAS_TIME_BANK_DRAWS"] = ' in l)
    if PLACEHOLDER_RE.search(draws_line):
        _fail("draws knob substituted", draws_line.strip())
    _ok(f"the run-cell draws line ships a literal: {draws_line.strip()[:72]}")

    # === 5. Dry-run the SUBMISSION branch -- the part Phase A never runs ===
    # Rebuild just the patched fragment and execute it against fakes. A bare
    # placeholder identifier raises NameError here, exactly as it did on
    # Kaggle, instead of silently passing a syntax check.
    patched = builder.substitute_knobs(builder.RUN_CELL_PATCH)
    body = "\n".join(line[8:] if line.startswith("        ") else line
                     for line in patched.splitlines())
    ns = {
        "bm": type("B", (), {"games": [], "n_passes": 0, "game_weights": None,
                             "solver": type("S", (), {"concurrency": 28,
                                                      "max_runtime_s_per_game": 7920.0})()})(),
        "_competition_games": lambda: ["g"] * 55,
        "atlas_fit_game_cap": lambda n: None,
    }
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(body, "<run-cell submission branch>", "exec",
                         flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), ns)
    except NameError as exc:
        _fail("submission branch executes", f"{exc} -- this is the 31.08 failure, exactly")
    except Exception as exc:  # pragma: no cover - fakes are minimal
        _fail("submission branch executes", f"unexpected {exc!r}")
    import os as _os
    if _os.environ.get("ATLAS_TIME_BANK_DRAWS") not in {"0", "1"}:
        _fail("submission branch sets the draws env", "expected '0' or '1'")
    _ok(f"the submission-only branch runs clean against fakes "
        f"(ATLAS_TIME_BANK_DRAWS={_os.environ.get('ATLAS_TIME_BANK_DRAWS')})")

    print("\nAll notebook-placeholder checks passed.")


if __name__ == "__main__":
    main()

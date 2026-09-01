"""Guard: the own-games knob points Phase A at our 24 games, or fails loudly.

01.09. Local A/B on the 25 public games is no longer valid evidence: our layer
transfers to unseen games at x0.42 while stock transfers at x1.01, because the
thresholds were tuned against failures in those very games. `ATLAS_OWN_GAMES_BUILD`
switches Phase A to our own 24-game testbed instead.

Three failure modes this guards, all of which would produce a GREEN run:

  * a silent fallback to the public 25 when the directory is missing. That is
    the 31.08 shape of bug exactly -- nothing crashes, the log looks normal,
    and the number measures the wrong thing. The block must RAISE.
  * the knob leaking into a competition rerun. Own games in a real submission
    would score ~0 and burn a slot.
  * the game list silently coming back empty or partial.

The last section checks the claim the block's own comment makes about the
testbed's limitation, against the real game files: the own games' first levels
are short, so this set measures mechanics honestly but cannot reproduce the
long-exploration failure. If someone adds long games later, this test tells
them the comment needs updating.

Runs on CPU only -- no GPU, no Kaggle, no quota.
"""

from __future__ import annotations

import importlib.util
import json
import re
import statistics
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWN = ROOT / "our_games"

spec = importlib.util.spec_from_file_location(
    "build_atlas_notebook", ROOT / "scripts" / "build_atlas_notebook.py"
)
builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(builder)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}: {detail}")
        FAILURES.append(name)


def _block(own: bool) -> str:
    text = builder.substitute_knobs(
        builder.ATLAS_CELL,
        {"ATLAS_OWN_GAMES_BUILD": "1" if own else "0",
         "ATLAS_CHECKPOINTS_BUILD": "1", "ATLAS_STOCK_BUILD": "0"},
    )
    start = text.index("ATLAS_PHASE_A_OWN_GAMES = ")
    return text[start:text.index('print("atlas: effective solver config:")', start)]


def _run(block: str, *, true_submission: bool, dirs: list[str], games: list[str]):
    """Execute the block against fakes; return (bm, raised)."""
    bm = types.SimpleNamespace(games=["public"] * 25, n_passes=3,
                               game_weights=[1], label="anim")
    fake_api = types.ModuleType("taaf.game_api")
    fake_api.ArcadeSpec = lambda **kw: types.SimpleNamespace(**kw)
    fake_api.GameAPI = lambda **kw: types.SimpleNamespace(**kw)
    fake_taaf = types.ModuleType("taaf")
    fake_taaf.game_api = fake_api
    saved = {k: sys.modules.get(k) for k in ("taaf", "taaf.game_api")}
    sys.modules["taaf"] = fake_taaf
    sys.modules["taaf.game_api"] = fake_api
    fake_os = types.SimpleNamespace(
        listdir=lambda p: games,
        path=types.SimpleNamespace(isdir=lambda p: True, join=lambda a, b: f"{a}/{b}"),
    )
    ns = {
        "bm": bm,
        "true_submission": true_submission,
        "_atlas_glob": types.SimpleNamespace(glob=lambda pat, recursive=False: dirs),
        "print": lambda *a, **k: None,
        "os": fake_os,
    }
    # the block imports `os as _atlas_os_g`; hand it the fake through builtins
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "os":
            return fake_os
        return real_import(name, *a, **k)

    builtins.__import__ = fake_import
    raised = None
    try:
        exec(compile(block, "<own-games block>", "exec"), ns)
    except Exception as exc:  # noqa: BLE001
        raised = exc
    finally:
        builtins.__import__ = real_import
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return ns["bm"], raised


def main() -> None:
    ids = sorted(p.name for p in OWN.iterdir() if p.is_dir())

    # === 1. the knob substitutes and the OFF side is inert =================
    off = _block(False)
    if "ATLAS_PHASE_A_OWN_GAMES = False" not in off:
        check("knob substitutes", False, "expected the literal False")
    bm, raised = _run(off, true_submission=False, dirs=["/kaggle/input/x/our_games"],
                      games=ids)
    check("with the knob OFF Phase A keeps the public games",
          raised is None and bm.games == ["public"] * 25 and bm.n_passes == 3,
          f"games={len(bm.games)} passes={bm.n_passes} raised={raised!r}")

    # === 2. the ON side swaps in the own games ============================
    on = _block(True)
    if "ATLAS_PHASE_A_OWN_GAMES = True" not in on:
        check("knob substitutes", False, "expected the literal True")
    bm, raised = _run(on, true_submission=False, dirs=["/kaggle/input/x/our_games"],
                      games=ids)
    check(f"the knob ON loads all {len(ids)} own games",
          raised is None and len(bm.games) == len(ids),
          f"raised={raised!r} games={len(bm.games) if raised is None else '-'}")
    check("n_passes is forced to 1 and weights cleared",
          raised is None and bm.n_passes == 1 and bm.game_weights is None)
    check("the run label records the swap",
          raised is None and bm.label.endswith(f"-own{len(ids)}"),
          f"label={getattr(bm, 'label', None)!r}")

    # === 3. a missing directory is FATAL, never a silent fallback =========
    # This is the whole point of the guard: a green run on the wrong game set
    # is indistinguishable from a correct one until the score comes back.
    bm, raised = _run(on, true_submission=False, dirs=[], games=ids)
    check("a missing our_games directory RAISES instead of falling back",
          isinstance(raised, RuntimeError) and "refusing" in str(raised),
          f"raised={raised!r}, games would have been {len(bm.games)}")

    bm, raised = _run(on, true_submission=False,
                      dirs=["/kaggle/input/x/our_games"], games=[])
    check("an empty our_games directory RAISES too",
          isinstance(raised, RuntimeError),
          f"raised={raised!r}")

    # === 4. own games can never reach a competition rerun =================
    bm, raised = _run(on, true_submission=True, dirs=["/kaggle/input/x/our_games"],
                      games=ids)
    check("the knob is ignored during a competition rerun",
          raised is None and bm.games == ["public"] * 25,
          f"raised={raised!r} games={len(bm.games)}")

    # === 5. the games on disk are real and complete =======================
    check("our_games holds 24 game directories", len(ids) == 24, f"found {len(ids)}")
    missing = []
    first_level: list[int] = []
    for g in ids:
        metas = list((OWN / g).glob("*/metadata.json"))
        if not metas:
            missing.append(g)
            continue
        data = json.loads(metas[0].read_text(encoding="utf-8"))
        base = data.get("baseline_actions") or []
        if not base:
            missing.append(g)
        else:
            first_level.append(base[0])
    check("every own game carries metadata and baselines", not missing, f"{missing}")

    # === 6. the block's stated limitation is still true ===================
    # The comment claims own first levels are SHORT compared with the public
    # games we never take (median human baseline 54, range 21-78). If someone
    # adds long games, this check fires and the comment must be revised.
    med = statistics.median(first_level)
    check(f"own first levels are still short (median {med}, max {max(first_level)})",
          med < 21,
          "own games now reach the long-exploration class -- update the "
          "KNOWN LIMITATION comment in build_atlas_notebook.py")

    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) failed.")
        sys.exit(1)
    print("\nAll own-games checks passed.")


if __name__ == "__main__":
    main()

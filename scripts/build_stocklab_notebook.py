"""Build the stock-lab kernel: UNMODIFIED Duck, playing OUR testbed.

Why a separate kernel. From 02.09 the base is stock, not our fork -- measured
that evening: with every switch of ours off and the prompt back to three
tools, we still cost 1647 generated tokens per model call against stock's 815.
Stock is a different kernel, not ours with the knobs down.

This kernel exists to put a number on stock under OUR conditions. It mounts
the upstream source dataset (so the agent is untouched Duck) plus our own
dataset (only for the 30 testbed games), and swaps Phase A's game list. It is
a MEASUREMENT kernel and is never meant to be submitted: the own-games swap is
guarded on `not true_submission`, exactly as duck's own public-eval block is.

`arc3-duck-baseline` stays untouched -- version 1 of it is our best submission
asset (1.61 on the hidden set) and must not be disturbed.

usage:
    python scripts/build_stocklab_notebook.py
    ATLAS_STOCKLAB_CAP_S=1500 ATLAS_STOCKLAB_CONC=30 python scripts/build_stocklab_notebook.py
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_NB = ROOT / "notebooks_duck" / "submission.ipynb"
OUT_DIR = ROOT / "notebooks_stocklab"
OUT_NB = OUT_DIR / "submission.ipynb"

KERNEL_ID = "sergueimakarov/arc3-stock-lab"
OUR_DATASET = "sergueimakarov/arc3-atlas-src"

CAP_S = float(os.environ.get("ATLAS_STOCKLAB_CAP_S", "1500"))
CONC = int(os.environ.get("ATLAS_STOCKLAB_CONC", "30"))

# The one cell we add. It runs AFTER duck's own public-evaluation block, so it
# overrides whatever that set -- and only outside a competition rerun.
SWAP_CELL = '''# =====================================================================
# stock-lab: play OUR 30-game testbed instead of the 25 public games.
#
# Everything above this cell is UNMODIFIED Duck. This cell touches only the
# game list and two solver knobs, so the number it produces is stock's,
# measured under the same conditions as our own runs (V49/V50): 30 games,
# concurrency {conc}, {cap:.0f}s per game.
#
# NEVER reaches a submission: guarded on `not true_submission`, and the run
# cell replaces bm.games from Kaggle's gateway for a real rerun anyway.
# =====================================================================
if not true_submission:
    import glob as _lab_glob
    import os as _lab_os
    import taaf.game_api as _lab_api

    _own = sorted(_lab_glob.glob("/kaggle/input/*/our_games")) or sorted(
        _lab_glob.glob("/kaggle/input/**/our_games", recursive=True))
    if not _own:
        # fatal, never a silent fallback to the public 25 -- a green run on
        # the wrong game set is indistinguishable from a correct one until
        # the number comes back and quietly means nothing.
        raise RuntimeError(
            "stock-lab: no our_games/ under /kaggle/input -- refusing to "
            "fall back to the public game list"
        )
    _own_dir = _own[0]
    _ids = sorted(
        d for d in _lab_os.listdir(_own_dir)
        if _lab_os.path.isdir(_lab_os.path.join(_own_dir, d))
    )
    if not _ids:
        raise RuntimeError(f"stock-lab: {{_own_dir}} holds no game directories")

    _spec = _lab_api.ArcadeSpec(environments_dir=_own_dir)
    bm.games = [_lab_api.GameAPI(env_name=_g, arcade_spec=_spec) for _g in _ids]
    bm.n_passes = 1
    bm.game_weights = None
    bm.label = f"{{bm.label}}-stocklab{{len(_ids)}}"
    if hasattr(bm.solver, "concurrency"):
        bm.solver.concurrency = {conc}
    if hasattr(bm.solver, "max_runtime_s_per_game"):
        bm.solver.max_runtime_s_per_game = {cap}

    print(f"stock-lab: UNMODIFIED Duck on {{len(_ids)}} own games from {{_own_dir}}")
    print(f"stock-lab:   concurrency={{bm.solver.concurrency}} "
          f"cap={{bm.solver.max_runtime_s_per_game}}s passes={{bm.n_passes}}")
    print(f"stock-lab:   {{', '.join(_ids)}}")
'''


def build() -> None:
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))

    # place the swap right after duck's public-evaluation cell
    idx = next(
        i for i, c in enumerate(nb["cells"])
        if "Q38_P1_PUBLIC_GAME_IDS" in "".join(c["source"])
    )
    source = SWAP_CELL.format(conc=CONC, cap=CAP_S)
    nb["cells"].insert(idx + 1, {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    meta = json.loads((ROOT / "notebooks_duck" / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = "arc3 stock lab"
    if OUR_DATASET not in meta["dataset_sources"]:
        meta["dataset_sources"].append(OUR_DATASET)
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # sanity: the stock cells must be byte-identical to duck's
    orig = json.loads(SRC_NB.read_text(encoding="utf-8"))["cells"]
    built = [c for i, c in enumerate(nb["cells"]) if i != idx + 1]
    assert len(built) == len(orig), "cell count drifted"
    for a, b in zip(orig, built):
        assert "".join(a["source"]) == "".join(b["source"]), "a stock cell was modified"

    print(f"собрано: {OUT_NB}")
    print(f"  ячеек {len(nb['cells'])} (у стока {len(orig)}, добавлена одна)")
    print(f"  стоковые ячейки не тронуты: проверено побайтово")
    print(f"  игр 30, конкурентность {CONC}, потолок {CAP_S:.0f}с")
    print(f"  датасеты: {', '.join(meta['dataset_sources'])}")


if __name__ == "__main__":
    build()

"""Build a throwaway debug variant of the atlas notebook: same harness/config
as the real submission notebook, but restricted to a handful of specific
games with a generous per-game runtime cap, for deep manual transcript
analysis instead of a full 25-game calibration sweep.

Does not touch notebooks_atlas/ (the real submission pipeline) or its Kaggle
kernel slug -- writes to notebooks_atlas_debug/ under a separate slug so a
push here can never be confused with (or accidentally overwrite) the real
arc3-atlas kernel history.

Run scripts/build_atlas_notebook.py first so notebooks_atlas/submission.ipynb
reflects the current atlas_src changes -- this script patches THAT output,
it does not rebuild from the Duck bundle itself.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_NB = ROOT / "notebooks_atlas" / "submission.ipynb"
SRC_META = ROOT / "notebooks_atlas" / "kernel-metadata.json"
OUT_DIR = ROOT / "notebooks_atlas_debug"
OUT_NB = OUT_DIR / "submission.ipynb"
KERNEL_SLUG = "sergueimakarov/arc3-atlas-debug"
KERNEL_TITLE = "arc3 atlas debug"

# The 3 games picked for the reasoning-pattern deep-dive (see
# notebooks_colab/atlas_colab_debug.ipynb for the full rationale per game --
# this is the Kaggle-side rerun of the same investigation after Colab's A100
# turned out too slow/unreliable for an unattended multi-hour session).
DEBUG_GAME_IDS = [
    "cn04-2fe56bfb",  # position-drift hypothesis
    "vc33-5430563c",  # stalls after a real success (level-transition memory)
    "ka59-38d34dbb",  # high tokens / low actions (checkpoint adoption)
]

DEBUG_CAP_S = 3600.0  # 60 min/game, matching the Colab attempt's budget


def _source_of(cell: dict) -> str:
    return "".join(cell["source"])


def _set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def build() -> None:
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # --- restrict the game list (the "Inline customization hook" cell) -----
    hook_idx = next(
        i for i, c in enumerate(cells) if "Q38_P1_PUBLIC_GAME_IDS" in _source_of(c)
    )
    hook_src = _source_of(cells[hook_idx])
    if "Q38_P1_PUBLIC_GAME_IDS = [" not in hook_src:
        raise RuntimeError("Could not find Q38_P1_PUBLIC_GAME_IDS assignment to patch.")

    list_start = hook_src.index("Q38_P1_PUBLIC_GAME_IDS = [")
    list_end = hook_src.index("]", list_start) + 1
    new_list_src = "Q38_P1_PUBLIC_GAME_IDS = " + json.dumps(DEBUG_GAME_IDS)
    hook_src = hook_src[:list_start] + new_list_src + hook_src[list_end:]

    # The hook asserts len(...) == 25; relax that to match our smaller list.
    hook_src = hook_src.replace(
        "if len(Q38_P1_PUBLIC_GAME_IDS) != 25 or len(set(Q38_P1_PUBLIC_GAME_IDS)) != 25:",
        f"if len(Q38_P1_PUBLIC_GAME_IDS) != {len(DEBUG_GAME_IDS)} or "
        f"len(set(Q38_P1_PUBLIC_GAME_IDS)) != {len(DEBUG_GAME_IDS)}:",
    )
    hook_src = hook_src.replace(
        'raise RuntimeError("Q38 P1 public game list must contain exactly 25 unique games.")',
        'raise RuntimeError("Debug game list must contain '
        f'exactly {len(DEBUG_GAME_IDS)} unique games.")',
    )
    _set_source(cells[hook_idx], hook_src)

    # --- widen the calibration cap (the atlas v2 cell) ----------------------
    atlas_idx = next(
        i for i, c in enumerate(cells) if "ATLAS_CALIBRATION_CAP_S" in _source_of(c)
    )
    atlas_src = _source_of(cells[atlas_idx])
    marker = "ATLAS_CALIBRATION_CAP_S = "
    start = atlas_src.index(marker)
    line_end = atlas_src.index("\n", start)
    atlas_src = (
        atlas_src[:start]
        + f"ATLAS_CALIBRATION_CAP_S = {DEBUG_CAP_S}  # debug: 3-game deep-dive, not a calibration sweep"
        + atlas_src[line_end:]
    )
    _set_source(cells[atlas_idx], atlas_src)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n"
    )

    meta = json.loads(SRC_META.read_text(encoding="utf-8"))
    meta["id"] = KERNEL_SLUG
    meta["title"] = KERNEL_TITLE
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"wrote {OUT_NB} ({len(cells)} cells)")
    print(f"games: {DEBUG_GAME_IDS}")
    print(f"per-game cap: {DEBUG_CAP_S:.0f}s")
    print(f"wrote {OUT_DIR / 'kernel-metadata.json'} -> {KERNEL_SLUG}")


if __name__ == "__main__":
    build()

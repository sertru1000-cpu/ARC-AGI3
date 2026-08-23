"""Prepare (but never push) the atlas_src/ Kaggle dataset bundle.

Validates the bundle layout inherited from Duck's August source
(atlas_src/taaf-kaggle-bundle.json), writes dataset-metadata.json next to
it, and prints the exact command plus a size/file summary for a human to
review. This script never calls the Kaggle API -- publishing a new dataset
is an external write to the user's account and needs explicit per-push
consent, same spirit as scripts/build_atlas_notebook.py never calling
`kaggle kernels push` itself.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = ROOT / "atlas_src"
MARKER = "taaf-kaggle-bundle.json"
DATASET_OWNER = "sergueimakarov"
DATASET_SLUG = "arc3-atlas-src"
DATASET_TITLE = "arc3 atlas source (Duck fork 08.07)"
OLD_DATASET_REF = "jakobbrggen/taaf-kaggle-source-anim-20260807-anim"


def _iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "dataset-metadata.json":
            continue
        if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
            continue
        yield path


def build() -> None:
    marker_path = BUNDLE_DIR / MARKER
    if not marker_path.is_file():
        raise SystemExit(f"missing bundle marker: {marker_path}")

    files = sorted(_iter_files(BUNDLE_DIR))
    total_bytes = sum(f.stat().st_size for f in files)

    metadata = {
        "title": DATASET_TITLE,
        "id": f"{DATASET_OWNER}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path = BUNDLE_DIR / "dataset-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"bundle dir:  {BUNDLE_DIR}")
    print(f"marker:      {marker_path.read_text(encoding='utf-8').strip()}")
    print(f"files:       {len(files)}")
    print(f"total size:  {total_bytes / 1024:.1f} KiB")
    print(f"wrote:       {meta_path}")
    print()
    print("NOT pushed. This only prepares the folder. To publish, needs")
    print("explicit per-push go-ahead, then run manually:")
    print(f"  .venv/Scripts/kaggle.exe datasets create -p {BUNDLE_DIR} --dir-mode zip")
    print()
    print("After that succeeds, in notebooks_duck/submission.ipynb cell 3")
    print("(DATASET_SOURCES) replace:")
    print(f'  "{OLD_DATASET_REF}"')
    print("with:")
    print(f'  "{DATASET_OWNER}/{DATASET_SLUG}"')
    print("then rebuild notebooks_atlas via scripts/build_atlas_notebook.py.")


if __name__ == "__main__":
    build()

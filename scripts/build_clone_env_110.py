"""Build a submission-shaped OFFLINE game set by cloning the 25 public games.

The real Kaggle Phase B evaluates a hidden set of ~110 games (55 semi-private
+ 55 fully-private, per the ARC-AGI-3 technical report), mounted into the
kernel as offline environment_files. To rehearse that SHAPE (game count,
concurrency, wave math, vLLM load) before Saturday's build, this script
clones the 25 public game dirs into N uniquely-id'd copies.

How a clone works (verified against arc_agi's loader, 28.08):
  - the arcade scans environments_dir recursively for metadata.json and takes
    game_id from the file, local_dir from its parent;
  - the game module is loaded from local_dir as {class_name.lower()}.py and
    the class is looked up by metadata's class_name -- so a clone keeps the
    ORIGINAL game .py untouched and only rewrites metadata.json with a new
    game_id ("k000-<version>") plus an explicit class_name pointing at the
    original class ("Ar25").
  - taaf's own clone_game_ids() (competition_arcade.py, the R2.57 110-run
    set) uses the same "k000".."k109" id scheme for the REST-server variant;
    this script is the offline-ArcadeSpec equivalent.

Usage:
  python scripts/build_clone_env_110.py --source environment_files \
      --dest environment_files_110 --count 110
Writes <dest>/game_ids.txt (comma-joined) for the deploy script's --game.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def derive_class_name(game_id: str) -> str:
    """Mirror arc_agi.models.EnvironmentInfo: first 4 chars, first upper."""
    first_four = game_id[:4]
    return first_four[0].upper() + first_four[1:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="environment_files")
    parser.add_argument("--dest", default="environment_files_110")
    parser.add_argument("--count", type=int, default=110)
    parser.add_argument("--prefix", default="k")
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # One (game_dir, version_dir, metadata) per public game, sorted for
    # a stable clone->source mapping.
    sources = []
    for metadata_file in sorted(source.rglob("metadata.json")):
        meta = json.loads(metadata_file.read_text(encoding="utf-8"))
        sources.append((metadata_file.parent, meta))
    if not sources:
        raise SystemExit(f"No games found under {source}")
    print(f"{len(sources)} source game(s); building {args.count} clone(s) -> {dest}")

    clone_ids: list[str] = []
    for i in range(args.count):
        src_dir, src_meta = sources[i % len(sources)]
        src_id = str(src_meta["game_id"])
        version = src_id.split("-", 1)[1] if "-" in src_id else "v0"
        clone_prefix = f"{args.prefix}{i:03d}"
        clone_id = f"{clone_prefix}-{version}"
        clone_dir = dest / clone_prefix / version
        shutil.copytree(src_dir, clone_dir)
        meta = dict(src_meta)
        meta["game_id"] = clone_id
        # The clone id would derive class_name "K000" -- pin the ORIGINAL
        # class so the loader finds e.g. class Ar25 in the untouched ar25.py.
        meta["class_name"] = str(src_meta.get("class_name") or derive_class_name(src_id))
        meta["title"] = f"{src_meta.get('title') or src_id} (clone of {src_id.split('-')[0]})"
        (clone_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        clone_ids.append(clone_id)

    (dest / "game_ids.txt").write_text(",".join(clone_ids), encoding="utf-8")
    print(f"wrote {len(clone_ids)} clone(s); ids in {dest / 'game_ids.txt'}")


if __name__ == "__main__":
    main()

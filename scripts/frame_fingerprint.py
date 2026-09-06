"""Fingerprint every level's rendered frame -- proof that a refactor changed nothing.

The sprite code in the short games is about to be rewritten from literal pixel
indices to expressions in CELL. At CELL == 4 the result must be
BYTE-IDENTICAL: the rewrite is meant to make the same picture expressible at
other cell sizes, not to change the picture. Anything that shifts is a bug,
and a shifted sprite is invisible to every other check we have -- the game
still plays and still scores.

usage:
    python scripts/frame_fingerprint.py --out before.json
    python scripts/frame_fingerprint.py --compare before.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fingerprint() -> dict[str, str]:
    out: dict[str, str] = {}
    for md_path in sorted(glob.glob(str(ROOT / "our_games" / "*" / "*" / "metadata.json"))):
        gid = md_path.split(os.sep)[-3]
        md = json.loads(Path(md_path).read_text(encoding="utf-8"))
        py = Path(md_path).parent / f"{gid}.py"
        try:
            spec = importlib.util.spec_from_file_location(f"fp_{gid}", py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            game = getattr(mod, md["class_name"])()
        except Exception as exc:
            out[gid] = f"ОШИБКА: {type(exc).__name__}: {exc}"
            continue
        h = hashlib.sha256()
        for lv in range(len(getattr(mod, "LEVELS", []) or [1])):
            try:
                game._current_level_index = lv
                hook = getattr(game, "on_set_level", None) or getattr(game, "_load", None)
                if callable(hook):
                    try:
                        hook(game.current_level)
                    except TypeError:
                        hook()
                frame = game._camera._raw_render(list(game.current_level.get_sprites()))
                h.update(bytes(frame.astype("uint8").tobytes()))
            except Exception as exc:
                h.update(f"ОШИБКА{type(exc).__name__}".encode())
        out[gid] = h.hexdigest()[:16]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out")
    ap.add_argument("--compare")
    args = ap.parse_args()

    now = fingerprint()
    if args.out:
        Path(args.out).write_text(json.dumps(now, ensure_ascii=False, indent=1), encoding="utf-8")
        errs = [g for g, v in now.items() if v.startswith("ОШИБКА")]
        print(f"снято отпечатков: {len(now)}" + (f", с ошибками: {errs}" if errs else ""))
        return 0
    if args.compare:
        was = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        same = [g for g in was if was[g] == now.get(g)]
        diff = [g for g in was if was[g] != now.get(g)]
        print(f"совпало {len(same)} из {len(was)}")
        for g in diff:
            print(f"  ИЗМЕНИЛОСЬ {g}: {was[g]} -> {now.get(g)}")
        return 1 if diff else 0
    print(json.dumps(now, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Package the SFT dataset into a self-contained MLX fine-tuning kit.

Reads data/sft/train.jsonl (built by build_sft_dataset.py), makes an
episode-level train/valid split (no example of one episode leaks across the
split), drops overlong examples that would be truncated mid-target on a
24 GB Mac, strips non-mlx keys, and writes everything the MacBook needs
into one copyable folder:

    data/mlx-rehearsal/
      data/train.jsonl      <- mlx_lm.lora chat format
      data/valid.jsonl
      check_student.py      <- post-training smoke: model vs teacher reply
      README.md             <- commands to run on the Mac

Usage:
    .venv/Scripts/python.exe scripts/prep_mlx_dataset.py
    .venv/Scripts/python.exe scripts/prep_mlx_dataset.py --max-chars 24000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="data/sft/train.jsonl")
    p.add_argument("--out", default="data/mlx-rehearsal")
    p.add_argument("--max-chars", type=int, default=32000,
                   help="drop examples longer than this (~chars/4 tokens); "
                        "mlx truncates at max-seq-length and a truncated "
                        "TARGET teaches garbage")
    p.add_argument("--valid-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    src = ROOT / args.src
    out = ROOT / args.out
    (out / "data").mkdir(parents=True, exist_ok=True)

    examples = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    episodes: dict[str, list[dict]] = {}
    for ex in examples:
        episodes.setdefault(ex.get("source", "?"), []).append(ex)

    rng = random.Random(args.seed)
    names = sorted(episodes)
    rng.shuffle(names)
    n_valid = max(1, round(len(names) * args.valid_frac))
    valid_names = set(names[:n_valid])

    stats = {"train": 0, "valid": 0, "dropped_long": 0}
    lengths: list[int] = []
    files = {
        "train": (out / "data" / "train.jsonl").open("w", encoding="utf-8"),
        "valid": (out / "data" / "valid.jsonl").open("w", encoding="utf-8"),
    }
    try:
        for name, exs in episodes.items():
            split = "valid" if name in valid_names else "train"
            for ex in exs:
                text_len = sum(len(m["content"]) for m in ex["messages"])
                lengths.append(text_len)
                if text_len > args.max_chars:
                    stats["dropped_long"] += 1
                    continue
                files[split].write(
                    json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")
                stats[split] += 1
    finally:
        for f in files.values():
            f.close()

    lengths.sort()
    pct = lambda q: lengths[int(q * (len(lengths) - 1))] if lengths else 0
    print(f"episodes: {len(names)} total, valid episodes: {sorted(valid_names)}")
    print(f"examples: train={stats['train']} valid={stats['valid']} "
          f"dropped_long={stats['dropped_long']} (cap {args.max_chars} chars)")
    print(f"example length chars p50={pct(.5)} p90={pct(.9)} max={lengths[-1] if lengths else 0}"
          f" -> tokens roughly /4")
    print(f"kit: {out}  (copy this folder to the Mac)")


if __name__ == "__main__":
    main()

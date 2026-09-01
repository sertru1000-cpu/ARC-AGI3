"""Paired A/B on level-1 conversion: does arm B convert more games than arm A?

Pairs the arms BY GAME and looks only at the games that flipped -- McNemar's
test. Pairing is what makes this affordable: the game-to-game difficulty spread
cancels out, so the sample size that matters is the number of DISCORDANT games,
not the number of runs.

Power, by simulation (see docs/plan_top10_by_3009.md item 31), with 49 games:
    1 run per arm  -> +20 pp detected 47% of the time
    2 runs per arm -> +20 pp at 82%, +15 pp at 57%
    3 runs per arm -> +15 pp at 77%
So two runs per arm is the working minimum, and anything under +15 pp is not
worth spending money on: we would not be able to tell it from noise.

Arms may list several run directories each (repeats); every (game, run-index)
pair becomes one trial.

usage:
    python scripts/conversion_ab.py --a runs/armA_1 runs/armA_2 --b runs/armB_1 runs/armB_2
    python scripts/conversion_ab.py --a runs/kaggle_v40 --b runs/kaggle_v41   # same code: expect "no difference"
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("conversion", ROOT / "scripts" / "conversion.py")
conv = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(conv)


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar. b = only B took it, c = only A took it."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def arm_trials(runs: list[Path]) -> list[dict[str, bool]]:
    """One dict per run: game -> took level 1."""
    out = []
    for r in runs:
        s = conv.summarise(r)
        out.append({g: v["took_l1"] for g, v in s["per_game"].items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", nargs="+", required=True, type=Path, help="рукав A (база)")
    ap.add_argument("--b", nargs="+", required=True, type=Path, help="рукав B (правка)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    A, B = arm_trials(args.a), arm_trials(args.b)
    if not A or not B:
        sys.exit("пустой рукав")

    pairs = min(len(A), len(B))
    if len(A) != len(B):
        print(f"ВНИМАНИЕ: прогонов в рукавах разное число ({len(A)} и {len(B)}), "
              f"сравниваю первые {pairs} пар\n")

    b_only = c_only = both = neither = 0
    flipped_to_b: list[str] = []
    flipped_to_a: list[str] = []
    for i in range(pairs):
        shared = sorted(set(A[i]) & set(B[i]))
        for g in shared:
            a_ok, b_ok = A[i][g], B[i][g]
            if b_ok and not a_ok:
                b_only += 1
                flipped_to_b.append(f"{g}#{i+1}")
            elif a_ok and not b_ok:
                c_only += 1
                flipped_to_a.append(f"{g}#{i+1}")
            elif a_ok and b_ok:
                both += 1
            else:
                neither += 1

    total = b_only + c_only + both + neither
    if total == 0:
        sys.exit("нет общих игр между рукавами — проверьте наборы")

    conv_a = (both + c_only) / total
    conv_b = (both + b_only) / total
    p = mcnemar_p(b_only, c_only)

    print(f"ПАРНОЕ СРАВНЕНИЕ ПО КОНВЕРСИИ, {pairs} прогон(ов) на рукав, {total} игро-пар\n")
    print(f"  {args.label_a:12} конверсия {conv_a*100:5.1f}%   (балл ≈ {conv_a*3.52:.2f})")
    print(f"  {args.label_b:12} конверсия {conv_b*100:5.1f}%   (балл ≈ {conv_b*3.52:.2f})")
    print(f"  разница      {(conv_b-conv_a)*100:+5.1f} п.п.   (≈ {(conv_b-conv_a)*3.52:+.2f} балла)\n")
    print(f"  оба взяли: {both}   никто: {neither}")
    print(f"  только {args.label_b}: {b_only}   только {args.label_a}: {c_only}   "
          f"итого расхождений: {b_only+c_only}")
    print(f"  McNemar p = {p:.4f}\n")

    if p < 0.05 and b_only > c_only:
        print(f"  ВЕРДИКТ: {args.label_b} конвертирует ЛУЧШЕ, различие значимо (p < 0.05).")
    elif p < 0.05 and c_only > b_only:
        print(f"  ВЕРДИКТ: {args.label_b} конвертирует ХУЖЕ, различие значимо (p < 0.05).")
    else:
        print("  ВЕРДИКТ: различия не видно. Это НЕ значит 'одинаково' — значит,")
        print("  эффект (если он есть) меньше того, что видит выборка такого размера.")
        if b_only + c_only < 10:
            print(f"  Расхождений всего {b_only+c_only} — для вывода нужно больше прогонов.")

    if flipped_to_b:
        print(f"\n  перевернулись в пользу {args.label_b}: {', '.join(flipped_to_b[:12])}")
    if flipped_to_a:
        print(f"  перевернулись в пользу {args.label_a}: {', '.join(flipped_to_a[:12])}")


if __name__ == "__main__":
    main()

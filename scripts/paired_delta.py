"""Paired delta test for two polygon runs (Gemini r22, 06.09).

The polygon's mean RHAE has sigma ~6 points on 30 games (one run each), so
comparing means only detects huge effects. This compares per game instead:
  * drop "dead" games (0 levels in both) and "ceiling" games (all levels in both);
  * delta = RHAE_new - RHAE_base per remaining game;
  * verdict criteria (written before any run): new wins in >= 4 games, loses in
    <= 1 game, median of the non-zero deltas > +10 points. Wins/losses are also
    reported by levels, which is the robust signal on wall games.

usage:
    python scripts/paired_delta.py runs/lab_polygon_stock runs/lab_polygon_core
"""
from __future__ import annotations
import json, sys
from pathlib import Path

def load(d: str) -> dict[str, tuple[float, int, int]]:
    b = json.load(open(Path(d) / "benchmark.json", encoding="utf-8"))
    out = {}
    for r in b["game_runs"]:
        out[r["game_id"][:4]] = (float(r.get("final_score") or 0.0), int(r.get("levels_completed") or 0), int(r.get("number_of_levels") or 0))
    return out

def main() -> int:
    base, new = load(sys.argv[1]), load(sys.argv[2])
    games = sorted(set(base) & set(new))
    dead = [g for g in games if base[g][1] == 0 and new[g][1] == 0]
    ceil = [g for g in games if base[g][2] and base[g][1] >= base[g][2] and new[g][1] >= new[g][2]]
    keep = [g for g in games if g not in dead and g not in ceil]
    rows = []
    for g in keep:
        d = new[g][0] - base[g][0]; dl = new[g][1] - base[g][1]
        rows.append((g, base[g][0], new[g][0], d, base[g][1], new[g][1], dl))
    wins = [r for r in rows if r[3] > 0.5]; losses = [r for r in rows if r[3] < -0.5]
    lw = [r for r in rows if r[6] > 0]; ll = [r for r in rows if r[6] < 0]
    nz = sorted(r[3] for r in rows if abs(r[3]) > 0.5); med = nz[len(nz) // 2] if nz else 0.0
    print(f"{sys.argv[1]} -> {sys.argv[2]}: games {len(games)}, dead {len(dead)} {dead}, ceiling {len(ceil)} {ceil}, compared {len(keep)}")
    print(f"{'game':5s} {'base':>6s} {'new':>6s} {'delta':>7s}  {'lv base':>7s} {'lv new':>6s}")
    for r in sorted(rows, key=lambda r: -r[3]):
        print(f"{r[0]:5s} {r[1]:6.1f} {r[2]:6.1f} {r[3]:+7.1f}  {r[4]:7d} {r[5]:6d}")
    print(f"RHAE: wins {len(wins)}, losses {len(losses)}, median non-zero delta {med:+.1f}")
    print(f"levels: wins {len(lw)} {[r[0] for r in lw]}, losses {len(ll)} {[r[0] for r in ll]}")
    ok = len(wins) >= 4 and len(losses) <= 1 and med > 10
    print("VERDICT (>=4 wins, <=1 loss, median > +10):", "WIN" if ok else ("LOSS" if len(losses) >= 4 and len(wins) <= 1 else "NOISE"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

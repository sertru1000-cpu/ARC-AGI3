"""Delta table for the own-games testbed (Gemini round-7 protocol).

Compares two builds' benchmark.json (build A vs build B) over the 8
original games: levels completed and actions-vs-baseline per game
(aggregated over passes), plus a harness-feature census grepped from the
run logs. Per the protocol, only DELTAS are meaningful -- absolute
numbers on these games say nothing about the Kaggle score.

Usage:
  python scripts/owng_delta_table.py \
      --a runs/<A>/benchmark.json --b runs/<B>/benchmark.json \
      [--log-a <A.log> --log-b <B.log>]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# feature markers: label -> regex matching the harness's actual stdout
# lines (verified against tool_agent.py print() calls, 29.08). NOTE: the
# zombie gate has no stdout line -- it is invisible to this census.
LOG_MARKERS = {
    "probe ration nudge": r"probe ration nudge injected",
    "probe hard gate": r"probe hard gate blocked try_actions",
    "plan_real found plan": r"plan_real found a plan",
    "plan_real auto-exec": r"harness auto-executed the model-found plan_real",
    "proactive plan_real": r"auto-plan_real note injected|proactive plan_real found no plan",
    "level auto-replay": r"auto-replay note injected",
    "mechanic handoff": r"mechanic-handoff note injected",
    "hail mary": r"hail-mary",
    "rollbacks (model)": r"model called rollback\(",
    "rollbacks (forced)": r"force-rollback checkpoint injected",
    "sys_start anchors": r"auto-anchor created \(sys_start\)",
}


def load_runs(path: Path) -> dict[str, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    per_game: dict[str, list[dict]] = defaultdict(list)
    for run in data.get("game_runs", []):
        per_game[str(run.get("game_id"))].append(run)
    return per_game


def summarize(runs: list[dict]) -> dict:
    """Aggregate one game's passes: best + mean levels, actions on completed
    levels vs baseline (efficiency), wallclock, tokens."""
    if not runs:
        return {}
    levels = [int(r.get("levels_completed") or 0) for r in runs]
    eff = []       # per pass: sum(actions)/sum(baseline) over COMPLETED levels
    for r in runs:
        base = list(r.get("base_actions_per_level") or [])
        acts = list(r.get("actions_per_level") or [])
        done = int(r.get("levels_completed") or 0)
        if done and len(base) >= done and len(acts) >= done:
            b = sum(base[:done])
            a = sum(acts[:done])
            if b > 0 and a > 0:
                eff.append(a / b)
    return {
        "passes": len(runs),
        "levels_best": max(levels),
        "levels_mean": sum(levels) / len(levels),
        "n_levels": int(runs[0].get("number_of_levels") or 0),
        "eff": (sum(eff) / len(eff)) if eff else None,   # 1.0 = at baseline
        "wallclock_mean": sum(float(r.get("final_wallclock_seconds") or 0.0)
                              for r in runs) / len(runs),
        "tokens_mean": sum(int(r.get("final_generated_tokens") or 0)
                           for r in runs) / len(runs),
    }


def census(log_path: Path | None) -> dict[str, int]:
    if not log_path or not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return {label: len(re.findall(pattern, text, flags=re.IGNORECASE))
            for label, pattern in LOG_MARKERS.items()}


def fmt(v, spec=".2f", none="-"):
    return none if v is None else format(v, spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, type=Path, help="build A benchmark.json")
    ap.add_argument("--b", required=True, type=Path, help="build B benchmark.json")
    ap.add_argument("--log-a", type=Path, default=None)
    ap.add_argument("--log-b", type=Path, default=None)
    args = ap.parse_args()

    games_a = load_runs(args.a)
    games_b = load_runs(args.b)
    all_ids = sorted(set(games_a) | set(games_b))

    print(f"{'game':<16} {'lv best A/B':>12} {'lv mean A/B':>14} "
          f"{'eff A/B (1=base)':>18} {'min/game A/B':>14}")
    print("-" * 78)
    tot_a = tot_b = 0.0
    for gid in all_ids:
        sa = summarize(games_a.get(gid, []))
        sb = summarize(games_b.get(gid, []))
        tot_a += sa.get("levels_mean", 0.0) if sa else 0.0
        tot_b += sb.get("levels_mean", 0.0) if sb else 0.0
        name = gid.split("-")[0]
        n = (sa or sb).get("n_levels", "?")
        print(f"{name:<16} "
              f"{sa.get('levels_best', 0)}/{sb.get('levels_best', 0)} of {n:<4} "
              f"{fmt(sa.get('levels_mean'))}/{fmt(sb.get('levels_mean')):<8} "
              f"{fmt(sa.get('eff')):>8}/{fmt(sb.get('eff')):<9} "
              f"{fmt((sa.get('wallclock_mean') or 0)/60, '.0f'):>6}/{fmt((sb.get('wallclock_mean') or 0)/60, '.0f')}")
    print("-" * 78)
    print(f"{'TOTAL levels (mean over passes)':<34} A={tot_a:.2f}  B={tot_b:.2f}  "
          f"delta={tot_b - tot_a:+.2f}")

    ca, cb = census(args.log_a), census(args.log_b)
    if ca or cb:
        print("\nHarness-feature census (log line counts, A vs B):")
        for label in LOG_MARKERS:
            print(f"  {label:<22} {ca.get(label, 0):>6}  ->  {cb.get(label, 0):>6}")


if __name__ == "__main__":
    main()

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

Two rulers (29.08): benchmark.json's levels_completed is the STRICT
current-play state (rollbacks/resets zero it -- the 24-conc run read
all-zero on it while sys_level anchors showed 11 real level-ups). When
<exp_dir>/artifacts/*_events.jsonl sits next to the benchmark, the table
also computes OFFICIAL levels = max score over plays, which is what the
Kaggle metric uses. Official is the headline; strict is context.
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


def official_from_events(events_file: Path) -> tuple[int, int]:
    """(max levels over plays, number of plays) from an events jsonl.
    A score drop between consecutive action rows = a new play."""
    best = prev = 0
    plays = 1
    with events_file.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") != "action":
                continue
            score = int(d.get("score") or 0)
            if score < prev:
                plays += 1
            best = max(best, score)
            prev = score
    return best, plays


def official_per_game(benchmark_path: Path) -> dict[str, list[tuple[int, int]]]:
    """Scan <exp_dir>/artifacts/*_events.jsonl next to benchmark.json."""
    artifacts = benchmark_path.parent / "artifacts"
    per_game: dict[str, list[tuple[int, int]]] = defaultdict(list)
    if not artifacts.is_dir():
        return per_game
    for f in sorted(artifacts.glob("*_events.jsonl")):
        gid = f.name.split("_p")[0]
        per_game[gid].append(official_from_events(f))
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
    off_a = official_per_game(args.a)
    off_b = official_per_game(args.b)
    all_ids = sorted(set(games_a) | set(games_b))

    print(f"{'game':<8} {'OFFICIAL lv A':>16} {'OFFICIAL lv B':>16} "
          f"{'plays A/B':>10} {'strict A/B':>11} {'min A/B':>9}")
    print("-" * 78)
    tot_a = tot_b = 0
    for gid in all_ids:
        sa = summarize(games_a.get(gid, []))
        sb = summarize(games_b.get(gid, []))
        oa = off_a.get(gid, [])
        ob = off_b.get(gid, [])
        lv_a = [x[0] for x in oa] or [0]
        lv_b = [x[0] for x in ob] or [0]
        pl_a = sum(x[1] for x in oa)
        pl_b = sum(x[1] for x in ob)
        tot_a += sum(lv_a)
        tot_b += sum(lv_b)
        name = gid.split("-")[0]
        n = (sa or sb).get("n_levels", "?")
        print(f"{name:<8} "
              f"{'+'.join(map(str, sorted(lv_a, reverse=True))):>12} /{n:<3}"
              f"{'+'.join(map(str, sorted(lv_b, reverse=True))):>12} /{n:<3}"
              f"{pl_a:>5}/{pl_b:<5}"
              f"{fmt(sa.get('levels_mean')):>5}/{fmt(sb.get('levels_mean')):<6}"
              f"{fmt((sa.get('wallclock_mean') or 0)/60, '.0f'):>4}/{fmt((sb.get('wallclock_mean') or 0)/60, '.0f')}")
    print("-" * 78)
    print(f"{'TOTAL OFFICIAL levels (sum over passes)':<42} A={tot_a}  B={tot_b}  "
          f"delta={tot_b - tot_a:+d}")

    ca, cb = census(args.log_a), census(args.log_b)
    if ca or cb:
        print("\nHarness-feature census (log line counts, A vs B):")
        for label in LOG_MARKERS:
            print(f"  {label:<22} {ca.get(label, 0):>6}  ->  {cb.get(label, 0):>6}")


if __name__ == "__main__":
    main()

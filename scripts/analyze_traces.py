"""Trace analyzer: one-command report over a directory of *.jsonl traces.

Usage:
    .venv/Scripts/python.exe scripts/analyze_traces.py <dir-with-traces>
    .venv/Scripts/python.exe scripts/analyze_traces.py <dir> --game tn36 --dump-turns
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def game_report(name: str, recs: list[dict], dump_turns: bool = False) -> dict:
    turns = [r for r in recs if r.get("turn", 0) > 0]
    errors = [r for r in turns if r.get("sandbox_error")]
    err_types = Counter()
    for r in errors:
        m = re.findall(r"(\w+Error|Timeout\w*)", r["sandbox_error"] or "")
        err_types[m[-1] if m else "other"] += 1

    ts = [r["ts"] for r in turns]
    deltas = [b - a for a, b in zip(ts, ts[1:])]
    med_turn_s = sorted(deltas)[len(deltas) // 2] if deltas else 0

    # Level timeline: (turn, level) at each change.
    timeline = []
    lvl = 0
    for r in turns:
        if r.get("level", 0) != lvl:
            lvl = r.get("level", 0)
            timeline.append((r["turn"], lvl))

    wm = next((r.get("world_model") for r in reversed(turns) if r.get("world_model")), None)
    stalls = sum(1 for r in turns if r.get("actions_executed") == 0)
    acts = sum(r.get("actions_executed", 0) for r in turns)

    rep = {
        "game": name,
        "turns": len(turns),
        "llm_actions": acts,
        "max_level": max((r.get("level", 0) for r in turns), default=0),
        "level_timeline": timeline,
        "errors": len(errors),
        "error_types": dict(err_types),
        "no_action_turns": stalls,
        "median_turn_s": round(med_turn_s, 1),
        "final_world_model": wm,
        "no_code": sum(1 for r in turns if r.get("code") is None),
    }

    print(f"\n== {name} ==")
    print(f"  turns={rep['turns']}  llm_actions={acts}  max_level={rep['max_level']}"
          f"  errors={rep['errors']} {rep['error_types']}  no-code={rep['no_code']}")
    print(f"  median turn {rep['median_turn_s']}s, turns without actions: {stalls}"
          f" ({stalls * 100 // max(1, len(turns))}%)")
    if timeline:
        print(f"  level-ups at turns: {timeline}")
    if wm:
        print(f"  final WORLD_MODEL: {json.dumps(wm, ensure_ascii=False)[:300]}")
    if dump_turns:
        for r in turns:
            flag = "E" if r.get("sandbox_error") else (" " if r.get("actions_executed") else ".")
            print(f"   [{flag}] t{r['turn']:>3} a={r.get('actions_executed', 0):>3} "
                  f"lvl={r.get('level', 0)} | {(r.get('reply') or '')[:100].replace(chr(10), ' ')}")
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dir", help="directory containing *.jsonl traces")
    p.add_argument("--game", default=None, help="filter by game prefix")
    p.add_argument("--dump-turns", action="store_true", help="per-turn one-liners")
    args = p.parse_args()

    files = sorted(Path(args.dir).glob("**/*.jsonl"))
    if args.game:
        files = [f for f in files if f.name.startswith(args.game)]
    if not files:
        raise SystemExit(f"no traces found in {args.dir}")

    reports = [game_report(f.stem.split("-")[0], load(f), args.dump_turns) for f in files]

    total_lvls = sum(r["max_level"] for r in reports)
    total_errs = sum(r["errors"] for r in reports)
    total_turns = sum(r["turns"] for r in reports)
    print("\n===== TOTAL =====")
    print(f"  games={len(reports)}  levels={total_lvls}  turns={total_turns}"
          f"  errors={total_errs} ({total_errs * 100 // max(1, total_turns)}%)")


if __name__ == "__main__":
    main()

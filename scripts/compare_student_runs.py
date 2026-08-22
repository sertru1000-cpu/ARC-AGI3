"""Side-by-side loop metrics for student runs (the docs/student_vs_teacher_analysis.md table).

Usage:
    .venv/Scripts/python.exe scripts/compare_student_runs.py LABEL=dir [LABEL=dir ...]

Per game and run: turns, level, % zero-action turns, actions/turn, unique code
blocks, max consecutive identical code block, verify_theory calls, stall-force
turns, % turns with goal=unknown. Dirs are searched recursively for *.jsonl;
several traces of one game (stand reps) are averaged.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def metrics(recs: list[dict]) -> dict:
    turns = [r for r in recs if r.get("turn", 0) > 0 and "turn_failed" not in r]
    if not turns:
        return {}
    n = len(turns)
    acts = [r.get("actions_executed", 0) or 0 for r in turns]
    codes = [(r.get("code") or "").strip() for r in turns]
    max_rep = rep = 1
    for a, b in zip(codes, codes[1:]):
        rep = rep + 1 if a == b and a else 1
        max_rep = max(max_rep, rep)
    unk = sum(1 for r in turns
              if "unknown" in str((r.get("world_model") or {}).get("goal", "")).lower())
    return {
        "turns": n,
        "level": max(r.get("level", 0) for r in turns),
        "zero_act": sum(1 for a in acts if a == 0) / n,
        "act_per_turn": sum(acts) / n,
        "uniq_code": len(set(c for c in codes if c)),
        "max_rep": max_rep,
        "verify": sum(c.count("verify_theory(") for c in codes),
        "forced": sum(1 for r in turns if r.get("forced")),
        "unknown": unk / n,
        "actions": sum(acts),
    }


def main() -> None:
    runs = [a.split("=", 1) for a in sys.argv[1:]]
    table: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for label, d in runs:
        for p in Path(d).rglob("*.jsonl"):
            if p.name.startswith("_"):
                continue
            m = metrics([json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()])
            if m:
                table[p.stem.split("-")[0]][label].append(m)

    hdr = "| game | run | turns | lvl | 0-act | act/turn | uniq code | max rep | verify | forced | goal=unk |"
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    tot: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for game in sorted(table):
        for label, _ in runs:
            ms = table[game].get(label)
            if not ms:
                continue
            avg = {k: sum(m[k] for m in ms) / len(ms) for k in ms[0]}
            best = max(m["level"] for m in ms)
            tot[label][0] += best
            tot[label][1] += 1
            print(f"| {game} | {label} | {avg['turns']:.0f} | {best} | {avg['zero_act']:.0%} | "
                  f"{avg['act_per_turn']:.1f} | {avg['uniq_code']:.0f}/{avg['turns']:.0f} | "
                  f"{avg['max_rep']:.0f} | {avg['verify']:.0f} | {avg['forced']:.0f} | {avg['unknown']:.0%} |")
    for label, (lv, g) in tot.items():
        print(f"\n{label}: {lv} levels / {g} games")


if __name__ == "__main__":
    main()

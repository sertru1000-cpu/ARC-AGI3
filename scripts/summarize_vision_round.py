"""Summarize one or more teacher rounds from their trace dirs.

Per game: best level, turns, real actions spent, actions per level gained,
turn at which each level was reached, zero-action turns, error turns, lost
turns (API failures), and the pure-text Gemini-Pro ceiling from
docs/gemini_pro_stats.md for comparison.

Usage:
    .venv/Scripts/python.exe scripts/summarize_vision_round.py data/teacher/<dir> [more dirs] [--md out.md]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_pure_ceiling() -> dict[str, tuple[int, int]]:
    """game -> (pure best, +hint best) from docs/gemini_pro_stats.md."""
    out: dict[str, tuple[int, int]] = {}
    md = ROOT / "docs" / "gemini_pro_stats.md"
    if not md.exists():
        return out
    for line in md.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*([a-z0-9]{4})\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", line)
        if m:
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


def summarize_trace(path: Path) -> dict:
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    turns = [r for r in recs if r.get("turn", 0) > 0 and "turn_failed" not in r]
    failed = [r for r in recs if "turn_failed" in r]
    levels = [r.get("level") or 0 for r in turns]
    best = max(levels, default=0)
    reached: dict[int, int] = {}
    for r in turns:
        lv = r.get("level") or 0
        if lv > 0 and lv not in reached:
            reached[lv] = r["turn"]
    actions = sum(r.get("actions_executed") or 0 for r in turns)
    # actions spent up to the turn that reached the best level
    last_useful_turn = reached.get(best, 0)
    actions_to_best = sum((r.get("actions_executed") or 0) for r in turns
                          if r["turn"] <= last_useful_turn) if best else 0
    zero = sum(1 for r in turns if not (r.get("actions_executed") or 0))
    errors = sum(1 for r in turns if r.get("sandbox_error"))
    nocode = sum(1 for r in turns if not r.get("code"))
    first_ts = recs[0]["ts"] if recs else 0
    last_ts = recs[-1]["ts"] if recs else 0
    backend = next((r.get("backend") for r in recs if r.get("turn") == 0), None)
    return {
        "game": path.stem.split("-")[0], "file": path.name, "best": best,
        "win_levels": None, "turns": len(turns), "actions": actions,
        "actions_to_best": actions_to_best,
        "actions_per_level": round(actions_to_best / best, 1) if best else None,
        "reached": reached, "zero_action_turns": zero, "error_turns": errors,
        "nocode_turns": nocode, "lost_turns": len(failed),
        "minutes": round((last_ts - first_ts) / 60, 1), "backend": backend,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dirs", nargs="+")
    p.add_argument("--md", default=None, help="also write a markdown table here")
    args = p.parse_args()
    ceiling = load_pure_ceiling()

    rows: list[dict] = []
    for d in args.dirs:
        d = Path(d)
        cfg = {}
        if (d / "_config.json").exists():
            cfg = json.loads((d / "_config.json").read_text(encoding="utf-8"))
        results = {}
        if (d / "_results.jsonl").exists():
            for l in (d / "_results.jsonl").read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    results[(r["game"], r.get("rep", 0))] = r
        for f in sorted(d.glob("*.jsonl")):
            if f.name.startswith("_"):
                continue
            s = summarize_trace(f)
            s["round"] = d.name
            s["model"] = cfg.get("model")
            rep = int(f.stem.split("-r")[-1]) if "-r" in f.stem else 0
            res = results.get((s["game"], rep))
            if res:
                s["win_levels"] = res.get("win_levels")
                s["finished"] = True
                s["state"] = res.get("state")
            else:
                s["finished"] = False
            pc = ceiling.get(s["game"])
            s["pro_text_pure"] = pc[0] if pc else None
            s["pro_text_hint"] = pc[1] if pc else None
            rows.append(s)

    hdr = ("game", "model", "best", "win_levels", "pro_text_pure", "pro_text_hint", "turns",
           "actions", "actions_per_level", "reached", "zero_action_turns", "error_turns",
           "lost_turns", "minutes", "finished")
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for s in sorted(rows, key=lambda r: (r["round"], r["game"])):
        lines.append("| " + " | ".join(
            str(s.get(h) if s.get(h) is not None else "") for h in hdr) + " |")
    table = "\n".join(lines)
    print(table)
    total = sum(r["best"] for r in rows)
    improved = sum(1 for r in rows if r["pro_text_pure"] is not None and r["best"] > r["pro_text_pure"])
    print(f"\nepisodes {len(rows)}, levels total {total}, games above pure-text Pro ceiling: {improved}")
    if args.md:
        Path(args.md).write_text(table + "\n", encoding="utf-8")
        print("written", args.md)


if __name__ == "__main__":
    main()

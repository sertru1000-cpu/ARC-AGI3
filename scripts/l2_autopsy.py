"""Where does the agent die on level 2? Read a stock-lab run's event logs.

The question the 03.09 measurement was launched to answer. Reads
`artifacts/*_events.jsonl` (one file per game play) and reports, per game:
levels completed, actions and model calls spent on EACH level, how the play
ended, and the model's own words ([ASSISTANT] blocks) from its last analyses
on the deepest level reached -- what it believed when progress stopped.

Sources are the per-play event streams, never `summary.txt` or `per-level=`
counters (both known to lie, see scripts/rhae.py). Also counts `[atlas`
strings in the prompts: a stock run must show zero.

usage:
    python scripts/l2_autopsy.py runs/stocklab_v7_11games
    python scripts/l2_autopsy.py runs/stocklab_v7_11games --game ac01 --tail 1200
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

ASSISTANT = re.compile(r"\[ASSISTANT\]\n(.*?)(?=\n\[[A-Z ]+(?::|\])|\Z)", re.S)


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def per_game(events: list[dict]) -> dict:
    acts = [e for e in events if e.get("type") == "action"]
    anas = [e for e in events if e.get("type") == "analysis"]
    per_level_actions: dict[int, int] = defaultdict(int)
    per_level_calls: dict[int, int] = defaultdict(int)
    for e in acts:
        per_level_actions[e.get("level")] += 1
    for e in anas:
        per_level_calls[e.get("level")] += 1
    completed = sum(1 for e in acts if e.get("level_completed"))
    last = acts[-1] if acts else None
    end = "no actions"
    if last is not None:
        if last.get("game_over"):
            end = "game_over"
        elif last.get("run_complete"):
            end = "run_complete"
        elif last.get("state") == "WIN":
            end = "WIN"
        else:
            end = "stopped (time/gave_up)"
    deepest = max(per_level_actions) if per_level_actions else 1
    deep_anas = [e for e in anas if e.get("level") == deepest]
    said = []
    for e in deep_anas[-2:]:
        for m in ASSISTANT.finditer(e.get("transcript", "")):
            said.append(m.group(1).strip())
    atlas_marks = sum(e.get("transcript", "").count("[atlas") for e in anas)
    return dict(
        actions=len(acts), calls=len(anas), completed=completed, deepest=deepest,
        per_level_actions=dict(per_level_actions), per_level_calls=dict(per_level_calls),
        end=end, score=last.get("score") if last else 0,
        said="\n---\n".join(said), atlas_marks=atlas_marks,
        no_change=sum(1 for e in acts if e.get("board_changed") is False),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("--game", action="append")
    ap.add_argument("--tail", type=int, default=900, help="chars of the model's words to show")
    args = ap.parse_args()
    files = sorted(glob.glob(str(Path(args.run) / "artifacts" / "*_events.jsonl")))
    if not files:
        print("no artifacts/*_events.jsonl under", args.run)
        return 1
    rows = []
    for f in files:
        gid = Path(f).name.split("-")[0]
        if args.game and gid not in args.game:
            continue
        rows.append((gid, per_game(load(f))))

    print(f"{'игра':6}{'ур.':>4}{'глуб':>5}{'ходов':>7}{'вызов':>6}{'пустых':>7}  "
          f"{'ходы по уровням':28} {'вызовы по уровням':22} конец")
    for gid, g in rows:
        pla = " ".join(f"L{k}:{v}" for k, v in sorted(g["per_level_actions"].items()))
        plc = " ".join(f"L{k}:{v}" for k, v in sorted(g["per_level_calls"].items()))
        print(f"{gid:6}{g['completed']:>4}{g['deepest']:>5}{g['actions']:>7}{g['calls']:>6}"
              f"{g['no_change']:>7}  {pla:28} {plc:22} {g['end']}")

    marks = sum(g["atlas_marks"] for _, g in rows)
    print(f"\nстрок '[atlas' в промптах: {marks} ({'НЕ чистый сток!' if marks else 'сток чистый'})")
    total_l1 = sum(1 for _, g in rows if g["completed"] >= 1)
    total_l2 = sum(1 for _, g in rows if g["completed"] >= 2)
    print(f"игр {len(rows)}: взят L1 в {total_l1}, L2 в {total_l2}; "
          f"всего уровней {sum(g['completed'] for _, g in rows)}")

    print("\n=== что модель говорила на самом глубоком уровне ([ASSISTANT] двух последних анализов) ===")
    for gid, g in rows:
        if g["deepest"] < 2 and not args.game:
            continue
        print(f"\n--- {gid}: глубина L{g['deepest']}, на нём ходов {g['per_level_actions'].get(g['deepest'], 0)}, "
              f"вызовов {g['per_level_calls'].get(g['deepest'], 0)}, конец: {g['end']}")
        txt = g["said"]
        print(txt[-args.tail:] if txt else "(нет блоков [ASSISTANT])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

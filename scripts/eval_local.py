"""Local evaluation harness: per-level action counts + run journal.

Runs agent/my_agent.py against the public games and records, for every game,
how many actions each completed level took — the quantity RHAE scoring is
built on. Each run is appended to runs/ as a timestamped JSON so successive
agent versions can be compared.

Usage:
    .venv/Scripts/python.exe scripts/eval_local.py --tag explorer-v1
    .venv/Scripts/python.exe scripts/eval_local.py --game ls20,vc33 --max-steps 300
    .venv/Scripts/python.exe scripts/eval_local.py --compare            # last two runs
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

RUNS_DIR = ROOT / "runs"


def load_my_agent_class():
    spec = importlib.util.spec_from_file_location("user_agent_module", ROOT / "agent" / "my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyAgent


def per_level_actions(frames) -> list[int]:
    """Actions spent inside each completed level, from the frame trace.

    frames[0] is the seed frame; each later frame is the result of one action.
    A level is 'closed' when levels_completed increments between frames.
    """
    counts: list[int] = []
    spent = 0
    prev_level = 0
    for f in frames[1:]:
        spent += 1
        lvl = int(f.levels_completed or 0)
        if lvl > prev_level:
            counts.append(spent)
            spent = 0
            prev_level = lvl
    return counts


def run(args) -> None:
    import arc_agi
    from arc_agi import OperationMode

    logging.basicConfig(level=logging.WARNING)
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    all_ids = [e.game_id.split("-")[0] for e in arc.get_environments()]
    game_ids = all_ids
    if args.game:
        wanted = {g.strip() for g in args.game.split(",")}
        game_ids = [g for g in all_ids if g in wanted]

    MyAgentCls = load_my_agent_class()
    if args.max_steps:
        MyAgentCls.MAX_ACTIONS = args.max_steps

    results = []
    t_start = time.time()
    for i, gid in enumerate(game_ids, 1):
        env = arc.make(gid)
        if env is None:
            print(f"[{i}/{len(game_ids)}] {gid}: env creation failed, skipped")
            continue
        agent = MyAgentCls(
            card_id="local-eval", game_id=gid, agent_name=f"eval.{gid}",
            ROOT_URL="http://localhost", record=False, arc_env=env, tags=["local-eval"],
        )
        t0 = time.time()
        agent.main()
        final = agent.frames[-1]
        lvl_actions = per_level_actions(agent.frames)
        res = {
            "game": gid,
            "state": str(final.state).split(".")[-1],
            "levels": int(final.levels_completed or 0),
            "win_levels": int(final.win_levels or 0),
            "actions": agent.action_counter,
            "per_level_actions": lvl_actions,
            "seconds": round(time.time() - t0, 2),
        }
        results.append(res)
        lvls = f"{res['levels']}/{res['win_levels']}"
        print(f"[{i}/{len(game_ids)}] {gid}: levels {lvls:>5}  actions {res['actions']:>5}  "
              f"per-level {lvl_actions}  {res['state']}")

    total_levels = sum(r["levels"] for r in results)
    total_win = sum(r["win_levels"] for r in results)
    completion = sum(
        (r["levels"] / r["win_levels"]) if r["win_levels"] else 0 for r in results
    ) / max(len(results), 1)

    summary = {
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "max_steps": MyAgentCls.MAX_ACTIONS,
        "games": len(results),
        "total_levels": total_levels,
        "total_win_levels": total_win,
        "mean_completion": round(completion, 4),
        "wall_seconds": round(time.time() - t_start, 1),
        "results": results,
    }

    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n========= EVAL SUMMARY =========")
    print(f"tag={args.tag}  games={len(results)}  levels={total_levels}/{total_win}  "
          f"mean_completion={completion:.1%}  wall={summary['wall_seconds']}s")
    print(f"saved → {out.relative_to(ROOT)}")


def compare() -> None:
    files = sorted(RUNS_DIR.glob("*.json"))
    if len(files) < 2:
        print("Need at least two runs in runs/ to compare.")
        return
    a, b = (json.loads(f.read_text(encoding="utf-8")) for f in files[-2:])
    print(f"comparing {a['tag']} ({a['timestamp']})  →  {b['tag']} ({b['timestamp']})\n")
    ra = {r["game"]: r for r in a["results"]}
    rb = {r["game"]: r for r in b["results"]}
    for gid in sorted(set(ra) | set(rb)):
        la = ra.get(gid, {}).get("levels", "-")
        lb = rb.get(gid, {}).get("levels", "-")
        mark = "  " if la == lb else ("↑ " if str(lb) > str(la) else "↓ ")
        print(f"  {mark}{gid:8} levels {la} → {lb}")
    print(f"\n  total levels {a['total_levels']} → {b['total_levels']}   "
          f"mean completion {a['mean_completion']:.1%} → {b['mean_completion']:.1%}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default=None, help="comma-separated game ids (default: all)")
    p.add_argument("--max-steps", type=int, default=None, help="override MAX_ACTIONS")
    p.add_argument("--tag", default="dev", help="label stored in the run journal")
    p.add_argument("--compare", action="store_true", help="diff the two latest runs")
    args = p.parse_args()
    if args.compare:
        compare()
    else:
        run(args)


if __name__ == "__main__":
    main()

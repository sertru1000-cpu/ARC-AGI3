"""Experiment stand: many offline games CONCURRENTLY against one LLM server.

One invocation = one harness/model configuration (variants differ by env,
so run one process per variant). Games x reps fan out over a thread pool
(Duck-style: explicit pool, shared soft deadline), all hitting the same
OpenAI-compatible endpoint — vLLM batches the streams.

Usage (on a RunPod pod next to a vLLM server, or anywhere):
    export LLM_BASE_URL=http://127.0.0.1:1234/v1 LLM_MODEL=student LLM_API_KEY=x
    python scripts/run_stand.py --games lp85,sb26,wa30 --reps 3 --concurrency 9 \\
        --label student-v1 --max-turns 60 --game-seconds 2400

Results: data/stand/<label>_<stamp>/results.jsonl + per-game traces + summary.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", required=True, help="comma-separated ids")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--label", default="stand")
    p.add_argument("--max-actions", type=int, default=400)
    p.add_argument("--max-turns", type=int, default=60)
    p.add_argument("--game-seconds", type=float, default=2400)
    p.add_argument("--mutation", default="",
                   help="env transform: colors:<seed>|mirror_h|mirror_v")
    p.add_argument("--total-seconds", type=float, default=0,
                   help="shared soft deadline for the whole matrix (0 = off)")
    args = p.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "data" / "stand" / f"{args.label}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("AGENT_BRAIN", "llm")
    os.environ.setdefault("ONLY_RESET_LEVELS", "true")
    os.environ["MY_AGENT_PARALLEL"] = "1"
    os.environ["MY_AGENT_MAX_ACTIONS"] = str(args.max_actions)
    os.environ["MY_AGENT_MAX_TURNS"] = str(args.max_turns)
    os.environ["MY_AGENT_GAME_SECONDS"] = str(args.game_seconds)
    if args.total_seconds > 0:
        os.environ["MY_AGENT_TOTAL_SECONDS"] = str(args.total_seconds)
    os.environ["MY_AGENT_TRACE_DIR"] = str(out_dir / "traces")
    os.environ.setdefault("MY_AGENT_CROSS_MEMORY_PATH", str(out_dir / "cross_memory.json"))

    import arc_agi
    from arc_agi import OperationMode

    spec = importlib.util.spec_from_file_location(
        "user_agent_module", ROOT / "agent" / "my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE,
                         environments_dir=str(ROOT / "environment_files"))
    available = {e.game_id.split("-")[0]: e.game_id for e in arc.available_environments}

    jobs = [(g.strip(), r) for r in range(args.reps)
            for g in args.games.split(",") if g.strip()]
    results_path = out_dir / "results.jsonl"
    write_lock = threading.Lock()

    def play(job: tuple[str, int]) -> dict:
        gid, rep = job
        t0 = time.time()
        row = {"game": gid, "rep": rep, "label": args.label}
        try:
            env = arc.make(available.get(gid, gid))
            if args.mutation:
                from agent.harness.mutations import make_mutated_env
                env = make_mutated_env(env, args.mutation)
            agent = module.MyAgent(
                card_id="stand", game_id=gid, agent_name=f"{args.label}.{gid}.r{rep}",
                ROOT_URL="http://localhost", record=False, arc_env=env,
                tags=["stand", args.label])
            agent.main()
            f = agent.frames[-1]
            row.update(levels=int(f.levels_completed or 0),
                       win_levels=int(f.win_levels or 0),
                       actions=agent.action_counter,
                       turns=agent.policy.turns if agent.policy else 0,
                       state=str(f.state).split(".")[-1])
        except Exception as exc:  # noqa: BLE001
            row.update(error=repr(exc)[:300], levels=-1)
        row["seconds"] = round(time.time() - t0, 1)
        with write_lock:
            with results_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{args.label}] {gid} r{rep}: levels={row.get('levels')} "
              f"actions={row.get('actions')} ({row['seconds']}s)")
        return row

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(play, jobs))

    # Summary: per-game best + mean over reps.
    by_game: dict[str, list[int]] = {}
    for r in rows:
        by_game.setdefault(r["game"], []).append(max(0, r.get("levels", 0)))
    summary = {g: {"best": max(v), "mean": round(sum(v) / len(v), 2), "reps": len(v)}
               for g, v in sorted(by_game.items())}
    total = {"label": args.label, "episodes": len(rows),
             "levels_total": sum(max(0, r.get("levels", 0)) for r in rows),
             "errors": sum(1 for r in rows if r.get("levels", -1) < 0),
             "wall_seconds": round(time.time() - t0, 1),
             "per_game": summary}
    (out_dir / "summary.json").write_text(json.dumps(total, indent=1), encoding="utf-8")
    print(json.dumps(total, indent=1))


if __name__ == "__main__":
    main()

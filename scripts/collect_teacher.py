"""Collect teacher trajectories: a strong API model plays public games
through OUR harness, producing traces in exactly our training format.

The traces (WORLD_MODEL + code + sandbox feedback per turn) are the raw
material for distilling Qwen via LoRA SFT. Runs locally on CPU — the game
engine is light and the LLM lives behind an HTTP endpoint.

Setup (any OpenAI-compatible endpoint; Vertex via LLM_AUTH=gcloud-adc):
    export LLM_BASE_URL=https://openrouter.ai/api/v1
    export LLM_API_KEY=sk-or-...
    export LLM_MODEL=anthropic/claude-opus-5      # or any strong model

Usage:
    .venv/Scripts/python.exe scripts/collect_teacher.py --games ls20,vc33 --repeats 2
    .venv/Scripts/python.exe scripts/collect_teacher.py --all --repeats 1
    # multimodal teacher, 4 games at once:
    .venv/Scripts/python.exe scripts/collect_teacher.py --games sk48,tn36,bp35,cd82 \\
        --vision --parallel 4 --max-turns 200 --max-actions 800 --game-seconds 7200

Each (game, rep) writes its own trace file <game>-r<rep>.jsonl (so parallel
reps never interleave in one file) plus a running _results.jsonl, so a killed
run still leaves everything finished so far on disk.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
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

DATA_DIR = ROOT / "data" / "teacher"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", default=None, help="comma-separated ids")
    p.add_argument("--all", action="store_true", help="all 25 public games")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-actions", type=int, default=250)
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--game-seconds", type=float, default=900)
    p.add_argument("--mutation", default="",
                   help="env transform: colors:<seed>|mirror_h|mirror_v, '+'-combinable")
    p.add_argument("--vision", action="store_true",
                   help="attach the current board as a PNG to every LLM request "
                        "(MY_AGENT_VISION=1; traces stay text-only)")
    p.add_argument("--vision-scale", type=int, default=8,
                   help="nearest-neighbor upscale factor (8 -> 512x512)")
    p.add_argument("--parallel", type=int, default=1,
                   help="episodes played concurrently (API-bound, CPU engine is cheap)")
    p.add_argument("--label", default="", help="extra tag in the output dir name")
    p.add_argument("--hints-file", default="",
                   help="JSON of human hints (data/teacher/human_hints.json): sets "
                        "MY_AGENT_HINT_<ID> for every selected game that has one "
                        "(all batches merged; <ID>_v2 variants appended)")
    p.add_argument("--hint-games", default="",
                   help="comma-separated subset of games that get hints (default: all "
                        "selected games that have one)")
    args = p.parse_args()

    if not os.getenv("LLM_BASE_URL"):
        raise SystemExit("Set LLM_BASE_URL / LLM_MODEL (+ LLM_API_KEY or LLM_AUTH=gcloud-adc).")
    if not os.getenv("LLM_API_KEY") and os.getenv("LLM_AUTH") != "gcloud-adc":
        raise SystemExit("Set LLM_API_KEY, or LLM_AUTH=gcloud-adc for Vertex.")

    model = os.getenv("LLM_MODEL", "unknown").replace("/", "_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = ("_" + args.label if args.label else "") + ("_vision" if args.vision else "")
    out_dir = DATA_DIR / f"{model}{tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["AGENT_BRAIN"] = "llm"
    # Death must cost a level, not the whole game — same semantics the student
    # faces on Kaggle. setdefault so an explicit override still wins.
    os.environ.setdefault("ONLY_RESET_LEVELS", "true")
    # Soft harness for expert collection (user decision 21.08): never kill an
    # episode for the model's own mistakes -- a code error comes back as
    # [python error] text and the next turn can fix it; no-code replies get
    # the corrective nudge as many times as needed; lost turns (API hiccups)
    # are noted to the model and logged, not counted toward a kill. The
    # harness also never takes over (no stall-force probes) -- traces must be
    # the teacher's own decisions. All are setdefault: explicit env wins.
    os.environ.setdefault("MY_AGENT_MAX_NO_CODE_STRIKES", "1000")
    os.environ.setdefault("MY_AGENT_MAX_TURN_FAILURES", "50")
    os.environ.setdefault("MY_AGENT_STALL_FORCE", "1000")
    os.environ["MY_AGENT_MAX_ACTIONS"] = str(args.max_actions)
    os.environ["MY_AGENT_MAX_TURNS"] = str(args.max_turns)
    os.environ["MY_AGENT_GAME_SECONDS"] = str(args.game_seconds)
    os.environ["MY_AGENT_TRACE_DIR"] = str(out_dir)
    if args.vision:
        os.environ["MY_AGENT_VISION"] = "1"
        os.environ["MY_AGENT_VISION_SCALE"] = str(args.vision_scale)
    if args.parallel > 1:
        # Each episode gets the full wall window; don't divide it among games.
        os.environ["MY_AGENT_PARALLEL"] = "1"

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import arc_agi
    from arc_agi import OperationMode

    spec = importlib.util.spec_from_file_location("user_agent_module", ROOT / "agent" / "my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    all_ids = [e.game_id.split("-")[0] for e in arc.get_environments()]
    game_ids = all_ids if args.all else [
        g for g in all_ids if g in {x.strip() for x in (args.games or "").split(",")}
    ]
    if not game_ids:
        raise SystemExit("No games selected: use --games ls20,vc33 or --all")

    if args.hints_file:
        hints_raw = json.loads(Path(args.hints_file).read_text(encoding="utf-8"))
        merged: dict[str, str] = {}
        for batch, items in hints_raw.items():
            if not isinstance(items, dict):
                continue
            for key, text in items.items():
                base = key.split("_")[0].upper()
                merged[base] = (merged.get(base, "") + " " + text).strip()
        want = {g.strip().upper() for g in args.hint_games.split(",") if g.strip()} or                {g.upper() for g in game_ids}
        for gid in want:
            if gid in merged:
                os.environ.setdefault(f"MY_AGENT_HINT_{gid}", merged[gid])
        print("hints set for:", sorted(g for g in want if g in merged))

    config = {
        "model": os.getenv("LLM_MODEL"), "games": game_ids, "repeats": args.repeats,
        "max_actions": args.max_actions, "max_turns": args.max_turns,
        "game_seconds": args.game_seconds, "vision": args.vision,
        "vision_scale": args.vision_scale if args.vision else None,
        "parallel": args.parallel, "mutation": args.mutation or None,
        "hints": sorted(k for k in os.environ if k.startswith("MY_AGENT_HINT_")),
        "soft_harness": {k: os.environ[k] for k in (
            "MY_AGENT_MAX_NO_CODE_STRIKES", "MY_AGENT_MAX_TURN_FAILURES", "MY_AGENT_STALL_FORCE")},
        "started": stamp,
    }
    (out_dir / "_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results_path = out_dir / "_results.jsonl"
    lock = threading.Lock()

    def play(job: tuple[str, int]) -> dict | None:
        gid, rep = job
        env = arc.make(gid)
        if env is None:
            return None
        if args.mutation:
            from agent.harness.mutations import make_mutated_env
            env = make_mutated_env(env, args.mutation)
        agent = module.MyAgent(
            # "-r<rep>" suffix -> separate trace file per rep; my_agent only
            # uses the part before '-' for hints/logs.
            card_id="teacher", game_id=f"{gid}-r{rep}", agent_name=f"teacher.{gid}.r{rep}",
            ROOT_URL="http://localhost", record=False, arc_env=env,
            tags=["teacher"],
        )
        t0 = time.time()
        try:
            agent.main()
        except Exception as exc:
            print(f"[{gid} r{rep}] CRASHED: {exc!r}")
            res = {"game": gid, "rep": rep, "levels": -1, "error": repr(exc)[:300],
                   "seconds": round(time.time() - t0, 1)}
        else:
            f = agent.frames[-1]
            res = {
                "game": gid, "rep": rep,
                "levels": int(f.levels_completed or 0),
                "win_levels": int(f.win_levels or 0),
                "actions": agent.action_counter,
                "state": str(f.state).split(".")[-1],
                "seconds": round(time.time() - t0, 1),
                "turns": agent.policy.turns if agent.policy else 0,
            }
            print(f"[{gid} r{rep}] levels={res['levels']}/{res['win_levels']} "
                  f"actions={res['actions']} turns={res['turns']} {res['state']} "
                  f"({res['seconds']}s)")
        with lock:
            with results_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
        return res

    jobs = [(gid, rep) for rep in range(args.repeats) for gid in game_ids]
    if args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            results = [r for r in pool.map(play, jobs) if r]
    else:
        results = [r for r in map(play, jobs) if r]

    usage = None
    try:
        from agent.harness.llm import OpenAICompatLLM
        usage = {"prompt_tokens": OpenAICompatLLM.total_prompt_tokens,
                 "completion_tokens": OpenAICompatLLM.total_completion_tokens}
        print(f"Token usage: {usage['prompt_tokens']:,} in / {usage['completion_tokens']:,} out")
    except Exception:
        pass
    (out_dir / "_summary.json").write_text(
        json.dumps({"model": os.getenv("LLM_MODEL"), "config": config,
                    "results": results, "usage": usage}, indent=2),
        encoding="utf-8",
    )
    won_levels = sum(max(0, r["levels"]) for r in results)
    print(f"\nTeacher run done: {len(results)} episodes, {won_levels} levels, traces -> {out_dir}")


if __name__ == "__main__":
    main()

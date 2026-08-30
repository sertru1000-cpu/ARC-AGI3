"""Mock-LLM stress run of the full harness on the local corpus (backlog 19.3/19.4).

Runs the REAL inference harness (tool_agent + taaf local engine) against
our_games_gen / our_games with a scripted OpenAI-compatible mock server in
place of vllm. No GPU, no quota, no code changes to the battle harness.

What it is for:
  - infrastructure bugs surfaced at scale (hundreds of levels, real threads)
  - false/missed firings of the automation: proactive plan_real, draft
    speedrun, entropy cull path, mechanic handoff, auto-replay, hail-mary
  - regression smoke before every battle push (19.4): run this, diff census

Usage (from repo root):
  uv run python scripts/mock_llm_stress.py --n-games 12 --minutes 5 \
      --policy mix --concurrent-jobs 12 --llm-gate 4 --astar

Census + tracebacks land in runs/mock_stress/<run-name>/stress_summary.json
and the full combined log in harness_stdout.log.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy", default="mix",
                   choices=["wander", "spam", "batch", "reset", "chaos", "mix"])
    p.add_argument("--n-games", type=int, default=12,
                   help="How many corpus games to sample (each has 5 levels).")
    p.add_argument("--games", default="",
                   help="Explicit comma-separated game ids (overrides --n-games sampling).")
    p.add_argument("--env-dir", default=str(ROOT / "our_games_gen"),
                   help="Environments dir (our_games_gen or our_games).")
    p.add_argument("--minutes", type=float, default=5.0, help="Wall cap per game.")
    p.add_argument("--max-actions", type=int, default=250, help="Action cap per game.")
    p.add_argument("--concurrent-jobs", type=int, default=12)
    p.add_argument("--llm-gate", type=int, default=4,
                   help="ATLAS_LLM_MAX_CONCURRENT_REQUESTS (0 = no gate). "
                        "Below concurrent-jobs => scarcity, exercises the zombie gate.")
    p.add_argument("--astar", action="store_true",
                   help="Enable the A* heuristic (runs/search_lab/h_model.pkl).")
    p.add_argument("--delay-ms", type=float, default=150.0,
                   help="Mock inference latency; nonzero lets proactive search hide behind it.")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--port", type=int, default=8399)
    p.add_argument("--run-name", default="")
    p.add_argument("--cpus", type=int, default=0,
                   help="Pin the process (and children) to the first N logical CPUs "
                        "via the Windows affinity mask -- emulates the ~4-vCPU Kaggle kernel. 0 = no limit.")
    p.add_argument("--l1-nodes", type=int, default=0,
                   help="Override proactive L1 search max_nodes (0 = keep battle value).")
    p.add_argument("--deep-nodes", type=int, default=0,
                   help="Override deep/proactive L2+ search max_nodes (0 = keep battle value).")
    return p.parse_args()


def _limit_cpus(n: int) -> None:
    import ctypes
    mask = (1 << n) - 1
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.kernel32.SetProcessAffinityMask(handle, mask):
        raise OSError("SetProcessAffinityMask failed")
    print(f"CPU affinity limited to {n} logical core(s) (mask={mask:#x}); "
          "child processes inherit it", flush=True)


def _sample_games(env_dir: Path, n: int, seed: int) -> list[str]:
    ids = []
    for meta in sorted(env_dir.glob("*/*/metadata.json")):
        try:
            ids.append(json.loads(meta.read_text(encoding="utf-8"))["game_id"])
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    if not ids:
        raise SystemExit(f"no games found under {env_dir}")
    random.Random(seed).shuffle(ids)
    return ids[: max(1, n)]


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


_CENSUS_PATTERNS = {
    "astar_loaded": r"atlas: A\* heuristic loaded",
    "astar_unavailable": r"atlas: A\* heuristic UNAVAILABLE",
    "plan_real_found": r"atlas: plan_real found a plan",
    "plan_real_none": r"atlas: plan_real found no plan",
    "proactive_none_on_entry": r"atlas: proactive plan_real found no plan on level entry",
    # proactive plans are executed by the harness and reported to the model
    # via this one-shot note -- this IS the "plan auto-executed" counter
    "auto_plan_exec": r"atlas: auto-plan_real note injected",
    "entropy_dead": r"atlas: entropy-dead detected",
    "zombie_gate": r"atlas: zombie gate engaged",
    "auto_replay_note": r"atlas: auto-replay note injected",
    "handoff_note": r"atlas: mechanic-handoff note injected",
    "hail_mary_exec": r"atlas: hail-mary plan executed",
    "hail_mary_none": r"atlas: hail-mary search found nothing",
    "speedrun_fired": r"atlas: draft-speedrun firing",
    "speedrun_aborted": r"atlas: draft-speedrun aborted",
    "action_effect": r"atlas: action-effect summary injected",
    "tracebacks": r"Traceback \(most recent call last",
}


def _census(text: str) -> dict:
    out = {k: len(re.findall(pat, text)) for k, pat in _CENSUS_PATTERNS.items()}
    # Unique exception headlines right after tracebacks, for the report.
    errors: dict[str, int] = {}
    for m in re.finditer(
        r"Traceback \(most recent call last[\s\S]{0,4000}?\n(\w[\w.]*(?:Error|Exception)[^\n]*)",
        text,
    ):
        line = m.group(1).strip()[:160]
        errors[line] = errors.get(line, 0) + 1
    out["unique_errors"] = errors
    return out


def _plays_from_events(path: Path) -> list[list[int]]:
    plays: list[list[int]] = []
    cur: list[int] = []
    prev = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if row.get("type") != "action":
            continue
        score = int(row.get("score", 0) or 0)
        if prev is not None and score < prev:
            plays.append(cur)
            cur = []
        cur.append(score)
        prev = score
    if cur:
        plays.append(cur)
    return plays


def _levels_census(run_dir: Path) -> dict:
    per_game: dict[str, int] = {}
    for f in sorted(run_dir.glob("artifacts/*_events.jsonl")):
        gid = f.name.split("_p")[0]
        best = 0
        for play in _plays_from_events(f):
            if play:
                best = max(best, max(play))
        per_game[gid] = max(per_game.get(gid, 0), best)
    return {
        "games_with_levels": sum(1 for v in per_game.values() if v > 0),
        "total_levels": sum(per_game.values()),
        "per_game": per_game,
    }


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("PYTHONUTF8", "1")

    base_url = f"http://127.0.0.1:{args.port}/v1"
    env = {
        "OPENAI_BASE_URL": base_url,
        "LOCAL_ANALYZER_BASE_URL": base_url,
        "OPENAI_PROVIDER": "vllm",
        "LOCAL_ANALYZER_PROVIDER": "vllm",
        "OPENAI_API_KEY": "mock",
        "LOCAL_ANALYZER_API_KEY": "mock",
        "MOCK_POLICY": args.policy,
        "MOCK_SEED": str(args.seed),
        "MOCK_DELAY_MS": str(args.delay_ms),
        "ATLAS_LLM_MAX_CONCURRENT_REQUESTS": str(max(0, args.llm_gate)),
    }
    if args.astar:
        model = ROOT / "runs" / "search_lab" / "h_model.pkl"
        if not model.is_file():
            raise SystemExit(f"--astar requested but {model} is missing")
        env["ATLAS_ASTAR_MODEL"] = str(model)
    os.environ.update(env)

    # Import order matters: tool_agent reads the ATLAS_* env at module import.
    sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "tufa-arc-agi-framework" / "src"))
    sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "ARC3-Inference"))
    from mock_llm_server import STATS, serve_in_thread  # noqa: E402  (scripts/ on path via __file__)
    from inference.framework import run as harness_run  # noqa: E402

    if args.cpus > 0:
        _limit_cpus(args.cpus)
    if args.l1_nodes > 0 or args.deep_nodes > 0:
        from inference.agent import tool_agent as _ta  # noqa: E402
        if args.l1_nodes > 0:
            _ta._ATLAS_PROACTIVE_L1_BUDGET = dict(_ta._ATLAS_PROACTIVE_L1_BUDGET, max_nodes=args.l1_nodes)
        if args.deep_nodes > 0:
            _ta._ATLAS_PROACTIVE_DEEP_BUDGET = dict(_ta._ATLAS_PROACTIVE_DEEP_BUDGET, max_nodes=args.deep_nodes)
        print(f"search budgets overridden: L1={_ta._ATLAS_PROACTIVE_L1_BUDGET} "
              f"DEEP={_ta._ATLAS_PROACTIVE_DEEP_BUDGET}", flush=True)

    env_dir = Path(args.env_dir)
    games = ([g.strip() for g in args.games.split(",") if g.strip()]
             if args.games else _sample_games(env_dir, args.n_games, args.seed))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"mock-{args.policy}-{stamp}"
    experiments_dir = ROOT / "runs" / "mock_stress"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    server, _ = serve_in_thread(args.port)
    print(f"mock LLM up at {base_url} (policy={args.policy}, seed={args.seed}, "
          f"delay={args.delay_ms}ms, gate={args.llm_gate})", flush=True)
    print(f"games ({len(games)}): {', '.join(games)}", flush=True)

    log_path = experiments_dir / f"{run_name}_stdout.log"
    argv = [
        "inference-taaf-run",
        "--agent", "inference",
        "--game", ",".join(games),
        "--environments-dir", str(env_dir),
        "--model", f"mock-{args.policy}",
        "--run-name", run_name,
        "--experiments-dir", str(experiments_dir),
        "--concurrent-jobs", str(args.concurrent_jobs),
        "--max-runtime-minutes", str(args.minutes),
        "--max-actions", str(args.max_actions),
        "--timeout", "60",
    ]
    print("argv:", " ".join(argv), flush=True)

    old_argv, old_out, old_err = sys.argv, sys.stdout, sys.stderr
    started = time.time()
    exit_code = 0
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_f:
        sys.argv = argv
        sys.stdout = _Tee(old_out, log_f)
        sys.stderr = _Tee(old_err, log_f)
        try:
            harness_run.main()
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
        except Exception:
            import traceback
            traceback.print_exc()
            exit_code = 1
        finally:
            sys.argv, sys.stdout, sys.stderr = old_argv, old_out, old_err
            server.shutdown()
    wall = time.time() - started

    text = log_path.read_text(encoding="utf-8", errors="replace")
    census = _census(text)

    # The runner puts each run in its own dir under experiments_dir.
    run_dirs = [d for d in experiments_dir.iterdir() if d.is_dir() and run_name in d.name]
    run_dir = max(run_dirs, key=lambda d: d.stat().st_mtime) if run_dirs else None
    levels = _levels_census(run_dir) if run_dir else {"note": "run dir not found"}

    summary = {
        "run_name": run_name,
        "policy": args.policy,
        "seed": args.seed,
        "games": games,
        "wall_seconds": round(wall, 1),
        "exit_code": exit_code,
        "mock_requests": STATS.requests,
        "mock_by_kind": STATS.by_kind,
        "census": census,
        "levels": levels,
        "env": {k: env[k] for k in sorted(env)},
    }
    out_path = (run_dir or experiments_dir) / "stress_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ MOCK-LLM STRESS SUMMARY ================")
    print(f"run: {run_name}  wall={wall:.0f}s  exit={exit_code}")
    print(f"mock requests: {STATS.requests}  by kind: {STATS.by_kind}")
    print(f"levels: {levels.get('total_levels')} across "
          f"{levels.get('games_with_levels')}/{len(games)} games")
    for key in _CENSUS_PATTERNS:
        print(f"  {key}: {census.get(key)}")
    if census.get("unique_errors"):
        print("errors:")
        for line, n in sorted(census["unique_errors"].items(), key=lambda kv: -kv[1]):
            print(f"  {n}x {line}")
    print(f"summary -> {out_path}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()

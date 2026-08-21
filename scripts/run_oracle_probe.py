"""Oracle-probe diagnostic (20.08): replay a known-good teacher opening
(real code from a successful teacher trace) against a fresh env, then hand
control to OUR OWN LLM policy for the remaining budget.

Tells apart two very different failure modes that both look like "0 levels":
  - the model can't find its footing (exploration/discovery problem) -- in
    which case, once handed a correct grounding, it should continue sanely
  - the model can't reason its way to a solution even when grounded
    correctly -- in which case it should flail just the same as usual

Does NOT touch agent/my_agent.py (the actual submission file). Reuses
Sandbox/LLMPolicy/Agent env-stepping directly, standalone -- same pattern as
scripts/run_stand.py and the scripts/test_*.py smoke tests.

Usage (against a live vLLM server, same env vars as run_stand.py):
  export LLM_BASE_URL=http://127.0.0.1:1234/v1 LLM_MODEL=base LLM_API_KEY=x
  python scripts/run_oracle_probe.py --game sb26 \\
      --teacher-trace data/teacher/sb26.jsonl \\
      --oracle-turns 6 --max-turns 100 --label oracle-sb26
  # (stand-kit ships data/teacher/{sb26,ft09,m0r0}.jsonl -- real teacher
  # traces confirmed to reach a level-up, picked 20.08)

Also works with AGENT_BRAIN=llm unset -- MockLLM smoke test with no server.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time as _time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

os.environ.setdefault("AGENT_BRAIN", "llm")


def load_oracle_code(trace_path: Path, n_turns: int) -> list[str]:
    """First N turns' real code blocks from a teacher trace (turn 0 is the
    opening-probe summary, not code -- skip it; skip turns with no code,
    e.g. no-code-strike retries, since replaying those teaches nothing)."""
    codes: list[str] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("turn", 0) == 0:
            continue
        code = rec.get("code")
        if code:
            codes.append(code)
        if len(codes) >= n_turns:
            break
    return codes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True)
    p.add_argument("--teacher-trace", required=True, type=Path)
    p.add_argument("--oracle-turns", type=int, default=6,
                   help="how many of the teacher's opening turns to replay verbatim")
    p.add_argument("--max-turns", type=int, default=100,
                   help="LLM turn budget AFTER the oracle handoff")
    p.add_argument("--max-actions", type=int, default=400)
    p.add_argument("--game-seconds", type=float, default=2400)
    p.add_argument("--label", default="oracle-probe")
    p.add_argument("--dry-run", action="store_true",
                   help="force MockLLM regardless of .env -- for smoke-testing "
                   "the plumbing without spending a single real API call")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "oracle",
                   help="where to persist the result JSON + turn trace -- "
                   "console output alone is easy to lose (SSH drops, closed "
                   "terminals), this is the durable record")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _time.strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"{args.label}_{args.game}_{stamp}.json"
    os.environ.setdefault("MY_AGENT_TRACE_DIR", str(args.out_dir / f"{args.label}_{args.game}_{stamp}_traces"))

    def _save(result: dict) -> None:
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"result saved -> {out_path}")

    import arc_agi
    from arc_agi import OperationMode
    from arcengine import GameState

    spec = importlib.util.spec_from_file_location("user_agent_module", ROOT / "agent" / "my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from agent.harness.sandbox import Sandbox
    from agent.harness.llm_policy import LLMPolicy
    from agent.harness.llm import default_backend

    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE,
                         environments_dir=str(ROOT / "environment_files"))
    available = {e.game_id.split("-")[0]: e.game_id for e in arc.available_environments}
    env = arc.make(available.get(args.game, args.game))

    agent = module.MyAgent(
        card_id="oracle", game_id=args.game, agent_name=f"{args.label}.{args.game}",
        ROOT_URL="http://localhost", record=False, arc_env=env, tags=["oracle-probe"],
    )
    agent.MAX_ACTIONS = args.max_actions

    sandbox = Sandbox(env_step=agent._env_step, budget_left=agent._budget_left)
    first = agent._env_step("RESET", None)
    sandbox.update_frame(first)
    win_levels = int(first.win_levels or 0)

    oracle_codes = load_oracle_code(args.teacher_trace, args.oracle_turns)
    print(f"loaded {len(oracle_codes)} oracle turns from {args.teacher_trace}")
    if not oracle_codes:
        raise SystemExit(
            f"ERROR: 0 oracle turns loaded from {args.teacher_trace} -- this trace "
            "file has no turn>0 code (empty/broken run, e.g. the episode never "
            "started). Pick a different trace file for --game (check `wc -l` and "
            "grep for a real max level first) instead of burning a paid LLM run "
            "on a plain from-scratch episode with no oracle grounding at all."
        )

    replayed = 0
    won_during_replay = False
    for code in oracle_codes:
        if agent._budget_left() <= 0:
            break
        res = sandbox.run_code(code)
        replayed += 1
        level = sandbox.current.level if sandbox.current else "?"
        print(f"  oracle turn {replayed}: actions={res.actions_executed} "
              f"level={level} error={bool(res.error)}")
        if res.interrupted == "WIN":
            won_during_replay = True
            break

    print(f"oracle replay done: {replayed} turns, "
          f"level={sandbox.current.level if sandbox.current else 0}, "
          f"budget_left={agent._budget_left()}")

    result = {
        "game": args.game, "label": args.label, "oracle_turns": replayed,
        "oracle_level": sandbox.current.level if sandbox.current else 0,
    }

    if won_during_replay:
        print("oracle replay already won the game before any LLM handoff -- "
              "pick a smaller --oracle-turns to leave something for the model.")
        result.update(llm_turns=0, final_level=win_levels, win=True, note="won_during_replay")
        print(json.dumps(result, indent=2))
        _save(result)
        return

    if args.dry_run:
        from agent.harness.llm import MockLLM
        backend = MockLLM(["```python\nprint('dry-run: no real LLM call was made')\n```"])
        print("--dry-run: forcing MockLLM, no real API call will be made")
    else:
        backend = default_backend()
        print(f"backend: {backend.name}")
    policy = LLMPolicy(
        backend=backend, sandbox=sandbox, game_id=args.game,
        win_levels=win_levels, max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "8192")),
    )
    cross_note = (
        f"[oracle note] An expert opening ({replayed} turns) already explored this "
        "game and correctly identified its controls and objects -- see history above "
        "for what that revealed. You are taking over from here. Build on it; don't "
        "restart blind exploration from scratch."
    )
    policy.start("", cross_note)
    agent.policy = policy

    turns = 0
    t0 = _time.time()
    while (agent.frames[-1].state is not GameState.WIN
           and agent._budget_left() > 0
           and turns < args.max_turns
           and (_time.time() - t0) < args.game_seconds):
        if agent.frames[-1].state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            sandbox.update_frame(agent._env_step("RESET", None))
        try:
            info = policy.play_turn()
        except Exception as exc:
            print(f"turn failed: {exc!r}")
            break
        turns += 1
        level = sandbox.current.level if sandbox.current else 0
        print(f"  llm turn {turns}: actions={info['actions']} level={level}/{win_levels} "
              f"err={bool(info['error'])}")
        # Save after every turn, not just at the end -- an interrupted
        # terminal (today's recurring failure mode) must not lose progress.
        _save({**result, "llm_turns": turns, "final_level": level, "win": False, "in_progress": True})
        if info["win"]:
            break

    result.update({
        "llm_turns": turns,
        "final_level": sandbox.current.level if sandbox.current else 0,
        "win": agent.frames[-1].state is GameState.WIN,
        "total_actions": agent.action_counter,
        "seconds": round(_time.time() - t0, 1),
    })
    print(json.dumps(result, indent=2))
    _save(result)


if __name__ == "__main__":
    main()

"""Thread-safety smoke: N offline games concurrently on the Mock brain.

Validates the parallel Phase B path locally (no GPU, no API):
threads share the cross-game journal and the global time budget; the run
must finish without exceptions, journal must stay valid JSON, and every
game must record turns.

Usage:
    .venv/Scripts/python.exe scripts/test_parallel.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

os.environ.update({
    "AGENT_BRAIN": "llm",           # MockLLM engages when no LLM_BASE_URL
    "LLM_BASE_URL": "",
    "MY_AGENT_PARALLEL": "1",
    "MY_AGENT_MAX_TURNS": "6",
    "MY_AGENT_MAX_ACTIONS": "40",
    "MY_AGENT_GAME_SECONDS": "120",
    "MY_AGENT_TOTAL_SECONDS": "120",
    "MY_AGENT_EXPECTED_GAMES": "3",
    "MY_AGENT_CROSS_MEMORY_PATH": str(ROOT / "traces_test" / "cross_parallel.json"),
    "MY_AGENT_TRACE_DIR": str(ROOT / "traces_test" / "parallel"),
    "ONLY_RESET_LEVELS": "true",
})

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402

spec = importlib.util.spec_from_file_location("user_agent_module", ROOT / "agent" / "my_agent.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

GAMES = ["ls20", "vc33", "ft09"]

arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE,
                     environments_dir=str(ROOT / "environment_files"))

errors: list[str] = []
results: dict[str, int] = {}


def play(gid: str) -> None:
    try:
        env = arc.make(gid)
        agent = module.MyAgent(card_id="par", game_id=gid, agent_name=f"par.{gid}",
                               ROOT_URL="http://localhost", record=False, arc_env=env,
                               tags=["parallel-smoke"])
        agent.main()
        results[gid] = agent.policy.turns if agent.policy else -1
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{gid}: {exc!r}")


t0 = time.time()
threads = [threading.Thread(target=play, args=(g,), daemon=True) for g in GAMES]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=300)

print(f"elapsed {time.time()-t0:.1f}s, results: {results}")
assert not errors, errors
assert len(results) == len(GAMES), f"missing games: {set(GAMES)-set(results)}"
assert all(v > 0 for v in results.values()), results

journal = json.loads(Path(os.environ["MY_AGENT_CROSS_MEMORY_PATH"]).read_text(encoding="utf-8"))
assert journal.get("games", 0) >= len(GAMES), journal
print("journal valid:", {k: journal[k] for k in ("games", "levels")})
print("PARALLEL SMOKE PASSED")

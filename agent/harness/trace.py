"""JSONL trace of every LLM turn — the raw material for prompt iteration.

One file per game. Each line is one record: the model's reply, the code we
extracted, what the sandbox did, and where the game stood afterwards.
Enabled by env MY_AGENT_TRACE_DIR; silently off otherwise.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


class TraceWriter:
    def __init__(self, game_id: str):
        trace_dir = os.getenv("MY_AGENT_TRACE_DIR", "").strip()
        self.path: Path | None = None
        if trace_dir:
            d = Path(trace_dir)
            d.mkdir(parents=True, exist_ok=True)
            safe = game_id.replace("/", "_")
            self.path = d / f"{safe}.jsonl"

    def write(self, record: dict) -> None:
        if self.path is None:
            return
        record = {"ts": round(time.time(), 2), **record}
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass  # tracing must never kill a run

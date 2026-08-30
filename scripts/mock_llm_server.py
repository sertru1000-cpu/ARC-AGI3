"""OpenAI-compatible mock LLM server for harness stress runs (backlog 19.3).

Serves POST /v1/chat/completions with scripted tool calls instead of a real
model. The point is NOT to be smart: the scripted policies exercise the
harness automation (proactive plan_real, zombie/entropy cull, mechanic
handoff, draft speedrun, auto-replay, noop guard) and its error paths on
hundreds of local levels with zero GPU and zero quota.

Policies (env MOCK_POLICY, or per-request via the model name suffix
"mock-<policy>"):
  wander  random valid action each turn (MOUSE gets random in-frame coords)
  spam    the same first valid action every turn (entropy-death by design:
          the cull MUST fire on unsolvable-by-spam levels)
  batch   6 random actions per turn in one action([...]) call
  reset   wander + RESET every 20th turn, double RESET every 60th
          (stresses full-restart bookkeeping + auto-replay + speedrun paths)
  chaos   wander 70%; 10% text-only reply (no tool call); 10% arguments
          that are not valid JSON; 10% code that raises at runtime
  mix     rotates wander/spam/batch/reset/chaos by turn index (default)

Deterministic: the per-turn seed is derived from MOCK_SEED + the number of
assistant messages already in the conversation, so reruns reproduce.

Run standalone:  python scripts/mock_llm_server.py --port 8399
Or import and call serve_in_thread(port) from an orchestrator.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_POLICIES = ("wander", "spam", "batch", "reset", "chaos", "mix")

# Generic action snippet: introspects the sandbox (valid_actions, frame
# dims) instead of hardcoding per-game knowledge, so one template covers
# every game in the corpus including MOUSE-gated ones.
_SNIPPET = """\
import random
random.seed({seed})
va = [str(a) for a in (valid_actions or [])]
moves = [a for a in va if a != "RESET"] or (va or ["ACTION1"])
try:
    g = getattr(current_frame, "grid", current_frame)
    H = len(g); W = len(g[0])
except Exception:
    H = W = 8
picks = []
for _ in range({count}):
    a = random.choice(moves)
    if a == "MOUSE":
        picks.append({{"action": "MOUSE", "row": random.randrange(H), "col": random.randrange(W)}})
    else:
        picks.append(a)
action(picks)
"""

_SPAM_SNIPPET = """\
va = [str(a) for a in (valid_actions or [])]
moves = [a for a in va if a not in ("RESET", "MOUSE")] or (va or ["ACTION1"])
action([moves[0]])
"""

_RESET_SNIPPET = 'action(["RESET"])\n'
_DOUBLE_RESET_SNIPPET = 'action(["RESET"])\naction(["RESET"])\n'
_RAISING_SNIPPET = "x = 1 / 0  # mock chaos: deliberate runtime error\n"


class MockStats:
    """Thread-safe counters, readable by the orchestrator after the run."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests = 0
        self.by_kind: dict[str, int] = {}

    def bump(self, kind: str) -> None:
        with self.lock:
            self.requests += 1
            self.by_kind[kind] = self.by_kind.get(kind, 0) + 1


STATS = MockStats()


def _turn_index(messages: list[dict]) -> int:
    return sum(1 for m in messages if m.get("role") == "assistant")


def _pick_policy(policy: str, turn: int) -> str:
    if policy != "mix":
        return policy
    return ("wander", "wander", "batch", "spam", "wander", "reset", "chaos")[turn % 7]


def build_reply(payload: dict, *, policy: str, seed: int) -> tuple[dict, str]:
    """Return (message dict, kind label) for one /chat/completions call."""
    messages = payload.get("messages") or []
    turn = _turn_index(messages)
    rng = random.Random(seed * 1_000_003 + turn)
    eff = _pick_policy(policy, turn)
    kind = eff
    code: str | None = None
    content = ""
    bad_args = False

    if eff == "wander":
        code = _SNIPPET.format(seed=rng.randrange(10**9), count=1)
    elif eff == "batch":
        code = _SNIPPET.format(seed=rng.randrange(10**9), count=6)
    elif eff == "spam":
        code = _SPAM_SNIPPET
    elif eff == "reset":
        if turn > 0 and turn % 60 == 0:
            code, kind = _DOUBLE_RESET_SNIPPET, "reset-double"
        elif turn > 0 and turn % 20 == 0:
            code, kind = _RESET_SNIPPET, "reset-single"
        else:
            code, kind = _SNIPPET.format(seed=rng.randrange(10**9), count=1), "wander"
    elif eff == "chaos":
        roll = rng.random()
        if roll < 0.10:
            content, kind = "World model: mock text-only turn, no tool call.", "chaos-text"
        elif roll < 0.20:
            bad_args, kind = True, "chaos-badjson"
        elif roll < 0.30:
            code, kind = _RAISING_SNIPPET, "chaos-raise"
        else:
            code, kind = _SNIPPET.format(seed=rng.randrange(10**9), count=1), "wander"
    else:
        code = _SNIPPET.format(seed=rng.randrange(10**9), count=1)

    message: dict = {"role": "assistant", "content": content or ""}
    if bad_args:
        message["tool_calls"] = [{
            "id": f"mock-{turn}",
            "type": "function",
            "function": {"name": "python", "arguments": '{"code": "action([\'ACTION1\'"'},
        }]
    elif code is not None:
        message["tool_calls"] = [{
            "id": f"mock-{turn}",
            "type": "function",
            "function": {"name": "python", "arguments": json.dumps({"code": code})},
        }]
    return message, kind


class _Handler(BaseHTTPRequestHandler):
    policy = os.environ.get("MOCK_POLICY", "mix")
    seed = int(os.environ.get("MOCK_SEED", "12345") or "12345")
    delay_ms = float(os.environ.get("MOCK_DELAY_MS", "0") or "0")

    def log_message(self, fmt, *args):  # silence per-request stderr noise
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            payload = {}
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)
        model = str(payload.get("model", ""))
        policy = self.policy
        if model.startswith("mock-") and model[5:] in _POLICIES:
            policy = model[5:]
        message, kind = build_reply(payload, policy=policy, seed=self.seed)
        STATS.bump(kind)
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        body = json.dumps({
            "id": "mock-chatcmpl",
            "object": "chat.completion",
            "model": model or "mock",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve_in_thread(port: int) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="mock-llm")
    thread.start()
    return server, thread


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8399)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    print(f"mock LLM listening on http://127.0.0.1:{args.port}/v1 "
          f"(policy={_Handler.policy}, seed={_Handler.seed})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

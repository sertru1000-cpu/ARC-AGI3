"""Test for the multimodal teacher input (agent/harness/vision.py).

Checks, all offline (MockLLM, free):
  1. PNG rendering: 64x64 grid x8 -> 512x512, nearest-neighbor exact (every
     8x8 block is one flat color from the palette, no blended pixels).
  2. VisionLLM attaches the image ONLY to the last user message of the
     request, as OpenAI multipart content; the policy's own message history
     and the JSONL trace stay pure text (what the student trains on).
  3. End-to-end through MyAgent with MY_AGENT_VISION=1: every LLM call that
     the mock receives carries exactly one image.

Run:  .venv/Scripts/python.exe scripts/test_vision.py
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

TRACE_DIR = Path(tempfile.mkdtemp(prefix="vision_trace_"))
os.environ["AGENT_BRAIN"] = "llm"
os.environ["MY_AGENT_VISION"] = "1"
os.environ["MY_AGENT_VISION_SCALE"] = "8"
os.environ["MY_AGENT_MAX_TURNS"] = "3"
os.environ["MY_AGENT_MAX_ACTIONS"] = "30"
os.environ["MY_AGENT_FLOOR_ACTIONS"] = "0"
os.environ["MY_AGENT_TRACE_DIR"] = str(TRACE_DIR)
os.environ.pop("LLM_BASE_URL", None)  # never touch a paid endpoint from a test

REPLY = """WORLD_MODEL:
controls: probing
prior: unknown yet
goal: unknown
plan: press UP once
recent_findings: nothing new
open_questions: what moves
cross_level_notes: none yet

```python
r = action(['UP'])
print(r['board_changed'])
```"""


def test_render() -> None:
    from PIL import Image

    from agent.harness.vision import ARC_COLOR_MAP, grid_to_png_bytes

    rng = np.random.default_rng(0)
    grid = rng.integers(0, 16, size=(64, 64))
    png = grid_to_png_bytes(grid, scale=8)
    img = Image.open(io.BytesIO(png)).convert("RGB")
    assert img.size == (512, 512), img.size
    arr = np.asarray(img)
    # Every 8x8 block must be one flat palette color == the source cell.
    for r in range(0, 64, 7):
        for c in range(0, 64, 5):
            block = arr[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8].reshape(-1, 3)
            expect = np.array(ARC_COLOR_MAP[int(grid[r, c])])
            assert (block == expect).all(), (r, c, block[0], expect)
    # Exactly the palette colors, nothing blended.
    uniq = {tuple(x) for x in arr.reshape(-1, 3)}
    assert uniq <= set(ARC_COLOR_MAP.values()), uniq - set(ARC_COLOR_MAP.values())
    print(f"render ok: 512x512, {len(png)} bytes, {len(uniq)} colors")


def test_wrapper_shape() -> None:
    from agent.harness.llm import MockLLM
    from agent.harness.vision import VisionLLM

    seen: list[list] = []

    def script(messages):
        seen.append(messages)
        return "ok"

    grid = np.zeros((64, 64), dtype=int)
    v = VisionLLM(MockLLM(script), lambda: grid, scale=8)
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"}]
    snapshot = json.dumps(msgs)
    assert v.chat(msgs) == "ok"
    assert json.dumps(msgs) == snapshot, "caller's messages were mutated"
    sent = seen[-1]
    assert isinstance(sent[1]["content"], str), "history message got an image"
    last = sent[-1]["content"]
    assert isinstance(last, list) and last[0]["type"] == "image_url" and last[1]["type"] == "text"
    assert last[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert last[1]["text"].endswith("u2")
    base64.b64decode(last[0]["image_url"]["url"].split(",", 1)[1])  # valid base64
    # Tool-loop shape: last message is a tool result -> image goes in a new user turn.
    msgs2 = msgs + [{"role": "assistant", "content": "", "tool_calls": []},
                    {"role": "tool", "tool_call_id": "x", "name": "run_python", "content": "out"}]
    v.chat_tools(msgs2, tools=[])
    sent = seen[-1]
    assert sent[-2]["role"] == "tool" and isinstance(sent[-2]["content"], str)
    assert sent[-1]["role"] == "user" and sent[-1]["content"][0]["type"] == "image_url"
    # grid_provider None -> plain passthrough.
    v2 = VisionLLM(MockLLM(script), lambda: None)
    v2.chat(msgs)
    assert isinstance(seen[-1][-1]["content"], str)
    print("wrapper ok: image only on the current request, history & caller untouched")


def test_end_to_end() -> None:
    import importlib.util

    import arc_agi
    from arc_agi import OperationMode

    import agent.harness.llm as llm_mod
    from agent.harness.llm import MockLLM

    calls: list[list] = []

    def script(messages):
        calls.append(messages)
        return REPLY

    mock = MockLLM(script)
    llm_mod.default_backend = lambda: mock

    spec = importlib.util.spec_from_file_location("user_agent_module", ROOT / "agent" / "my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.VISION_ENABLED

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("ls20")
    agent = module.MyAgent(card_id="t", game_id="ls20-r0", agent_name="vision-test",
                           ROOT_URL="http://localhost", record=False, arc_env=env, tags=["test"])
    agent.main()

    assert calls, "LLM was never called"
    for m in calls:
        imgs = sum(1 for msg in m if isinstance(msg.get("content"), list)
                   for part in msg["content"] if part.get("type") == "image_url")
        assert imgs == 1, f"expected exactly 1 image per request, got {imgs}"
    # Policy history is text-only.
    for msg in agent.policy.messages:
        assert isinstance(msg["content"], str), "policy history contains non-text content"
    # Trace is text-only and records the vision backend.
    trace = TRACE_DIR / "ls20-r0.jsonl"
    recs = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines()]
    assert recs[0]["backend"].startswith("Vision(x8)+"), recs[0]
    assert "image_url" not in trace.read_text(encoding="utf-8")
    print(f"end-to-end ok: {len(calls)} LLM calls, each with 1 image; trace text-only "
          f"({len(recs)} records)")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    test_render()
    test_wrapper_shape()
    test_end_to_end()
    print("ALL VISION TESTS PASS")


if __name__ == "__main__":
    main()

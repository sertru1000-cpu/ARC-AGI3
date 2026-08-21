"""Offline test for the primary -> fallback endpoint switch in OpenAICompatLLM.

Fakes urllib so no network: primary answers HTTP 429 "quota exceeded" twice,
the fallback answers 200. Checks: (1) a transient single 429 does NOT flip;
(2) the second billing-worded 429 flips process-wide; (3) 403 flips at once;
(4) after the flip, requests go to the fallback URL with the fallback model
and bearer.

Run:  .venv/Scripts/python.exe scripts/test_llm_fallback.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["LLM_BASE_URL"] = "https://primary.test/v1"
os.environ["LLM_MODEL"] = "models/gemini-primary"
os.environ["LLM_API_KEY"] = "primary-key"
os.environ.pop("LLM_AUTH", None)
os.environ["LLM_FALLBACK_BASE_URL"] = "https://fallback.test/v1"
os.environ["LLM_FALLBACK_MODEL"] = "google/gemini-fallback"
os.environ["LLM_FALLBACK_AUTH"] = "key"
os.environ["LLM_FALLBACK_API_KEY"] = "fallback-key"
os.environ["LLM_RETRIES"] = "3"

from agent.harness.llm import OpenAICompatLLM  # noqa: E402

calls: list[dict] = []
primary_script: list = []


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_urlopen(req, timeout=None):
    body = json.loads(req.data)
    calls.append({"url": req.full_url, "model": body["model"],
                  "auth": req.get_header("Authorization")})
    if req.full_url.startswith("https://primary"):
        item = primary_script.pop(0)
        if isinstance(item, int):
            msg = {429: "Your prepayment credits are depleted.", 403: "billing disabled",
                   4290: "You exceeded your current quota, please check your plan"}[item]
            item = 429 if item == 4290 else item
            raise urllib.error.HTTPError(req.full_url, item, "err", {}, io.BytesIO(msg.encode()))
    return _Resp(json.dumps({"choices": [{"message": {"content": "hi"}}],
                             "usage": {}}).encode())


urllib.request.urlopen = fake_urlopen
import time  # noqa: E402
time.sleep = lambda s: None  # don't actually wait out fake 429s


def reset():
    OpenAICompatLLM.fallback_active = False
    OpenAICompatLLM._billing_strikes = 0
    calls.clear()


def main():
    llm = OpenAICompatLLM()
    msgs = [{"role": "user", "content": "x"}]

    # (1) one transient 429 then success on primary: no flip.
    reset(); primary_script[:] = [429, "ok"]
    assert llm.chat(msgs) == "hi"
    assert not OpenAICompatLLM.fallback_active
    assert all(c["url"].startswith("https://primary") for c in calls)
    print("single 429 -> stays on primary: ok")

    # (2) two billing 429s: flip, same call completes on fallback.
    reset(); primary_script[:] = [429, 429]
    assert llm.chat(msgs) == "hi"
    assert OpenAICompatLLM.fallback_active
    last = calls[-1]
    assert last["url"].startswith("https://fallback.test/v1"), last
    assert last["model"] == "google/gemini-fallback" and last["auth"] == "Bearer fallback-key"
    # subsequent calls (any instance) go straight to fallback
    n = len(calls)
    OpenAICompatLLM().chat(msgs)
    assert calls[n]["url"].startswith("https://fallback"), calls[n]
    print("two quota 429s -> permanent flip to fallback: ok")

    # (2b) generic rate-limit 429s never flip, even repeated.
    reset(); primary_script[:] = [4290, 4290, "ok"]
    assert llm.chat(msgs) == "hi" and not OpenAICompatLLM.fallback_active
    assert all(c["url"].startswith("https://primary") for c in calls)
    print("rate-limit 429s -> wait, no flip: ok")

    # (3) 403 flips immediately.
    reset(); primary_script[:] = [403]
    assert llm.chat(msgs) == "hi" and OpenAICompatLLM.fallback_active
    assert len(calls) == 2 and calls[1]["url"].startswith("https://fallback")
    print("403 -> immediate flip: ok")
    print("FALLBACK TEST PASSED")


if __name__ == "__main__":
    main()

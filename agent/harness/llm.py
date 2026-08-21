"""LLM backend abstraction.

The agent talks to `LLMBackend.chat(messages) -> str` (single-shot code-fence
protocol) or `LLMBackend.chat_tools(messages, tools) -> {'content', 'tool_calls'}`
(native OpenAI-style tool-calling, used by the tool-loop). Implementations:
  - MockLLM: scripted responses (or a callable), for GPU-less local testing.
  - OpenAICompatLLM: any OpenAI-compatible server — vLLM inside the Kaggle
    notebook, or an Ollama/llama.cpp endpoint during development.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


class _AdcToken:
    """Access token from gcloud Application Default Credentials, cached.

    Enabled via LLM_AUTH=gcloud-adc (Vertex AI path). Tokens live ~1h; we
    refresh 5 minutes early by shelling out to gcloud.
    """

    _token: str | None = None
    _expires_at: float = 0.0

    @classmethod
    def get(cls) -> str:
        import subprocess
        import time as _t

        if cls._token is None or _t.time() > cls._expires_at - 300:
            gcloud = os.getenv("LLM_GCLOUD_PATH", "gcloud")
            out = subprocess.check_output(
                [gcloud, "auth", "application-default", "print-access-token"],
                text=True, timeout=60,
            )
            cls._token = out.strip().splitlines()[-1]
            cls._expires_at = _t.time() + 3600
        return cls._token


# One requested tool invocation, backend-agnostic (parsed out of whatever
# wire format the provider uses): {"id": str, "name": str, "arguments": dict}.
ToolCall = dict


class LLMBackend:
    def chat(self, messages: list[Message], max_tokens: int = 2048, temperature: float = 0.6) -> str:
        raise NotImplementedError

    def chat_tools(
        self,
        messages: list[Message],
        tools: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.6,
        tool_choice: str = "auto",
    ) -> dict:
        """Native tool-calling chat call. Returns {'content': str|None,
        'tool_calls': list[ToolCall]} -- content may be present ALONGSIDE
        tool_calls (most APIs allow reasoning text + a call in one message)."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class MockLLM(LLMBackend):
    """Deterministic backend for tests: a list of canned replies or a hook.

    `script` may be a list of items (returned in order; the last one repeats)
    or a callable `(messages) -> item` for smarter fakes. Each item is either
    a plain string (a text-only reply) or a dict `{"content": str|None,
    "tool": name, "arguments": dict}` describing one tool call (used by
    `chat_tools`; `chat` ignores the tool fields and returns "content" as-is,
    or "" if the item has none).
    """

    def __init__(self, script: list) -> None:
        self.script = script
        self.calls = 0

    def _next(self, messages: list[Message]):
        self.calls += 1
        if callable(self.script):
            return self.script(messages)
        idx = min(self.calls - 1, len(self.script) - 1)
        return self.script[idx]

    def chat(self, messages: list[Message], max_tokens: int = 2048, temperature: float = 0.6) -> str:
        item = self._next(messages)
        if isinstance(item, dict):
            return item.get("content") or ""
        return item

    def chat_tools(
        self,
        messages: list[Message],
        tools: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.6,
        tool_choice: str = "auto",
    ) -> dict:
        item = self._next(messages)
        if isinstance(item, dict):
            return {
                "content": item.get("content"),
                "tool_calls": [{
                    "id": f"mock-{self.calls}",
                    "name": item.get("tool", "run_python"),
                    "arguments": item.get("arguments", {}),
                }],
            }
        return {"content": item, "tool_calls": []}


class OpenAICompatLLM(LLMBackend):
    """Minimal OpenAI-compatible chat client (urllib only — no extra deps)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "default")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "none")
        self.timeout = timeout if timeout is not None else float(os.getenv("LLM_TIMEOUT_S", "300"))

    RETRIES = int(os.getenv("LLM_RETRIES", "3"))
    # Cumulative usage across all calls in this process (budget watchdog).
    total_prompt_tokens = 0
    total_completion_tokens = 0
    # Fallback endpoint (21.08): play on a prepaid AI Studio key first and,
    # when its balance/quota runs dry, switch PERMANENTLY (process-wide, all
    # threads) to a second endpoint (Vertex) without interrupting episodes.
    # Configured via LLM_FALLBACK_BASE_URL / LLM_FALLBACK_MODEL /
    # LLM_FALLBACK_AUTH (gcloud-adc | key) / LLM_FALLBACK_API_KEY.
    fallback_active = False
    _billing_strikes = 0

    def _endpoint(self) -> tuple[str, str, str]:
        """(base_url, model, bearer) for the endpoint currently in use."""
        if OpenAICompatLLM.fallback_active and os.getenv("LLM_FALLBACK_BASE_URL"):
            auth = os.getenv("LLM_FALLBACK_AUTH", "key")
            bearer = _AdcToken.get() if auth == "gcloud-adc" else os.getenv("LLM_FALLBACK_API_KEY", "none")
            return (os.getenv("LLM_FALLBACK_BASE_URL").rstrip("/"),
                    os.getenv("LLM_FALLBACK_MODEL", self.model), bearer)
        bearer = _AdcToken.get() if os.getenv("LLM_AUTH") == "gcloud-adc" else self.api_key
        return self.base_url, self.model, bearer

    @staticmethod
    def _http_status_and_body(exc: Exception) -> tuple[int, str]:
        import urllib.error
        if isinstance(exc, urllib.error.HTTPError):
            try:
                body = exc.read().decode("utf-8", "replace")
            except Exception:
                body = ""
            return exc.code, body
        return 0, str(exc)

    def _maybe_switch_to_fallback(self, status: int, body: str) -> bool:
        """Billing/quota exhaustion on the primary -> flip to the fallback.
        402/403 flip at once; a quota-worded 429 is a strike and 2 strikes
        flip (a single 429 can be a transient per-minute rate limit)."""
        if OpenAICompatLLM.fallback_active or not os.getenv("LLM_FALLBACK_BASE_URL"):
            return False
        low = body.lower()
        # Only BALANCE exhaustion flips. AI Studio's generic "You exceeded your
        # current quota" is a per-minute rate limit under many parallel
        # streams (21.08 round 4: 15 threads flipped both processes to Vertex
        # within 2 min while a single probe succeeded) -- that one must just
        # wait and retry. "Your prepayment credits are depleted" is the real
        # signal (seen live 21.08 round 2).
        # NOT "billing"/"payment": the generic rate-limit text says "check your
        # plan and billing details" (flipped round 4 a second time, 21.08).
        billing_words = ("depleted", "prepayment", "insufficient", "credit")
        hit = status in (402, 403) or (status == 429 and any(w in low for w in billing_words))
        if not hit:
            return False
        OpenAICompatLLM._billing_strikes += 1
        if OpenAICompatLLM._billing_strikes >= 2 or status in (402, 403):
            OpenAICompatLLM.fallback_active = True
            import logging
            logging.getLogger(__name__).warning(
                "PRIMARY LLM ENDPOINT EXHAUSTED (HTTP %s: %s) -> switching to fallback %s / %s",
                status, " ".join(body[:200].split()),
                os.getenv("LLM_FALLBACK_BASE_URL"), os.getenv("LLM_FALLBACK_MODEL"))
            return True
        return False

    def _request(self, payload: dict) -> dict:
        """POST /chat/completions with retry/backoff. Returns the parsed
        response JSON (the raw `message` dict lives at data['choices'][0]['message'])."""
        # Reasoning models (gpt-oss/harmony via vLLM): thinking depth knob.
        effort = os.getenv("LLM_REASONING_EFFORT")
        if effort:
            payload["reasoning_effort"] = effort
        last_exc: Exception | None = None
        attempt = 0
        while attempt < self.RETRIES:
            attempt += 1
            base_url, model, bearer = self._endpoint()
            payload["model"] = model
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {bearer}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                usage = data.get("usage") or {}
                OpenAICompatLLM.total_prompt_tokens += usage.get("prompt_tokens", 0)
                OpenAICompatLLM.total_completion_tokens += usage.get("completion_tokens", 0)
                return data
            except Exception as exc:  # timeout, connection reset, 5xx, bad JSON
                last_exc = exc
                status, body = self._http_status_and_body(exc)
                # Server rejects the reasoning knob (non-harmony model / old
                # vLLM): drop it once and keep going rather than dying on 400s.
                if "reasoning_effort" in payload and status == 400:
                    payload.pop("reasoning_effort")
                    continue
                if self._maybe_switch_to_fallback(status, body):
                    attempt -= 1  # the switch itself doesn't cost a retry
                    continue
                import time as _t

                # Rate limits (429) live on per-minute windows -- wait them out.
                if status == 429:
                    _t.sleep(min(90.0, 20.0 * attempt))
                else:
                    _t.sleep(min(10.0, 2.0 * attempt))
        raise RuntimeError(f"LLM backend failed after {self.RETRIES} attempts: {last_exc!r}")

    def chat(self, messages: list[Message], max_tokens: int = 2048, temperature: float = 0.6) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = self._request(payload)
        # AI Studio's compat layer may omit `content` entirely when thinking
        # consumed the whole budget; Vertex sends null. Vertex can also return
        # a choice with NO `message` at all (seen 21.08 on a multimodal call:
        # KeyError('message') killed the turn) -- treat both as an empty
        # reply (a normal no-code strike with a corrective nudge), and log the
        # raw shape once so the cause is diagnosable from the run log.
        choices = data.get("choices") or []
        msg = (choices[0] if choices else {}).get("message")
        if not isinstance(msg, dict):
            import logging
            logging.getLogger(__name__).warning(
                "LLM reply without message: %s", json.dumps(data, default=str)[:600])
            return ""
        return msg.get("content") or ""

    def chat_tools(
        self,
        messages: list[Message],
        tools: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.6,
        tool_choice: str = "auto",
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        data = self._request(payload)
        msg = data["choices"][0]["message"]
        parsed_calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}
            parsed_calls.append({"id": tc.get("id"), "name": fn.get("name"), "arguments": args})
        return {"content": msg.get("content"), "tool_calls": parsed_calls}

    @property
    def name(self) -> str:
        fb = " +fallback" if os.getenv("LLM_FALLBACK_BASE_URL") else ""
        return f"OpenAICompat({self.model}@{self.base_url}{fb})"


def default_backend() -> LLMBackend:
    """Pick a backend from the environment: real server if configured, else mock."""
    if os.getenv("LLM_BASE_URL"):
        return OpenAICompatLLM()
    return MockLLM(["NOTHING TO DO"])

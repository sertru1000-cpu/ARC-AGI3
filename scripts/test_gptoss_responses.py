"""Проверка адаптера Responses API для gpt-oss (перевод запроса и ответа) на настоящем модуле бандла.
usage: .venv/bin/python scripts/test_gptoss_responses.py --bundle <бандл>"""
import argparse, json, sys, types
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--bundle", required=True); a = ap.parse_args()
sys.path.insert(0, str(Path(a.bundle) / "src" / "ARC3-Inference")); sys.path.insert(0, str(Path(a.bundle) / "src" / "tufa-arc-agi-framework" / "src"))
import inference.agent.tool_agent as wta, requests as real_requests
fails = []
def check(b, m):
    print(("ok   " if b else "СБОЙ ") + m)
    if not b: fails.append(m)
nb = json.load(open("kernels/notebooks_duck_gptoss/submission.ipynb")); c9 = "".join(nb["cells"][9]["source"])
seg = c9[c9.index("_g_orig_build_payload = _wta.build_chat_payload"): c9.index('print("gpt-oss: адаптер Responses API')]
captured = {}
class FakeReal:
    HTTPError = real_requests.HTTPError; RequestException = real_requests.RequestException
    @staticmethod
    def post(url, headers=None, json=None, timeout=None, **kw):
        captured["url"] = url; captured["body"] = json
        data = {"id": "resp_1", "status": "completed", "model": "gpt-oss-120b", "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                "output": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "думаю"}]},
                           {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Plan: probe"}]},
                           {"type": "function_call", "call_id": "call_9", "name": "python", "arguments": "{\"code\": \"print(1)\"}"}]}
        return types.SimpleNamespace(status_code=200, json=lambda: data, text=json_dumps(data))
json_dumps = json.dumps
fake_wta = types.SimpleNamespace(requests=FakeReal, build_chat_payload=wta.build_chat_payload)
import os
ns = {"_wta": fake_wta, "os": os, "SETUP_ENV_PATH": types.SimpleNamespace(read_text=lambda: "{}", write_text=lambda s: None)}
exec("import json as _g_json\n" + seg, ns)
msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": [{"type": "text", "text": "board"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
        {"role": "assistant", "content": None, "reasoning": "r", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "python", "arguments": "{\"code\": \"x\"}"}},
                                                                             {"id": "c2", "type": "function", "function": {"name": "python", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "{\"stdout\": \"1\"}"}, {"role": "user", "content": "next"}]
tools = [{"type": "function", "function": {"name": "python", "description": "run", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}}}]
resp = ns["_g_post"]("http://127.0.0.1:1234/v1/chat/completions", headers={"Authorization": "Bearer EMPTY"}, json={"model": "gpt-oss-120b", "messages": msgs, "tools": tools, "max_tokens": 500, "temperature": 0.6}, timeout=10)
b = captured["body"]
check(captured["url"].endswith("/v1/responses"), "запрос ушёл в /v1/responses")
check(b["instructions"] == "SYS" and b["input"][0]["role"] == "user" and b["input"][0]["content"] == [{"type": "input_text", "text": "board"}], "system -> instructions; картинка отброшена; текст -> input_text")
kinds = [it["type"] for it in b["input"]]
check(kinds == ["message", "function_call", "function_call_output", "message"], "история: только отвеченный вызов c1, результат, следующий user: %s" % kinds)
check(b["input"][1]["call_id"] == "c1" and b["input"][2]["call_id"] == "c1", "call_id сохранён")
check(b["tools"][0]["name"] == "python" and b["max_output_tokens"] == 500 and b["parallel_tool_calls"] is False and b["reasoning"]["effort"] == "medium", "tools/max_output_tokens/parallel_tool_calls/reasoning")
d = resp.json(); m = d["choices"][0]["message"]
check(resp.status_code == 200 and m["content"] == "Plan: probe" and m["reasoning"] == "думаю" and m["tool_calls"][0]["id"] == "call_9" and m["tool_calls"][0]["function"]["arguments"] == "{\"code\": \"print(1)\"}", "ответ переведён в chat-форму: content, reasoning, tool_calls")
check(d["choices"][0]["finish_reason"] == "tool_calls" and d["usage"]["prompt_tokens"] == 100, "finish_reason и usage")
check(ns["_g_stats"]["ok"] == 1 and ns["_g_stats"]["tool_calls"] == 1, "статистика адаптера")
r2 = ns["_g_post"]("http://127.0.0.1:1234/v1/models", headers={}, json=None, timeout=5) if False else None
print("\nИТОГ: %d сбоев" % len(fails) if fails else "\nВСЕ ПРОВЕРКИ ПРОШЛИ"); sys.exit(1 if fails else 0)

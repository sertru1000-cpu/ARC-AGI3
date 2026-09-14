"""Проверка санитайзера истории под harmony (gpt-oss на Duck) на настоящем модуле бандла.
usage: .venv/bin/python scripts/test_gptoss_patch.py --bundle <бандл>"""
import argparse, json, sys, types
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--bundle", required=True); a = ap.parse_args()
sys.path.insert(0, str(Path(a.bundle) / "src" / "ARC3-Inference")); sys.path.insert(0, str(Path(a.bundle) / "src" / "tufa-arc-agi-framework" / "src"))
import inference.agent.tool_agent as wta
fails = []
def check(b, m):
    print(("ok   " if b else "СБОЙ ") + m)
    if not b: fails.append(m)
nb = json.load(open("kernels/notebooks_duck_gptoss/submission.ipynb"))
c9 = "".join(nb["cells"][9]["source"])
start = c9.index("_g_orig_build_payload = _wta.build_chat_payload"); end = c9.index('print("gpt-oss: санитайзер')
ns = {"_wta": wta}; exec(c9[start:end], ns)
san = ns["_g_sanitize_messages"]
hist = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
        {"role": "assistant", "content": None, "reasoning": "думаю", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "python", "arguments": "{}"}},
                                                                                  {"id": "b", "type": "function", "function": {"name": "python", "arguments": "{}"}},
                                                                                  {"id": "c", "type": "function", "function": {"name": "python", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": None},
        {"role": "assistant", "reasoning_content": "x", "content": None, "tool_calls": [{"id": "z", "type": "function", "function": {"name": "python", "arguments": "{}"}}]},
        {"role": "user", "content": "next"}]
out = san(hist)
check(len(out) == 6 and "reasoning" not in out[2] and "reasoning_content" not in out[4], "reasoning убран из assistant-сообщений")
check([c["id"] for c in out[2]["tool_calls"]] == ["a"], "из трёх вызовов оставлен только отвеченный")
check(out[2]["content"] == "" and out[3]["content"] == "", "content None -> строка")
check("tool_calls" not in out[4] and out[4]["content"] == "(tool call omitted)", "вызов без результата убран, вместо пустого content -- пометка")
payload = ns["_g_build_payload"](provider="openrouter", model="gpt-oss-120b", messages=hist, max_tokens=None, temperature=0.6, top_p=0.95, top_k=20, thinking=True,
                                 tools=[{"type": "function", "function": {"name": "python", "parameters": {}}}], tool_choice="auto", seed=None)
check(payload.get("parallel_tool_calls") is False and "top_k" not in payload and "chat_template_kwargs" not in payload, "payload: parallel_tool_calls=False, без top_k/chat_template_kwargs (openrouter)")
check(len(payload["messages"][2]["tool_calls"]) == 1, "payload собран из очищенной истории")
print("\nИТОГ: %d сбоев" % len(fails) if fails else "\nВСЕ ПРОВЕРКИ ПРОШЛИ"); sys.exit(1 if fails else 0)

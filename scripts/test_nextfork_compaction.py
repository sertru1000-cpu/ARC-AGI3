"""Сжатие истории (NEXTFORK_COMPACT=1, inference/agent/compaction.py) — проверки без сервера (29.09).
Ставится импортом solver.py, как в ноутбуке; вызов модели подменён — проверяем, что уходит в запрос и что остаётся в истории.
usage: .venv/bin/python scripts/test_nextfork_compaction.py
"""
import json, logging, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "nextfork/src"
sys.path[:0] = [str(B / "ARC3-Inference"), str(B / "tufa-arc-agi-framework/src")]
os.environ.update(MULTIMODAL_UPSCALE="4", MULTIMODAL_CONTEXT="current_grid", NEXTFORK_COMPACT="1", NEXTFORK_COMPACT_TOKENS="3000")
logging.disable(logging.ERROR)
import inference.framework.solver  # noqa: F401,E402
import inference.agent.compaction as cp  # noqa: E402
import inference.agent.tool_agent as ta  # noqa: E402
from inference.agent.runtime_state import RUNTIME_STATE_FILENAME  # noqa: E402

TMP = ROOT / ".tmp_compact"; TMP.mkdir(exist_ok=True); STATE = TMP / RUNTIME_STATE_FILENAME
A = [[0] * 8 for _ in range(8)]; Bd = [[0] * 8 for _ in range(8)]; Bd[3][3] = 5
hist = [{"action": "", "frame": {"grid": A, "step": 0, "level": 1}}]
for i, (act, g) in enumerate([("RIGHT", Bd), ("RIGHT", Bd), ("UP", A), ("UP", A), ("LEFT", Bd)]):
    hist.append({"action": act, "frame": {"grid": g, "step": i + 1, "level": 1}})
STATE.write_text(json.dumps({"current_frame": hist[-1]["frame"], "history": hist}))

def make_agent():
    a = ta.ToolAgent.__new__(ta.ToolAgent)
    a.__dict__.update(_session_runtime_dir=TMP, _session_total_tokens=0, _session_generated_tokens=0,
                      _system_prompt="SYSTEM " * 50, _max_output_tokens=None)
    return a

def turns(n, size=1600):
    h = []
    for k in range(n):
        h += [{"role": "user", "content": f"turn {k} observation " + "x" * size},
              {"role": "assistant", "content": f"thinking about turn {k}", "tool_calls": [{"id": f"c{k}", "type": "function", "function": {"name": "python", "arguments": "{}"}}]},
              {"role": "tool", "tool_call_id": f"c{k}", "content": f"result {k}"}]
    return h

class Fake:
    def __init__(self, content="CONTROLS: RIGHT moves the block, verified.\nTRIED: UP twice, no change.\nPLAN: go right.", finish="stop", raise_exc=None):
        self.calls = []; self.content = content; self.finish = finish; self.raise_exc = raise_exc
    def __call__(self, messages, **kw):
        self.calls.append((messages, kw))
        if self.raise_exc: raise self.raise_exc
        return ta._ChatCompletionResult(message={"content": self.content}, finish_reason=self.finish, usage={"prompt_tokens": 1, "completion_tokens": 1})

ok = True
def check(cond, text):
    global ok; ok &= bool(cond); print("%s %s" % ("ok  " if cond else "FAIL", text))

check(ta.ToolAgent.analyze.__module__ == "inference.agent.compaction", "analyze обёрнут сжатием (ставится из solver.py)")

a = make_agent(); f = Fake(); a._chat_completion = f; a._history_messages = turns(2, 200)
check(cp._compact(a, STATE) is False and not f.calls, "короткая история — сжатия нет, модель не вызывается")

a = make_agent(); f = Fake(); a._chat_completion = f; a._history_messages = turns(8)
done = cp._compact(a, STATE)
req = f.calls[0][0] if f.calls else []
last = req[-1]["content"] if req else ""
check(done and len(f.calls) == 1, "длинная история — один запрос сжатия")
check(f.calls and f.calls[0][1].get("tools") is None, "запрос сжатия без инструментов")
check("CONTEXT COMPACTION" in last and "<state_snapshot>" in last and "<tried_and_failed>" in last and "<goal>" in last, "в запросе шаблон разделов (формат qwen-code)")
check("level 1; moves made on this level: 5" in last and "RIGHT: changed the board 1 time(s), no change 1" in last
      and "UP: changed the board 1 time(s), no change 1" in last, "факты из журнала посчитаны обвязкой верно")
h = a._history_messages
check(h[0]["role"] == "user" and h[0]["content"].startswith(cp.MARK) and "verified" in h[0]["content"], "история начинается с пересказа")
check(sum(1 for m in h if m["role"] == "assistant") == 1 + cp.KEEP, "дословно сохранены последние %d хода" % cp.KEEP)
check("turn 7 observation" in json.dumps(h) and "turn 6 observation" in json.dumps(h) and "turn 0 observation" not in json.dumps(h), "старые ходы убраны, свежие на месте")
check(a._max_output_tokens is None, "предел ответа восстановлен после сжатия")

a._history_messages += turns(8)[3:]
f2 = Fake(); a._chat_completion = f2
cp._compact(a, STATE)
check(f2.calls and cp.MARK in json.dumps(f2.calls[0][0], ensure_ascii=False), "второе сжатие видит прежний пересказ (память копится)")

a = make_agent(); a._chat_completion = Fake(content="<analysis>думаю долго " + "z" * 300 + "</analysis>\n<state_snapshot>\n<goal>reach the door</goal>\n<controls>RIGHT moves, verified</controls>\n<tried_and_failed>UP twice</tried_and_failed>\n</state_snapshot>")
a._history_messages = turns(8); cp._compact(a, STATE); note = a._history_messages[0]["content"]
check("<state_snapshot>" in note and "<analysis>" not in note and "думаю" not in note, "блок <analysis> выброшен, в истории только снимок")
for name, fake in (("обрыв по длине", Fake(finish="length")), ("ошибка сервера", Fake(raise_exc=RuntimeError("boom"))), ("пустой ответ", Fake(content=""))):
    a = make_agent(); a._chat_completion = fake; before = turns(8); a._history_messages = list(before)
    check(cp._compact(a, STATE) is False and a._history_messages == before, "%s — история не тронута, остаётся обрезка" % name)

print("\nИТОГ:", "все проверки пройдены" if ok else "ЕСТЬ ПРОВАЛЫ")
sys.exit(0 if ok else 1)

"""Синтетическая проверка согласования двух ответов (ToolAgent._consensus_pass), 26.09.

Подставная модель отдаёт заранее заданные ответы; сухой прогон идёт через настоящий
_run_python_tool с настоящей песочницей. Проверяем:
  1. ответы совпали -> 2 вызова модели, исполняется A, заметки нет;
  2. разошлись -> 3 вызова, возвращается C и заметка с обоими ходами;
  3. код A без action() -> 1 вызов, A без изменений;
  4. NEXTFORK_CONSENSUS=0 -> механизм не хочет работать;
  5. ход номер 4 и дальше -> не хочет;
  6. сухой прогон не трогает среду и не портит обратный вызов.
usage: .venv/bin/python scripts/test_consensus.py
"""
from __future__ import annotations
import json, os, sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nextfork/src/ARC3-Inference"))
import inference.agent.tool_agent as TA
from inference.agent.runtime_state import RUNTIME_STATE_FILENAME
TMP = ROOT / ".tmp_consensus"; TMP.mkdir(exist_ok=True)
STATE = TMP / RUNTIME_STATE_FILENAME
STATE.write_text(json.dumps({"current_frame": {"grid": [[0]*8 for _ in range(8)], "step": 0, "level": 1}, "history": []}))

def resp(code):
    return types.SimpleNamespace(message={"role": "assistant", "content": "", "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": code})}}]},
        finish_reason="tool_calls", usage=None)
def tcs(code): return resp(code).message["tool_calls"]

class Env:
    def __init__(self): self.seen = []
    def __call__(self, req): self.seen.append(req); return {"executed": True, "board_changed": True}

def agent(scripted):
    a = TA.ToolAgent.__new__(TA.ToolAgent)
    a._session_runtime_dir = None; a._history_messages = []; a._session_total_tokens = 0; a._session_generated_tokens = 0
    a._last_step_summary = None; a._last_action_result = None; a._summarized_knowledge = TA._empty_world_model()
    a._noop_guard = TA.NoopGuard(); a._noop_guard_blocked = 0; a._python_timeout = 30.0; a._tool_output_chars = 20000
    a._current_valid_actions = ["UP", "DOWN"]; a._step_env_callback = Env(); a._ensure_session(STATE)
    calls = []
    def fake(messages, **kw): calls.append(messages); return resp(scripted[len(calls)-1])
    a._chat_completion = fake; a._calls = calls
    return a

ok = True
def check(label, cond):
    global ok; ok &= bool(cond); print("%s %s" % ("ОК  " if cond else "ПРОВАЛ", label))
os.environ["NEXTFORK_CONSENSUS"] = "1"
UP = "action([{'action': 'UP'}])"; RIGHT = "action([{'action': 'RIGHT'}])"; NOACT = "x = current_frame.ascii"

a = agent([UP]); r = a._consensus_pass(state_path=STATE, messages=[{"role":"user","content":"s"}], request_kwargs={}, action_num=0, reasoning="r", content="", tool_calls=tcs(UP))
check("1. совпали: вызовов %d, исход %s, заметки нет" % (len(a._calls), r[4]), len(a._calls) == 1 and r[4] == "agree" and r[3] is None and r[2] == tcs(UP))
check("6. среда не тронута сухим прогоном: %d ходов" % len(a._step_env_callback.seen), len(a._step_env_callback.seen) == 0 and isinstance(a._step_env_callback, Env))

a = agent([RIGHT, RIGHT]); r = a._consensus_pass(state_path=STATE, messages=[{"role":"user","content":"s"}], request_kwargs={}, action_num=2, reasoning="r", content="", tool_calls=tcs(UP))
note = r[3]
check("2. разошлись: вызовов %d, исход %s, C исполняется, заметка с обоими ходами" % (len(a._calls), r[4]),
      len(a._calls) == 2 and r[4] == "resolved_by_C" and r[2] == tcs(RIGHT) and note and "UP" in note["content"] and "RIGHT" in note["content"])
check("2b. заметка ушла ТРЕТЬЕМУ вызову", a._calls[1][-1] is note or a._calls[1][-1]["content"] == note["content"])

a = agent([UP]); r = a._consensus_pass(state_path=STATE, messages=[], request_kwargs={}, action_num=0, reasoning="r", content="", tool_calls=tcs(NOACT))
check("3. код без action(): вызовов %d, исход %s" % (len(a._calls), r[4]), len(a._calls) == 0 and r[4] == "a_no_action")

os.environ["NEXTFORK_CONSENSUS"] = "0"; check("4. выключатель", not agent([])._consensus_wanted(0))
os.environ["NEXTFORK_CONSENSUS"] = "1"; check("5. ход 4 и дальше не согласуется, ход 3 — да", not agent([])._consensus_wanted(4) and agent([])._consensus_wanted(3))
print("\nИТОГ:", "все проверки пройдены" if ok else "ЕСТЬ ПРОВАЛЫ"); sys.exit(0 if ok else 1)

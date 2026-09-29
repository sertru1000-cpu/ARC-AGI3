"""Ячейка согласования исполняется против исходников ВЕРСИИ 3 с Kaggle (/tmp/nf_kaggle), как в бою.
Подставная модель, настоящая песочница. Те же случаи, что у теста исходной версии."""
from __future__ import annotations
import json, os, sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
V3 = Path("/tmp/nf_kaggle/src/ARC3-Inference")
assert (V3 / "inference/agent/tool_agent.py").is_file(), "скачайте версию 3: kaggle datasets download sergueimakarov/arc3-nextfork -p /tmp/nf_kaggle --unzip"
sys.path.insert(0, str(V3))
import inference.agent.tool_agent as TA
assert not hasattr(TA.ToolAgent, "_consensus_pass"), "это не версия 3 — в ней уже есть согласование"
from inference.agent.runtime_state import RUNTIME_STATE_FILENAME
CELL = open(ROOT / "kernels/notebooks_nextfork_cons/cell15.py", encoding="utf-8").read()
exec(compile(CELL, "cell15", "exec"), {})                     # накладываем ячейку на v3
TMP = ROOT / ".tmp_conscell"; TMP.mkdir(exist_ok=True); STATE = TMP / RUNTIME_STATE_FILENAME
STATE.write_text(json.dumps({"current_frame": {"grid": [[0]*8 for _ in range(8)], "step": 0, "level": 1}, "history": []}))
def resp(code): return types.SimpleNamespace(message={"role":"assistant","content":"","tool_calls":[{"id":"t","type":"function","function":{"name":"python","arguments":json.dumps({"code":code})}}]}, finish_reason="tool_calls", usage=None)
class Env:
    def __init__(self): self.seen=[]
    def __call__(self, req): self.seen.append(req); return {"executed": True, "board_changed": True}
def agent(scripted, action_num):
    a = TA.ToolAgent.__new__(TA.ToolAgent)
    a._session_runtime_dir=None; a._history_messages=[]; a._session_total_tokens=0; a._session_generated_tokens=0
    a._last_step_summary=None; a._last_action_result=None; a._summarized_knowledge=TA._empty_world_model()
    a._noop_guard=TA.NoopGuard(); a._noop_guard_blocked=0; a._python_timeout=30.0; a._tool_output_chars=20000
    a._current_valid_actions=["UP","DOWN"]; a._step_env_callback=Env(); a._ensure_session(STATE)
    a._cons_action_num=action_num; a._cons_state_path=STATE; a._cons_done=False
    it = iter(scripted); calls=[]
    def orig(self, messages, **kw): calls.append(messages); return resp(next(it))
    a._orig_calls = calls
    import builtins
    globals()["_ORIG"] = orig
    return a
ok=True
def check(l,c):
    global ok; ok&=bool(c); print("%s %s"%("ОК  " if c else "ПРОВАЛ", l))
UP="action([{'action': 'UP'}])"; RIGHT="action([{'action': 'RIGHT'}])"; NOACT="x = current_frame.ascii"
# подменяем «оригинальный» вызов модели внутри ячейки на подставной
import inspect
cell_ns = {}
exec(compile(CELL, "cell15", "exec"), cell_ns)
def run_case(scripted, action_num, first):
    a = agent(scripted, action_num)
    cell_ns["_orig_chat"] = lambda self, messages, **kw: (a._orig_calls.append(messages), resp(scripted.pop(0)))[1]
    r = cell_ns["_cons_chat"](a, [{"role":"user","content":"s"}])
    return a, r
a,r = run_case([UP, UP], 0, UP);      check("1. совпали: вызовов %d, отдан A" % len(a._orig_calls), len(a._orig_calls)==2 and json.loads(r.message["tool_calls"][0]["function"]["arguments"])["code"]==UP)
check("6. среда не тронута: %d ходов" % len(a._step_env_callback.seen), len(a._step_env_callback.seen)==0 and isinstance(a._step_env_callback, Env))
a,r = run_case([UP, RIGHT, RIGHT], 2, UP); check("2. разошлись: вызовов %d, отдан C, заметка третьему" % len(a._orig_calls), len(a._orig_calls)==3 and json.loads(r.message["tool_calls"][0]["function"]["arguments"])["code"]==RIGHT and "UP" in a._orig_calls[2][-1]["content"] and "RIGHT" in a._orig_calls[2][-1]["content"])
a,r = run_case([NOACT, UP], 0, NOACT); check("3. код без action(): вызовов %d, отдан A" % len(a._orig_calls), len(a._orig_calls)==1)
a,r = run_case([UP, RIGHT], 4, UP);   check("5. ход 4: вызовов %d — согласования нет" % len(a._orig_calls), len(a._orig_calls)==1)
a = agent([UP], 0); a._cons_done=True; cell_ns["_orig_chat"]=lambda self,m,**kw:(a._orig_calls.append(m),resp(UP))[1]; cell_ns["_cons_chat"](a,[]); check("7. второй вызов в том же ходу не согласуется: %d" % len(a._orig_calls), len(a._orig_calls)==1)
import shutil; shutil.rmtree(TMP, ignore_errors=True)
print("\nИТОГ:", "все проверки пройдены" if ok else "ЕСТЬ ПРОВАЛЫ"); sys.exit(0 if ok else 1)

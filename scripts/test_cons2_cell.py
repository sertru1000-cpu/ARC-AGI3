"""Ячейка cons2 против исходников версии 3 с Kaggle: подставная модель, настоящая песочница и runtime_state."""
from __future__ import annotations
import json, os, sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
V3 = Path("/tmp/nf_kaggle/src/ARC3-Inference"); sys.path.insert(0, str(V3))
import inference.agent.tool_agent as TA
from inference.agent.runtime_state import RUNTIME_STATE_FILENAME
assert not hasattr(TA.ToolAgent, "_consensus_pass")
CELL = open(ROOT / "kernels/notebooks_nextfork_cons2/cell15.py", encoding="utf-8").read()
ns = {}; exec(compile(CELL, "cell15", "exec"), ns)
TMP = ROOT / ".tmp_cons2"; TMP.mkdir(exist_ok=True); STATE = TMP / RUNTIME_STATE_FILENAME
CH = ".WwgGcBMPRbSYOrNp"
def grid(shift):
    g = [[0]*10 for _ in range(10)]
    for r in range(2, 4):
        for c in range(2+shift, 4+shift): g[r][c] = 3
    return g
def write_state(g, level=1, hist=None):
    fr = {"grid": g, "step": 0, "level": level, "shape": [10, 10], "ascii": "\n".join("".join(CH[v] for v in r) for r in g)}
    STATE.write_text(json.dumps({"current_frame": fr, "history": hist or []}))
def resp(code): return types.SimpleNamespace(message={"role":"assistant","content":"","tool_calls":[{"id":"t","type":"function","function":{"name":"python","arguments":json.dumps({"code":code})}}]}, finish_reason="tool_calls", usage=None)
def agent():
    a = TA.ToolAgent.__new__(TA.ToolAgent)
    a._session_runtime_dir=None; a._history_messages=[]; a._session_total_tokens=0; a._session_generated_tokens=0
    a._last_step_summary=None; a._last_action_result=None; a._summarized_knowledge=TA._empty_world_model()
    a._noop_guard=TA.NoopGuard(); a._noop_guard_blocked=0; a._python_timeout=30.0; a._tool_output_chars=20000
    a._current_valid_actions=["UP","RIGHT"]; a._step_env_callback=lambda req: {"executed": True, "board_changed": True}
    a._ensure_session(STATE); a._c2_state_path=STATE; a._c2_done=False; a._c2_fork=True; a._c2_pending=""
    return a
ok=True
def check(l,c):
    global ok; ok&=bool(c); print("%s %s"%("ОК  " if c else "ПРОВАЛ", l))
UP  = "print('PREDICT: blue block moves 1 cell up')\naction([{'action': 'UP'}])"
RGT = "print('PREDICT: blue block moves 2 cells right')\naction([{'action': 'RIGHT'}])"
write_state(grid(0))
# 1. согласие: 2 вызова, A, предсказание запомнено, в сообщение добавлена просьба PREDICT
a=agent(); calls=[]; sc=[UP, UP]
ns["_c2_orig_chat"]=lambda self,m,**kw:(calls.append(m), resp(sc.pop(0)))[1]
r=ns["_c2_chat"](a,[{"role":"user","content":"state"}])
check("1. согласие: вызовов %d, A, pending=%r" % (len(calls), a._c2_pending), len(calls)==2 and a._c2_pending.startswith("blue block moves 1 cell up") and "PREDICT:" in calls[0][-1]["content"])
# 2. расхождение: 3 вызова, арбитру ушли оба предсказания, исполняется C, pending — предсказание C
a=agent(); calls=[]; sc=[UP, RGT, RGT]
ns["_c2_orig_chat"]=lambda self,m,**kw:(calls.append(m), resp(sc.pop(0)))[1]
r=ns["_c2_chat"](a,[{"role":"user","content":"state"}])
note=calls[2][-1]["content"]
check("2. расхождение: вызовов %d, C исполняется, арбитр видит оба предсказания" % len(calls), len(calls)==3 and "moves 1 cell up" in note and "moves 2 cells right" in note and json.loads(r.message["tool_calls"][0]["function"]["arguments"])["code"]==RGT and "right" in a._c2_pending)
# 3. вне развилки — один вызов, без просьбы PREDICT
a=agent(); a._c2_fork=False; calls=[]; sc=[UP]
ns["_c2_orig_chat"]=lambda self,m,**kw:(calls.append(m), resp(sc.pop(0)))[1]
ns["_c2_chat"](a,[{"role":"user","content":"state"}])
check("3. вне развилки: вызовов %d, просьбы нет" % len(calls), len(calls)==1 and "PREDICT:" not in calls[0][-1]["content"])
# 4. проверка после хода: предсказали сдвиг вправо, доска сдвинулась вправо -> hit; в recent_findings — факт
a=agent(); a._c2_level=1; a._c2_level_start=0
ns["_c2_orig_analyze"]=lambda self,sp,an,*x,**k: (setattr(self,"_c2_pending","blue block moves 2 cells right"), setattr(self,"_last_action_result",{"executed":True,"board_changed":True}), write_state(grid(2)))[0]
ns["_c2_analyze"](a, STATE, 1)
rf=a._summarized_knowledge["recent_findings"]
check("4. проверка после хода: hit, факт в recent_findings: %r" % rf[:60], "Prediction held" in rf and "right" in rf)
# 5. промах: предсказали «ничего не изменится», доска изменилась -> MISS
write_state(grid(0)); a=agent(); a._c2_level=1; a._c2_level_start=0
ns["_c2_orig_analyze"]=lambda self,sp,an,*x,**k: (setattr(self,"_c2_pending","nothing changes"), setattr(self,"_last_action_result",{"executed":True,"board_changed":True}), write_state(grid(2)))[0]
ns["_c2_analyze"](a, STATE, 1)
check("5. промах предсказания -> MISS в фактах", "Prediction failed" in a._summarized_knowledge["recent_findings"])
# 6. окно развилки: ходы 0-3 уровня — развилка, 4-й — нет; смена уровня открывает окно заново
write_state(grid(0), level=1); a=agent(); ns["_c2_orig_analyze"]=lambda self,sp,an,*x,**k: None
ns["_c2_analyze"](a, STATE, 0); f0=a._c2_fork; ns["_c2_analyze"](a, STATE, 4); f4=a._c2_fork
write_state(grid(0), level=2); ns["_c2_analyze"](a, STATE, 30); f30=a._c2_fork
check("6. окно: ход0=%s ход4=%s новый уровень ход30=%s" % (f0,f4,f30), f0 and not f4 and f30)
import shutil; shutil.rmtree(TMP, ignore_errors=True)
print("\nИТОГ:", "все проверки пройдены" if ok else "ЕСТЬ ПРОВАЛЫ"); sys.exit(0 if ok else 1)

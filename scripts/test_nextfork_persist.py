"""nextfork: память функций (persist_defs) ставится импортом solver.py поверх AGENTFIX и работает в настоящей песочнице (27.09)."""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "nextfork/src/ARC3-Inference"), str(ROOT / "nextfork/src/tufa-arc-agi-framework/src")]
os.environ.update(MULTIMODAL_UPSCALE="4", MULTIMODAL_CONTEXT="current_grid")
import inference.framework.solver  # noqa: F401 — AGENTFIX + PERSIST
import inference.agent.tool_agent as ta
from inference.agent.runtime_state import RUNTIME_STATE_FILENAME
TMP = ROOT / ".tmp_nfpersist"; TMP.mkdir(exist_ok=True); STATE = TMP / RUNTIME_STATE_FILENAME
STATE.write_text(json.dumps({"current_frame": {"grid": [[0, 1], [2, 3]], "step": 0, "level": 1}, "history": []}))
a = ta.ToolAgent.__new__(ta.ToolAgent)
a.__dict__.update(_session_runtime_dir=TMP, _history_messages=[], _session_total_tokens=0, _session_generated_tokens=0,
                  _last_step_summary=None, _last_action_result=None, _summarized_knowledge=ta._empty_world_model(),
                  _step_env_callback=lambda r: {}, _current_valid_actions=["UP"], _python_timeout=30.0, _tool_output_chars=20000,
                  _noop_guard=ta.NoopGuard(), _noop_guard_blocked=0)
run = lambda code: json.loads(a._run_python_tool(STATE, {"code": code}).content)
r1 = run("from collections import deque\nK = 3\ndef f():\n    return len(deque([1, 2])) + K\nprint(f())")
r2 = run("print(f())")
ok = r1.get("stdout", "").strip() == "5" and r2.get("stdout", "").strip() == "5"
print(("ok   " if ok else "FAIL ") + "функция с импортом и константой переживает вызов (%r / %r)" % (r1.get("stdout"), r2.get("stdout")))
p = a._build_user_prompt(1, valid_actions=["UP"], current_frame=None, history_entries=[])
print(("ok   " if "Currently kept: f" in p else "FAIL ") + "строка о сохранённом в сообщении")
print(("ok   " if getattr(ta.ToolAgent._persistent_history_messages, "_agentfix", False) else "FAIL ") + "AGENTFIX Скотта тоже стоит")

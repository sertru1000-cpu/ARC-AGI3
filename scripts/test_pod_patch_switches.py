"""Проверка выключателей общего патча для пода: встаёт ровно то, что просили.

ЗАЧЕМ. Прогон на поде проверяет одно изменение — слой модели мира ИЛИ расписание.
Патч один на оба, поэтому важно, что WM_LAYER/WM_SCHED действительно решают, какая
подмена произойдёт, а какая нет. Проверяется на заглушках, без движка и без сети.

usage:  .venv/bin/python scripts/test_pod_patch_switches.py
"""
import importlib
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load(layer: str, sched: str):
    for k in [k for k in sys.modules if k.startswith("inference") or k == "wm_pod_patch"]:
        del sys.modules[k]
    os.environ["WM_LAYER"], os.environ["WM_SCHED"] = layer, sched

    ta = types.ModuleType("inference.agent.tool_agent")

    class ToolAgent:
        def _run_python_tool(self, *a, **k):  # исходный метод
            return {}

        def _build_user_prompt(self, action_num, *a, **k):  # исходный метод
            return ""
    ta.ToolAgent = ToolAgent

    class _R:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    ta._ToolDispatchResult = _R
    ta.load_runtime_state = lambda p: (None, [])
    ta._ascii_frame_view_payload = lambda f: None
    ta._ascii_history_view_payload = lambda h: []
    ta._normalize_message_content = lambda c: c if isinstance(c, str) else ""
    ta._extract_reasoning_text = lambda m: ""
    ta._recover_tool_calls_from_markup = lambda *c: []

    sb = types.ModuleType("inference.agent.python_tool_sandbox")
    sb.run_sandboxed_python = lambda **k: {"stdout": "", "error": ""}
    sb.SAFE_BUILTINS = {}

    sv = types.ModuleType("inference.framework.solver")

    class _HarnessGameSession:
        def runtime_limit_reached(self):  # исходный метод
            return False
    sv._HarnessGameSession = _HarnessGameSession

    for name, mod in (("inference", types.ModuleType("inference")),
                      ("inference.agent", types.ModuleType("inference.agent")),
                      ("inference.framework", types.ModuleType("inference.framework")),
                      ("inference.agent.tool_agent", ta),
                      ("inference.agent.python_tool_sandbox", sb),
                      ("inference.framework.solver", sv)):
        sys.modules[name] = mod
    importlib.import_module("wm_pod_patch")
    return (ta.ToolAgent._run_python_tool.__name__,
            ta.ToolAgent._build_user_prompt.__name__,
            sv._HarnessGameSession.runtime_limit_reached.__name__)


ok = lambda b, m: print(("ok   " if b else "СБОЙ ") + m)
for layer, sched, want_l, want_s in (("1", "0", True, False),
                                     ("0", "1", False, True),
                                     ("0", "0", False, False),
                                     ("1", "1", True, True)):
    ln, pn, sn = load(layer, sched)
    got_l = ln == "_wm_run" and pn == "_wm_prompt"
    got_s = sn == "_s_limit"
    ok(got_l == want_l and got_s == want_s,
       "WM_LAYER=%s WM_SCHED=%s -> слой %s, расписание %s" % (layer, sched, got_l, got_s))

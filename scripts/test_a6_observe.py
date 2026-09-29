"""Проверка a6: блок наблюдения на настоящих кадрах ls20 (локальный движок), поверх AGENTFIX Скотта (27.09)."""
import json, logging, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "reference/keith-upstream/src"
sys.path[:0] = [str(B / "ARC3-Inference"), str(B / "tufa-arc-agi-framework/src")]
os.environ.update(MULTIMODAL_UPSCALE="4", MULTIMODAL_CONTEXT="current_grid")
logging.disable(logging.ERROR)
exec(open("/tmp/scott_cell10.py").read(), {"__name__": "scott"})
import inference.agent.tool_agent as ta
from inference.agent.runtime_state import Frame, HistoryEntry
ARM = sys.argv[1] if len(sys.argv) > 1 else "a6_observe"
nb = json.load(open(ROOT / f"pod_arms/{ARM}.ipynb"))
exec("".join(next(c for c in nb["cells"] if "ARM a6" in "".join(c["source"]) and "ARM P" not in "".join(c["source"]))["source"]), {"__name__": "a6"})
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
env = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT / "environment_files")).make("ls20-9607627b")
fr = env.reset(); hist = [HistoryEntry(action="", frame=Frame(grid=tuple(tuple(r) for r in fr.frame[-1]), step=0, level=1))]
for i, a in enumerate(["ACTION4", "ACTION4", "ACTION1"]):
    fr = env.step(GameAction[a]); hist.append(HistoryEntry(action=a, frame=Frame(grid=tuple(tuple(r) for r in fr.frame[-1]), step=i + 1, level=1)))
agent = ta.ToolAgent.__new__(ta.ToolAgent); agent._summarized_knowledge = ta._empty_world_model()
agent.__dict__.update(_last_step_summary=None, _session_runtime_dir=ROOT, _agentfix_last_exec=None)
text = agent._build_user_prompt(3, valid_actions=["UP", "DOWN", "LEFT", "RIGHT"], current_frame=hist[-1].frame, history_entries=hist, previous_step_summary=None)
block = text[text.index("Harness observation"):]
print(block); print("\nзнаков блока:", len(block))
if len(sys.argv) > 1 and sys.argv[1] == "a6b_observe":
    ok = "Board around the objects that moved recently" in text and "act in this same python call" in text
    print("\nv2 вырезка и строка «действуй»:", "ok" if ok else "FAIL")
    # пустой последний ход: вырезка вокруг игрока всё равно есть
    hist2 = hist + [HistoryEntry(action="ACTION3", frame=hist[-1].frame)]
    t2 = agent._build_user_prompt(4, valid_actions=["UP"], current_frame=hist2[-1].frame, history_entries=hist2, previous_step_summary=None)
    print("v2 пустой ход — вырезка вокруг игрока есть:", "ok" if "Board around the objects that moved recently" in t2 else "FAIL")
    agent._context_budget_tokens = 10 ** 6
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": [{"type": "text", "text": text + "\n\nCurrent grid image:"}, {"type": "image_url", "image_url": {"url": "data:x"}}]},
            {"role": "assistant", "content": "a"}, {"role": "user", "content": "next"}]
    out = agent._persistent_history_messages(msgs, tools=None)
    u = [m for m in out if m["role"] == "user"][0]["content"][0]["text"]
    print("v2 копия наблюдения вынута из истории:", "ok" if "Harness observation" not in u and "Current state:" in u and u.endswith("Current grid image:") else "FAIL", "| знаков: было %d, стало %d" % (len(text), len(u)))

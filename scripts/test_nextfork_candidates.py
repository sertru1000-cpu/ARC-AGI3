"""nextfork: кандидаты a6b (NEXTFORK_OBSERVE) и a7 (NEXTFORK_ZOOM) ставятся импортом solver.py поверх AGENTFIX + памяти функций (27.09)."""
import base64, io, json, logging, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "nextfork/src/ARC3-Inference"), str(ROOT / "nextfork/src/tufa-arc-agi-framework/src")]
os.environ.update(MULTIMODAL_UPSCALE="4", MULTIMODAL_CONTEXT="current_grid")
logging.disable(logging.ERROR)
import inference.framework.solver  # noqa: F401
import inference.agent.tool_agent as ta
from inference.agent.runtime_state import Frame, HistoryEntry
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
on = lambda k: os.environ.get(k) == "1"
env = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT / "environment_files")).make("ls20-9607627b")
fr = env.reset(); mk = lambda f, i: Frame(grid=tuple(tuple(int(v) for v in r) for r in f.frame[-1]), step=i, level=1)
hist = [HistoryEntry(action="", frame=mk(fr, 0))]
for i, a in enumerate(["ACTION4", "ACTION4", "ACTION1"]):
    fr = env.step(GameAction[a]); hist.append(HistoryEntry(action=a, frame=mk(fr, i + 1)))
ag = ta.ToolAgent.__new__(ta.ToolAgent); ag._summarized_knowledge = ta._empty_world_model()
ag.__dict__.update(_last_step_summary=None, _session_runtime_dir=ROOT, _agentfix_last_exec=None, _context_budget_tokens=10 ** 6)
p = ag._build_user_prompt(3, valid_actions=["UP"], current_frame=hist[-1].frame, history_entries=hist, previous_step_summary=None)
m = ag._build_user_message(p, hist[-1].frame)
imgs = [x for x in m["content"] if isinstance(x, dict) and x.get("type") == "image_url"] if isinstance(m["content"], list) else []
mods = sys.modules
res = [("память функций стоит", "inference.agent.persist_defs" in mods),
       ("наблюдение: модуль %s" % ("загружен" if on("NEXTFORK_OBSERVE") else "НЕ загружен"), ("inference.agent.observe_v2" in mods) == on("NEXTFORK_OBSERVE")),
       ("наблюдение в сообщении", ("Harness observation" in p and "Board around the objects" in p) == on("NEXTFORK_OBSERVE")),
       ("крупная картинка: модуль %s" % ("загружен" if on("NEXTFORK_ZOOM") else "НЕ загружен"), ("inference.agent.zoom_player" in mods) == on("NEXTFORK_ZOOM")),
       ("картинок в сообщении %d" % len(imgs), len(imgs) == (2 if on("NEXTFORK_ZOOM") else 1))]
for w, ok in res:
    print(("ok   " if ok else "FAIL ") + w)

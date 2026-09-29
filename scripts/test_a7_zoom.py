"""Проверка a7: вторая картинка — окрестность игрока x32, на настоящих кадрах ls20 поверх AGENTFIX Скотта (27.09)."""
import base64, io, json, logging, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "reference/keith-upstream/src"
sys.path[:0] = [str(B / "ARC3-Inference"), str(B / "tufa-arc-agi-framework/src")]
os.environ.update(MULTIMODAL_UPSCALE="4", MULTIMODAL_CONTEXT="current_grid")
logging.disable(logging.ERROR)
exec(open("/tmp/scott_cell10.py").read(), {"__name__": "scott"})
import inference.agent.tool_agent as ta
from inference.agent.runtime_state import Frame, HistoryEntry
from PIL import Image
nb = json.load(open(ROOT / "pod_arms/a7_zoom.ipynb"))
exec("".join(next(c for c in nb["cells"] if "ARM a7" in "".join(c["source"]))["source"]), {"__name__": "a7"})
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
env = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT / "environment_files")).make("ls20-9607627b")
fr = env.reset(); mk = lambda f, i: Frame(grid=tuple(tuple(r) for r in f.frame[-1]), step=i, level=1)
hist = [HistoryEntry(action="", frame=mk(fr, 0))]
agent = ta.ToolAgent.__new__(ta.ToolAgent); agent._summarized_knowledge = ta._empty_world_model()
agent.__dict__.update(_last_step_summary=None, _session_runtime_dir=ROOT, _agentfix_last_exec=None)
ok = bad = 0
def check(c, w):
    global ok, bad
    print(("ok   " if c else "FAIL ") + w); ok += bool(c); bad += (not c)
agent._build_user_prompt(0, valid_actions=["UP"], current_frame=hist[-1].frame, history_entries=hist, previous_step_summary=None)
m0 = agent._build_user_message("x", hist[-1].frame)
check(sum(1 for p in m0["content"] if p.get("type") == "image_url") == 1, "начало уровня (никто не двигался): вырезки нет, одна картинка")
for i, a in enumerate(["ACTION4", "ACTION4", "ACTION1"]):
    fr = env.step(GameAction[a]); hist.append(HistoryEntry(action=a, frame=mk(fr, i + 1)))
agent._build_user_prompt(3, valid_actions=["UP"], current_frame=hist[-1].frame, history_entries=hist, previous_step_summary=None)
m = agent._build_user_message("prompt text", hist[-1].frame)
imgs = [p for p in m["content"] if p.get("type") == "image_url"]
check(len(imgs) == 2, "после ходов: две картинки (вырезка + вся доска)")
z = Image.open(io.BytesIO(base64.b64decode(imgs[0]["image_url"]["url"].split(",", 1)[1])))
full = Image.open(io.BytesIO(base64.b64decode(imgs[1]["image_url"]["url"].split(",", 1)[1])))
check(z.size == (512, 512), "вырезка 512x512 (16 клеток x32): %s" % (z.size,))
check(m["content"][-1] is imgs[1] and full.size[0] >= 256, "вся доска — последней картинкой: %s" % (full.size,))
lab = [p["text"] for p in m["content"] if p.get("type") == "text" and "Zoomed" in p.get("text", "")]
print("  подпись:", lab[0].strip()[:160] if lab else None)
# игрок в кадре: на вырезке есть оранжевый и синий
cols = set(z.getdata())
check(len(cols) >= 3, "в вырезке несколько цветов (игрок + пол + стены): %d" % len(cols))
print("\nитого: ок %d, провалов %d" % (ok, bad))

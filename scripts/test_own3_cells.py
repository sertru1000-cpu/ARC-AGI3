"""Проверка ячеек open/fresh/short на НАСТОЯЩЕМ коде датасета v3 (/tmp/nf_kaggle), без модели (26.09)."""
import importlib, os, sys
from pathlib import Path
# OWN3_BASE — какой бандл (по умолчанию боевой v3); OWN3_PRE — ячейка, исполняемая ДО наших (например, AGENTFIX Скотта)
_BASE = os.environ.get("OWN3_BASE", str(Path(__file__).resolve().parents[1] / "reference/nextfork-v3"))
sys.path.insert(0, _BASE + "/src/ARC3-Inference"); sys.path.insert(0, _BASE + "/src/tufa-arc-agi-framework/src")
if os.environ.get("OWN3_PRE"):
    os.environ.setdefault("MULTIMODAL_UPSCALE", "4"); os.environ.setdefault("MULTIMODAL_CONTEXT", "current_grid")
    exec(open(os.environ["OWN3_PRE"]).read(), {"__name__": "pre"})
import inference.agent.tool_agent as ta
from inference.agent.runtime_state import Frame, HistoryEntry
ROOT = Path(__file__).resolve().parents[1]

def cell(name):
    return (ROOT / f"kernels/notebooks_nextfork_{name}1h/cell15.py").read_text()

def agent():
    a = object.__new__(ta.ToolAgent)
    a._session_runtime_dir = Path("/tmp/g1"); a._summarized_knowledge = ta._empty_world_model()
    a._history_messages = [{"role": "user", "content": "x"}]; a._context_budget_tokens = 10 ** 6
    return a

def fr(level, step):
    return Frame(grid=((0,) * 4,) * 4, step=step, level=level)

ok = 0; bad = 0
def check(cond, what):
    global ok, bad
    print(("ok   " if cond else "FAIL ") + what); ok += bool(cond); bad += (not cond)

orig_build = ta.ToolAgent._build_user_prompt; orig_persist = ta.ToolAgent._persistent_history_messages
# OPEN
exec(cell("open"), {"__name__": "c"})
a = agent()
t1 = a._build_user_prompt(0, valid_actions=["UP"], current_frame=fr(1, 0), history_entries=[])
t2 = a._build_user_prompt(1, valid_actions=["UP"], current_frame=fr(1, 1), history_entries=[])
t3 = a._build_user_prompt(2, valid_actions=["UP"], current_frame=fr(1, 2), history_entries=[])
t4 = a._build_user_prompt(3, valid_actions=["UP"], current_frame=fr(2, 3), history_entries=[])
check("Opening protocol" in t1 and "Opening protocol" in t2 and "Opening protocol" not in t3, "open: блок в вызовах 1-2 уровня, не в 3-м")
check("Opening protocol" in t4, "open: новый уровень — блок снова")
a._session_runtime_dir = Path("/tmp/g2")
check("Opening protocol" in a._build_user_prompt(0, valid_actions=["UP"], current_frame=fr(2, 0), history_entries=[]), "open: новая игра на том же номере уровня — блок снова")
ta.ToolAgent._build_user_prompt = orig_build
# FRESH
exec(cell("fresh"), {"__name__": "c"})
a = agent(); a._summarized_knowledge.update(goal_model="reach the red door", current_plan="go right", world_model="block slides 5")
h29 = [HistoryEntry(action="RIGHT", frame=fr(1, i)) for i in range(29)]
t = a._build_user_prompt(29, valid_actions=["UP"], current_frame=fr(1, 29), history_entries=h29)
check("FRESH START" not in t and a._history_messages, "fresh: 29 ходов — сброса нет")
h30 = h29 + [HistoryEntry(action="UP", frame=fr(1, 29))]
t = a._build_user_prompt(30, valid_actions=["UP"], current_frame=fr(1, 30), history_entries=h30)
check("FRESH START" in t and "reach the red door" in t and "RIGHT x29" in t, "fresh: 30 ходов — сброс, отброшенная цель и испробованное в тексте")
check(a._history_messages == [] and a._summarized_knowledge["goal_model"] == "" and a._summarized_knowledge["world_model"] == "block slides 5", "fresh: история очищена, цель стёрта, механика сохранена")
check("reach the red door" not in t.split("FRESH START")[0], "fresh: старая цель не попала в блок модели мира")
t = a._build_user_prompt(31, valid_actions=["UP"], current_frame=fr(1, 31), history_entries=h30 + h30[:1])
check("FRESH START" not in t, "fresh: сразу после сброса повтора нет")
h90 = [HistoryEntry(action="UP", frame=fr(1, i)) for i in range(90)]
for n in (60, 90):
    a._build_user_prompt(n, valid_actions=["UP"], current_frame=fr(1, n), history_entries=h90[:n])
check(a._f_state["resets"] == 2, "fresh: не больше 2 сбросов на уровень")
t = a._build_user_prompt(91, valid_actions=["UP"], current_frame=fr(2, 91), history_entries=h90 + [HistoryEntry(action="UP", frame=fr(2, 91))])
check("FRESH START" not in t and a._f_state["resets"] == 0, "fresh: новый уровень — счёт заново")
ta.ToolAgent._build_user_prompt = orig_build
# SHORT
exec(cell("short"), {"__name__": "c"})
a = agent()
full = orig_build(a, 5, valid_actions=["UP", "DOWN"], current_frame=fr(1, 5), history_entries=[])
msgs = [{"role": "system", "content": "S"},
        {"role": "user", "content": [{"type": "text", "text": full + "\n\nCurrent grid image:"}, {"type": "image_url", "image_url": {"url": "data:x"}}]},
        {"role": "assistant", "content": "ok", "tool_calls": []},
        {"role": "user", "content": "You have not acted yet. Investigate first. Then investigate and revise ..."},
        {"role": "assistant", "content": "ok2"}]
out = a._persistent_history_messages(msgs, tools=None)
u = [m for m in out if m["role"] == "user"]
txt = u[0]["content"][0]["text"]
check("Current state:" in txt and "Valid actions right now:" in txt and "Only tool" not in txt and txt.endswith("Current grid image:"), "short: оставлены строки состояния, инструкция вынута, подпись картинки на месте")
check(u[0]["content"][1]["type"] == "image_url", "short: картинка не тронута")
check(u[1]["content"] == "You have not acted yet", "short: напоминание сжато до первой фразы")
check([m for m in out if m["role"] == "assistant"][0]["content"] == "ok", "short: ответы модели не тронуты")
print("короче в %.1f раза (%d -> %d знаков)" % (len(full) / len(txt), len(full), len(txt)))
print("\nитого: ок %d, провалов %d" % (ok, bad))

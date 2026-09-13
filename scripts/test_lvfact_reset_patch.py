"""Проверки слоёв «факт о завершении уровня» и «RESET при застревании» на настоящем ToolAgent бандла.

lvfact: при росте уровня из истории берётся завершивший ход, последние ходы перед ним, число ходов на
уровне и доска перед завершением; полный текст -- один раз, дальше строка; без роста уровня промпт стоковый.
reset: после STALL ходов без взятия обвязка вызывает _execute_auto_reset у сессии (через bound-метод
step_env), кладёт сводку во вход на NOTE_TURNS ходов, не больше MAX сбросов на уровень; после взятия
уровня счётчики сбрасываются; при level_completed/game_over в последнем результате сброса нет.
Боевая ветка не тронута; ячейки компилируются.

usage:  .venv/bin/python scripts/test_lvfact_reset_patch.py --bundle <бандл keithtyser>
"""
import argparse
import ast
import sys
import types
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True)
a = ap.parse_args()
sys.path.insert(0, str(Path(a.bundle) / "src" / "ARC3-Inference"))
sys.path.insert(0, str(Path(a.bundle) / "src" / "tufa-arc-agi-framework" / "src"))
import inference.agent.tool_agent as wta  # noqa: E402

fails = []


def check(b, m):
    print(("ok   " if b else "СБОЙ ") + m)
    if not b:
        fails.append(m)


class Frame:
    def __init__(self, level, tag):
        self.level = level; self.ascii = "board-L%d-%s" % (level, tag)


def ent(action, level, tag):
    return types.SimpleNamespace(action=action, frame=Frame(level, tag))


orig_prompt, orig_run = wta.ToolAgent._build_user_prompt, wta.ToolAgent._run_python_tool

# ================= lvfact =================
cell = open("kernels/notebooks_stockflash_lvfact/cell15.py", encoding="utf-8").read()
compile(cell, "lvfact", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
ns = {"TRUE_SUBMISSION": True}; exec(cell, ns)
check(wta.ToolAgent._build_user_prompt is orig_prompt, "lvfact: боевая ветка не тронута")
ns = {"TRUE_SUBMISSION": False}; exec(cell, ns)
check(wta.ToolAgent._build_user_prompt is not orig_prompt, "lvfact: обёртка установлена")
ns["_lf_orig_prompt"] = lambda self, n, *a, **k: "STOCK"
agent = types.SimpleNamespace()
hist = [ent("RIGHT", 1, "a"), ent("UP", 1, "b"), ent("MOUSE(row=31, col=43)", 1, "c"), ent("LEFT", 1, "d"), ent("DOWN", 2, "first")]
t1 = wta.ToolAgent._build_user_prompt(agent, 3, current_frame=Frame(1, "x"), history_entries=hist[:3])
check(t1 == "STOCK", "уровень 1: промпт стоковый")
t2 = wta.ToolAgent._build_user_prompt(agent, 6, current_frame=Frame(2, "first"), history_entries=hist)
check(t2.startswith("LEVEL 2 FACT") and "completing move was DOWN" in t2 and "RIGHT, UP, MOUSE(row=31, col=43), LEFT" in t2
      and "after 5 moves" in t2 and "board-L1-d" in t2 and t2.endswith("STOCK"), "переход на уровень 2: полный факт с доской перед завершением")
t3 = wta.ToolAgent._build_user_prompt(agent, 7, current_frame=Frame(2, "y"), history_entries=hist + [ent("UP", 2, "z")])
check(t3.startswith("LEVEL 2 FACT") and "board-L1-d" not in t3 and "completing move was DOWN" in t3, "следующий ход: только строка, без доски")
check(ns["_lf_stats"]["levels"] == 1 and ns["_lf_stats"]["facts"] == 1, "lvfact: статистика levels=1, facts=1")
wta.ToolAgent._build_user_prompt = orig_prompt

# ================= reset =================
cell = open("kernels/notebooks_stockflash_reset/cell15.py", encoding="utf-8").read()
compile(cell, "reset", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
ns = {"TRUE_SUBMISSION": True}; exec(cell, ns)
check(wta.ToolAgent._build_user_prompt is orig_prompt and wta.ToolAgent._run_python_tool is orig_run, "reset: боевая ветка не тронута")
ns = {"TRUE_SUBMISSION": False}; exec(cell, ns)
check(wta.ToolAgent._build_user_prompt is not orig_prompt and wta.ToolAgent._run_python_tool is not orig_run, "reset: обе обёртки установлены")
ns["_rs_orig_prompt"] = lambda self, n, *a, **k: "STOCK"
ns["_rs_orig_run"] = lambda self, p, args: wta._ToolDispatchResult(content="{}", step_executed=True)


class Sess:
    def __init__(self):
        self.resets = 0
    def step_env(self, arguments):
        return {}
    def _execute_auto_reset(self):
        self.resets += 1


sess = Sess()
agent = types.SimpleNamespace(_step_env_callback=sess.step_env, _last_action_result={})
h = [ent("RIGHT", 1, "a%d" % (i % 5)) for i in range(120)]   # 120 ходов, 5 различных досок, много возвратов
wta.ToolAgent._build_user_prompt(agent, 0, current_frame=Frame(1, "s"), history_entries=[])
agent._last_action_result = {"action_num": 50, "level": 1}
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "x"})
check(sess.resets == 0, "50 ходов: сброса нет")
wta.ToolAgent._build_user_prompt(agent, 50, current_frame=Frame(1, "s"), history_entries=h[:50])
agent._last_action_result = {"action_num": 101, "level": 1, "level_completed": True}
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "x"})
check(sess.resets == 0, "101 ход, но последний результат level_completed: сброса нет")
agent._last_action_result = {"action_num": 101, "level": 1}
wta.ToolAgent._build_user_prompt(agent, 101, current_frame=Frame(1, "s"), history_entries=h[:101])
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "x"})
check(sess.resets == 1 and ns["_rs_stats"]["resets"] == 1, "101 ход без взятия: обвязка сделала RESET")
t = wta.ToolAgent._build_user_prompt(agent, 102, current_frame=Frame(1, "s"), history_entries=h[:102])
check(t.startswith("HARNESS RESET (attempt 1 of 2)") and "RIGHT x" in t and "5 distinct boards" in t and "returned to an already-seen board" in t
      and t.endswith("STOCK"), "сводка во входе: гистограмма, доски, возвраты")
for i in range(3):
    t = wta.ToolAgent._build_user_prompt(agent, 103 + i, current_frame=Frame(1, "s"), history_entries=h[:103 + i])
check(t == "STOCK", "сводка показана 3 хода и убрана")
agent._last_action_result = {"action_num": 150, "level": 1}
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "x"})
check(sess.resets == 1, "150 ходов (49 после сброса): второго сброса ещё нет")
agent._last_action_result = {"action_num": 205, "level": 1}
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "x"})
check(sess.resets == 2, "205 ходов (104 после первого сброса): второй сброс")
agent._last_action_result = {"action_num": 320, "level": 1}
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "x"})
check(sess.resets == 2 and ns["_rs_stats"]["blocked_max"] == 1, "лимит 2 сброса на уровень: третьего нет")
wta.ToolAgent._build_user_prompt(agent, 321, current_frame=Frame(2, "n"), history_entries=h)
agent._last_action_result = {"action_num": 330, "level": 2}
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "x"})
check(sess.resets == 2 and agent._rs_state["resets"] == 0 and agent._rs_state["level_start"] == 321, "новый уровень: счётчики сброшены, сброса нет")
agent._last_action_result = {"action_num": 425, "level": 2}
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "x"})
check(sess.resets == 3, "уровень 2, 104 хода без взятия: сброс снова доступен")
# второй триггер: 40 вызовов без уровня при малом числе ходов
sess2 = Sess(); agent2 = types.SimpleNamespace(_step_env_callback=sess2.step_env, _last_action_result={})
wta.ToolAgent._build_user_prompt(agent2, 0, current_frame=Frame(1, "s"), history_entries=[])
for i in range(39):   # первый промпт обнуляет счёт (новый уровень), дальше 39 вызовов
    wta.ToolAgent._build_user_prompt(agent2, 5, current_frame=Frame(1, "s"), history_entries=h[:5])
agent2._last_action_result = {"action_num": 5, "level": 1}
wta.ToolAgent._run_python_tool(agent2, Path("/tmp/x"), {"code": "x"})
check(sess2.resets == 0, "триггер по вызовам: 39 вызовов и 5 ходов -- сброса нет")
wta.ToolAgent._build_user_prompt(agent2, 5, current_frame=Frame(1, "s"), history_entries=h[:5])
wta.ToolAgent._run_python_tool(agent2, Path("/tmp/x"), {"code": "x"})
check(sess2.resets == 1 and ns["_rs_stats"].get("by_calls") == 1, "триггер по вызовам: 40 вызовов при 5 ходах -- сброс [по вызовам]")
t = wta.ToolAgent._build_user_prompt(agent2, 6, current_frame=Frame(1, "s"), history_entries=h[:6])
check("5 moves and 40 model calls" in t, "сводка называет и ходы, и вызовы")
for i in range(38):   # один промпт уже ушёл на проверку сводки
    wta.ToolAgent._build_user_prompt(agent2, 6, current_frame=Frame(1, "s"), history_entries=h[:6])
wta.ToolAgent._run_python_tool(agent2, Path("/tmp/x"), {"code": "x"})
check(sess2.resets == 1, "после сброса счёт вызовов начинается заново: 39 -- второго сброса нет")
wta.ToolAgent._build_user_prompt(agent2, 6, current_frame=Frame(1, "s"), history_entries=h[:6])
wta.ToolAgent._run_python_tool(agent2, Path("/tmp/x"), {"code": "x"})
check(sess2.resets == 2, "40 вызовов после первого сброса -- второй сброс")
wta.ToolAgent._build_user_prompt, wta.ToolAgent._run_python_tool = orig_prompt, orig_run

# ================= explore =================
cell = open("kernels/notebooks_stockflash_explore/cell15.py", encoding="utf-8").read()
compile(cell, "explore", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
ns = {"TRUE_SUBMISSION": True}; exec(cell, ns)
check(wta.ToolAgent._build_user_prompt is orig_prompt and wta.ToolAgent._run_python_tool is orig_run, "explore: боевая ветка не тронута")
ns = {"TRUE_SUBMISSION": False}; exec(cell, ns)
check(wta.ToolAgent._build_user_prompt is not orig_prompt and wta.ToolAgent._run_python_tool is not orig_run, "explore: обе обёртки установлены")
ns["_ex_orig_prompt"] = lambda self, n, *a, **k: "STOCK"
ns["_ex_orig_run"] = lambda self, p, args: wta._ToolDispatchResult(content="{}", step_executed=True)
grid = [[0] * 64 for _ in range(64)]
for r in range(10, 13):
    for c in range(10, 13): grid[r][c] = 5          # объект 3x3
for r in range(30, 32):
    for c in range(40, 45): grid[r][c] = 7          # объект 2x5
objs = ns["_ex_objects"](grid)
check(len(objs) == 2 and objs[0]["size"] == 10 and objs[0]["color"] == 7 and objs[1]["row"] == 11 and objs[1]["col"] == 11,
      "сегментация: два объекта без фона, крупнейший первым, центры верные")


class GFrame:
    def __init__(self): self.grid = grid; self.ascii = "x"; self.level = 1; self.step = 0; self.shape = [64, 64]


class ESess:
    def __init__(self): self.calls = []; self.resets = 0
    def step_env(self, arguments):
        acts = arguments["actions"]; self.calls.append(acts)
        a = acts[0]["action"]
        return {"executed": True, "executed_count": len(acts), "board_changed": a in ("RIGHT", "MOUSE"), "level_completed": False, "game_over": False}
    def _execute_auto_reset(self): self.resets += 1


wta.load_runtime_state = lambda p: (GFrame(), [])
wta._ascii_frame_view_payload = lambda f: {"grid": f.grid, "ascii": f.ascii, "level": f.level, "step": 0, "shape": [64, 64]}
esess = ESess()
agent = types.SimpleNamespace(_step_env_callback=esess.step_env, _last_action_result={}, _current_valid_actions=["UP", "DOWN", "LEFT", "RIGHT", "MOUSE"])
wta.ToolAgent._build_user_prompt(agent, 0, current_frame=Frame(1, "s"), history_entries=[])
agent._last_action_result = {"action_num": 60, "level": 1}
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "x"})
check(len(esess.calls) == 0, "60 ходов: исследования нет")
agent._last_action_result = {"action_num": 101, "level": 1}
wta.ToolAgent._build_user_prompt(agent, 101, current_frame=Frame(1, "s"), history_entries=[])
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "x"})
arrows = [c[0]["action"] for c in esess.calls if c[0]["action"] != "MOUSE"]; clicks = [c[0] for c in esess.calls if c[0]["action"] == "MOUSE"]
check(arrows.count("UP") == 3 and arrows.count("RIGHT") == 3 and "SPACE" not in arrows and len(clicks) == 2 and clicks[0]["row"] == 30,
      "исследование: 4 стрелки x3 (SPACE не в valid -- пропущен), 2 клика по центрам объектов")
t = wta.ToolAgent._build_user_prompt(agent, 115, current_frame=Frame(1, "s"), history_entries=[])
check(t.startswith("HARNESS EXPLORATION (attempt 1 of 2)") and "RIGHT changed the board 3/3" in t and "UP changed the board 0/3" in t
      and "CHANGED the board" in t and "no level completed" in t and t.endswith("STOCK"), "отчёт во входе: стрелки, клики, итог")
check(ns["_ex_stats"]["explorations"] == 1 and ns["_ex_stats"]["moves"] == 14, "статистика: 1 исследование, 14 ходов")
n_before = len(esess.calls)
agent._last_action_result = {"action_num": 130, "level": 1}
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "x"})
check(len(esess.calls) == n_before, "через 29 ходов после исследования второго нет")
agent._last_action_result = {"action_num": 215, "level": 1}
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "x"})
clicks2 = [c[0] for c in esess.calls[n_before:] if c[0]["action"] == "MOUSE"]
check(ns["_ex_stats"]["explorations"] == 2 and len(clicks2) == 0, "второе исследование: те же объекты повторно не кликаются")
agent._last_action_result = {"action_num": 330, "level": 1}
wta.ToolAgent._run_python_tool(agent, Path("/tmp/x"), {"code": "x"})
check(ns["_ex_stats"]["explorations"] == 2 and ns["_ex_stats"]["blocked_max"] == 1, "лимит 2 исследования на уровень")
esess2 = ESess(); agent3 = types.SimpleNamespace(_step_env_callback=esess2.step_env, _last_action_result={}, _current_valid_actions=["UP", "MOUSE"])
wta.ToolAgent._build_user_prompt(agent3, 0, current_frame=Frame(1, "s"), history_entries=[])
for i in range(40):
    wta.ToolAgent._build_user_prompt(agent3, 3, current_frame=Frame(1, "s"), history_entries=[])
agent3._last_action_result = {"action_num": 3, "level": 1}
wta.ToolAgent._run_python_tool(agent3, Path("/tmp/x"), {"code": "x"})
check(len(esess2.calls) > 0 and ns["_ex_stats"].get("by_calls") == 1, "explore: триггер по вызовам (40 вызовов при 3 ходах) -- исследование [по вызовам]")
wta.ToolAgent._build_user_prompt, wta.ToolAgent._run_python_tool = orig_prompt, orig_run

print("\nИТОГ: %d сбоев" % len(fails) if fails else "\nВСЕ ПРОВЕРКИ ПРОШЛИ")
sys.exit(1 if fails else 0)

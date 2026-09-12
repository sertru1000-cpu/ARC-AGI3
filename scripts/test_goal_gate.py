"""Проверки ворот цели (12.09) на поддельном движке: без цели — только короткие пробы и их мало;
set_goal открывает action(); измеритель считается до/после пачки; после N пачек без роста цель
опровергается и action() закрыт до новой цели; смена уровня подтверждает цель и сбрасывает пробы.
usage: MY_AGENT_GOAL_GATE=1 .venv/bin/python scripts/test_goal_gate.py
"""
import os, sys
os.environ.setdefault("MY_AGENT_GOAL_GATE", "1"); os.environ.setdefault("MY_AGENT_VERIFY_GATE_ATTEMPTS", "0")  # старые ворота теории не мешают проверке
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agent.harness.sandbox import Sandbox


class Frame:
    def __init__(self, grid, level=0, state="NOT_FINISHED"):
        self.frame = [grid.tolist()]; self.levels_completed = level; self.state = state
        self.available_actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"]


class Env:
    """Поле 64x64, «агент» — клетка цвета 5, RIGHT двигает его вправо; уровень берётся на col 10."""
    def __init__(self):
        self.col = 0; self.level = 0
    def grid(self):
        g = np.zeros((64, 64), dtype=int); g[5, self.col] = 5; g[63, :] = self.level; return g
    def step(self, engine, payload):
        if engine == "ACTION4": self.col = min(63, self.col + 1)
        if engine == "ACTION1": pass
        if self.col >= 10 and self.level == 0: self.level = 1; self.col = 0
        return Frame(self.grid(), self.level)


def make():
    env = Env(); sb = Sandbox(env_step=env.step, budget_left=lambda: 500)
    sb.update_frame(Frame(env.grid(), 0), None); return env, sb


checks = 0
def ok(cond, msg):
    global checks; checks += 1
    if not cond: raise SystemExit("FAIL %d: %s" % (checks, msg))
    print("ok %2d  %s" % (checks, msg))

env, sb = make()
ok(sb.goal_gate, "ворота включены через MY_AGENT_GOAL_GATE=1")
r = sb.run_code("r = action(['RIGHT']); print(r['board_changed'])")
ok(r.error is None and r.actions_executed == 1, "проба из 1 хода без цели проходит")
r = sb.run_code("action(['RIGHT','RIGHT','RIGHT'])")
ok(r.error and "goal gate" in r.error, "пачка из 3 ходов без цели заблокирована: " + str(r.error)[:60])
for i in range(3):
    r = sb.run_code("action(['UP'])")
ok(r.error is None, "пробы 2–4 без цели проходят")
r = sb.run_code("action(['UP'])")
ok(r.error and "goal gate" in r.error, "пятая проба без цели заблокирована (лимит 4)")
ok(sb.goal_stats["blocked"] == 2 and sb.goal_stats["probes"] == 4, "счётчики: blocked=2, probes=4")
r = sb.run_code("set_goal('short', lambda g: 0)")
ok(r.error and "checkable condition" in r.error, "слишком короткий текст цели отвергнут")
r = sb.run_code("set_goal('move the 5-cell to column 10', 'not callable')")
ok(r.error and "must be a function" in r.error, "неисполняемый измеритель отвергнут")
r = sb.run_code("print(set_goal('move the 5-cell to column 10', lambda g: int(np.argmax(g[5]))))")
ok(r.error is None and "progress_now" in (r.output or ""), "цель с измерителем принята: " + (r.output or "").strip()[:70])
r = sb.run_code("r = action(['RIGHT','RIGHT','RIGHT']); print(r['goal_progress'])")
ok(r.error is None and "'delta': 3.0" in (r.output or ""), "после пачки измеритель вырос на 3: " + (r.output or "").strip()[:80])
ok("progress values" in sb.goal_status(), "строка статуса содержит значения: " + sb.goal_status()[:80])
# три пачки без роста -> опровержение
for i in range(3):
    r = sb.run_code("r = action(['UP','UP']); print(r['goal_progress'])")
ok("'falsified': True" in (r.output or ""), "после 3 пачек без роста цель опровергнута")
r = sb.run_code("action(['RIGHT'])")
ok(r.error and "FALSIFIED" in r.error, "после опровержения action() закрыт: " + str(r.error)[:60])
r = sb.run_code("set_goal('move the 5-cell to column 10', lambda g: int(np.argmax(g[5])))")
ok(r.error and "already falsified" in r.error, "та же цель дословно повторно отвергнута")
r = sb.run_code("set_goal('reach column 10 with the 5-cell (measure: column index)', lambda g: int(np.argmax(g[5])))")
ok(r.error and "same measure under a new name" in r.error, "А: новая формулировка с ТЕМ ЖЕ измерителем отвергнута")
r = sb.run_code("set_goal('reach column 10 with the 5-cell (measure: column index)', lambda g: 100 + int(np.argmax(g[5])))")
ok(r.error is None, "А: другой измеритель (другое число на текущей доске) принят")
r = sb.run_code("for _ in range(3):\n    r = action(['RIGHT','RIGHT','RIGHT'])\n    if r.get('level_completed'): break\nprint(r.get('level_completed'), r['level'])")
ok("True 1" in (r.output or ""), "уровень взят (пачки по 3 — старые ворота теории режут длиннее): " + (r.output or "").strip()[:40] + str(r.error or "")[-80:])
ok(sb.goal is None and sb.goal_stats["confirmed"] == 1 and sb.goal_probes_used == 0, "цель подтверждена, сброшена, пробы обнулены")
r = sb.run_code("action(['RIGHT','RIGHT','RIGHT'])")
ok(r.error and "goal gate" in r.error, "на новом уровне длинная пачка снова требует цель")
# В: лимит ходов на гипотезу
os.environ["MY_AGENT_GOAL_MOVES"] = "4"
env3, sb3 = make()
ok(sb3.goal_moves_cap == 4, "В: лимит ходов на гипотезу включён через MY_AGENT_GOAL_MOVES=4")
r = sb3.run_code("set_goal('move the 5-cell to column 10', lambda g: int(np.argmax(g[5])))")
r = sb3.run_code("r = action(['UP','UP','UP']); print(r['goal_progress']['moves_without_progress'])")
ok("3" in (r.output or "") and sb3.goal.get("closed") is None, "В: 3 хода без роста -- ещё не опровергнута")
r = sb3.run_code("r = action(['UP','UP']); print(r['goal_progress'].get('falsified'))")
ok("True" in (r.output or ""), "В: 5 ходов без роста при лимите 4 -- опровергнута по лимиту ходов раньше терпения пачек")
os.environ["MY_AGENT_GOAL_MOVES"] = "0"
os.environ["MY_AGENT_GOAL_GATE"] = "0"
env2, sb2 = make()
ok(not sb2.goal_gate and sb2.goal_status() == "", "с выключенными воротами поведение прежнее, статус пуст")
r = sb2.run_code("action(['RIGHT']*5)")
ok(r.error is None and r.actions_executed == 5, "без ворот пачка из 5 ходов проходит")
print("ВСЕ %d ПРОВЕРОК ПРОШЛИ" % checks)

"""Проверка патча «проверка на петлю» на НАСТОЯЩЕМ ToolAgent из бандла Duck.

Урок 09.09: код по памяти о чужом коде падает в бою. Здесь импортируется настоящий
`inference.agent.tool_agent`, патч ложится на настоящий класс, проверяется сигнатура
(`_build_user_prompt(self, action_num, *, valid_actions, current_frame=None, ...)`),
поведение при первом заходе, при возврате, при повторном возврате, и то, что выход
модели не затрагивается.

usage:  .venv/bin/python scripts/test_loop_patch.py
"""
import sys, types
from pathlib import Path

BUNDLE = Path("/tmp/duckbundle")
sys.path.insert(0, str(BUNDLE / "src" / "ARC3-Inference"))
sys.path.insert(0, str(BUNDLE / "src" / "tufa-arc-agi-framework" / "src"))
import inference.agent.tool_agent as ta  # noqa: E402

Agent = ta.ToolAgent
ok = lambda b, m: print(("ok   " if b else "СБОЙ ") + m)

import inspect
sig = inspect.signature(Agent._build_user_prompt)
ok(list(sig.parameters)[:2] == ["self", "action_num"], "сигнатура: (self, action_num, ...)")
ok("current_frame" in sig.parameters and "previous_step_summary" in sig.parameters,
   "в сигнатуре есть current_frame и previous_step_summary")

# исходный метод подменяем на предсказуемый, чтобы мерить ТОЛЬКО нашу добавку
Agent._build_user_prompt = lambda self, action_num, **kw: "БАЗОВЫЙ ПРОМПТ"

TRUE_SUBMISSION = False
bm = types.SimpleNamespace(solver=types.SimpleNamespace(max_runtime_s_per_game=7920.0))
ns = {"TRUE_SUBMISSION": TRUE_SUBMISSION, "bm": bm, "print": print}
exec(compile(Path("kernels/notebooks_stockflash_loop/cell15.py").read_text(encoding="utf-8"),
             "cell15", "exec"), ns)
ok(bm.solver.max_runtime_s_per_game == 1500.0, "проба: потолок 25 минут выставлен")

agent = object.__new__(Agent)
F = lambda a, lvl, st: types.SimpleNamespace(ascii=a, level=lvl, step=st)
call = lambda frame, summ=None: Agent._build_user_prompt(
    agent, 0, valid_actions=["UP"], current_frame=frame, history_entries=[],
    previous_step_summary=summ)

A, B = "доска-А", "доска-Б"
t1 = call(F(A, 1, 1))
ok(t1 == "БАЗОВЫЙ ПРОМПТ", "первый заход: ничего не дописано")
t2 = call(F(B, 1, 2), {"executed_actions": ["UP", "UP"]})
ok(t2 == "БАЗОВЫЙ ПРОМПТ", "новое состояние: ничего не дописано")
t3 = call(F(A, 1, 3), {"executed_actions": ["LEFT"]})
ok("LOOP CHECK" in t3, "возврат в виденное: блок дописан")
ok("step 1" in t3, "назван шаг, на котором состояние было впервые")
ok("UP, UP" in t3, "названо, что делали из этого состояния")
ok(t3.startswith("БАЗОВЫЙ ПРОМПТ"), "исходный промпт сохранён целиком")
t4 = call(F(A, 1, 4), {"executed_actions": ["DOWN"]})
ok("3-th time" in t4 or "3-" in t4, "счётчик повторов растёт")
t5 = call(F(A, 2, 5), {"executed_actions": ["DOWN"]})
ok(t5 == "БАЗОВЫЙ ПРОМПТ", "та же доска на ДРУГОМ уровне — не считается возвратом")

# устойчивость: кадра нет, ascii пустой, сводка кривая
ok(call(None) == "БАЗОВЫЙ ПРОМПТ", "кадра нет — молча пропускаем")
ok(call(F("", 1, 6)) == "БАЗОВЫЙ ПРОМПТ", "пустая доска — молча пропускаем")
ok(call(F(A, 1, 7), {"executed_actions": None}) .startswith("БАЗОВЫЙ ПРОМПТ"),
   "кривая сводка не роняет учёт")
ok(ns["_LOOP_STATS"]["fail"] == 0, "сбоев учёта: %d" % ns["_LOOP_STATS"]["fail"])
ok(ns["_LOOP_STATS"]["hits"] >= 3, "возвратов насчитано: %d" % ns["_LOOP_STATS"]["hits"])

# боевая ветка: потолок не трогаем
ns2 = {"TRUE_SUBMISSION": True,
       "bm": types.SimpleNamespace(solver=types.SimpleNamespace(max_runtime_s_per_game=7920.0)),
       "print": lambda *a, **k: None}
exec(compile(Path("kernels/notebooks_stockflash_loop/cell15.py").read_text(encoding="utf-8"),
             "cell15", "exec"), ns2)
ok(ns2["bm"].solver.max_runtime_s_per_game == 7920.0, "бой: потолок на игру остался стоковым")

# --- ИСПРАВЛЕНИЕ 10.09: перезаход солвера в тот же шаг — не возврат ---
Agent._build_user_prompt = lambda self, action_num, **kw: "БАЗОВЫЙ ПРОМПТ"
ns3 = {"TRUE_SUBMISSION": False,
       "bm": types.SimpleNamespace(solver=types.SimpleNamespace(max_runtime_s_per_game=7920.0)),
       "print": lambda *a, **k: None}
exec(compile(Path("kernels/notebooks_stockflash_loop/cell15.py").read_text(encoding="utf-8"),
             "cell15", "exec"), ns3)
ag = object.__new__(Agent)
c3 = lambda frame, summ=None: Agent._build_user_prompt(
    ag, 0, valid_actions=["UP"], current_frame=frame, history_entries=[], previous_step_summary=summ)
C, D = "доска-В", "доска-Г"
c3(F(C, 1, 10), {"executed_actions": ["RIGHT"]})            # первый визит C, пришли через RIGHT
r1 = c3(F(C, 1, 10), {"executed_actions": ["RIGHT"]})       # перезаход того же шага
ok(r1 == "БАЗОВЫЙ ПРОМПТ", "перезаход того же шага: предупреждения НЕТ")
r2 = c3(F(C, 1, 10), {"executed_actions": ["RIGHT"]})       # и ещё раз
ok(r2 == "БАЗОВЫЙ ПРОМПТ", "второй перезаход: предупреждения НЕТ")
ok(ns3["_LOOP_STATS"]["reentry"] == 2, "перезаходы посчитаны отдельно: %d" % ns3["_LOOP_STATS"]["reentry"])
ok(ns3["_LOOP_STATS"]["hits"] == 0, "возвратов не насчитано: %d" % ns3["_LOOP_STATS"]["hits"])
c3(F(D, 1, 11), {"executed_actions": ["DOWN"]})             # из C ушли через DOWN
r3 = c3(F(C, 1, 12), {"executed_actions": ["LEFT"]})        # настоящий возврат в C
ok("LOOP CHECK" in r3, "настоящий возврат (шаг больше): предупреждение есть")
ok("DOWN" in r3 and "RIGHT" not in r3,
   "«что делали» — ходы ИЗ состояния (DOWN), а не ходы В него (RIGHT)")

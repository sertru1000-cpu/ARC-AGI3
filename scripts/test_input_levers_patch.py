"""Проверка сборки 2а (разбор перехода + проверка на петлю v2) на НАСТОЯЩЕМ ToolAgent из бандла.

Проверяется: разбор не появляется на первом шаге; появляется со второго и описывает сдвиг верно;
перезаход того же шага получает ТОТ ЖЕ разбор (а не «ничего не изменилось») и не даёт ложной петли;
холостой ход даёт «NO»; смена уровня даёт строку уровня; массовое изменение даёт сводку; полоса
по краю поля помечается как HUD; настоящий возврат даёт и разбор, и блок петли, причём разбор ПЕРВЫМ;
отсутствие кадра не роняет учёт; выход модели не затронут; боевой потолок не тронут.
Плюс сверка на 25 снимках пробы: сколько разборов стали сводками и сколько полос помечено HUD.
"""
import sys, types, glob
from pathlib import Path

B = Path("/tmp/duckbundle")
sys.path.insert(0, str(B / "src" / "ARC3-Inference")); sys.path.insert(0, str(B / "src" / "tufa-arc-agi-framework" / "src"))
import inference.agent.tool_agent as ta  # noqa: E402

Agent = ta.ToolAgent
ok = lambda b, m: print(("ok   " if b else "СБОЙ ") + m)
Agent._build_user_prompt = lambda self, action_num, **kw: "БАЗОВЫЙ ПРОМПТ"
cell = Path("kernels/notebooks_stockflash_input/cell15.py").read_text(encoding="utf-8")
bm = types.SimpleNamespace(solver=types.SimpleNamespace(max_runtime_s_per_game=7920.0))
ns = {"TRUE_SUBMISSION": False, "bm": bm, "print": lambda *a, **k: None}
exec(compile(cell, "cell15", "exec"), ns)
ok(bm.solver.max_runtime_s_per_game == 7920.0, "потолок на игру стоковый 7920 (полный прогон)")
ok("_run_python_tool" not in cell and "_chat_completion" not in cell, "выход модели не затронут")

def G(objs, h=64, w=64, bg=0):
    g = [[bg] * w for _ in range(h)]
    for (r, c, hh, ww, col) in objs:
        for y in range(r, r + hh):
            for x in range(c, c + ww): g[y][x] = col
    return tuple(tuple(row) for row in g)
F = lambda grid, step, level=1: types.SimpleNamespace(grid=grid, ascii="\n".join("".join("WwgGcBMPRbSYOrNp"[v] for v in row) for row in grid), step=step, level=level)
ag = object.__new__(Agent)
call = lambda frame, acts=None: Agent._build_user_prompt(ag, 0, valid_actions=["UP"], current_frame=frame,
                                                         history_entries=[], previous_step_summary={"executed_actions": acts or []})
g0 = G([(10, 10, 3, 3, 9)]); g1 = G([(10, 13, 3, 3, 9)])
t1 = call(F(g0, 1))
ok(t1 == "БАЗОВЫЙ ПРОМПТ", "шаг 1: разбора нет (не с чем сравнивать)")
t2 = call(F(g1, 2), ["RIGHT"])
ok("HARNESS DIFF" in t2 and "executed: RIGHT" in t2, "шаг 2: разбор есть, названы исполненные действия")
ok("b 3x3 at (10,10) -> (10,13) [+0,+3]" in t2, "сдвиг описан верно: b 3x3 (10,10)->(10,13) [+0,+3]")
t2r = call(F(g1, 2), ["RIGHT"])
ok(t2r == t2, "перезаход того же шага: тот же разбор, без ложной петли")
t3 = call(F(g1, 3), ["UP"])
ok("board changed: NO" in t3, "холостой ход: «NO»")
t4 = call(F(g0, 4), ["LEFT"])
ok("LOOP CHECK" in t4 and "HARNESS DIFF" in t4, "настоящий возврат: и разбор, и петля")
ok(t4.index("HARNESS DIFF") < t4.index("LOOP CHECK"), "разбор идёт ПЕРВЫМ, петля после")
t5 = call(F(G([(5, 5, 4, 4, 11)]), 5, level=2), ["UP"])
ok("level changed" in t5, "смена уровня: строка уровня вместо объектов")
big = G([(r, c, 1, 1, 8) for r in range(0, 40, 2) for c in range(0, 40, 2)])
call(F(G([]), 6, level=2), ["UP"])
t7 = call(F(big, 7, level=2), ["SPACE"])
ok("large change" in t7 and "moved:" not in t7, "массовое изменение: сводка вместо списка")
hud0 = G([(63, 0, 1, 40, 14)]); hud1 = G([(63, 0, 1, 30, 14)])
call(F(hud0, 8, level=3), ["UP"])
t9 = call(F(hud1, 9, level=3), ["UP"])
ok("edge bars changed (likely HUD/timer" in t9, "полоса по краю поля помечена как HUD")
ok(call(None) == "БАЗОВЫЙ ПРОМПТ", "кадра нет: молча пропускаем")
ok(ns["_DIFF_STATS"]["fail"] == 0 and ns["_LOOP_STATS"]["fail"] == 0, "сбоев учёта: разбор %d, петля %d" % (ns["_DIFF_STATS"]["fail"], ns["_LOOP_STATS"]["fail"]))
ok(ns["_DIFF_STATS"]["reuse"] >= 1 and ns["_LOOP_STATS"]["reentry"] >= 1, "перезаходы учтены: разбор %d, петля %d" % (ns["_DIFF_STATS"]["reuse"], ns["_LOOP_STATS"]["reentry"]))

# сверка на 25 снимках пробы 11.09: во что превращаются шумные разборы
sys.path.insert(0, "scripts")
from prompt_replay_diff import boards_for
summ = hud = noisy_before = 0
for f in sorted(glob.glob("runs/flash_v1_phaseA/prompts/*.log")):
    snap, b, a, ex, lvl = boards_for(f)
    d = ns["_diff_text"](tuple(map(tuple, b)), tuple(map(tuple, a)), ex, 1, 2 if lvl else 1)
    summ += "large change" in d; hud += "edge bars" in d
print("     снимки пробы: сводок %d из 25, с помеченной полосой HUD %d из 25" % (summ, hud))

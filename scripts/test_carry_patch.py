"""Проверки слоя «перенос модели мира через взятие уровня + путь уровня» на НАСТОЯЩЕМ ToolAgent бандла.

Сценарий: модель мира уровня 1 заполнена всеми полями; ход берёт уровень (level_transition) -- world/goal/action
сохраняются с пометкой, findings/questions/plan стёрты; первый промпт уровня 2 (настоящий _build_user_prompt)
содержит блок LEVEL 1 SOLUTION PATH с путём от последнего RESET и перенесённую модель между стоковыми маркерами;
следующий промпт уровня 2 -- тот же блок; повторный перенос не дублирует пометку; run_complete и game_over --
стоковое стирание; уровень 1 -- промпт стоковый; CARRY_LAYER=0 -- ничего не патчится; длинный путь режется.

usage:  .venv/bin/python scripts/test_carry_patch.py --bundle runs/peer_kernels/duck_base/x
"""
import argparse
import ast
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True)
a = ap.parse_args()
sys.path.insert(0, str(Path(a.bundle) / "src" / "ARC3-Inference"))
sys.path.insert(0, str(Path(a.bundle) / "src" / "tufa-arc-agi-framework" / "src"))
os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "Qwen/Qwen3.8-Flash-Next-NVFP4")
import inference.agent.tool_agent as wta  # noqa: E402
from inference.agent.runtime_state import Frame, HistoryEntry  # noqa: E402

fails = []


def check(b, m):
    print(("ok   " if b else "СБОЙ ") + m)
    if not b:
        fails.append(m)


def fr(level, step):
    return Frame(grid=((0, 1), (1, 0)), step=step, level=level)


cell = open("kernels/notebooks_stockflash_carry/cell15.py", encoding="utf-8").read()
compile(cell, "carry", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
orig_upd, orig_prompt = wta.ToolAgent._update_summarized_knowledge_from_step_summary, wta.ToolAgent._build_user_prompt

os.environ["CARRY_LAYER"] = "0"
exec(cell, {"TRUE_SUBMISSION": False})
check(wta.ToolAgent._update_summarized_knowledge_from_step_summary is orig_upd and wta.ToolAgent._build_user_prompt is orig_prompt,
      "CARRY_LAYER=0: ничего не патчится")
os.environ["CARRY_LAYER"] = "1"
ns = {"TRUE_SUBMISSION": True}
exec(cell, ns)
check(wta.ToolAgent._update_summarized_knowledge_from_step_summary is not orig_upd and wta.ToolAgent._build_user_prompt is not orig_prompt,
      "слой установлен (и в боевом режиме)")

agent = wta.ToolAgent(base_url="http://127.0.0.1:1/v1", provider="vllm")
WM = {"world_model": "4 grids of tiles", "goal_model": "make grid match the mini-map", "action_model": "MOUSE toggles a tile",
      "recent_findings": "tile (2,3) toggled", "open_questions": "is the bar a timer?", "current_plan": "click (46,38)",
      "cross_level_notes": ""}
agent._summarized_knowledge = dict(WM)

hist = [HistoryEntry(action="", frame=fr(1, 0))]
seq = ["UP", "UP", "MOUSE(row=4, col=43)", "RESET", "LEFT", "LEFT", "LEFT", "MOUSE(row=46, col=38)", "SPACE"]
for i, act in enumerate(seq):
    hist.append(HistoryEntry(action=act, frame=fr(2 if i == len(seq) - 1 else 1, i + 1)))

p1 = wta.ToolAgent._build_user_prompt(agent, 8, valid_actions=["MOUSE"], current_frame=fr(1, 8), history_entries=hist[:-1],
                                      previous_step_summary={"executed_count": 1, "executed_actions": ["MOUSE(row=46, col=38)"], "level": 1})
check("SOLUTION PATH" not in p1 and "make grid match the mini-map" in p1 and "[carried" not in p1, "уровень 1: промпт стоковый, модель мира на месте")

summary = {"executed_count": 1, "executed_actions": ["SPACE"], "level": 2, "level_transition": True, "run_complete": False, "game_over": False}
agent._last_step_summary = summary
wta.ToolAgent._update_summarized_knowledge_from_step_summary(agent)
k = agent._summarized_knowledge
check(k["goal_model"] == "[carried from level 1 -- re-check on this board] make grid match the mini-map"
      and k["world_model"].endswith("4 grids of tiles") and k["action_model"].endswith("MOUSE toggles a tile"),
      "переход: world/goal/action перенесены с пометкой уровня 1")
check(k["recent_findings"] == "" and k["open_questions"] == "" and k["current_plan"] == "", "переход: findings/questions/plan стёрты, как в стоке")
check(ns["_cr_stats"]["transitions"] == 1 and ns["_cr_stats"]["carried_nonempty"] == 1 and ns["_cr_stats"]["carried_fields"] == 3,
      "статистика переноса: 1 переход, 3 поля")

p2 = wta.ToolAgent._build_user_prompt(agent, 9, valid_actions=["MOUSE"], current_frame=fr(2, 9), history_entries=hist,
                                      previous_step_summary=summary)
print("---- первый промпт уровня 2 (начало) ----\n" + p2[:900] + "\n----")
check(p2.startswith("LEVEL 1 SOLUTION PATH"), "уровень 2: блок пути в начале промпта")
check("LEFT x3, MOUSE(row=46, col=38), SPACE." in p2 and "UP x2" not in p2, "путь -- от последнего RESET, сжатая запись")
check("(5 moves since the last RESET of that level; 9 moves spent on the level in total" in p2, "числа ходов: 5 от RESET, 9 всего")
check("You have progressed to a new level!" in p2, "стоковый текст на месте")
wm_part = p2.split("Below you are provided with the current world model")[1].split("end of world model")[0]
check("Working world model carried from earlier turns:" in wm_part and "[carried from level 1 -- re-check on this board] make grid match the mini-map" in wm_part
      and "click (46,38)" not in wm_part, "перенесённая модель -- между стоковыми маркерами, старого плана нет")
p3 = wta.ToolAgent._build_user_prompt(agent, 10, valid_actions=["MOUSE"], current_frame=fr(2, 10), history_entries=hist + [HistoryEntry(action="UP", frame=fr(2, 10))],
                                      previous_step_summary={"executed_count": 1, "executed_actions": ["UP"], "level": 2})
check(p3.startswith("LEVEL 1 SOLUTION PATH") and "LEFT x3, MOUSE(row=46, col=38), SPACE." in p3, "следующий промпт уровня 2: блок пути сохраняется")
check(ns["_cr_stats"]["paths"] == 1 and ns["_cr_stats"]["path_moves"] == [5] and ns["_cr_stats"]["level_moves"] == [9], "статистика пути: один путь, 5 из 9")

# модель обновила цель на уровне 2 -> второй переход: пометка не дублируется
agent._summarized_knowledge["goal_model"] = "[carried from level 1 -- re-check on this board] make grid match the mini-map"
agent._last_step_summary = dict(summary, level=3)
wta.ToolAgent._update_summarized_knowledge_from_step_summary(agent)
g = agent._summarized_knowledge["goal_model"]
check(g.count("[carried") == 1 and g.startswith("[carried from level 2"), "второй переход: одна пометка, уровень 2")

agent._last_step_summary = dict(summary, run_complete=True)
wta.ToolAgent._update_summarized_knowledge_from_step_summary(agent)
check(all(agent._summarized_knowledge[x] == "" for x in ("world_model", "goal_model", "action_model")), "run_complete: стоковое стирание")
agent._summarized_knowledge = dict(WM)
agent._last_step_summary = dict(summary, game_over=True)
wta.ToolAgent._update_summarized_knowledge_from_step_summary(agent)
check(agent._summarized_knowledge["goal_model"] == "", "game_over: стоковое стирание")
agent._summarized_knowledge = dict(WM)
agent._last_step_summary = {"executed_count": 1, "level": 2, "level_transition": False}
wta.ToolAgent._update_summarized_knowledge_from_step_summary(agent)
check(agent._summarized_knowledge == WM, "без перехода: модель мира не тронута")

# длинный путь режется, берутся последние ходы
agent2 = wta.ToolAgent(base_url="http://127.0.0.1:1/v1", provider="vllm")
h2 = [HistoryEntry(action="", frame=fr(1, 0))] + [HistoryEntry(action=("UP" if i % 2 else "DOWN"), frame=fr(1, i + 1)) for i in range(400)]
h2.append(HistoryEntry(action="MOUSE(row=1, col=2)", frame=fr(2, 401)))
wta.ToolAgent._build_user_prompt(agent2, 400, valid_actions=["UP"], current_frame=fr(1, 400), history_entries=h2[:-1])
p4 = wta.ToolAgent._build_user_prompt(agent2, 401, valid_actions=["UP"], current_frame=fr(2, 401), history_entries=h2)
blk = p4.split("\n\n")[0]
check("(first %d moves omitted)" % (401 - ns["_CR_PATH_MAX"]) in blk and blk.count("MOUSE(row=1, col=2)") == 1 and len(blk) < 2500,
      "длинный путь: последние %d ходов, блок %d знаков" % (ns["_CR_PATH_MAX"], len(blk)))

# в ноутбуке слой стоит в ячейке 15 до запуска прогона, потолок пробы 3600 только вне боя
import json  # noqa: E402
nb = json.load(open("kernels/notebooks_stockflash_carry/submission.ipynb", encoding="utf-8"))
c15 = "".join(nb["cells"][15]["source"])
check(c15.find("_cr_stats = ") < c15.find("await bm.run(") and "if not TRUE_SUBMISSION:\n    bm.solver.max_runtime_s_per_game = 3600.0" in c15,
      "ноутбук: слой до запуска, потолок 3600 с только вне боя")

print("\nИТОГ: %s" % ("все проверки пройдены" if not fails else "СБОИ: %d" % len(fails)))
sys.exit(1 if fails else 0)

"""Проверка макроходов на НАСТОЯЩЕЙ песочнице Duck (python_tool_sandbox) и настоящем модуле tool_agent.
usage: .venv/bin/python scripts/test_macro_patch.py --bundle <бандл>"""
import argparse, json, sys, types
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--bundle", required=True); a = ap.parse_args()
sys.path.insert(0, str(Path(a.bundle) / "src" / "ARC3-Inference")); sys.path.insert(0, str(Path(a.bundle) / "src" / "tufa-arc-agi-framework" / "src"))
import inference.agent.tool_agent as wta
fails = []
def check(b, m):
    print(("ok   " if b else "СБОЙ ") + m)
    if not b: fails.append(m)

cell = open("kernels/notebooks_stockflash_macro/cell15.py", encoding="utf-8").read()
seg = cell[cell.index("import re as _mc_re"):]
orig_prompt, orig_run, orig_sandbox = wta.ToolAgent._build_user_prompt, wta.ToolAgent._run_python_tool, wta.run_sandboxed_python
ns = {"TRUE_SUBMISSION": True}
exec(seg, ns)
check(wta.ToolAgent._run_python_tool is orig_run and wta.run_sandboxed_python is orig_sandbox, "TRUE_SUBMISSION=True: боевая ветка не тронута")
ns = {"TRUE_SUBMISSION": False}
exec(seg, ns)
check(wta.ToolAgent._run_python_tool is not orig_run and wta.run_sandboxed_python is not orig_sandbox and wta.ToolAgent._build_user_prompt is not orig_prompt,
      "TRUE_SUBMISSION=False: три обёртки установлены")

# --- имитация движка: доска 64x64, объект цвета 3 (4x4) в (10..13, 20..23), объект цвета 5 (2x2) в (40..41, 40..41)
env = {"obj_col": 20, "obj_color": 3, "level": 0, "n": 0, "log": []}
def grid():
    g = [[0] * 64 for _ in range(64)]
    for r in range(10, 14):
        for c in range(env["obj_col"], env["obj_col"] + 4):
            g[r][c] = env["obj_color"]
    for r in range(40, 42):
        for c in range(40, 42):
            g[r][c] = 5
    return g
def frame_payload():
    fr = wta.Frame(grid=grid(), step=env["n"], level=env["level"])
    return wta._ascii_frame_view_payload(fr)
def state():
    return {"current_frame": frame_payload(), "history": [], "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT", "MOUSE"], "last_action_result": {}}
def handler(actions):
    changed = False; lvl = False
    for act in actions:
        env["n"] += 1; name = act["action"].upper(); env["log"].append(act)
        if name == "RIGHT" and env["obj_col"] < 60:
            env["obj_col"] += 1; changed = True
        elif name == "MOUSE":
            r, c = int(act.get("row")), int(act.get("col"))
            if 10 <= r <= 13 and env["obj_col"] <= c < env["obj_col"] + 4 and env["obj_color"] == 3:
                env["obj_color"] = 4; changed = True
            elif 40 <= r <= 41 and 40 <= c <= 41:
                env["level"] += 1; changed = True; lvl = True
    res = {"executed": True, "action_num": env["n"], "level": env["level"], "score": 0, "reward": 0, "state": "NOT_FINISHED",
           "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT", "MOUSE"], "board_changed": changed, "done": False,
           "level_completed": lvl, "game_over": False, "run_complete": False}
    return {"action_result": res, "state": state()}

def run(code):
    return wta.run_sandboxed_python(code=ns["_MC_HELPERS"] + "\n" + code, timeout_seconds=10, initial_state=state(), action_handler=handler)

# 1. move_until_stuck: 40 сдвигов до столбца 60, потом ход без изменения -> стоп
env.update(obj_col=20, obj_color=3, level=0, n=0, log=[])
r = run("r = move_until_stuck('RIGHT', max_moves=100)\nprint('RES', r['moves'], r['stopped_by'], r['changed_moves'])")
check(not r.get("error"), "песочница: без ошибок (%s)" % (r.get("error") or "")[:200])
check("RES 40 max_moves 40" in r.get("stdout", ""), "move_until_stuck: потолок 40 ходов сработал: %r" % r.get("stdout", "").strip()[:80])
env.update(obj_col=55, n=0, log=[])
r = run("r = move_until_stuck('RIGHT')\nprint('RES', r['moves'], r['stopped_by'], r['changed_moves'])")
check("RES 6 no_change 5" in r.get("stdout", ""), "move_until_stuck: стоп на первом ходе без изменения (5 сдвигов + 1): %r" % r.get("stdout", "").strip()[:80])
check("[[MACRO]]" not in r.get("stdout", ""), "маркер [[MACRO]] вырезан из stdout для модели")
ev = getattr(ns["_mc_tls"], "events", None)
check(ev and ev[0]["macro"] == "move_until_stuck" and ev[0]["moves"] == 6, "обвязка прочитала маркер: %r" % (ev,))

# 2. click_objects: два объекта (фон пропущен); клик по объекту 3 меняет цвет, клик по объекту 5 берёт уровень
env.update(obj_col=20, obj_color=3, level=0, n=0, log=[])
r = run("out = click_objects()\nprint('OUT', [(o['color'], o['row'], o['col'], o['changed'], o['level_completed']) for o in out])")
so = r.get("stdout", "")
check(not r.get("error"), "click_objects: без ошибок (%s)" % (r.get("error") or "")[:200])
check("'level_completed'" not in so and so.count("True") >= 2 and env["level"] == 1 and len(env["log"]) == 2,
      "click_objects: 2 клика по центрам, доска менялась, уровень взят: %s | log=%s" % (so.strip()[:120], env["log"]))
ev = getattr(ns["_mc_tls"], "events", None)
check(ev and ev[0]["macro"] == "click_objects" and ev[0]["level_completed"] is True and ev[0]["moves"] == 2, "маркер click_objects с level_completed: %r" % (ev,))
env.update(obj_col=20, obj_color=3, level=0, n=0, log=[])
r = run("out = click_objects(color='S', stop_on_change=True)\nprint('N', len(out))")
# буква цвета 3: берём из легенды песочницы через саму сегментацию
r2 = run("print(sorted(set(n['color'] for n in current_frame.segmentation['nodes'])))")
check(not r2.get("error"), "сегментация в песочнице читается: %s" % r2.get("stdout", "").strip())

# 3. sweep_moves: 4 хода (без MOUSE), только RIGHT меняет доску
env.update(obj_col=20, obj_color=3, level=0, n=0, log=[])
r = run("res = sweep_moves()\nprint('SW', res)")
check("'RIGHT': True" in r.get("stdout", "") and "'UP': False" in r.get("stdout", "") and len(env["log"]) == 4, "sweep_moves: 4 хода, изменил только RIGHT: %s" % r.get("stdout", "").strip()[:100])

# 4. учёт в _mc_run: подменяем исходный _run_python_tool на прогон песочницы
class FakeSelf: pass
def fake_orig(self, state_path, arguments):
    res = wta.run_sandboxed_python(code=arguments["code"], timeout_seconds=10, initial_state=state(), action_handler=handler)
    return types.SimpleNamespace(content=res.get("stdout", ""), step_executed=bool(res.get("action_results")))
ns["_mc_orig_run"] = fake_orig
fs = FakeSelf()
env.update(obj_col=20, obj_color=3, level=0, n=0, log=[])
ns["_mc_run"](fs, None, {"code": "move_until_stuck('RIGHT', 3)\nsweep_moves()"})
ns["_mc_run"](fs, None, {"code": "print(len(current_frame.segmentation['nodes']))"})
st = ns["_mc_stats"]
check(st["games"] == 1 and st["turns"] == 2 and st["macro_turns"] == 1 and st["macro_calls"] == 2 and st["macro_moves"] == 3 + 4 and st["games_used"] == 1
      and st["by_name"] == {"move_until_stuck": 1, "sweep_moves": 1}, "учёт: %r" % st)
# 5. промпт: описание макросов спереди
wta.ToolAgent._build_user_prompt = lambda self, n, *a, **k: "BASE"
ns3 = {"TRUE_SUBMISSION": False}; exec(seg, ns3)
check(ns3["_mc_prompt"](fs, 1).startswith("MACRO HELPERS") and ns3["_mc_prompt"](fs, 1).endswith("BASE"), "описание макросов во входе перед штатным промптом")
check(len(ns["_MC_NOTE"]) < 900, "описание короткое: %d знаков" % len(ns["_MC_NOTE"]))
wta.ToolAgent._build_user_prompt, wta.ToolAgent._run_python_tool, wta.run_sandboxed_python = orig_prompt, orig_run, orig_sandbox
print("\nИТОГ: %d сбоев" % len(fails) if fails else "\nВСЕ ПРОВЕРКИ ПРОШЛИ"); sys.exit(1 if fails else 0)

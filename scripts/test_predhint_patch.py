"""Проверки слоя «цель из кадра взятия уровня МОДЕЛЬЮ, без перебора» (kernels/notebooks_stockflash_predhint/cell15.py).

A. Обёртка хода: ходов движка НЕ добавляет (оригинал вызывается ровно один раз на ход), копит образцы состояний уровня,
   а при level_completed снимает кадр взятия и выводит цель. Кадры берутся из настоящего solver._raw_frames.
B. Индукция на НАСТОЯЩИХ данных: кадр взятия уровня 1 из записи боевого прогона (локальный движок повторяет её точно)
   + образцы состояний того же уровня -> утверждения истинны в цели и ложны во всех образцах.
C. Настоящий ToolAgent: блок цели попадает во вход, пока на сессии есть _ph_goal; PREDHINT=0 ничего не патчит.

usage:  .venv/bin/python scripts/test_predhint_patch.py --bundle runs/peer_kernels/duck_base/x
"""
import argparse, json, os, sys, types
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--bundle", required=True)
ap.add_argument("--cell", default="kernels/notebooks_stockflash_predhint/cell15.py")
ap.add_argument("--run", default="runs/flash_v1_phaseA"); ap.add_argument("--games", default="ar25,ft09,vc33,re86,su15")
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(a.bundle) / "src" / "ARC3-Inference")); sys.path.insert(0, str(Path(a.bundle) / "src" / "tufa-arc-agi-framework" / "src"))
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "Qwen/Qwen3.8-Flash-Next-NVFP4")
import logging; logging.disable(logging.ERROR)
import numpy as np
import inference.framework.solver as solver
import inference.agent.tool_agent as wta
from inference.agent.runtime_state import Frame as RFrame
import arc_agi
from arc_agi import OperationMode
fails = []
def check(b, m):
    print(("ok   " if b else "СБОЙ ") + m)
    if not b: fails.append(m)

cell = open(a.cell, encoding="utf-8").read()
orig_prompt, orig_exec = wta.ToolAgent._build_user_prompt, solver._HarnessGameSession._execute_action
os.environ["PREDHINT"] = "0"; exec(cell, {"TRUE_SUBMISSION": False})
check(wta.ToolAgent._build_user_prompt is orig_prompt and solver._HarnessGameSession._execute_action is orig_exec, "PREDHINT=0: ничего не патчится")
os.environ["PREDHINT"] = "1"; ns = {"TRUE_SUBMISSION": True}; exec(cell, ns)
check(wta.ToolAgent._build_user_prompt is not orig_prompt and solver._HarnessGameSession._execute_action is not orig_exec, "слой установлен (в боевом режиме тоже)")

# ---------------- A: обёртка хода на подделке сессии ----------------
calls = {"n": 0}
class FakeState:
    def __init__(self, grid, lvl, frames):
        self.frame = types.SimpleNamespace(data=grid); self.levels_completed = lvl
        self.raw = types.SimpleNamespace(state="NOT_FINISHED", frame=frames)
class FakeSess:
    def __init__(self): self.game = types.SimpleNamespace(current_state=None)
def fake_exec(self, action, *x, **k):
    calls["n"] += 1
    return self._next_payload
ns["_ph_orig_exec"] = fake_exec
sess = FakeSess()
g1 = np.zeros((8, 8), dtype=np.int16); g1[0, 0] = 5
for i in range(6):
    sess.game.current_state = FakeState(g1 + (i % 2), 0, [g1])
    sess._next_payload = {"level_completed": False}
    ns["_ph_exec"](sess, None)
check(calls["n"] == 6, "обёртка вызывает оригинал ровно один раз на ход (ходов движка не добавляет)")
check(len(sess._ph_state["samples"]) == 3, "образцы состояний уровня копятся (каждый второй ход): %d" % len(sess._ph_state["samples"]))
goal = np.zeros((8, 8), dtype=np.int16); goal[2, 2] = 7   # цвет 5 исчез -> отделяет цель от образцов
sess.game.current_state = FakeState(goal, 1, [goal, np.zeros((8, 8), dtype=np.int16)])
sess._next_payload = {"level_completed": True}
ns["_ph_exec"](sess, None)
check(getattr(sess, "_ph_goal", None) and sess._ph_goal["stmts"], "при взятии уровня цель выведена: %s" % (getattr(sess, "_ph_goal", {}) or {}).get("stmts"))
check(sess._ph_state["samples"] == [], "после взятия уровня образцы сброшены (новый уровень считается заново)")

# ---------------- B: индукция на настоящей записи ----------------
from goal_cross_level_replay import load_events, replay_collect
arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT / "environment_files"))
envs = {e.game_id[:4]: e.game_id for e in arc.get_environments()}
ok_games = 0
for g in a.games.split(","):
    gid = envs.get(g)
    evs = load_events(a.run, gid)
    if not evs:
        continue
    per = replay_collect(arc, gid, evs)
    if 1 not in per:
        print("     %s: в записи уровень 1 не взят (пропуск)" % g); continue
    goal_g = per[1]["goal"]; samples = per[1]["negs"][-200:]
    stmts = ns["_gh_infer"](goal_g, samples)
    print("     %s: цель по записи -> %s" % (g, " | ".join(stmts) or "—"))
    if stmts:
        ok_games += 1
        fg = ns["_gh_feat"](goal_g); fn = [ns["_gh_feat"](x) for x in samples if x.shape == goal_g.shape]
        good = all(f(fg) and not any(f(x) for x in fn) for _r, t, f in ns["_gh_preds"](fg) if t in stmts)
        check(good, "%s: каждое утверждение истинно в цели и ложно во всех %d образцах" % (g, len(fn)))
check(ok_games >= 2, "цель выведена по настоящим записям минимум в двух играх (получилось %d)" % ok_games)

# ---------------- C: промпт ----------------
agent = wta.ToolAgent(base_url="http://127.0.0.1:1/v1", provider="vllm")
fs = types.SimpleNamespace(_ph_goal={"level_done": 1, "stmts": ["no cells of colour 9 are left on the board"]})
fs.step_env = types.MethodType(lambda self, arguments: {}, fs)
agent._step_env_callback = fs.step_env
p = wta.ToolAgent._build_user_prompt(agent, 7, valid_actions=["UP"], current_frame=RFrame(grid=((0,),), step=7, level=2),
                                     history_entries=[], previous_step_summary={"executed_count": 1, "executed_actions": ["UP"], "level": 2})
print("---- промпт (начало) ----\n" + p[:420] + "\n----")
check(p.startswith("HARNESS-INFERRED GOAL OF THE PREVIOUS LEVEL") and "no cells of colour 9" in p and "share the KIND of goal" in p,
      "блок цели во входе: утверждение и оговорка про числа")
nb = json.load(open(ROOT / "kernels/notebooks_stockflash_predhint/submission.ipynb", encoding="utf-8"))
c15 = "".join(nb["cells"][15]["source"])
check(c15.find("_ph_stats = ") < c15.find("await bm.run(") and "max_runtime_s_per_game = 4800.0" in c15 and "_bf_stats" not in c15,
      "ноутбук: слой до запуска, потолок 4800 с вне боя, перебора в сборке НЕТ")
print("ИТОГ:", "все проверки пройдены" if not fails else "сбоев %d: %s" % (len(fails), fails))
sys.exit(1 if fails else 0)

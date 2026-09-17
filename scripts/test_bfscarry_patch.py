"""Проверки связки «перебор после застоя + carry» (kernels/notebooks_stockflash_bfscarry/cell15.py).

A. Настоящий модуль solver бандла + локальный движок + поддельный цикл play (как в test_bfs_patch.py): режим stall включает
   перебор после застоя один раз на уровень и только при достаточном остатке; найденный путь пишется на сессию в записи
   модели; этот путь, повторённый на ЧИСТОЙ игре от RESET, действительно берёт уровень (проверка перевода ходов).
B. Настоящий ToolAgent бандла: запись перебора на сессии (через _step_env_callback.__self__) даёт блок LEVEL N SOLVED BY
   HARNESS SEARCH с путём, перенос world/goal/action с пометкой, стирание findings/questions/plan; блок держится на
   следующих промптах; второй раз та же запись не потребляется; без записи промпт уровня 1 стоковый.

usage:  .venv/bin/python scripts/test_bfscarry_patch.py --bundle runs/peer_kernels/duck_base/x
"""
import argparse, json, os, random, re, sys, time, types
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--bundle", required=True)
ap.add_argument("--cell", default="kernels/notebooks_stockflash_bfscarry/cell15.py"); a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(a.bundle) / "src" / "ARC3-Inference")); sys.path.insert(0, str(Path(a.bundle) / "src" / "tufa-arc-agi-framework" / "src"))
os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "Qwen/Qwen3.8-Flash-Next-NVFP4")
import logging; logging.disable(logging.ERROR)
import numpy as np
import inference.framework.solver as solver
import inference.agent.tool_agent as wta
from inference.agent.runtime_state import Frame as RFrame
import arc_agi, arcengine
from arc_agi import OperationMode
from arcengine import GameAction
fails = []
def check(b, m):
    print(("ok   " if b else "СБОЙ ") + m)
    if not b: fails.append(m)

cell = open(a.cell, encoding="utf-8").read()
orig_play, orig_prompt, orig_upd = solver._HarnessGameSession.play, wta.ToolAgent._build_user_prompt, wta.ToolAgent._update_summarized_knowledge_from_step_summary
ns = {"TRUE_SUBMISSION": True}; exec(cell, ns)
check(solver._HarnessGameSession.play is not orig_play and wta.ToolAgent._build_user_prompt is not orig_prompt
      and wta.ToolAgent._update_summarized_knowledge_from_step_summary is not orig_upd, "оба слоя установлены (боевой режим тоже)")
check(ns["_BF_MODE"] == "stall", "режим перебора: stall (застой %.0f с, остаток %.0f с)" % (ns["_BF_STALL_S"], ns["_BF_MIN_LEFT"]))

# ---------------- A: локальный движок ----------------
class Raw:
    def __init__(self, st): self.state = st
class EFrame:
    def __init__(self, data): self.data = data
class State:
    def __init__(self, fr):
        self.frame = EFrame(np.asarray(fr.frame[-1]) if fr is not None and fr.frame else np.zeros((64, 64), int))
        self.levels_completed = int(fr.levels_completed or 0) if fr is not None else 0
        self.raw = Raw(str(fr.state).split(".")[-1] if fr is not None else "GAME_OVER")
        self.available_actions = [int(x) for x in (fr.available_actions or [])] if fr is not None else []
        self.won = self.raw.state == "WIN"
class Game:
    def __init__(self, env, gid): self.env = env; self.game_id = gid; self.current_state = State(None); self.n = 0
    def execute_action(self, action, generated_tokens=0, uncached_input_tokens=0):
        if action.id != GameAction.RESET and action.id.value not in self.current_state.available_actions:
            raise ValueError("not available")
        fr = self.env.reset() if action.id == GameAction.RESET else self.env.step(action.id, data=dict(action.data) or None)
        self.n += 1; self.current_state = State(fr); return self.current_state
class FakeAnalyzer:
    def __init__(self, sess, rng, levels_seen): self.sess = sess; self.rng = rng; self.calls = 0; self.levels_seen = levels_seen
    def analyze(self, *x, **k):
        self.calls += 1; av = self.sess.game.current_state.available_actions
        self.levels_seen.append(self.sess.game.current_state.levels_completed)
        simple = [v for v in av if 1 <= v <= 5]
        if simple: self.sess.game.execute_action(arcengine.ActionInput(id=GameAction.from_id(self.rng.choice(simple)), data={}))
        return {"ok": True}
class Sess:
    def __init__(self, game, cap): self.game = game; self.cap = cap; self.t0 = time.monotonic(); self.last_engine_action = None; self.analyzer = None
    def should_stop(self): return (time.monotonic() - self.t0) >= self.cap or self.game.current_state.raw.state == "WIN"
    def timing_payload(self): return {"time_remaining_seconds": max(0.0, self.cap - (time.monotonic() - self.t0))}
    def step_env(self, arguments): return {}

def fake_orig_play(self):
    while not self.should_stop():
        self.analyzer.analyze(); time.sleep(0.2)
ns["_bf_orig_play"] = fake_orig_play
ns["_BF_STALL_S"] = 3.0; ns["_BF_MIN_LEFT"] = 4.0; ns["_BF_SECONDS"] = 5.0

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT / "environment_files"))
envs = {e.game_id[:4]: e.game_id for e in arc.get_environments()}

def replay_model_path(gid, path):
    env = arc.make(gid); fr = env.reset()
    for s in path:
        m = re.match(r"MOUSE\(row=(\d+), col=(\d+)\)", s)
        if m:
            fr = env.step(GameAction.ACTION6, data={"x": int(m.group(2)), "y": int(m.group(1))})
        else:
            name = {"UP": "ACTION1", "DOWN": "ACTION2", "LEFT": "ACTION3", "RIGHT": "ACTION4", "SPACE": "ACTION5"}.get(s, s)
            fr = env.step(GameAction[name], data=None)
    return int(fr.levels_completed or 0)

for g, cap in (("sp80", 40.0), ("sc25", 40.0), ("ls20", 30.0)):
    env = arc.make(envs[g]); game = Game(env, envs[g]); sess = Sess(game, cap=cap)
    game.execute_action(arcengine.ActionInput(id=GameAction.RESET, data={}))
    seen = []; sess.analyzer = FakeAnalyzer(sess, random.Random(0), seen)
    n0 = ns["_bf_stats"]["games"]; ns["_bf_play"](sess)
    runs = ns["_bf_stats"]["games"] - n0; found = list(getattr(sess, "_bf_found", []) or [])
    print(f"  {g}: потолок {cap} с, переборов {runs}, найдено {[(r['level_done'], len(r['path'])) for r in found]}, вызовов модели {sess.analyzer.calls}, уровень {game.current_state.levels_completed}")
    if g == "sp80":
        check(runs == 1 and len(found) == 1 and found[0]["level_done"] == 1, "sp80: один перебор после застоя, путь уровня 1 записан на сессию")
        check(all(re.fullmatch(r"UP|DOWN|LEFT|RIGHT|SPACE|MOUSE\(row=\d+, col=\d+\)", s) for s in found[0]["path"]), "sp80: путь в записи модели")
        check(replay_model_path(envs[g], found[0]["path"]) >= 1, "sp80: путь из записи модели на чистой игре берёт уровень 1")
        check(any(x == 1 for x in seen), "sp80: модель вызывалась и после взятия уровня перебором")
    if g == "sc25":
        check(runs == 1 and not found, "sc25: не найдено — ровно один перебор на уровень, второго нет")
    if g == "ls20":
        check(runs == 0, "ls20: потолок 30 с — к застою остаток < остаток модели + 30 с, перебор не включался")

# ---------------- B: настоящий ToolAgent ----------------
agent = wta.ToolAgent(base_url="http://127.0.0.1:1/v1", provider="vllm")
WM = {"world_model": "a maze with a key", "goal_model": "bring the key to the door", "action_model": "arrows move the player",
      "recent_findings": "wall at left", "open_questions": "what is the bar?", "current_plan": "go right x3", "cross_level_notes": ""}
agent._summarized_knowledge = dict(WM)
fs = types.SimpleNamespace(_bf_found=[])
fs.step_env = types.MethodType(lambda self, arguments: {}, fs)
agent._step_env_callback = fs.step_env
stats = ns["_cr_stats"]; bf0 = int(stats.get("bfs_paths", 0))
p1 = wta.ToolAgent._build_user_prompt(agent, 30, valid_actions=["UP"], current_frame=RFrame(grid=((0,),), step=30, level=1), history_entries=[],
                                      previous_step_summary={"executed_count": 1, "executed_actions": ["UP"], "level": 1})
check("SOLVED BY HARNESS SEARCH" not in p1 and "[carried" not in p1, "без записи перебора промпт уровня 1 стоковый")
fs._bf_found.append({"level_done": 1, "path": ["UP", "UP", "UP", "MOUSE(row=10, col=20)", "SPACE"], "moves": 990})
summ = {"executed_count": 1, "executed_actions": ["UP"], "level": 1}
p2 = wta.ToolAgent._build_user_prompt(agent, 31, valid_actions=["UP"], current_frame=RFrame(grid=((0,),), step=1021, level=2), history_entries=[],
                                      previous_step_summary=summ)
print("---- промпт после перебора (начало) ----\n" + p2[:700] + "\n----")
check(p2.startswith("LEVEL 1 SOLVED BY HARNESS SEARCH") and "UP x3, MOUSE(row=10, col=20), SPACE." in p2 and "You are now on level 2" in p2,
      "блок перебора: путь в сжатой записи и «ты на уровне 2»")
wm_part = p2.split("Below you are provided with the current world model")[1].split("end of world model")[0]
check("[carried from level 1 -- re-check on this board] bring the key to the door" in wm_part and "go right x3" not in wm_part and "wall at left" not in wm_part,
      "модель мира перенесена с пометкой, план и находки стёрты")
p3 = wta.ToolAgent._build_user_prompt(agent, 32, valid_actions=["UP"], current_frame=RFrame(grid=((0,),), step=1022, level=2), history_entries=[],
                                      previous_step_summary=summ)
check(p3.startswith("LEVEL 1 SOLVED BY HARNESS SEARCH") and p3.count("[carried from level") == p2.count("[carried from level"),
      "следующий промпт: блок держится, запись не потреблена второй раз, пометка не удвоена")
check(int(stats.get("bfs_paths", 0)) - bf0 == 1, "статистика: bfs_paths +1")

nb = json.load(open("kernels/notebooks_stockflash_bfscarry/submission.ipynb", encoding="utf-8")); c15 = "".join(nb["cells"][15]["source"])
check(c15.find("_bf_stats = ") < c15.find("_cr_stats = ") < c15.find("await bm.run(") and "bm.solver.max_runtime_s_per_game = 3600.0" in c15,
      "ноутбук: оба слоя до запуска, потолок пробы 3600 с вне боя")
print("ИТОГ:", "все проверки пройдены" if not fails else f"сбоев {len(fails)}: {fails}")
sys.exit(1 if fails else 0)

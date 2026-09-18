"""Проверки слоя «перебор берёт уровень 1 -> цель во входе модели» (kernels/notebooks_stockflash_goalhint/cell15.py).

A. Локальный движок + настоящий модуль solver бандла: перебор в режиме pre берёт уровень 1, кладёт на сессию кадр ЦЕЛИ
   и образцы не-целевых состояний; индукция по этому кадру даёт непустые утверждения, и КАЖДОЕ из них ложно во всех
   образцах (то есть отделяет цель), а также истинно в самом кадре цели.
B. Настоящий ToolAgent бандла: блок HARNESS-INFERRED GOAL попадает в промпт; при переходе на следующий уровень текст
   меняется на «тот же вид цели, другие числа»; GOALHINT=0 ничего не патчит.

usage:  .venv/bin/python scripts/test_goalhint_patch.py --bundle runs/peer_kernels/duck_base/x
"""
import argparse, json, os, sys, time, types
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--bundle", required=True)
ap.add_argument("--cell", default="kernels/notebooks_stockflash_goalhint/cell15.py")
ap.add_argument("--games", default="sp80,ls20,cd82"); a = ap.parse_args()
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
orig_play, orig_prompt = solver._HarnessGameSession.play, wta.ToolAgent._build_user_prompt
os.environ["GOALHINT"] = "0"; os.environ["BFS_LAYER"] = "0"
exec(cell, {"TRUE_SUBMISSION": False})
check(wta.ToolAgent._build_user_prompt is orig_prompt and solver._HarnessGameSession.play is orig_play, "выключатели: ничего не патчится")
os.environ["GOALHINT"] = "1"; os.environ["BFS_LAYER"] = "1"
ns = {"TRUE_SUBMISSION": True}; exec(cell, ns)
check(wta.ToolAgent._build_user_prompt is not orig_prompt and solver._HarnessGameSession.play is not orig_play, "оба слоя установлены (боевой режим тоже)")
check(ns["_BF_MODE"] == "pre", "режим перебора pre (до первого вызова модели)")


class Raw:
    def __init__(self, st): self.state = st
class EF:
    def __init__(self, d): self.data = d
class State:
    def __init__(self, fr):
        self.frame = EF(np.asarray(fr.frame[-1]) if fr is not None and fr.frame else np.zeros((64, 64), int))
        self.levels_completed = int(fr.levels_completed or 0) if fr is not None else 0
        self.raw = types.SimpleNamespace(state=(str(fr.state).split(".")[-1] if fr is not None else "GAME_OVER"),
                                         frame=(list(fr.frame) if fr is not None and fr.frame else []))
        self.available_actions = [int(x) for x in (fr.available_actions or [])] if fr is not None else []
        self.won = self.raw.state == "WIN"
class Game:
    def __init__(self, env, gid): self.env = env; self.game_id = gid; self.current_state = State(None); self.n = 0
    def execute_action(self, action, generated_tokens=0, uncached_input_tokens=0):
        if action.id != GameAction.RESET and action.id.value not in self.current_state.available_actions:
            raise ValueError("not available")
        fr = self.env.reset() if action.id == GameAction.RESET else self.env.step(action.id, data=dict(action.data) or None)
        self.n += 1; self.current_state = State(fr); return self.current_state
class Sess:
    def __init__(self, game): self.game = game; self.last_engine_action = None; self.t0 = time.monotonic()
    def should_stop(self): return False
    def timing_payload(self): return {"time_remaining_seconds": 1e9}
    def step_env(self, arguments): return {}

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT / "environment_files"))
envs = {e.game_id[:4]: e.game_id for e in arc.get_environments()}
sess_ok = None
for g in a.games.split(","):
    env = arc.make(envs[g]); game = Game(env, envs[g]); sess = Sess(game)
    game.execute_action(arcengine.ActionInput(id=GameAction.RESET, data={}))
    rec = ns["_bf_prephase"](sess)
    raw = getattr(sess, "_bf_goal_raw", None)
    if not rec["found"]:
        print("     %s: перебор уровень не взял (пропуск)" % g); continue
    check(raw is not None and raw.get("goal") is not None and len(raw.get("samples") or []) > 0,
          "%s: кадр цели и образцы не-целевых состояний сняты (%d образцов)" % (g, len(raw.get("samples") or []) if raw else 0))
    stmts = ns["_gh_infer"](raw["goal"], raw["samples"])
    print("     %s: цель -> %s" % (g, " | ".join(stmts) or "—"))
    if not stmts:
        print("     %s: ни одно утверждение не отделило цель — слой ничего не добавит (ожидаемый исход в части игр)" % g)
    # каждое утверждение обязано быть истинным в цели и ложным во всех образцах
    fg = ns["_gh_feat"](raw["goal"]); fn = [ns["_gh_feat"](x) for x in raw["samples"] if x.shape == raw["goal"].shape]
    good = True
    for _r, text, f in ns["_gh_preds"](fg):
        if text in stmts:
            good = good and f(fg) and not any(f(x) for x in fn)
    check(good, "%s: каждое утверждение истинно в цели и ложно во всех образцах" % g)
    if sess_ok is None and stmts:
        sess_ok = (sess, stmts, raw)

check(sess_ok is not None, "хотя бы одна игра дала цель для проверки промпта")
if sess_ok:
    sess, stmts, raw = sess_ok
    agent = wta.ToolAgent(base_url="http://127.0.0.1:1/v1", provider="vllm")
    agent._step_env_callback = sess.step_env
    p1 = wta.ToolAgent._build_user_prompt(agent, 5, valid_actions=["UP"], current_frame=RFrame(grid=((0,),), step=5, level=2), history_entries=[],
                                          previous_step_summary={"executed_count": 1, "executed_actions": ["UP"], "level": 2})
    print("---- промпт (начало) ----\n" + p1[:600] + "\n----")
    check(p1.startswith("HARNESS-INFERRED GOAL") and all(s in p1 for s in stmts), "блок цели в начале промпта, все утверждения на месте")
    check("Use this as the goal of the current level" in p1, "первый промпт: цель текущего уровня")
    p2 = wta.ToolAgent._build_user_prompt(agent, 6, valid_actions=["UP"], current_frame=RFrame(grid=((0,),), step=6, level=3), history_entries=[],
                                          previous_step_summary={"executed_count": 1, "executed_actions": ["UP"], "level": 3})
    check("share the KIND of goal" in p2 and all(s in p2 for s in stmts), "после взятия уровня: тот же вид цели, другие числа")
    check(ns["_gh_stats"]["goals"] >= 1 and ns["_gh_stats"]["prompts"] >= 2, "статистика слоя: цели и промпты посчитаны")

nb = json.load(open(ROOT / "kernels/notebooks_stockflash_goalhint/submission.ipynb", encoding="utf-8"))
c15 = "".join(nb["cells"][15]["source"])
check(c15.find("_bf_stats = ") < c15.find("_gh_stats = ") < c15.find("await bm.run(") and "max_runtime_s_per_game = 4800.0" in c15,
      "ноутбук: оба слоя до запуска, потолок пробы 4800 с вне боя")
print("ИТОГ:", "все проверки пройдены" if not fails else "сбоев %d: %s" % (len(fails), fails))
sys.exit(1 if fails else 0)

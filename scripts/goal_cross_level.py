"""Переносится ли ЦЕЛЬ между уровнями одной игры (17.09, тезис владельца: «все уровни в игре подобны, как только
определена цель уровня — её надо передавать на другой уровень; разница только в сложности: множественность объектов,
препятствия, помехи»).

Проверка. Для каждой игры с известным путём уровня 1 (runs/bfs_originals.json):
  1. проходим уровень 1 по известному пути, снимаем ЦЕЛЕВЫЕ кадры (кадры завершающего хода до последнего — последний
     принадлежит уже следующему уровню) и не-цели уровня 1 (состояния вдоль пути + случайное блуждание);
  2. выводим предикаты, отделяющие цель уровня 1 от всех не-целей уровня 1 (словарь goal_predicates.templates);
  3. ищем перебором по снимкам среды взятие уровня 2 (тот же алфавит и бюджеты, что у engine_bfs2), снимаем целевые
     кадры и не-цели уровня 2;
  4. МЕРА ПЕРЕНОСА: (а) сколько предикатов уровня 1 истинны в цели уровня 2; (б) сколько из них ещё и отделяют её
     от всех не-целей уровня 2; (в) совпадает ли ШАБЛОН (тип цели) у уровней 1 и 2, даже если параметры разные.
Контроль: те же предикаты уровня 1 на СЛУЧАЙНОМ состоянии уровня 2 — сколько отделяют его (ожидание: почти ноль).

usage:  .venv/bin/python scripts/goal_cross_level.py [--timeout 180] [--out runs/goal_cross_level_17_09.json]
"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import logging; logging.disable(logging.ERROR)
import copy
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
from engine_bfs2 import Snap, Counter, step, simple_actions, active_clicks, KeyOf
from goal_predicates import features, templates


def walk_states(env, n, seed=0):
    """случайное блуждание от ТЕКУЩЕГО состояния: не-цели + маска часов."""
    rng = random.Random(seed); acts = [GameAction[x] for x in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")]
    base = copy.deepcopy(env)
    fr = base.step(rng.choice(acts), data=None)
    prev = np.asarray(fr.frame[-1], dtype=np.int16); ch = np.zeros_like(prev, dtype=np.int32); k = 0; out = [prev]
    lvl0 = int(fr.levels_completed or 0)
    for _ in range(n):
        fr = base.step(rng.choice(acts), data=None)
        g = np.asarray(fr.frame[-1], dtype=np.int16)
        if int(fr.levels_completed or 0) != lvl0 or str(fr.state).split(".")[-1] == "GAME_OVER":
            base = copy.deepcopy(env); fr = base.step(rng.choice(acts), data=None); prev = np.asarray(fr.frame[-1], dtype=np.int16); continue
        if g.shape == prev.shape and (g != prev).any():
            ch += (g != prev); k += 1
        out.append(g); prev = g
    mask = (ch >= 0.8 * max(1, k)) & (ch >= 3)
    return out, mask


def bfs_next_level(env, timeout, max_states=3000, max_moves=60000):
    """перебор в ширину по снимкам до взятия СЛЕДУЮЩЕГО уровня; возвращает (целевые кадры, состояния пути)."""
    cnt = Counter(max_moves, time.time() + timeout)
    fr = env.step(GameAction.ACTION1, data=None) if False else None
    root = Snap(copy.deepcopy(env), None)
    g0 = np.asarray(env.state.frame[-1], dtype=np.int16) if hasattr(env, "state") else None
    return None, None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", default="runs/goal_cross_level_17_09.json"); a = ap.parse_args()
    sol = json.load(open(ROOT / "runs/bfs_originals.json", encoding="utf-8"))
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT / "environment_files"))
    rows = []
    for gid, rec in sol.items():
        if not rec.get("solved"):
            continue
        env = arc.make(gid); env.reset()
        # --- уровень 1: путь, не-цели, цель ---
        negs1 = []
        for name, payload in rec["path"][:-1]:
            fr = env.step(GameAction[name] if name != "RESET" else GameAction.RESET, data=payload)
            negs1.append(np.asarray(fr.frame[-1], dtype=np.int16))
        w1, mask = walk_states(arc.make(gid), 80)
        env1 = copy.deepcopy(env)
        fr = env.step(GameAction[rec["path"][-1][0]], data=rec["path"][-1][1])
        goal1 = [np.asarray(x, dtype=np.int16) for x in fr.frame][:-1] or [np.asarray(fr.frame[0], dtype=np.int16)]
        shape = goal1[-1].shape
        f_neg1 = [features(g, mask) for g in negs1 + w1 if g.shape == shape]
        f_goal1 = features(goal1[-1], mask)
        preds1 = []
        for nm, tid, fn in templates(f_goal1):
            try:
                if fn(f_goal1) and not any(fn(x) for x in f_neg1):
                    preds1.append((nm, tid, fn))
            except Exception:
                pass
        # --- уровень 2: перебор от состояния после взятия уровня 1 ---
        cnt = Counter(60000, time.time() + a.timeout)
        root = Snap(copy.deepcopy(env), fr)
        key_of = KeyOf(mask if mask.shape == root.grid.shape else np.zeros_like(root.grid, dtype=bool))
        acts = simple_actions(root.avail) + active_clicks(root, cnt)
        lvl0 = root.lvl
        seen = {key_of(root.grid): []}
        from collections import deque
        q = deque([(root, [])]); found = None; negs2 = [root.grid]
        while q and found is None and not cnt.exhausted() and len(seen) < 3000:
            snap, path = q.popleft()
            for act in acts:
                if cnt.exhausted():
                    break
                nxt = step(snap, act, cnt)
                if nxt is None:
                    continue
                if nxt.lvl > lvl0:
                    found = (snap, act); break
                if nxt.state == "GAME_OVER":
                    continue
                k = key_of(nxt.grid)
                if k in seen:
                    continue
                seen[k] = path + [act]; negs2.append(nxt.grid); q.append((nxt, path + [act]))
        row = {"game": gid[:4], "preds_level1": len(preds1), "templates_level1": sorted({t for _, t, _ in preds1}),
               "level2_solved": found is not None, "states_level2": len(seen)}
        if found is not None:
            snap, act = found
            env2 = copy.deepcopy(snap.env)
            fr2 = env2.step(GameAction[act[0]] if act[0] != "RESET" else GameAction.RESET, data=dict(act[1]) if act[1] else None)
            goal2 = [np.asarray(x, dtype=np.int16) for x in fr2.frame][:-1] or [np.asarray(fr2.frame[0], dtype=np.int16)]
            f_goal2 = features(goal2[-1], mask if mask.shape == goal2[-1].shape else np.zeros_like(goal2[-1], dtype=bool))
            f_neg2 = [features(g, mask if mask.shape == g.shape else np.zeros_like(g, dtype=bool)) for g in negs2 if g.shape == goal2[-1].shape]
            hold = sep = 0
            for nm, tid, fn in preds1:
                try:
                    if fn(f_goal2):
                        hold += 1
                        if not any(fn(x) for x in f_neg2):
                            sep += 1
                except Exception:
                    pass
            # какие шаблоны отделяют цель уровня 2 сами по себе (параметры из уровня 2)
            t2 = set()
            for nm, tid, fn in templates(f_goal2):
                try:
                    if fn(f_goal2) and not any(fn(x) for x in f_neg2):
                        t2.add(tid)
                except Exception:
                    pass
            # контроль: те же предикаты уровня 1 на случайном состоянии уровня 2
            ctrl = 0
            if f_neg2:
                rnd = random.Random(0).choice(f_neg2)
                for nm, tid, fn in preds1:
                    try:
                        if fn(rnd) and not any(fn(x) for x in f_neg2 if x is not rnd):
                            ctrl += 1
                    except Exception:
                        pass
            row.update({"preds1_hold_at_goal2": hold, "preds1_separate_goal2": sep, "templates_level2": sorted(t2),
                        "templates_common": sorted(set(row["templates_level1"]) & t2), "control_preds1_on_random_state2": ctrl,
                        "negatives_level2": len(f_neg2)})
        rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    ok = [r for r in rows if r.get("level2_solved")]
    if ok:
        print("\nИТОГ: игр с взятым уровнем 2 — %d из %d; предикат уровня 1 ОТДЕЛЯЕТ цель уровня 2 в %d играх; "
              "тип цели (шаблон) совпал у уровней 1 и 2 в %d играх; контроль (те же предикаты на случайном состоянии) — %d"
              % (len(ok), len(rows), sum(1 for r in ok if r["preds1_separate_goal2"] > 0),
                 sum(1 for r in ok if r["templates_common"]), sum(1 for r in ok if r["control_preds1_on_random_state2"] > 0)))
    json.dump(rows, open(ROOT / a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print("записано:", a.out)


if __name__ == "__main__":
    main()

"""Тот же тест прогресса, но с целью, ПЕРЕНЕСЁННОЙ с предыдущего уровня (18.09, замер №4 линии целей).

Почему это боевая постановка. В scripts/goal_progress_test.py мера считалась до НАСТОЯЩЕГО кадра взятия уровня
(AUC 0.629, P(приблизил|выигрышный) 0.69 против 0.49) -- но в бою этого кадра нет, пока уровень не взят.
Единственное, что есть на уровне k+1, -- цель, выведенная на уровне k. Здесь мера прогресса строится ИМЕННО из неё:
берём предикаты, отделившие цель уровня k, переводим в плотный остаток (scripts/goal_search.candidates) и меряем,
отличает ли Δостаток выигрышные ходы уровня k+1 от ходов-братьев.

Данные: записи настоящих траекторий (шесть прогонов), уровни k и k+1 одной игры; выигрышные ходы -- те, что
стоят в записи; братья -- прочие простые действия, разворачиваются на снимке локального движка.

Критерий (задан до замера): AUC >= 0.60 и P(приблизил|выигрышный) - P(приблизил|брат) >= 0.10 -- функция прогресса
переносится, строим поиск по ней; иначе линия целей закрывается как рычаг балла.

usage:  .venv/bin/python scripts/goal_progress_transfer.py [--out runs/goal_progress_transfer_18_09.json]
"""
import argparse, copy, json, random, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import logging; logging.disable(logging.ERROR)
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
from goal_cross_level_replay import load_events, DEFAULT_RUNS
from goal_search import state_features, candidates
from goal_predicates import features as pfeatures, templates

MASK = None


def feats(g):
    return state_features(g, np.zeros_like(g, dtype=bool))


def level_goal_residual(f_goal_prev, f_start_prev, f_negs_prev):
    """плотный остаток, построенный по цели ПРЕДЫДУЩЕГО уровня: берём кандидатов, истинных в его цели и ложных
    в его состояниях, и оставляем те, у которых есть плотная мера (goal_search.candidates)."""
    cands = candidates(f_start_prev)
    keep = []
    for nm, fn in cands:
        try:
            if fn(f_goal_prev) <= 0.0 and not any(fn(x) <= 0.0 for x in f_negs_prev):
                keep.append((nm, fn))
        except Exception:
            continue
    return keep


def residual(keep, f):
    vals = []
    for _nm, fn in keep:
        try:
            vals.append(fn(f))
        except Exception:
            pass
    return min(vals) if vals else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="runs/goal_progress_transfer_18_09.json")
    ap.add_argument("--max-steps", type=int, default=60); a = ap.parse_args()
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT / "environment_files"))
    rng = random.Random(0)
    rows = []
    win_better = win_n = sib_better = sib_n = 0
    auc_pairs = []
    ranks = []
    for e in arc.get_environments():
        gid = e.game_id
        # лучшая запись: наибольшее число взятых уровней
        best = None
        for run in DEFAULT_RUNS:
            evs = load_events(run, gid)
            if not evs:
                continue
            done = sum(1 for x in evs if x["done"])
            if done >= 2 and (best is None or done > best[0]):
                best = (done, evs)
        if best is None:
            continue
        evs = best[1]
        env = arc.make(gid); fr = env.reset()
        if fr is None or not fr.frame:
            continue
        # первый проход: собрать состояния уровня 1 и кадр его взятия
        lvl = 0; negs1 = []; f_goal1 = None; idx_after_l1 = None
        snaps = []
        for i, ev in enumerate(evs[:1500]):
            try:
                act = GameAction.RESET if ev["name"] == "RESET" else GameAction[ev["name"]]
                fr = env.step(act, data=ev["data"])
            except Exception:
                break
            if fr is None or not fr.frame:
                break
            frames = [np.asarray(x, dtype=np.int16) for x in fr.frame]
            if int(fr.levels_completed or 0) > lvl:
                lvl += 1
                if lvl == 1:
                    f_goal1 = feats((frames[:-1] or frames)[-1]); idx_after_l1 = i + 1
                    break
            else:
                negs1.append(frames[-1])
        if f_goal1 is None or not negs1:
            continue
        f_negs1 = [feats(g) for g in negs1[-120:] if g.shape == negs1[-1].shape]
        if not f_negs1:
            continue
        keep = level_goal_residual(f_goal1, f_negs1[0], f_negs1)
        if not keep:
            continue
        # второй проход: идём по уровню 2 и меряем Δостаток на выигрышных ходах и братьях
        env2 = arc.make(gid); env2.reset()
        for ev in evs[:idx_after_l1]:
            try:
                env2.step(GameAction.RESET if ev["name"] == "RESET" else GameAction[ev["name"]], data=ev["data"])
            except Exception:
                break
        per = {"game": gid[:4], "candidates": len(keep), "steps": 0, "win": 0, "sib": 0, "sib_n": 0}
        cur = env2
        for ev in evs[idx_after_l1:idx_after_l1 + a.max_steps]:
            st = cur.state if hasattr(cur, "state") else None
            try:
                g_now = np.asarray((st.frame or [[0]])[-1], dtype=np.int16) if st is not None else None
            except Exception:
                g_now = None
            if g_now is None:
                e_now = copy.deepcopy(cur); fr_now = e_now.step(GameAction.ACTION1, data=None)
                if fr_now is None or not fr_now.frame:
                    break
                g_now = np.asarray(fr_now.frame[-1], dtype=np.int16)
            f_now = feats(g_now); d0 = residual(keep, f_now)
            if d0 is None:
                break
            e_win = copy.deepcopy(cur)
            try:
                fr_w = e_win.step(GameAction.RESET if ev["name"] == "RESET" else GameAction[ev["name"]], data=ev["data"])
            except Exception:
                break
            if fr_w is None or not fr_w.frame:
                break
            d_w = residual(keep, feats(np.asarray(fr_w.frame[-1], dtype=np.int16)))
            if d_w is None:
                break
            win_n += 1; per["steps"] += 1
            if d_w < d0 - 1e-12:
                win_better += 1; per["win"] += 1
            _ds_list = []   # 19.09: Δd братьев этого состояния -- для МЕСТА выигрышного хода (пункт 3 критика раунда 10)
            for sn in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"):
                if sn == str(ev["name"]):
                    continue
                e_s = copy.deepcopy(cur)
                try:
                    fr_s = e_s.step(GameAction[sn], data=None)
                except Exception:
                    continue
                if fr_s is None or not fr_s.frame:
                    continue
                d_s = residual(keep, feats(np.asarray(fr_s.frame[-1], dtype=np.int16)))
                if d_s is None:
                    continue
                sib_n += 1; per["sib_n"] += 1
                if d_s < d0 - 1e-12:
                    sib_better += 1; per["sib"] += 1
                auc_pairs.append(1.0 if d_w < d_s else (0.5 if d_w == d_s else 0.0))
                _ds_list.append(d_s)
            if _ds_list:
                _b = sum(1 for x in _ds_list if x < d_w - 1e-12); _t = sum(1 for x in _ds_list if abs(x - d_w) <= 1e-12)
                ranks.append((1 + _b, 1 + _b + _t, 1 + len(_ds_list)))
            cur = e_win
            if int(fr_w.levels_completed or 0) > 1:
                break
        if per["steps"]:
            rows.append(per)
            print("%s: кандидатов %d, шагов %d, выигрышный приближает %d, братья %d из %d"
                  % (per["game"], per["candidates"], per["steps"], per["win"], per["sib"], per["sib_n"]), flush=True)
    auc = float(np.mean(auc_pairs)) if auc_pairs else 0.5
    pw = win_better / max(win_n, 1); ps = sib_better / max(sib_n, 1)
    print("\nИТОГ (%d игр, цель ПЕРЕНЕСЕНА с уровня 1 на уровень 2):" % len(rows))
    print("  P(остаток уменьшился | ВЫИГРЫШНЫЙ ход) = %d/%d = %.2f" % (win_better, win_n, pw))
    print("  P(остаток уменьшился | ход-брат)       = %d/%d = %.2f" % (sib_better, sib_n, ps))
    print("  разница %.2f; AUC = %.3f" % (pw - ps, auc))
    print("  КРИТЕРИЙ (задан до замера): AUC >= 0.60 И разница >= 0.10 -> %s" % ("ВЫПОЛНЕН" if auc >= 0.6 and pw - ps >= 0.1 else "НЕ ВЫПОЛНЕН"))
    if ranks:
        print("  МЕСТО выигрышного хода по Δd (перенесённая цель) среди %d состояний (вариантов: медиана %d):"
              % (len(ranks), int(np.median([r[2] for r in ranks]))))
        for k in (1, 2, 3):
            print("    top-%d: оптимистично %.2f, пессимистично %.2f, случайный угад %.2f" % (
                k, np.mean([r[0] <= k for r in ranks]), np.mean([r[1] <= k for r in ranks]),
                np.mean([min(1.0, k / r[2]) for r in ranks])))
    json.dump({"ranks": ranks, "per_game": rows, "win": [win_better, win_n], "sib": [sib_better, sib_n], "auc": auc},
              open(ROOT / a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("записано:", a.out)


if __name__ == "__main__":
    main()

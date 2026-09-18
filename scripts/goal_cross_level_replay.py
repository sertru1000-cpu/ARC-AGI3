"""Перенос ЦЕЛИ между уровнями на ВСЕХ 25 играх (17.09, вопрос владельца «почему всего 2? сделай 25 игр»).

Прежняя проверка (goal_cross_level.py) упиралась в поиск: до уровня 2 слепой перебор доходил лишь в 2 играх из 10.
Здесь источник другой и покрывает все игры: НАСТОЯЩИЕ траектории модели из наших прогонов. Локальный движок
детерминирован (память arc-agi-3-local-engine-replays-battle), поэтому записанную последовательность ходов можно
повторить и снять кадры, которых нет в записи: завершающий ход возвращает анимацию, и кадры до последнего показывают
решённое состояние уровня (последний принадлежит уже следующему уровню).

Для каждой игры и каждого прогона:
  * повторяем ходы из artifacts/<игра>_p0_events.jsonl;
  * для каждого взятия уровня k снимаем ЦЕЛЬ (кадр до переключения) и НЕ-ЦЕЛИ (все состояния, виденные на уровне k);
  * выводим шаблоны, отделяющие цель уровня k от всех не-целей уровня k (словарь goal_predicates.templates);
  * МЕРА ПЕРЕНОСА для пары соседних уровней k -> k+1: (а) совпал ли ТИП цели (шаблон), (б) переносится ли КОНКРЕТНЫЙ
    предикат (истинен в цели k+1 и отделяет её), (в) контроль: тот же предикат на случайном состоянии уровня k+1.

usage:  .venv/bin/python scripts/goal_cross_level_replay.py [--runs runs/flash_v1_phaseA,...] [--out runs/goal_cross_replay_17_09.json]
"""
import argparse, glob, json, random, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import logging; logging.disable(logging.ERROR)
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
from goal_predicates import features, templates

DEFAULT_RUNS = ("runs/flash_v1_phaseA", "runs/public_flash_keithtyser", "runs/public_flash_tufa",
                "runs/carry_probe_v1", "runs/carry_probe_v2", "runs/bfscarry_probe_v1")


def load_events(run, gid):
    f = glob.glob(str(ROOT / run / "artifacts" / f"{gid}_p0_events.jsonl"))
    if not f:
        return []
    out = []
    for line in open(f[0], encoding="utf-8"):
        e = json.loads(line)
        if e.get("type") != "action":
            continue
        m = re.match(r"MOUSE\(row=(\d+), col=(\d+)\)", str(e.get("action_display") or ""))
        out.append({"name": str(e.get("action_name")), "data": ({"x": int(m.group(2)), "y": int(m.group(1))} if m else None),
                    "done": e.get("level_completed") in (True, "True")})
    return out


def replay_collect(arc, gid, evs, max_actions=4000):
    """повтор записи: {уровень: {"goal": кадр цели, "negs": [состояния уровня]}}"""
    env = arc.make(gid)
    if env is None:
        return {}
    fr = env.reset(); per = {}; lvl = 0
    cur = [np.asarray(fr.frame[-1], dtype=np.int16)]
    for e in evs[:max_actions]:
        try:
            act = GameAction.RESET if e["name"] == "RESET" else GameAction[e["name"]]
            fr = env.step(act, data=e["data"])
        except Exception:
            break
        if fr is None:
            break
        frames = [np.asarray(x, dtype=np.int16) for x in (fr.frame or [])]
        if not frames:
            continue
        new_lvl = int(fr.levels_completed or 0)
        if new_lvl > lvl:
            goal = frames[:-1][-1] if len(frames) > 1 else frames[-1]
            per[new_lvl] = {"goal": goal, "negs": cur}
            cur = [frames[-1]]; lvl = new_lvl
        else:
            cur.append(frames[-1])
    return per


def build_lib(states):
    """библиотека форм, виденных В ЛЮБОЙ момент игры: {цвет: множество форм}."""
    lib = {}
    mask = None
    for g in states[:400]:
        if mask is None or mask.shape != g.shape:
            mask = np.zeros_like(g, dtype=bool)
        f = features(g, mask)
        for c, sh in f["shapes"].items():
            lib.setdefault(c, set()).update(sh)
    return lib


def separating(goal, negs, lib=None):
    mask = np.zeros_like(goal, dtype=bool)
    f_goal = features(goal, mask)
    f_negs = [features(g, mask) for g in negs if g.shape == goal.shape]
    f_start = f_negs[0] if f_negs else None      # первое состояние уровня = старт уровня
    out = []
    for nm, tid, fn in templates(f_goal, lib, f_start):
        try:
            if fn(f_goal) and not any(fn(x) for x in f_negs):
                out.append((nm, tid, fn))
        except Exception:
            pass
    return out, f_goal, f_negs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    ap.add_argument("--out", default="runs/goal_cross_replay_17_09.json"); a = ap.parse_args()
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT / "environment_files"))
    ids = [e.game_id for e in arc.get_environments()]
    rng = random.Random(0)
    rows = []
    for gid in ids:
        best = {}
        for run in a.runs.split(","):
            evs = load_events(run, gid)
            if not evs:
                continue
            per = replay_collect(arc, gid, evs)
            for k, v in per.items():
                if k not in best or len(v["negs"]) > len(best[k]["negs"]):
                    best[k] = v
        lv = sorted(best)
        row = {"game": gid[:4], "levels_with_goal": lv}
        pairs = []
        seps = {}
        lib = build_lib([g for k in lv for g in best[k]["negs"]] + [best[k]["goal"] for k in lv])
        for k in lv:
            s, fg, fn_ = separating(best[k]["goal"], best[k]["negs"], lib)
            seps[k] = (s, fg, fn_)
            row["types_level_%d" % k] = sorted({t for _, t, _ in s})
        for k in lv:
            if k + 1 not in seps:
                continue
            s1, _, _ = seps[k]; s2, fg2, fn2 = seps[k + 1]
            t1 = {t for _, t, _ in s1}; t2 = {t for _, t, _ in s2}
            hold = sep = 0
            per_type = {}
            for nm, tid, fn in s1:
                d = per_type.setdefault(tid, {"n": 0, "sep": 0}); d["n"] += 1
                try:
                    if fn(fg2):
                        hold += 1
                        if not any(fn(x) for x in fn2):
                            sep += 1; d["sep"] += 1
                except Exception:
                    pass
            ctrl = 0
            if fn2:
                rnd = rng.choice(fn2)
                for nm, tid, fn in s1:
                    try:
                        if fn(rnd) and not any(fn(x) for x in fn2 if x is not rnd):
                            ctrl += 1
                    except Exception:
                        pass
            pairs.append({"per_type": per_type, "from": k, "to": k + 1, "types_common": sorted(t1 & t2), "types_from": sorted(t1), "types_to": sorted(t2),
                          "preds_hold": hold, "preds_separate": sep, "control_random": ctrl,
                          "negatives": len(fn2), "preds_from": len(s1)})
        row["pairs"] = pairs
        rows.append(row)
        print("%s уровней с целью %s; пары: %s" % (row["game"], lv, json.dumps(pairs, ensure_ascii=False)), flush=True)
    allp = [p for r in rows for p in r["pairs"]]
    games_with_pairs = [r["game"] for r in rows if r["pairs"]]
    if allp:
        print("\nИТОГ: игр 25; игр с двумя и более уровнями %d (%s); пар соседних уровней %d"
              % (len(games_with_pairs), ",".join(games_with_pairs), len(allp)))
        print("  ТИП цели совпал: %d пар из %d (%.0f%%)" % (sum(1 for p in allp if p["types_common"]), len(allp),
                                                            100 * sum(1 for p in allp if p["types_common"]) / len(allp)))
        print("  конкретный предикат ОТДЕЛЯЕТ цель следующего уровня: %d пар (%.0f%%)"
              % (sum(1 for p in allp if p["preds_separate"] > 0), 100 * sum(1 for p in allp if p["preds_separate"] > 0) / len(allp)))
        print("  контроль (те же предикаты на случайном состоянии): %d пар" % sum(1 for p in allp if p["control_random"] > 0))
    json.dump(rows, open(ROOT / a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print("записано:", a.out)


if __name__ == "__main__":
    main()

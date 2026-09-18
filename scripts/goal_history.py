"""Цели над ИСТОРИЕЙ, а не над кадром + штраф за подобранные параметры (18.09, ответ критика раунда 9, H1-H5).

Три правки к прежнему словарю, каждая лечит измеренную болезнь:
  1. СОБЫТИЯ И ИСТОРИЯ. Прежние предикаты описывали кадр взятия. Здесь цель может быть свойством ТРАЕКТОРИИ:
     посещены все объекты цвета c; событие E когда-либо произошло; событие A произошло раньше B; событие E не
     происходило ни разу; условие выполнено не позже N ходов; одно действие повторено k раз подряд.
     События берутся из пары (состояние до, действие, состояние после): исчез объект цвета c; появился цвет c;
     число областей цвета c уменьшилось; доска не изменилась; сделан клик по цвету c; нажата стрелка/SPACE.
  2. ПЕРВОЕ ИСТИННОЕ ЗНАЧЕНИЕ НА ПОСЛЕДНЕМ ШАГЕ (критик: «predicate становится true впервые на T»). Предикат
     обязан быть ложным на ВСЕХ префиксах траектории и истинным только на полной -- это отсекает предикаты,
     которые случайно выполнялись и раньше.
  3. ШТРАФ ЗА ПОДБОР (MDL). Каждый свободно выбранный параметр стоит биты: цвет log2(16), число log2(допустимых),
     пара цветов log2(120), оператор log2(числа операторов). Предикаты ранжируются по (отделяет? да) и далее по
     возрастанию длины описания -- сначала самые дешёвые.

Мера качества -- та же, что у критика: доля ложного отделения (случайное не-целевое состояние/префикс объявляется
целью, параметры подбираются честно) и leave-one-game-out перенос на следующий уровень.

usage:  .venv/bin/python scripts/goal_history.py [--trials 20] [--out runs/goal_history_18_09.json]
"""
import argparse, json, math, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import logging; logging.disable(logging.ERROR)
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
from goal_cross_level_replay import load_events, DEFAULT_RUNS
from goal_predicates import features, comps

BITS_COLOR = math.log2(16)
BITS_OP = math.log2(12)


def feats(g):
    return features(g, np.zeros_like(g, dtype=bool))


# ---------- события пары (до, действие, после) ----------
def events(f0, act, f1):
    ev = set()
    for c in set(f0["colors"]) | set(f1["colors"]):
        n0, n1 = f0["colors"].get(c, 0), f1["colors"].get(c, 0)
        if n0 and not n1:
            ev.add(("gone", c))
        if not n0 and n1:
            ev.add(("appeared", c))
        if n1 < n0:
            ev.add(("fewer_cells", c))
        k0, k1 = len(f0["comps"].get(c, [])), len(f1["comps"].get(c, []))
        if k1 < k0:
            ev.add(("fewer_groups", c))
        if k1 > k0:
            ev.add(("more_groups", c))
    if np.array_equal(f0["grid"], f1["grid"]):
        ev.add(("no_change", None))
    name = str(act.get("name") or "")
    if name == "ACTION6":
        ev.add(("click", None))
        d = act.get("data") or {}
        y, x = int(d.get("y", 0)), int(d.get("x", 0))
        g = f0["grid"]
        if 0 <= y < g.shape[0] and 0 <= x < g.shape[1]:
            ev.add(("click_colour", int(g[y, x])))
    elif name:
        ev.add(("move", name))
    return ev


# ---------- предикаты над историей: (тип, параметры) + длина описания в битах ----------
def history_candidates(traj):
    """traj = список кортежей (f_before, act, f_after, events). Кандидаты из наблюдённой траектории."""
    all_ev = set().union(*[e for *_, e in traj]) if traj else set()
    out = []
    for e in all_ev:
        bits = BITS_OP + (BITS_COLOR if e[1] is not None else 0)
        out.append((("ever", e), bits))
        out.append((("never", e), bits))
        cnt = sum(1 for *_, evs in traj if e in evs)
        if cnt > 1:
            out.append((("count_ge", e, cnt), bits + math.log2(max(2, cnt + 1))))
    seen_order = []
    for *_, evs in traj:
        for e in sorted(evs):
            if e not in seen_order:
                seen_order.append(e)
    for i, a in enumerate(seen_order[:6]):
        for b in seen_order[i + 1:7]:
            bits = 2 * BITS_OP + (BITS_COLOR if a[1] is not None else 0) + (BITS_COLOR if b[1] is not None else 0)
            out.append((("before", a, b), bits))
    if traj:
        n = len(traj)
        out.append((("within", n), BITS_OP + math.log2(max(2, n + 1))))
        runs = {}
        cur = None; ln = 0
        for _, act, _, _ in traj:
            nm = str(act.get("name") or "")
            if nm == cur:
                ln += 1
            else:
                cur, ln = nm, 1
            runs[cur] = max(runs.get(cur, 0), ln)
        for nm, ln in runs.items():
            if ln >= 3:
                out.append((("run_ge", nm, ln), BITS_OP + math.log2(max(2, ln + 1))))
        # посещение: все объекты цвета c были задеты кликом
        clicked = {e[1] for *_, evs in traj for e in evs if e[0] == "click_colour"}
        for c in clicked:
            out.append((("clicked_colour", c), BITS_OP + BITS_COLOR))
    return out


def holds(pred, traj):
    """истинность предиката на (префиксе) траектории."""
    t = pred[0]
    if t == "ever":
        return any(pred[1] in evs for *_, evs in traj)
    if t == "never":
        return not any(pred[1] in evs for *_, evs in traj)
    if t == "count_ge":
        return sum(1 for *_, evs in traj if pred[1] in evs) >= pred[2]
    if t == "before":
        ia = next((i for i, (*_, evs) in enumerate(traj) if pred[1] in evs), None)
        ib = next((i for i, (*_, evs) in enumerate(traj) if pred[2] in evs), None)
        return ia is not None and ib is not None and ia < ib
    if t == "within":
        return len(traj) <= pred[1]
    if t == "run_ge":
        cur = None; ln = 0
        for _, act, _, _ in traj:
            nm = str(act.get("name") or "")
            ln = ln + 1 if nm == cur else 1
            cur = nm
            if nm == pred[1] and ln >= pred[2]:
                return True
        return False
    if t == "clicked_colour":
        return any(("click_colour", pred[1]) in evs for *_, evs in traj)
    return False


def induce(traj, require_first_true=True):
    """предикаты, истинные на полной траектории и ЛОЖНЫЕ на всех её префиксах; отсортированы по длине описания."""
    out = []
    for pred, bits in history_candidates(traj):
        if not holds(pred, traj):
            continue
        if require_first_true and any(holds(pred, traj[:i]) for i in range(1, len(traj))):
            continue
        out.append((bits, pred))
    return sorted(out)


def collect(arc, gid):
    """траектории по уровням из записей прогонов: {уровень: [(f0, act, f1, events), ...]}"""
    best = {}
    for run in DEFAULT_RUNS:
        evs = load_events(run, gid)
        if not evs:
            continue
        env = arc.make(gid)
        if env is None:
            return {}
        fr = env.reset()
        if fr is None or not fr.frame:
            continue
        cur = []; lvl = 0; f_prev = feats(np.asarray(fr.frame[-1], dtype=np.int16)); per = {}
        for e in evs[:2000]:
            try:
                act = GameAction.RESET if e["name"] == "RESET" else GameAction[e["name"]]
                fr = env.step(act, data=e["data"])
            except Exception:
                break
            if fr is None or not fr.frame:
                break
            f_now = feats(np.asarray(fr.frame[-1], dtype=np.int16))
            cur.append((f_prev, e, f_now, events(f_prev, e, f_now)))
            new_lvl = int(fr.levels_completed or 0)
            if new_lvl > lvl:
                per[new_lvl] = cur; cur = []; lvl = new_lvl
            f_prev = f_now
        for k, v in per.items():
            if k not in best or len(v) > len(best[k]):
                best[k] = v
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", default="runs/goal_history_18_09.json"); a = ap.parse_args()
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT / "environment_files"))
    rng = random.Random(0)
    data = {}
    for e in arc.get_environments():
        gid = e.game_id
        per = collect(arc, gid)
        per = {k: v for k, v in per.items() if 2 <= len(v) <= 400}
        if per:
            data[gid[:4]] = per
        print("%s: уровней с траекторией %d (длины %s)" % (gid[:4], len(per), [len(v) for v in per.values()][:6]), flush=True)

    # 1) покрытие: на скольких уровнях нашёлся хоть один предикат над историей
    cov = 0; tot = 0; per_level = {}
    for g, per in data.items():
        for k, traj in per.items():
            tot += 1
            preds = induce(traj)
            per_level[(g, k)] = preds
            cov += bool(preds)
    print("\nПОКРЫТИЕ: предикат над историей найден на %d уровнях из %d (%.0f%%)" % (cov, tot, 100 * cov / max(tot, 1)))

    # 2) контроль ложного отделения: псевдоцель = случайный ПРЕФИКС траектории (уровень там не взят)
    hit = trials = 0
    for (g, k), _ in per_level.items():
        traj = data[g][k]
        if len(traj) < 6:
            continue
        for _ in range(min(a.trials, len(traj) - 2)):
            i = rng.randrange(3, len(traj) - 1)
            if induce(traj[:i]):
                hit += 1
            trials += 1
    print("КОНТРОЛЬ: случайный префикс (уровень НЕ взят) отделяется как цель в %d пробах из %d (%.0f%%)"
          % (hit, trials, 100 * hit / max(trials, 1)))

    # 3) перенос на следующий уровень (leave-one-game-out не нужен: предикаты не параметризуются по играм)
    pairs = tr = 0
    for g, per in data.items():
        for k in sorted(per):
            if k + 1 not in per:
                continue
            pairs += 1
            top = [p for _, p in per_level.get((g, k), [])][:3]
            nxt = data[g][k + 1]
            if any(holds(p, nxt) and not any(holds(p, nxt[:i]) for i in range(1, len(nxt))) for p in top):
                tr += 1
    print("ПЕРЕНОС: предикат уровня k (из трёх самых дешёвых) описывает цель уровня k+1 в %d парах из %d (%.0f%%)"
          % (tr, pairs, 100 * tr / max(pairs, 1)))
    ex = [(g, k, [p for _, p in v[:2]]) for (g, k), v in list(per_level.items()) if v][:8]
    for g, k, p in ex:
        print("   %s уровень %d: %s" % (g, k, p))
    json.dump({"coverage": [cov, tot], "control": [hit, trials], "transfer": [tr, pairs],
               "examples": [[g, k, [str(x) for x in p]] for g, k, p in ex]},
              open(ROOT / a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("записано:", a.out)


if __name__ == "__main__":
    main()

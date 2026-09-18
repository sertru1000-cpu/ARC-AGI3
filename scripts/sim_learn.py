"""Сколько НАСТОЯЩИХ ходов нужно, чтобы выучить симулятор, пригодный для планирования (18.09, ночь).

Зачем. Метрика конкурса (ходы_базлайна / сделано_ходов)^2 запрещает перебор по настоящей среде, но не запрещает
перебор ВНУТРИ своей модели мира: в tu93 найденный перебором путь на все 9 уровней -- 185 ходов против базлайна 462,
то есть чистое исполнение дало бы почти полный балл за игру (+4 к итогу). Вся цена была в 8334 ходах поиска.
Значит, вопрос один: во сколько НАСТОЯЩИХ ходов обходится обучение симулятора, в котором потом можно искать даром.

Симулятор -- без модели и без обучения весами, объектный и дешёвый:
  * состояние раскладывается на объекты (связные области одного цвета, клетки-часы исключены);
  * для каждого действия копится статистика: сдвиг каждого объекта (dy, dx), исчезновение, появление, смена цвета;
  * предсказание = применить к текущему состоянию самое частое наблюдённое следствие этого действия
    (сдвиг «подвижного» объекта, остальные объекты неподвижны); если следствие не наблюдалось -- «не знаю».
Это заведомо грубая модель: она проверяет не «можно ли выучить игру», а «сколько ходов нужно до полезности».

Мера: доля точно предсказанных переходов (совпадение всей доски) на ОТЛОЖЕННЫХ переходах той же игры как функция
бюджета обучения N = 20, 50, 100, 200, 400 настоящих ходов. Отдельно считается доля «не знаю» (модель честно молчит).
Точка отсчёта: базлайн уровня 1 в этих играх -- 17-43 хода, то есть если полезность наступает позже 100 ходов,
схема «учись, потом планируй» не окупается по метрике.

usage:  .venv/bin/python scripts/sim_learn.py [--games all] [--holdout 120] [--out runs/sim_learn_18_09.json]
"""
import argparse, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import logging; logging.disable(logging.ERROR)
import arc_agi
from arc_agi import OperationMode
from engine_bfs import Env, frame_info
from graph_explorer import alphabet_for, clock_mask


def objects(g, mask):
    """объекты: связные области одного цвета (без клеток-часов), как (цвет, множество клеток)."""
    gg = g.copy()
    if mask.shape == gg.shape:
        gg[mask] = -1
    out = []
    seen = np.zeros_like(gg, dtype=bool)
    h, w = gg.shape
    vals, counts = np.unique(gg[gg >= 0], return_counts=True)
    bg = int(vals[counts.argmax()]) if len(vals) else 0
    for y0 in range(h):
        for x0 in range(w):
            c = int(gg[y0, x0])
            if c < 0 or c == bg or seen[y0, x0]:
                continue
            st = [(y0, x0)]; seen[y0, x0] = True; cells = []
            while st:
                y, x = st.pop(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and int(gg[yy, xx]) == c:
                        seen[yy, xx] = True; st.append((yy, xx))
            out.append((c, frozenset(cells)))
    return out, bg


def shape_of(cells):
    ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
    y0, x0 = min(ys), min(xs)
    return frozenset((y - y0, x - x0) for y, x in cells), (y0, x0)


def transition_effect(o0, o1):
    """следствие хода: для каждого объекта из «до» -- сдвиг (dy,dx), если такой же по форме и цвету объект есть в «после»."""
    idx = defaultdict(list)
    for c, cells in o1:
        sh, pos = shape_of(cells)
        idx[(c, sh)].append(pos)
    eff = []
    for c, cells in o0:
        sh, pos = shape_of(cells)
        cands = idx.get((c, sh), [])
        if not cands:
            eff.append(((c, sh), "gone")); continue
        best = min(cands, key=lambda p: abs(p[0] - pos[0]) + abs(p[1] - pos[1]))
        eff.append(((c, sh), (best[0] - pos[0], best[1] - pos[1])))
    return eff


class Sim:
    """выученный симулятор: сначала находим УПРАВЛЯЕМЫЙ объект (тот, что чаще других меняет положение),
    затем для каждого действия помним его самый частый сдвиг. Всё остальное считается неподвижным --
    это ближе к правде, чем двигать все объекты (измерено 18.09: «двигаем всё» ошибается на 520 клеток
    против 27 у опоры «ничего не меняется»)."""

    def __init__(self):
        self.stats = defaultdict(Counter)     # (действие, ключ объекта) -> следствие
        self.movers = Counter()               # ключ объекта -> сколько раз он двигался

    def learn(self, act, g0, g1, mask):
        o0, _ = objects(g0, mask); o1, _ = objects(g1, mask)
        for keyshape, eff in transition_effect(o0, o1):
            self.stats[(act, keyshape)][eff] += 1
            if eff not in ((0, 0), "gone"):
                self.movers[keyshape] += 1

    def player(self):
        return self.movers.most_common(1)[0][0] if self.movers else None

    def predict(self, act, g, mask):
        """предсказание доски после действия; None -- «не знаю»."""
        o0, bg = objects(g, mask)
        key = self.player()
        if key is None:
            return None                     # управляемый объект ещё не найден -- честно молчим
        out = g.copy()                      # всё остальное неподвижно
        for c, cells in o0:
            sh, pos = shape_of(cells)
            if (c, sh) != key:
                continue
            cnt = self.stats.get((act, key))
            if not cnt:
                return None
            eff, _n = cnt.most_common(1)[0]
            if eff == (0, 0):
                return out
            for (y, x) in cells:
                out[y, x] = bg
            if eff != "gone":
                dy, dx = eff
                for (y, x) in cells:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < g.shape[0] and 0 <= xx < g.shape[1]:
                        out[yy, xx] = c
            return out
        return out                          # управляемого объекта нет на доске -- считаем, что ничего не меняется


def run_game(arc, gid, budgets, holdout, seed=0):
    rng = random.Random(seed)
    env = Env(arc, gid)
    fr = env.reset_and_replay([]); g, lvl0, st, avail = frame_info(fr)
    acts = alphabet_for(g, avail)
    if not acts:
        return None
    mask = clock_mask(env, g, acts)
    # собираем поток переходов случайной игрой
    total = max(budgets) + holdout
    traj = []
    fr = env.reset_and_replay([]); prev, lvl, st, _ = frame_info(fr)
    for _ in range(total):
        act = rng.choice(acts)
        fr = env.do(act); g2, lvl2, st2, _ = frame_info(fr)
        if st2 == "GAME_OVER" or lvl2 > lvl:
            fr = env.reset_and_replay([]); prev, lvl, st, _ = frame_info(fr); continue
        traj.append((act, prev, g2)); prev = g2
    if len(traj) < max(budgets) + 20:
        return None
    test = traj[-holdout:] if len(traj) > holdout else traj[-20:]
    row = {"game": gid[:4], "transitions": len(traj), "curve": {}}
    for n in budgets:
        sim = Sim()
        for act, g0, g1 in traj[:n]:
            sim.learn(act, g0, g1, mask)
        ok = idk = 0; cells_sim = []; cells_noop = []; ok_noop = 0
        for act, g0, g1 in test:
            p = sim.predict(act, g0, mask)
            if p is None:
                idk += 1
                continue
            a_, b_ = p.copy(), g1.copy()
            if mask.shape == a_.shape:      # клетки-часы из сравнения исключаются: они меняются каждый ход
                a_[mask] = -1; b_[mask] = -1
            if np.array_equal(a_, b_):
                ok += 1
            cells_sim.append(int((a_ != b_).sum()))
        for act, g0, g1 in test:      # тривиальная опора: «ничего не меняется»
            a_, b_ = g0.copy(), g1.copy()
            if mask.shape == a_.shape:
                a_[mask] = -1; b_[mask] = -1
            cells_noop.append(int((a_ != b_).sum()))
            ok_noop += int(np.array_equal(a_, b_))
        row["curve"][n] = {"exact": ok / len(test), "unknown": idk / len(test),
                           "cells_sim": float(np.median(cells_sim)) if cells_sim else None,
                           "cells_noop": float(np.median(cells_noop)),
                           "exact_noop": ok_noop / len(test)}
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--games", default="all")
    ap.add_argument("--holdout", type=int, default=120); ap.add_argument("--out", default="runs/sim_learn_18_09.json")
    a = ap.parse_args()
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT / "environment_files"))
    ids = [e.game_id for e in arc.get_environments()]
    if a.games != "all":
        want = set(a.games.split(",")); ids = [g for g in ids if g[:4] in want]
    budgets = [20, 50, 100, 200, 400]
    rows = []
    for gid in ids:
        try:
            r = run_game(arc, gid, budgets, a.holdout)
        except Exception as exc:
            r = {"game": gid[:4], "error": repr(exc)[:160]}
        if r:
            rows.append(r)
            if "curve" in r:
                print("%s: " % r["game"] + " ".join("N=%d точно %.2f не знаю %.2f" % (n, r["curve"][n]["exact"], r["curve"][n]["unknown"]) for n in budgets), flush=True)
    if rows:
        print("\nИТОГ по %d играм (медиана доли ТОЧНЫХ предсказаний):" % len(rows))
        for n in budgets:
            vals = [r["curve"][n]["exact"] for r in rows if "curve" in r]
            idk = [r["curve"][n]["unknown"] for r in rows if "curve" in r]
            cs = [r["curve"][n]["cells_sim"] for r in rows if "curve" in r and r["curve"][n]["cells_sim"] is not None]
            cn = [r["curve"][n]["cells_noop"] for r in rows if "curve" in r]
            en = [r["curve"][n]["exact_noop"] for r in rows if "curve" in r]
            print("  N=%3d ходов: точно %.2f (опора «ничего не меняется» %.2f), не знаю %.2f; ошибка в клетках: симулятор %.0f, опора %.0f"
                  % (n, float(np.median(vals)), float(np.median(en)), float(np.median(idk)),
                     float(np.median(cs)) if cs else -1, float(np.median(cn))))
    json.dump(rows, open(ROOT / a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("записано:", a.out)


if __name__ == "__main__":
    main()

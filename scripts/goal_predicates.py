"""Индукция ЦЕЛИ уровня как проверяемого предиката (17.09, линия «новый подход, а не настройка обвязки»).

Замысел. Наши измерения: модель строит верную механику (86% предсказаний невиданных переходов, tn36 200/200), но цель
уровня не выводит (стратег 1/6; обученный детектор цели проиграл эвристике max-diff). Все прежние попытки давали модели
больше данных или больше перебора. Здесь цель представлена иначе: как ПРЕДИКАТ над состоянием из небольшого словаря
шаблонов, который обязан быть истинным в момент взятия уровня и ложным во всех виденных состояниях до него.
Проверяется не перенос весов (шесть нулей), а перенос ЯЗЫКА целей между играми.

Данные (локальный движок, бесплатно): пути уровня 1 из runs/bfs_originals.json (10 игр из 25 решены перебором).
  * ЦЕЛЬ: завершающий ход возвращает несколько кадров анимации; последний кадр -- уже следующий уровень, поэтому
    кандидаты в целевое состояние -- кадры до последнего (проверено: 2-9 кадров, все отличаются от последнего).
  * НЕ-ЦЕЛЬ: состояния вдоль пути и состояния случайного блуждания той же игры (--negatives).
  * Клетки-часы (счётчики/таймеры) исключаются: клетка считается часами, если меняется в >= 80% ходов блуждания.

Словарь шаблонов (язык целей), проверяется КАЖДЫЙ на каждой игре:
  T1  нет клеток цвета c                    T8  все клетки цвета c внутри рамки цвета d
  T2  клеток цвета c ровно k                T9  крупнейшая область цвета c занимает N клеток
  T3  клетки цвета c -- одна область        T10 строка/столбец целиком одного цвета
  T4  областей цвета c ровно m              T11 различных цветов на доске ровно k
  T5  доска симметрична (гориз./верт.)      T12 фигура цвета c совпала по форме с фигурой цвета d
  T6  клетки цвета c образуют прямоугольник T13 клетки цвета c совпали с клетками цвета d по позициям
  T7  цвет c занимает всю не-фоновую часть  T14 нет двух соседних клеток цвета c

Мера: для каждой игры -- какие шаблоны отделяют цель от ВСЕХ не-целей (параметры берутся из целевого состояния).
Перенос: для каждой игры проверяется, отделяет ли её цель хоть один шаблон, найденный на ОСТАЛЬНЫХ играх
(leave-one-game-out по словарю шаблонов, не по параметрам) -- это и есть вопрос «переносится ли язык целей».

usage:  .venv/bin/python scripts/goal_predicates.py [--negatives 200] [--out runs/goal_predicates_17_09.json]
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import logging; logging.disable(logging.ERROR)
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction


def comps(mask):
    """связные области (4-связность) булевой маски: список размеров и bbox."""
    h, w = mask.shape; seen = np.zeros_like(mask, dtype=bool); out = []
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            st = [(y0, x0)]; seen[y0, x0] = True; cells = []
            while st:
                y, x = st.pop(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True; st.append((yy, xx))
            ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
            out.append({"n": len(cells), "bbox": (min(ys), min(xs), max(ys), max(xs)), "cells": cells})
    return out


def shape_key(cells):
    ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
    y0, x0 = min(ys), min(xs)
    return frozenset((y - y0, x - x0) for y, x in cells)


def features(g, mask):
    """скалярные признаки состояния, на которых считаются предикаты (клетки-часы исключены)."""
    gg = g.copy(); gg[mask] = -1
    vals, counts = np.unique(gg[gg >= 0], return_counts=True)
    bg = int(vals[counts.argmax()]) if len(vals) else 0
    f = {"bg": bg, "colors": {int(v): int(c) for v, c in zip(vals, counts)}, "ncolors": int(len(vals)), "grid": gg}
    f["comps"] = {int(v): comps(gg == int(v)) for v in vals if int(v) != bg}
    return f


# ---- шаблоны: каждый по целевому состоянию даёт список конкретных предикатов (id, функция) ----
def templates(f_goal):
    out = []
    bg = f_goal["bg"]; cols = [c for c in f_goal["colors"] if c != bg]; gg = f_goal["grid"]

    for c in list(f_goal["colors"]) + [c for c in range(16) if c not in f_goal["colors"]]:
        if f_goal["colors"].get(c, 0) == 0:
            out.append(("T1 нет цвета %d" % c, "T1", lambda f, c=c: f["colors"].get(c, 0) == 0))
    for c in cols:
        k = f_goal["colors"][c]
        out.append(("T2 цвета %d ровно %d клеток" % (c, k), "T2", lambda f, c=c, k=k: f["colors"].get(c, 0) == k))
        cs = f_goal["comps"].get(c, [])
        if len(cs) == 1:
            out.append(("T3 цвет %d -- одна область" % c, "T3", lambda f, c=c: len(f["comps"].get(c, [])) == 1))
        out.append(("T4 областей цвета %d ровно %d" % (c, len(cs)), "T4", lambda f, c=c, m=len(cs): len(f["comps"].get(c, [])) == m))
        if cs:
            n = max(x["n"] for x in cs)
            out.append(("T9 крупнейшая область цвета %d = %d" % (c, n), "T9",
                        lambda f, c=c, n=n: bool(f["comps"].get(c)) and max(x["n"] for x in f["comps"][c]) == n))
            big = max(cs, key=lambda x: x["n"]); y0, x0, y1, x1 = big["bbox"]
            if big["n"] == (y1 - y0 + 1) * (x1 - x0 + 1):
                out.append(("T6 цвет %d -- прямоугольник" % c, "T6",
                            lambda f, c=c: bool(f["comps"].get(c)) and all(
                                x["n"] == (x["bbox"][2] - x["bbox"][0] + 1) * (x["bbox"][3] - x["bbox"][1] + 1) for x in f["comps"][c])))
        out.append(("T14 нет соседних клеток цвета %d" % c, "T14", lambda f, c=c: _no_adj(f, c)))
    ok = gg >= 0
    if ok.any():
        if np.array_equal(gg[:, ::-1][ok[:, ::-1]], gg[ok]) if gg.shape[1] > 1 else False:
            pass
        out.append(("T5 доска симметрична по вертикали", "T5", lambda f: _sym(f, 1)))
        out.append(("T5 доска симметрична по горизонтали", "T5", lambda f: _sym(f, 0)))
    for c in cols:
        out.append(("T7 цвет %d -- вся не-фоновая часть" % c, "T7",
                    lambda f, c=c: f["colors"].get(c, 0) > 0 and sum(v for k, v in f["colors"].items() if k != f["bg"]) == f["colors"].get(c, 0)))
    out.append(("T11 различных цветов ровно %d" % f_goal["ncolors"], "T11", lambda f, k=f_goal["ncolors"]: f["ncolors"] == k))
    out.append(("T10 есть строка одного не-фонового цвета", "T10", lambda f: _full_line(f, 0)))
    out.append(("T10 есть столбец одного не-фонового цвета", "T10", lambda f: _full_line(f, 1)))
    for c in cols:
        for d in cols:
            if c >= d:
                continue
            out.append(("T12 фигуры цветов %d и %d совпали по форме" % (c, d), "T12", lambda f, c=c, d=d: _same_shape(f, c, d)))
            out.append(("T13 клетки цвета %d совпали с клетками цвета %d" % (c, d), "T13", lambda f, c=c, d=d: _inside(f, c, d)))
            out.append(("T8 цвет %d внутри рамки цвета %d" % (c, d), "T8", lambda f, c=c, d=d: _in_bbox(f, c, d)))
    return out


def _no_adj(f, c):
    m = f["grid"] == c
    if not m.any():
        return False
    return not ((m[:-1, :] & m[1:, :]).any() or (m[:, :-1] & m[:, 1:]).any())


def _sym(f, axis):
    g = f["grid"]
    return bool(np.array_equal(g, np.flip(g, axis=axis)))


def _full_line(f, axis):
    g = f["grid"]; bg = f["bg"]
    lines = g if axis == 0 else g.T
    for row in lines:
        v = row[row >= 0]
        if len(v) and v[0] != bg and (v == v[0]).all():
            return True
    return False


def _same_shape(f, c, d):
    a = f["comps"].get(c, []); b = f["comps"].get(d, [])
    if not a or not b:
        return False
    return bool({shape_key(x["cells"]) for x in a} & {shape_key(x["cells"]) for x in b})


def _inside(f, c, d):
    """клетки цвета c образуют то же множество позиций, что и цвета d (совпадение по позициям невозможно --
    сравниваем формы+смещения областей: область цвета c совпала с областью цвета d по bbox)."""
    a = f["comps"].get(c, []); b = f["comps"].get(d, [])
    return bool({x["bbox"] for x in a} & {x["bbox"] for x in b})


def _in_bbox(f, c, d):
    a = f["comps"].get(c, []); b = f["comps"].get(d, [])
    if not a or not b:
        return False
    for bb in b:
        y0, x0, y1, x1 = bb["bbox"]
        if all(y0 <= x["bbox"][0] and x0 <= x["bbox"][1] and x["bbox"][2] <= y1 and x["bbox"][3] <= x1 for x in a):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solutions", default="runs/bfs_originals.json"); ap.add_argument("--negatives", type=int, default=200)
    ap.add_argument("--out", default="runs/goal_predicates_17_09.json"); a = ap.parse_args()
    sol = json.load(open(ROOT / a.solutions, encoding="utf-8"))
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT / "environment_files"))
    rng = random.Random(0)
    per_game = {}
    for gid, rec in sol.items():
        if not rec.get("solved"):
            continue
        env = arc.make(gid); fr = env.reset()
        path = [(n, p) for n, p in rec["path"]]
        # маска часов: случайное блуждание от старта
        walk = []; prev = np.asarray(fr.frame[-1], dtype=np.int16); ch = np.zeros_like(prev, dtype=np.int32); nch = 0
        acts = [GameAction[x] for x in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")]
        for _ in range(a.negatives):
            f2 = env.step(rng.choice(acts), data=None)
            g2 = np.asarray(f2.frame[-1], dtype=np.int16)
            if int(f2.levels_completed or 0) > 0 or str(f2.state).split(".")[-1] == "GAME_OVER":
                f2 = env.reset(); prev = np.asarray(f2.frame[-1], dtype=np.int16); continue
            if g2.shape == prev.shape and (g2 != prev).any():
                ch += (g2 != prev); nch += 1
            walk.append(g2); prev = g2
        mask = (ch >= 0.8 * max(1, nch)) & (ch >= 3)
        # путь: не-цели вдоль пути + целевые кадры завершающего хода
        fr = env.reset(); along = [np.asarray(fr.frame[-1], dtype=np.int16)]
        goal_frames = []
        for i, (name, payload) in enumerate(path):
            fr = env.step(GameAction[name] if name != "RESET" else GameAction.RESET, data=payload)
            frames = [np.asarray(x, dtype=np.int16) for x in fr.frame]
            if i == len(path) - 1:
                goal_frames = frames[:-1] or frames[:1]
            else:
                along.append(frames[-1])
        negs = [g for g in along + walk if g.shape == goal_frames[0].shape]
        f_negs = [features(g, mask) for g in negs]
        found = {}
        for gf in goal_frames:
            f_goal = features(gf, mask)
            for name, tid, fn in templates(f_goal):
                try:
                    if not fn(f_goal):
                        continue
                    if any(fn(fn2) for fn2 in f_negs):
                        continue
                except Exception:
                    continue
                found.setdefault(tid, []).append(name)
        per_game[gid[:4]] = {"templates": sorted(found), "examples": {k: v[0] for k, v in found.items()},
                             "negatives": len(negs), "goal_frames": len(goal_frames)}
        print("%s: отделяют цель шаблоны %s (не-целей %d)" % (gid[:4], sorted(found) or "—", len(negs)), flush=True)
    # перенос: покрывает ли словарь шаблонов, найденный на ОСТАЛЬНЫХ играх, цель этой игры
    cov = {}
    for g, r in per_game.items():
        others = set().union(*[set(x["templates"]) for gg, x in per_game.items() if gg != g]) if len(per_game) > 1 else set()
        cov[g] = sorted(set(r["templates"]) & others)
    n_sep = sum(1 for r in per_game.values() if r["templates"]); n_cov = sum(1 for v in cov.values() if v)
    print("\nИТОГ: игр с решением перебором %d; цель отделена хотя бы одним шаблоном в %d; словарь остальных игр покрывает цель в %d"
          % (len(per_game), n_sep, n_cov))
    from collections import Counter
    cnt = Counter(t for r in per_game.values() for t in r["templates"])
    print("частота шаблонов:", dict(cnt.most_common()))
    json.dump({"per_game": per_game, "leave_one_out": cov}, open(ROOT / a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("записано:", a.out)


if __name__ == "__main__":
    main()

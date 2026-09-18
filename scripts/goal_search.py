"""Поиск, ведомый ГИПОТЕЗОЙ О ЦЕЛИ (17.09). Проверка того, ради чего строился словарь предикатов:
знание «как выглядят цели вообще» должно вести поиск в НОВОЙ игре, где параметры цели неизвестны.

Отличие от engine_bfs2.py (слепой перебор в ширину, та же среда, те же снимки, те же бюджеты): порядок раскрытия узлов.
Здесь из словаря шаблонов (goal_predicates.py) берутся БЕСПАРАМЕТРИЧЕСКИЕ кандидаты в цель, каждый со своей мерой
близости (остаток), и первым раскрывается состояние, наиболее близкое хоть к одному кандидату:
  C1  не осталось клеток цвета c                  остаток = доля оставшихся клеток цвета c
  C3  клетки цвета c -- одна область              остаток = (областей - 1) / областей вначале
  C9  крупнейшая область цвета c -- весь цвет c    остаток = доля клеток вне крупнейшей области
  C8  все клетки цвета c внутри рамки цвета d      остаток = доля клеток c вне рамки
  C13 область цвета c совпала рамкой с областью d  остаток = нормированное расстояние между рамками
  C12 фигуры цветов c и d совпали по форме         остаток = 0/1
  C5  доска симметрична (гор./верт.)               остаток = доля несовпадающих не-фоновых клеток
  C10 есть строка/столбец одного не-фонового цвета остаток = минимальная доля «лишних» клеток в строке/столбце
  C11 различных цветов стало меньше                остаток = (цветов - 1) / цветов вначале
Ничья -- по новизне состояния и глубине. Цель поиска прежняя: взять уровень (движок сам сообщает об этом).

Сравнение честное: те же игры, те же бюджеты (--max-states / --max-moves / --timeout), та же среда и тот же алфавит,
что у engine_bfs2.py --mode exact (docs/bfs2_compare_16_09.txt: решено 15 из 25).

usage:  .venv/bin/python scripts/goal_search.py --games all [--max-states 3000 --max-moves 60000 --timeout 240] --out runs/goal_search.json
"""
import argparse, copy, heapq, json, sys, time
from collections import deque
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import logging; logging.disable(logging.ERROR)
import arc_agi
from arc_agi import OperationMode
from engine_bfs2 import Snap, Counter as Cnt, step, root_snap, simple_actions, active_clicks, clock_mask, KeyOf as Key
from goal_predicates import comps, shape_key
from goal_distance import dist_shapes, dist_bbox, dist_inside, dist_merge, match_cost, shape_gap, region_gap


def state_features(g, mask):
    gg = g.copy(); gg[mask] = -1
    vals, counts = np.unique(gg[gg >= 0], return_counts=True)
    if not len(vals):
        return {"bg": 0, "cnt": {}, "comps": {}, "grid": gg, "ncolors": 0}
    bg = int(vals[counts.argmax()])
    cnt = {int(v): int(c) for v, c in zip(vals, counts)}
    cm = {c: comps(gg == c) for c in cnt if c != bg}
    h, w = gg.shape
    holes = [x for x in comps(gg == bg)
             if x["bbox"][0] > 0 and x["bbox"][1] > 0 and x["bbox"][2] < h - 1 and x["bbox"][3] < w - 1 and x["n"] <= 0.25 * gg.size]
    from goal_predicates import regions as _regions
    return {"bg": bg, "cnt": cnt, "comps": cm, "grid": gg, "ncolors": len(cnt), "holes": holes,
            "regions": _regions(gg, bg),
            "shapes": {c: {shape_key(x["cells"]) for x in cs} for c, cs in cm.items()}}


def candidates(f0, lib=None):
    """беспараметрические кандидаты в цель + мера остатка (0 = цель достигнута), нормированные по стартовому состоянию."""
    out = []
    cols = [c for c in f0["cnt"] if c != f0["bg"]]
    for c in cols:
        n0 = max(1, f0["cnt"][c])
        out.append(("C1 нет цвета %d" % c, lambda f, c=c, n0=n0: f["cnt"].get(c, 0) / n0))
        out.append(("C3 цвет %d одной областью" % c, lambda f, c=c: dist_merge(f, c)))
        out.append(("C9 цвет %d весь в одной области" % c, lambda f, c=c: _outside_largest(f, c)))
        for d in cols:
            if c == d:
                continue
            out.append(("C8 цвет %d внутри рамки цвета %d" % (c, d), lambda f, c=c, d=d: dist_inside(f, c, d)))
            if c < d:
                out.append(("C13 рамки цветов %d и %d совпали" % (c, d), lambda f, c=c, d=d: dist_bbox(f, c, d)))
                out.append(("C12 формы цветов %d и %d совпали" % (c, d), lambda f, c=c, d=d: dist_shapes(f, c, d)))
    out.append(("C5 симметрия по вертикали", lambda f: _asym(f, 1)))
    out.append(("C5 симметрия по горизонтали", lambda f: _asym(f, 0)))
    out.append(("C10 строка одного цвета", lambda f: _line_residual(f, 0)))
    out.append(("C10 столбец одного цвета", lambda f: _line_residual(f, 1)))
    n0c = max(1, f0["ncolors"])
    out.append(("C11 меньше цветов", lambda f, n0c=n0c: max(0, f["ncolors"] - 1) / n0c))
    shapes_seen = {r["sub"].shape for r in f0.get("regions", [])}
    for sh in shapes_seen:   # C17: узор одной области сошёлся с узором другой такого же размера
        out.append(("C17 узоры областей %dx%d сошлись" % sh, lambda f, sh=sh: region_gap(f, sh)))
    for c in cols:   # C15: деталь встала в выемку (дополняющий объект)
        out.append(("C15 цвет %d в пустоту" % c, lambda f, c=c: match_cost(f["comps"].get(c, []), f.get("holes", []))))
    if lib:          # C16: объект принял форму образца, виденного в любой момент игры
        for c in cols:
            for d, sd in lib.items():
                if d == c or not sd:
                    continue
                out.append(("C16 цвет %d принимает форму образца %d" % (c, d),
                            lambda f, c=c, sd=frozenset(sd): shape_gap(f["shapes"].get(c, set()), sd)))
    return out


def _outside_largest(f, c):
    cs = f["comps"].get(c, []); n = f["cnt"].get(c, 0)
    if not cs or not n:
        return 1.0
    return (n - max(x["n"] for x in cs)) / n


def _outside_bbox(f, c, d):
    a = f["comps"].get(c, []); b = f["comps"].get(d, [])
    if not a or not b:
        return 1.0
    cells = [cell for x in a for cell in x["cells"]]
    best = 1.0
    for bb in b:
        y0, x0, y1, x1 = bb["bbox"]
        out = sum(1 for (y, x) in cells if not (y0 <= y <= y1 and x0 <= x <= x1))
        best = min(best, out / max(1, len(cells)))
    return best


def _bbox_dist(f, c, d):
    a = f["comps"].get(c, []); b = f["comps"].get(d, [])
    if not a or not b:
        return 1.0
    best = 1.0
    for x in a:
        for y in b:
            dist = sum(abs(p - q) for p, q in zip(x["bbox"], y["bbox"])) / 256.0
            best = min(best, min(1.0, dist))
    return best


def _shape_match(f, c, d):
    a = f["comps"].get(c, []); b = f["comps"].get(d, [])
    return bool(a and b and ({shape_key(x["cells"]) for x in a} & {shape_key(x["cells"]) for x in b}))


def _asym(f, axis):
    g = f["grid"]; ok = g >= 0
    if not ok.any():
        return 1.0
    fl = np.flip(g, axis=axis)
    bad = int(((g != fl) & ok).sum()); tot = int(ok.sum())
    return bad / max(1, tot)


def _line_residual(f, axis):
    g = f["grid"]; bg = f["bg"]
    lines = g if axis == 0 else g.T
    best = 1.0
    for row in lines:
        v = row[row >= 0]
        if len(v) < 4:
            continue
        vals, counts = np.unique(v, return_counts=True)
        for val, cn in zip(vals, counts):
            if int(val) == bg:
                continue
            best = min(best, (len(v) - int(cn)) / len(v))
    return best


def run_search(root, mask, acts, cands, cnt, max_states, mode="goal"):
    """Один цикл поиска. Режимы: goal -- очередь по остатку; bfs -- очередь в ширину;
    mix -- чередование (чётные раскрытия из очереди по цели, нечётные из очереди в ширину): покрытие не хуже
    слепого перебора при бюджете 2B, а выигранные целью игры добавляются сверху."""
    key_of = Key(mask)
    lvl0 = root.lvl
    seen = {key_of(root.grid): 0}
    heap = []; order = 0; fifo = deque([(root, [], 0)])
    heapq.heappush(heap, (0.0, 0, order, root, []))
    falsified = set(); solved = None; expansions = 0; best_res = 1.0; best_name = None; turn = 0
    while not cnt.exhausted() and len(seen) < max_states:
        if mode == "bfs" or (mode == "mix" and turn % 2 == 1):
            if not fifo:
                if mode == "bfs" or not heap:
                    break
            else:
                snap, path, depth = fifo.popleft()
        if mode == "goal" or (mode == "mix" and turn % 2 == 0):
            if not heap:
                if mode == "goal" or not fifo:
                    break
                snap, path, depth = fifo.popleft()
            else:
                _, depth, _, snap, path = heapq.heappop(heap)
        turn += 1; expansions += 1
        for act in acts:
            if cnt.exhausted():
                break
            nxt = step(snap, act, cnt)
            if nxt is None:
                continue
            if nxt.lvl > lvl0:
                solved = path + [act]; break
            if nxt.state == "GAME_OVER":
                continue
            k = key_of(nxt.grid)
            if k in seen:
                continue
            seen[k] = depth + 1
            res = 1.0
            if cands:
                f = state_features(nxt.grid, mask)
                vals = []
                for nm, fn in cands:
                    if nm in falsified:
                        continue
                    v = fn(f)
                    if v <= 0.0:
                        falsified.add(nm); continue
                    vals.append((v, nm))
                if vals:
                    res, name = min(vals, key=lambda x: x[0])
                    if res < best_res:
                        best_res, best_name = res, name
            order += 1
            if mode in ("goal", "mix"):
                heapq.heappush(heap, (res, depth + 1, order, nxt, path + [act]))
            if mode in ("bfs", "mix"):
                fifo.append((nxt, path + [act], depth + 1))
        if solved:
            break
    return {"solved": solved is not None, "path_len": len(solved) if solved else None, "states": len(seen),
            "expansions": expansions, "moves": cnt.moves, "best_residual": round(best_res, 3), "best_candidate": best_name,
            "candidates": len(cands), "falsified": len(falsified)}


def prepare(arc, gid, cnt, mode, allowed=None):
    """корень, маска часов, алфавит и кандидаты (allowed -- ограничение на типы целей, напр. перенесённые с уровня 1)."""
    root = root_snap(arc, gid, cnt)
    mask = clock_mask(root, cnt)
    acts = simple_actions(root.avail) + active_clicks(root, cnt)
    f0 = state_features(root.grid, mask)
    cands = []
    if mode in ("goal", "mix"):
        cands = [(nm, fn) for nm, fn in candidates(f0) if fn(f0) > 0
                 and (allowed is None or nm.split()[0] in allowed)]
    return root, mask, acts, cands


def solve(arc, gid, max_states, max_moves, timeout, mode="goal"):
    t0 = time.time(); cnt = Cnt(max_moves, t0 + timeout)
    root, mask, acts, cands = prepare(arc, gid, cnt, mode)
    r = run_search(root, mask, acts, cands, cnt, max_states, mode)
    r.update({"game": gid[:4], "seconds": round(time.time() - t0, 1)})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="all"); ap.add_argument("--mode", default="goal", choices=["goal", "bfs", "mix"])
    ap.add_argument("--max-states", type=int, default=3000); ap.add_argument("--max-moves", type=int, default=60000)
    ap.add_argument("--timeout", type=float, default=240.0); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT / "environment_files"))
    ids = [e.game_id for e in arc.get_environments()]
    if a.games != "all":
        want = set(a.games.split(",")); ids = [g for g in ids if g[:4] in want]
    rows = []
    for gid in ids:
        try:
            r = solve(arc, gid, a.max_states, a.max_moves, a.timeout, a.mode)
        except Exception as exc:
            r = {"game": gid[:4], "solved": False, "error": repr(exc)[:200]}
        rows.append(r); print(json.dumps(r, ensure_ascii=False), flush=True)
    print("\nИТОГ (%s): решено %d из %d; ходов на решённую игру медиана %s"
          % (a.mode, sum(1 for r in rows if r["solved"]), len(rows),
             int(np.median([r["moves"] for r in rows if r["solved"]])) if any(r["solved"] for r in rows) else "—"))
    if a.out:
        json.dump(rows, open(ROOT / a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print("записано:", a.out)


if __name__ == "__main__":
    main()

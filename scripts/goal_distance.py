"""Мера близости до цели «подобное к подобному» (17.09, слово владельца: «подобное к подобному и мера близости до цели —
так человек и мыслит»).

Зачем. Измерено (goal_heuristic_quality.py): прежние остатки ступенчатые — «формы совпали» даёт 1, пока не совпали,
и 0, когда совпали; промежуточных значений нет, поэтому поиску не за чем идти (медиана ро верного кандидата −0.02).
Здесь остаток считается как СТОИМОСТЬ СОПОСТАВЛЕНИЯ: каждый объект одного цвета сопоставляется с наиболее похожим
объектом другого (задача о назначениях, scipy.optimize.linear_sum_assignment), стоимость пары =
    несовпадение формы (доля клеток, не совпавших при наложении по центру) + нормированное расстояние между центрами.
Тогда «почти сошлись» < «далеко друг от друга», и у эвристики появляется градиент.

Функции:
  match_cost(objs_a, objs_b)      -- нормированная стоимость лучшего сопоставления двух наборов объектов;
  dist_shapes(f, c, d)            -- C12 плотно: фигуры цвета c подобны фигурам цвета d;
  dist_bbox(f, c, d)              -- C13 плотно: рамки совпали (расстояние по углам, нормировано на размер доски);
  dist_inside(f, c, d)            -- C8 плотно: клетки c вне рамки d, взвешенные расстоянием до рамки;
  dist_merge(f, c)                -- C3 плотно: во сколько «шагов сближения» области цвета c сливаются в одну
                                     (сумма расстояний между ближайшими областями), а не «число областей − 1».
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _norm_cells(cells):
    ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
    y0, x0 = min(ys), min(xs)
    return {(y - y0, x - x0) for y, x in cells}


def _centroid(cells):
    return (sum(c[0] for c in cells) / len(cells), sum(c[1] for c in cells) / len(cells))


def pair_cost(a, b, diag=90.0):
    """стоимость пары объектов: несовпадение формы (0..1) + расстояние центров (0..1)."""
    sa, sb = _norm_cells(a["cells"]), _norm_cells(b["cells"])
    inter = len(sa & sb); union = len(sa | sb)
    shape = 1.0 - inter / union if union else 1.0
    ca, cb = _centroid(a["cells"]), _centroid(b["cells"])
    dist = (abs(ca[0] - cb[0]) + abs(ca[1] - cb[1])) / diag
    return shape + min(1.0, dist)


def match_cost(A, B):
    """нормированная стоимость лучшего сопоставления (0 = наборы совпали, 1 = максимально непохожи)."""
    if not A or not B:
        return 1.0
    M = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            M[i, j] = pair_cost(a, b)
    r, c = linear_sum_assignment(M)
    matched = float(M[r, c].sum()); n = max(len(A), len(B))
    unmatched = n - len(r)          # лишние объекты штрафуются полной стоимостью
    return min(1.0, (matched + unmatched * 2.0) / (2.0 * n))


def dist_shapes(f, c, d):
    return match_cost(f["comps"].get(c, []), f["comps"].get(d, []))


def dist_bbox(f, c, d):
    A, B = f["comps"].get(c, []), f["comps"].get(d, [])
    if not A or not B:
        return 1.0
    M = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            M[i, j] = min(1.0, sum(abs(p - q) for p, q in zip(a["bbox"], b["bbox"])) / 128.0)
    r, cc = linear_sum_assignment(M)
    return float(M[r, cc].mean())


def dist_inside(f, c, d):
    """доля клеток цвета c вне рамки цвета d, взвешенная нормированным расстоянием до ближайшей рамки."""
    A, B = f["comps"].get(c, []), f["comps"].get(d, [])
    if not A or not B:
        return 1.0
    cells = [cell for x in A for cell in x["cells"]]
    best = 1.0
    for bb in B:
        y0, x0, y1, x1 = bb["bbox"]; tot = 0.0
        for (y, x) in cells:
            dy = max(0, y0 - y, y - y1); dx = max(0, x0 - x, x - x1)
            tot += min(1.0, (dy + dx) / 64.0)
        best = min(best, tot / max(1, len(cells)))
    return best


def dist_merge(f, c):
    """во сколько «шагов сближения» области цвета c сливаются в одну: сумма минимальных зазоров между областями."""
    A = f["comps"].get(c, [])
    if len(A) <= 1:
        return 0.0 if A else 1.0
    cents = [_centroid(x["cells"]) for x in A]
    gaps = []
    for i in range(len(A)):
        gaps.append(min(abs(cents[i][0] - cents[j][0]) + abs(cents[i][1] - cents[j][1]) for j in range(len(A)) if j != i))
    return min(1.0, float(np.mean(gaps)) / 64.0 + (len(A) - 1) / (len(A) + 4.0) * 0.25)

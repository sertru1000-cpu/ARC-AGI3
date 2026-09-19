"""Последняя проверка ветки обвязки (критик раунда 10Б): отличимо ли заранее «уместное» новое состояние (19.09).

Вопрос: 72% ходов застрявших игр ведут в НОВЫЕ состояния — агент исследует, но не отличает полезное новое от
бесполезного. Можно ли по признакам перехода в момент t предсказать, что уровень будет взят в следующие N ходов?

Данные: записанные партии трёх прогонов базы (события с доской после каждого хода). Для каждого хода t внутри уровня:
  метка y = 1, если в ходах t+1 .. t+N будет взят уровень;
  признаки ИЗМЕНЕНИЯ (ход t-1 -> t): сколько клеток изменилось; сколько объектов было/стало, появилось, исчезло;
    сколько цветов затронуто; холостой ли ход; возврат ли в виденное состояние; новая ли подпись события на уровне
    и в игре; доля «новых» объектов (цвет+форма, не встречавшиеся на уровне);
  признаки ВРЕМЕНИ (печатаются отдельно): ходов с начала уровня, номер уровня.
Модель: логистическая регрессия (numpy, L2), проверка по группам игр — все прогоны одной игры в одной части
(иначе меряется заучивание игр). Итог — AUC на отложенных играх.

Пороги (заданы критиком до замера): AUC < 0.60 — закрыть ветку обвязки до кода победителей; 0.60-0.70 — не больше
одного простого контроллера; > 0.70 — строить. Решение — по AUC признаков ИЗМЕНЕНИЯ.

usage:  .venv/bin/python scripts/relevance_test.py [--n 10]
"""
import argparse, glob, json, os
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["flash_v1_phaseA", "public_flash_keithtyser", "public_flash_tufa"]


def comps(b):
    """объекты: связные области одного цвета (без фона) как множество (цвет, форма-со-сдвигом-к-нулю)"""
    h, w = b.shape
    vals, cnt = np.unique(b, return_counts=True); bg = int(vals[cnt.argmax()])
    seen = np.zeros_like(b, dtype=bool); out = []
    for y0 in range(h):
        for x0 in range(w):
            c = int(b[y0, x0])
            if c == bg or seen[y0, x0]:
                continue
            st = [(y0, x0)]; seen[y0, x0] = True; cells = []
            while st:
                y, x = st.pop(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and int(b[yy, xx]) == c:
                        seen[yy, xx] = True; st.append((yy, xx))
            ys = [p[0] for p in cells]; xs = [p[1] for p in cells]
            out.append((c, frozenset((y - min(ys), x - min(xs)) for y, x in cells), (min(ys), min(xs))))
    return out


def build(n_ahead):
    X, Xt, y, grp = [], [], [], []
    for run in RUNS:
        for f in sorted(glob.glob(str(ROOT / "runs" / run / "artifacts" / "*_p0_events.jsonl"))):
            g = os.path.basename(f)[:4]
            ev = [json.loads(l) for l in open(f, encoding="utf-8")]
            ev = [e for e in ev if e.get("type") == "action" and isinstance(e.get("board"), list)]
            if len(ev) < 5:
                continue
            boards = [np.asarray(e["board"], dtype=np.int16) for e in ev]
            levels = [int(e.get("level") or 0) for e in ev]
            done = [str(e.get("level_completed")) == "True" for e in ev]
            cs = [comps(b) for b in boards]
            seen_states = set(); lvl_objs = set(); lvl_sigs = set(); game_sigs = set(); t_in_level = 0
            for t in range(1, len(ev)):
                if levels[t] != levels[t - 1]:
                    seen_states = set(); lvl_objs = set(); lvl_sigs = set(); t_in_level = 0
                t_in_level += 1
                b0, b1 = boards[t - 1], boards[t]
                if b0.shape != b1.shape:
                    continue
                ch = b0 != b1
                k0 = {(c, s) for c, s, _ in cs[t - 1]}; k1 = {(c, s) for c, s, _ in cs[t]}
                appeared = len(k1 - k0); vanished = len(k0 - k1)
                moved = len({(c, s, p) for c, s, p in cs[t]} - {(c, s, p) for c, s, p in cs[t - 1]}) - appeared
                novel_objs = len(k1 - lvl_objs) if lvl_objs else 0
                colors = len(set(np.unique(b0[ch]).tolist()) | set(np.unique(b1[ch]).tolist())) if ch.any() else 0
                sig = (appeared > 0, vanished > 0, moved > 0, colors)
                state = (levels[t], b1.tobytes())
                feats = [np.log1p(ch.sum()), len(cs[t]), len(cs[t]) - len(cs[t - 1]), appeared, vanished, max(0, moved),
                         colors, float(not ch.any()), float(state in seen_states),
                         float(sig not in lvl_sigs), float(sig not in game_sigs), novel_objs]
                seen_states.add(state); lvl_objs |= k1; lvl_sigs.add(sig); game_sigs.add(sig)
                if done[t]:
                    continue                      # сам ход взятия уровня не берём: он и есть событие
                # ЛОВУШКА (та же, что в truncate_run): у хода, взявшего уровень, в событии уже НОВЫЙ номер уровня,
                # поэтому ход u засчитывается уровню, на котором он НАЧАЛСЯ (levels[u-1])
                fut = any(done[u] for u in range(t + 1, min(len(ev), t + 1 + n_ahead)) if levels[u - 1] == levels[t])
                X.append(feats); Xt.append([np.log1p(t_in_level), levels[t]]); y.append(int(fut)); grp.append(g)
    return np.asarray(X, float), np.asarray(Xt, float), np.asarray(y), np.asarray(grp)


def auc(score, y):
    order = np.argsort(score); r = np.empty(len(score)); r[order] = np.arange(1, len(score) + 1)
    # средние ранги для ничьих
    s_sorted = score[order]; i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    pos = y == 1; n1 = pos.sum(); n0 = len(y) - n1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def logreg(X, y, l2=1.0, iters=300):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        z = X @ w + b; p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        gw = X.T @ (p - y) / len(y) + l2 * w / len(y); gb = float(np.mean(p - y))
        w -= 0.5 * gw; b -= 0.5 * gb
    return w, b


def cv_auc(X, y, grp):
    pred = np.zeros(len(y))
    for g in np.unique(grp):
        te = grp == g; tr = ~te
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9
        w, b = logreg((X[tr] - mu) / sd, y[tr])
        pred[te] = ((X[te] - mu) / sd) @ w + b
    return auc(pred, y)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=10); a = ap.parse_args()
    X, Xt, y, grp = build(a.n)
    print("ходов %d, из них перед взятием уровня (в следующие %d ходов) %d (%.1f%%), игр %d"
          % (len(y), a.n, y.sum(), 100 * y.mean(), len(np.unique(grp))))
    names = ["log(изменённых клеток)", "объектов", "Δобъектов", "появилось", "исчезло", "сдвинулось", "цветов затронуто",
             "холостой", "возврат", "новая подпись (уровень)", "новая подпись (игра)", "новых объектов на уровне"]
    print("по одному признаку (AUC, без обучения):")
    for i, nm in enumerate(names):
        v = auc(X[:, i], y); print("  %-28s %.3f" % (nm, max(v, 1 - v)))
    a_change = cv_auc(X, y, grp)
    a_time = cv_auc(Xt, y, grp)
    a_all = cv_auc(np.hstack([X, Xt]), y, grp)
    print("\nлогистическая регрессия, проверка на отложенных играх:")
    print("  признаки ИЗМЕНЕНИЯ: AUC %.3f" % a_change)
    print("  признаки ВРЕМЕНИ:   AUC %.3f" % a_time)
    print("  все вместе:         AUC %.3f" % a_all)
    verdict = "ЗАКРЫТЬ ветку обвязки" if a_change < 0.60 else ("не больше одного простого контроллера" if a_change <= 0.70 else "СТРОИТЬ")
    print("ПОРОГ КРИТИКА по признакам изменения: %.3f -> %s" % (a_change, verdict))
    json.dump({"n": a.n, "auc_change": a_change, "auc_time": a_time, "auc_all": a_all, "rows": len(y), "pos": int(y.sum())},
              open(ROOT / ("runs/relevance_test_n%d_19_09.json" % a.n), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

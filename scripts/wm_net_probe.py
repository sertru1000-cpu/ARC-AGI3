"""Сеть вместо программы: обобщается ли выученная модель переходов лучше написанной.

ВОПРОС. Вся измеренная цена ветки модели мира — в том, что программу пишет языковая модель:
ответ удлиняется с 1438 до 1974 токенов и срезает 27% ходов у ВСЕХ игр сразу. Сеть эту цену
обнуляет. Остаётся один вопрос — качество.

С ЧЕМ СРАВНИВАЕМ (обе цифры измерены нами раньше, тот же протокол «первые 70% на обучение,
последние 30% на проверку»):
  * таблица правил без языковой модели (08.09) — 24% новых переходов;
  * программы, написанные моделью, на ЧУЖОЙ траектории (09.09) — 56%.

ПОРОГИ, ЗАПИСАННЫЕ ДО ЗАМЕРА: выше 56% и медианный горизонт от 8 — идея живая; между 24% и
56% — сеть не лучше того, что модель уже умеет; ниже 24% — хуже нашей же таблицы, закрываем.
Третья цифра, инженерная: секунд на обучение одной игры. Больше 20 — в окно ожидания
(122 с в очереди на ход) не влезет и экономия пропадёт.

ЧТО ЗА СЕТЬ. Маленькая свёрточная: вход — доска 64x64 в one-hot по цветам плюс каналы
действия (пять кнопок одним числом на всю доску, для MOUSE — отметка в точке щелчка),
выход — цвет каждой клетки. Свёртки выбраны намеренно: 91% переходов перерисовывают объекты,
и таблица по типам на этом умирала, а сдвиговая инвариантность свёртки от этого не страдает.

usage:
    .venv/bin/python scripts/wm_net_probe.py --run runs/flash_v1_phaseA
    .venv/bin/python scripts/wm_net_probe.py --run runs/flash_v1_phaseA --game ar25 --epochs 400
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

NCOL = 16          # цветов на доске
NBTN = 6           # UP DOWN LEFT RIGHT SPACE MOUSE
BTNS = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE"]


def transitions(path: str):
    """(доска до, действие, доска после) по журналу одной игры. RESET рвёт цепочку."""
    out, prev = [], None
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        t, b = d.get("type"), d.get("board")
        if t not in ("action", "initial") or b is None:
            continue
        a = d.get("action_display") or ""
        name = a.split("(")[0].strip()
        if name == "RESET" or t == "initial":
            prev = b
            continue
        if prev is not None and name in BTNS:
            m = re.search(r"row=(\d+),\s*col=(\d+)", a)
            rc = (int(m.group(1)), int(m.group(2))) if m else None
            out.append((prev, (name, rc), b))
        prev = b
    return out


def encode(board, act, dev):
    """Доска в one-hot плюс каналы действия."""
    x = torch.tensor(board, dtype=torch.long, device=dev)
    oh = F.one_hot(x.clamp(0, NCOL - 1), NCOL).permute(2, 0, 1).float()
    name, rc = act
    extra = torch.zeros(NBTN + 1, x.shape[0], x.shape[1], device=dev)
    extra[BTNS.index(name)] = 1.0                    # какая кнопка — на всю доску
    if rc is not None:
        r, c = rc
        if 0 <= r < x.shape[0] and 0 <= c < x.shape[1]:
            extra[NBTN, r, c] = 1.0                  # точка щелчка — единственная клетка
    return torch.cat([oh, extra], 0)


class Net(nn.Module):
    """Свёрточная, с пропуском входа: предсказываем ИЗМЕНЕНИЕ, а не доску с нуля."""

    def __init__(self, ch=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(NCOL + NBTN + 1, ch, 5, padding=2), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, NCOL, 1),
        )

    def forward(self, x):
        # пропуск: логиты входного цвета получают фору, сеть учит только правку
        return self.body(x) + 4.0 * x[:, :NCOL]


def run_game(gid, trs, dev, epochs, ch, quiet=False):
    n = len(trs)
    k = max(1, int(n * 0.7))
    train, test = trs[:k], trs[k:]
    if not test:
        return None
    X = torch.stack([encode(b, a, dev) for b, a, _ in train])
    Y = torch.stack([torch.tensor(nb, dtype=torch.long, device=dev).clamp(0, NCOL - 1)
                     for _, _, nb in train])
    net = Net(ch).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(net(X), Y)
        loss.backward()
        opt.step()
    train_s = time.time() - t0

    # базовая линия «ничего не менять»: без неё число «точных» ни о чём не говорит
    id_exact = sum(1 for b, _, nb in test if b == nb) / len(test)
    id_cells = sum(sum(1 for r1, r2 in zip(b, nb) for c1, c2 in zip(r1, r2) if c1 == c2)
                   for b, _, nb in test) / (len(test) * len(test[0][0]) * len(test[0][0][0]))
    net.eval()
    exact = cells_ok = cells_tot = 0
    with torch.no_grad():
        for b, a, nb in test:
            p = net(encode(b, a, dev)[None]).argmax(1)[0]
            t = torch.tensor(nb, dtype=torch.long, device=dev).clamp(0, NCOL - 1)
            exact += int(bool((p == t).all()))
            cells_ok += int((p == t).sum())
            cells_tot += t.numel()
        # горизонт раскатки: ведём состояние ТОЛЬКО сетью по настоящим действиям.
        # Считаем лишь если сеть хоть раз предсказала доску целиком — иначе горизонт
        # заведомо ноль, а петля квадратична по выборке и стоит дороже обучения.
        hor = []
        starts = range(0, min(10, max(1, len(test) - 1))) if exact else []
        for s in starts:
            cur = torch.tensor(test[s][0], dtype=torch.long, device=dev).clamp(0, NCOL - 1)
            steps = 0
            for b, a, nb in test[s:]:
                p = net(encode(cur.tolist(), a, dev)[None]).argmax(1)[0]
                t = torch.tensor(nb, dtype=torch.long, device=dev).clamp(0, NCOL - 1)
                if not bool((p == t).all()):
                    break
                cur = p
                steps += 1
            hor.append(steps)
    with torch.no_grad():
        tp = net(X).argmax(1)
        fit = float((tp == Y).flatten(1).all(1).float().mean())
    res = {"game": gid, "n": n, "train": len(train), "test": len(test),
           "exact": exact / len(test), "cells": cells_ok / cells_tot, "fit": fit,
           "id_exact": id_exact, "id_cells": id_cells,
           "horizon": statistics.median(hor) if hor else 0, "sec": train_s}
    if not quiet:
        print("%-6s пер %3d  выучено %3.0f%%  ТОЧНЫХ %3.0f%% (не менять %3.0f%%)  "
              "клеток %5.1f%% (%5.1f%%)  гор %2.0f  %4.0f с" % (
                  gid, n, 100 * fit, 100 * res["exact"], 100 * id_exact,
                  100 * res["cells"], 100 * id_cells, res["horizon"], train_s))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/flash_v1_phaseA")
    ap.add_argument("--game", help="только одна игра, по четырём буквам")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--min-tr", type=int, default=20, help="игры с меньшим числом переходов пропускаем")
    ap.add_argument("--out", default="docs/wm_net_rows.json")
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print("устройство: %s, эпох %d, каналов %d\n" % (dev, a.epochs, a.ch))
    rows = []
    for f in sorted(glob.glob(os.path.join(a.run, "artifacts", "*_events.jsonl"))):
        gid = os.path.basename(f)[:4]
        if a.game and gid != a.game:
            continue
        trs = transitions(f)
        if len(trs) < a.min_tr:
            continue
        r = run_game(gid, trs, dev, a.epochs, a.ch)
        if r:
            rows.append(r)
    if not rows:
        print("нет игр с достаточным числом переходов")
        return 1
    json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    ex = statistics.median(r["exact"] for r in rows)
    print("\nИТОГ по %d играм" % len(rows))
    print("  выучено обучающих переходов: медиана %.0f%%"
          % (100 * statistics.median(r["fit"] for r in rows)))
    print("  точных предсказаний новых переходов: медиана %.0f%%, среднее %.0f%%"
          % (100 * ex, 100 * sum(r["exact"] for r in rows) / len(rows)))
    print("  то же у линии «ничего не менять»: медиана %.0f%%"
          % (100 * statistics.median(r["id_exact"] for r in rows)))
    print("  игр, где сеть ОБОШЛА эту линию: %d из %d"
          % (sum(1 for r in rows if r["exact"] > r["id_exact"]), len(rows)))
    print("  верных клеток: медиана %.1f%%" % (100 * statistics.median(r["cells"] for r in rows)))
    print("  горизонт раскатки: медиана %.0f" % statistics.median(r["horizon"] for r in rows))
    print("  обучение одной игры: медиана %.1f с, худшая %.1f с"
          % (statistics.median(r["sec"] for r in rows), max(r["sec"] for r in rows)))
    print("\nТОЧКИ СРАВНЕНИЯ: таблица без языковой модели 24%, программы модели на чужой траектории 56%")
    print("ПОРОГ: >56%% и горизонт >=8 — идея живая; 24-56%% — не лучше модели; <24%% — хуже таблицы")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

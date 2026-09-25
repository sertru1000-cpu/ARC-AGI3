"""Сколько ходов вернёт подсказка «ты уже пробовал такое сочетание ходов» (25.09).

Откуда вопрос. 19.09 измерено: у застрявших партий 72% ходов ведут в НОВЫЕ состояния, то есть агент не
ходит по кругу в смысле позиций — повторяются СОЧЕТАНИЯ ХОДОВ (64% троек против 21% у здоровых).
Подсказки по состояниям мы пробовали дважды и закрыли (12.09 «ты здесь уже был», 19.09 смысловая
похожесть). Подсказку по повторяющимся сочетаниям — ни разу.

Что считаем. По настоящим партиям (runs/flash_v1_phaseA/benchmark.json, 25 игр, 4380 ходов):
 1. доля ходов, входящих в ПОВТОРНОЕ появление тройки ходов (первое появление не в счёт — его не угадать);
 2. из них — сколько повторов оказались БЕСПОЛЕЗНЫ, то есть ни один ход тройки ничего не изменил
    (именно их и убрала бы подсказка; полезные повторы трогать нельзя — иногда повтор нужен);
 3. отдельно по застрявшим (0-1 уровень) и здоровым (2+) партиям.

Порог (записан ДО замера, как у прошлых двух веток): строить, если возвращается >= 15% ходов.

usage: .venv/bin/python scripts/repeated_patterns_bound.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT / "scripts"))
import logging  # noqa: E402

logging.disable(logging.ERROR)
import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

K = 3          # длина сочетания: те самые тройки из замера 19.09


def frame_of(fr):
    if fr is None or not fr.frame:
        return None
    return np.asarray(fr.frame[-1], dtype=np.int16)


def key_of(action: dict) -> str:
    name = action.get("id") or "?"
    data = action.get("data")
    if isinstance(data, dict) and "x" in data:
        return "%s(%s,%s)" % (name, data["x"], data["y"])
    return name


def main() -> None:
    bench = json.load(open(ROOT / "runs/flash_v1_phaseA/benchmark.json", encoding="utf-8"))
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL,
                         environments_dir=str(ROOT / "environment_files"))
    rows = []
    tot = rep_all = rep_useless = 0
    for gr in bench["game_runs"]:
        gid = gr["game_id"]
        env = arc.make(gid)
        fr = env.reset()
        g = frame_of(fr)
        if g is None:
            continue
        score = getattr(fr, "score", 0)

        keys, changed = [], []
        for rec in (gr.get("history") or []):
            act = (rec.get("action") or {})
            if not act.get("id"):
                continue
            try:
                fr2 = env.step(GameAction[act["id"]] if act["id"] != "RESET" else GameAction.RESET,
                               data=act.get("data"))
            except Exception:
                break
            g2 = frame_of(fr2)
            sc2 = getattr(fr2, "score", score) if fr2 is not None else score
            ch = (g2 is None) or (g2.shape != g.shape) or bool(np.any(g2 != g)) or sc2 != score
            keys.append(key_of(act)); changed.append(bool(ch))
            if g2 is None:
                break
            g, score = g2, sc2

        n = len(keys)
        seen: set[tuple] = set()
        rep_idx: list[int] = []          # ходы, входящие в ПОВТОРНОЕ появление тройки
        useless_idx: list[int] = []      # из них бесполезные: ни один ход тройки ничего не изменил
        for i in range(n - K + 1):
            trip = tuple(keys[i:i + K])
            if trip in seen:
                rep_idx.extend(range(i, i + K))
                if not any(changed[i:i + K]):
                    useless_idx.extend(range(i, i + K))
            else:
                seen.add(trip)
        r_all, r_use = len(set(rep_idx)), len(set(useless_idx))
        lv = gr.get("levels_completed") or 0
        rows.append((gid[:4], n, r_all, r_use, lv))
        tot += n; rep_all += r_all; rep_useless += r_use

    print("%-6s %6s %12s %14s %8s" % ("игра", "ходов", "в повторах", "бесполезных", "уровней"))
    for r in sorted(rows, key=lambda x: -x[3]):
        print("%-6s %6d %12d %14d %8d" % r)

    stuck = [r for r in rows if r[4] <= 1]
    ok = [r for r in rows if r[4] >= 2]
    print("\nВСЕГО ходов %d" % tot)
    print("  входят в повторяющиеся тройки:           %d (%.1f%%)" % (rep_all, 100 * rep_all / max(tot, 1)))
    print("  из них БЕСПОЛЕЗНЫЕ (ничего не изменили): %d (%.1f%%)" % (rep_useless, 100 * rep_useless / max(tot, 1)))
    for name, grp in (("застрявшие (0-1 уровень)", stuck), ("здоровые (2+)", ok)):
        t = sum(r[1] for r in grp); u = sum(r[3] for r in grp)
        print("  %-26s ходов %5d, бесполезных повторов %4d (%.1f%%)"
              % (name, t, u, 100 * u / max(t, 1)))

    p = rep_useless / max(tot, 1)
    print("\nподсказка «это сочетание ты уже пробовал, и оно ничего не дало» вернула бы %.1f%% ходов" % (100 * p))
    print("в баллах не более (1/(1-p))^2 = %+.1f%% к RHAE" % (100 * ((1 / (1 - p)) ** 2 - 1)))
    print("ПОРОГ 15%% ->", "СТРОИТЬ" if p >= 0.15 else "ЗАКРЫТЬ")


if __name__ == "__main__":
    main()

"""Снятие игр в ФАЗЕ B: имитация расписания на 110 играх по измеренным временным линиям.

ЗАЧЕМ ИМИТАЦИЯ, А НЕ ФОРМУЛА. В бою 110 игр на 28 мест и очередь из 82 ожидающих. Снятая
игра НЕ отдаёт своё время соседям — её место немедленно занимает следующая из очереди,
а суммарный поток ходов задан сервером и от числа игр не зависит. Значит эффект правила
зависит от того, чем занят освободившийся слот, и это надо проигрывать по часам.

ДАННЫЕ. Временные линии взятия уровней из полного базового прогона (25 публичных игр,
132 минуты каждая). Для 110 боевых игр линии берутся с повторением — это допущение, оно
названо: приватные игры тяжелее публичных примерно втрое по баллу.

ЗА ПРЕДЕЛАМИ 132 МИНУТ ДАННЫХ НЕТ. Поэтому считаются две границы: осторожная (после 132-й
минуты игра больше не берёт уровней) и оптимистичная (темп последней трети держится дальше).
Настоящий ответ между ними, и ровно его меряет субботний прогон с потолком 4 часа.

usage:  .venv/bin/python scripts/battle_drop_sim.py --stall 45 --cap 132
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from truncate_run import CAP_HARNESS, replay, score  # noqa: E402

RUN = Path("runs/flash_v1_phaseA")


def lines():
    """По каждой игре базы: моменты взятия уровней и ГОТОВАЯ кривая балла от времени жизни.

    Кривая считается один раз на сетке минут — иначе имитация перечитывает журналы
    сорок тысяч раз и не заканчивается.
    """
    b = json.load(open(RUN / "benchmark.json", encoding="utf-8"))["game_runs"]
    grid = list(range(5, 141, 5))
    out = []
    for r in b:
        hist = r["history"]
        f = glob.glob(str(RUN / "artifacts" / f"{r['game_id']}_p0_events.jsonl"))
        lv = []
        for line in open(f[0], encoding="utf-8"):
            d = json.loads(line)
            if d.get("type") == "action" and d.get("level_completed"):
                i = int(d["action_num"]) - 1
                if i < len(hist):
                    lv.append(float(hist[i].get("wallclock_seconds") or 0.0) / 60.0)
        base = json.loads(str(r["base_actions_per_level"]))
        n = int(r["number_of_levels"])
        curve = []
        for m in grid:
            per, done = replay(RUN, r, m * 60.0)
            curve.append((m, score(per, base, n, done, CAP_HARNESS)))
        out.append((sorted(lv), curve))
    return out


def at(curve, minutes):
    """Балл игры, прожившей столько минут (ступенчато по сетке)."""
    s = 0.0
    for m, v in curve:
        if m <= minutes:
            s = v
        else:
            break
    return s


def life(lv, stall, cap):
    """Сколько минут проживёт игра под правилом снятия."""
    mark = 0.0
    for t in lv:
        if t > cap:
            break
        if t - mark > stall:
            return mark + stall
        mark = t
    return min(cap, mark + stall) if cap - mark > stall else cap


def sim(games, stall, cap, slots, window, optimistic):
    """Проигрываем по часам: места, очередь, снятие. Возвращаем сумму баллов и время."""
    free = [0.0] * slots
    total, played, clock = 0.0, 0, 0.0
    for lv, curve in games:
        i = min(range(slots), key=lambda k: free[k])
        start = free[i]
        if start >= window:
            continue                # игра не успела начаться — честный ноль
        L = min(life(lv, stall, cap), window - start)
        free[i] = start + L
        clock = max(clock, free[i])
        s = at(curve, L)
        if optimistic and L > 132.0:
            rate = (at(curve, 132) - at(curve, 90)) / 42.0   # темп последней трети
            s += rate * (L - 132.0)
        total += s
        played += 1
    return total, played, clock


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stall", type=float, default=45.0, help="минут без нового уровня до снятия")
    ap.add_argument("--cap", type=float, default=132.0, help="потолок на игру, минут")
    ap.add_argument("--games", type=int, default=110)
    ap.add_argument("--slots", type=int, default=28)
    ap.add_argument("--window", type=float, default=530.0)
    ap.add_argument("--trials", type=int, default=40)
    a = ap.parse_args()

    src = lines()
    print("правило: снятие через %.0f мин без уровня, потолок %.0f мин, %d игр на %d мест, окно %.0f мин"
          % (a.stall, a.cap, a.games, a.slots, a.window))
    for label, stall, cap in (("БЕЗ ПРАВИЛА (как сейчас)", 10 ** 9, 132.0),
                              ("правило владельца", a.stall, a.cap),
                              ("правило + потолок 240", a.stall, 240.0),
                              ("правило + потолок 300", a.stall, 300.0)):
        for opt in (False, True):
            res = []
            for t in range(a.trials):
                rnd = random.Random(t)
                games = [src[rnd.randrange(len(src))] for _ in range(a.games)]
                total, played, clock = sim(games, stall, cap, a.slots, a.window, opt)
                res.append((total / a.games, played, clock))
            rh = statistics.mean(x[0] for x in res)
            pl = statistics.mean(x[1] for x in res)
            ck = statistics.mean(x[2] for x in res)
            print("  %-24s %-13s RHAE %5.2f  сыграло %5.1f/%d  прогон %5.0f мин"
                  % (label if not opt else "", "осторожно" if not opt else "оптимистично",
                     rh, pl, a.games, ck))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

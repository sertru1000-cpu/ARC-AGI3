"""Что даст правило снятия в БОЮ: 110 игр на 28 мест, а не 25 на 28.

ЗАЧЕМ. На публичных играх мест больше, чем игр, поэтому снятие ничего не освобождает —
там оно только режет. В бою наоборот: 110 игр, 28 мест, потолок 7920 с на игру, и всего
8 ч 50 мин. Слоты в дефиците, и время, которое держит застрявшая игра, отнимается
у остальных. Этот счёт отвечает: сколько слото-часов освобождает правило снятия и во что
их можно превратить.

КАК СЧИТАЕТСЯ. По полному базовому прогону (`runs/flash_v1_phaseA`, 25 публичных игр,
потолок 7920 с) для каждой игры восстанавливается временная линия взятых уровней:
момент действия берётся из `history[i].wallclock_seconds`, факт взятия — из событийного
журнала. Затем проигрывается правило: игра снимается, если STALL секунд не берёт НОВОГО
уровня (отсчёт от старта игры и от каждого уровня). Считается, сколько времени игра
прожила бы и сколько уровней успела бы взять до снятия.

ЧЕСТНОСТЬ. Публичные игры — не боевые, и доля застревающих там может отличаться. Поэтому
итог даётся не как обещание, а как перенос измеренной доли на 110 игр, с явно названной
ценой: уровни, которые в базе взяты ПОСЛЕ порога, правило потеряет.

usage:  .venv/bin/python scripts/sched_battle_math.py --stall 2700
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path


def timeline(run: Path, r: dict) -> list[float]:
    """Секунды, на которых игра брала уровни."""
    hist = r["history"]
    f = glob.glob(str(run / "artifacts" / f"{r['game_id']}_p0_events.jsonl"))
    if not f:
        return []
    out = []
    for line in open(f[0], encoding="utf-8"):
        d = json.loads(line)
        if d.get("type") != "action" or not d.get("level_completed"):
            continue
        i = int(d["action_num"]) - 1
        if i < len(hist):
            out.append(float(hist[i].get("wallclock_seconds") or 0.0))
    return sorted(out)


def measure(run: Path, stall: float, cap: float):
    """Средняя жизнь игры и потерянные уровни при данном пороге снятия."""
    b = json.load(open(run / "benchmark.json", encoding="utf-8"))
    runs = b["game_runs"] if isinstance(b.get("game_runs"), list) else list(b["game_runs"].values())
    alive_all, kept, total = [], 0, 0
    for r in runs:
        lv = timeline(run, r)
        total += len(lv)
        alive, taken, mark = cap, len(lv), 0.0
        for t in lv:
            if t - mark > stall:
                alive = mark + stall
                taken = sum(1 for x in lv if x <= alive)
                break
            mark = t
        else:
            if cap - mark > stall:
                alive = min(cap, mark + stall)
        alive_all.append(alive)
        kept += taken
    return sum(alive_all) / len(alive_all), kept, total


def sweep(a):
    """Главный вопрос боя: во что превратить освободившееся время — в проходы.

    Балл игры = МАКСИМУМ по попыткам (проверено по скореру, запись 07.09), поэтому
    освободившиеся слото-часы имеет смысл тратить на второй проход по всем 110 играм,
    а не на потолок: потолок сверх 132 минут почти никто не использует, а второй бросок
    измеренно даёт +37% к баллу (три одинаковых прогона, 07.09).
    """
    run = Path(a.run)
    have_min = a.window / 60.0
    print("%-8s %-12s %-14s %-16s %s" % ("порог", "жизнь, мин", "уровней", "1 проход, ч", "2 прохода, ч"))
    for stall in (1500, 1800, 2100, 2400, 2700, 3600):
        life, kept, total = measure(run, float(stall), a.cap)
        wall1 = a.games * life / a.slots / 3600.0
        print("%-8s %-12.0f %-14s %-16.2f %.2f" % (
            "%d мин" % (stall // 60), life / 60.0,
            "%d из %d (-%.0f%%)" % (kept, total, 100 * (total - kept) / total),
            wall1, 2 * wall1))
    life0, _, total0 = measure(run, 10 ** 9, a.cap)
    print("%-8s %-12.0f %-14s %-16.2f %s" % (
        "без", life0 / 60.0, "%d из %d (-0%%)" % (total0, total0),
        a.games * life0 / a.slots / 3600.0, "не влезает"))
    print("\nдоступно окно: %.2f ч" % (a.window / 3600.0))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/flash_v1_phaseA")
    ap.add_argument("--stall", type=float, default=2700.0, help="секунд без нового уровня до снятия")
    ap.add_argument("--cap", type=float, default=7920.0, help="боевой потолок на игру")
    ap.add_argument("--games", type=int, default=110, help="игр в бою")
    ap.add_argument("--slots", type=int, default=28, help="мест одновременно")
    ap.add_argument("--window", type=float, default=31800.0, help="окно боя, с (9 ч минус 10 мин)")
    ap.add_argument("--sweep", action="store_true",
                    help="перебрать пороги снятия и посчитать, сколько кругов влезает")
    a = ap.parse_args()

    if a.sweep:
        return sweep(a)
    b = json.load(open(Path(a.run) / "benchmark.json", encoding="utf-8"))
    runs = b["game_runs"] if isinstance(b.get("game_runs"), list) else list(b["game_runs"].values())

    rows = []
    for r in runs:
        lv = timeline(Path(a.run), r)
        # проигрываем правило: снятие, если STALL секунд без НОВОГО уровня
        alive, taken, mark = a.cap, len(lv), 0.0
        for t in lv:
            if t - mark > a.stall:
                alive = mark + a.stall
                taken = sum(1 for x in lv if x <= alive)
                break
            mark = t
        else:
            if a.cap - mark > a.stall:
                alive = min(a.cap, mark + a.stall)
                taken = len(lv)
        rows.append({"game": str(r["game_id"])[:4], "levels": len(lv), "kept": taken,
                     "alive": alive, "first": lv[0] if lv else None})

    n = len(rows)
    lost = sum(r["levels"] - r["kept"] for r in rows)
    total_lv = sum(r["levels"] for r in rows)
    mean_alive = sum(r["alive"] for r in rows) / n
    saved = a.cap - mean_alive

    print(f"БАЗА: {n} игр, уровней {total_lv}, средняя жизнь при потолке {a.cap:.0f} с")
    print(f"  снятых досрочно: {sum(1 for r in rows if r['alive'] < a.cap)} из {n}")
    print(f"  средняя жизнь при снятии: {mean_alive:.0f} с (экономия {saved:.0f} с на игру)")
    print(f"  ПОТЕРЯНО уровней правилом: {lost} из {total_lv}")
    for r in sorted(rows, key=lambda x: x["alive"]):
        if r["levels"] - r["kept"]:
            print(f"    {r['game']}: уровней {r['levels']}, снята на {r['alive']/60:.0f} мин, "
                  f"успела {r['kept']}")

    print()
    waves = math.ceil(a.games / a.slots)
    print(f"БОЙ БЕЗ ПРАВИЛА: {a.games} игр / {a.slots} мест = {waves} круга(ов) x {a.cap:.0f} с "
          f"= {waves * a.cap / 3600:.2f} ч при доступных {a.window / 3600:.2f} ч")
    need = a.games * a.cap
    have = a.slots * a.window
    print(f"  слото-время: нужно {need/3600:.0f} ч, есть {have/3600:.0f} ч "
          f"(занятость {100*need/have:.0f}%)")

    need2 = a.games * mean_alive
    print(f"БОЙ С ПРАВИЛОМ (снятие через {a.stall/60:.0f} мин без уровня):")
    print(f"  слото-время: нужно {need2/3600:.0f} ч из {have/3600:.0f} ч "
          f"(занятость {100*need2/have:.0f}%), освобождается {(need-need2)/3600:.0f} ч")
    # во что превратить освободившееся: поднять потолок так, чтобы снова упереться в окно
    # доля игр, доживающих до потолка, и доля снятых берутся из базы
    frac_full = sum(1 for r in rows if r["alive"] >= a.cap) / n
    frac_cut = 1 - frac_full
    mean_cut = (sum(r["alive"] for r in rows if r["alive"] < a.cap) / max(1, n - int(frac_full * n))
                if frac_cut else 0.0)
    # ищем потолок C: games * (frac_full*C + frac_cut*mean_cut) = have
    C = (have / a.games - frac_cut * mean_cut) / max(frac_full, 1e-9)
    print(f"  доживают до потолка {100*frac_full:.0f}% игр, снятые живут в среднем {mean_cut/60:.0f} мин")
    print(f"  ПОТОЛОК, который влезает при том же окне: {C:.0f} с = {C/60:.0f} мин "
          f"(сейчас {a.cap/60:.0f} мин, рост x{C/a.cap:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

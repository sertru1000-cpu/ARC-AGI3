"""Снятие игр в бою, посчитанное в ХОДАХ, а не в минутах.

ОШИБКА, КОТОРУЮ ЭТОТ СЧЁТ ИСПРАВЛЯЕТ. Первая имитация (`battle_drop_sim.py`) считала жизнь
игры в минутах и брала балл по кривой «балл от времени». Но время само по себе ничего не даёт:
игра продвигается ХОДАМИ, а суммарный поток ходов задан сервером и делится между живыми
играми. При 28 местах и очереди из 82 снятая игра отдаёт свою долю следующей из очереди —
переноса живым нет. Если же играть все 110 одновременно, очереди нет вовсе, и доля снятой
игры достаётся именно тем, кто ещё берёт уровни. Это разные схемы, и мерить их надо потоком.

ЧТО ИЗМЕРЕНО И ВЗЯТО КАК ДАНО:
  * поток сервера 1973 действия в час, от числа одновременных игр не зависит выше ~4
    (`scripts/concurrency_math.py`, инвариант 900 тыс. токенов/ч по семи прогонам);
  * для каждой игры базы — на каком по счёту ДЕЙСТВИИ взят каждый уровень и сколько
    действий потрачено на уровень (из событийных журналов);
  * балл считается штатной формулой харнесса по действиям на уровень.

ДОПУЩЕНИЕ, НАЗВАННОЕ ЯВНО: продвижение игры зависит только от числа полученных ходов,
а не от того, за какое время они получены. Проверить его нечем — все наши прогоны шли
при одной конкурентности.

usage:  .venv/bin/python scripts/battle_flow_sim.py
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
from truncate_run import CAP_HARNESS  # noqa: E402

RUN = Path("runs/flash_v1_phaseA")
FLOW = 1973.0 / 60.0          # действий в минуту на весь прогон


def games():
    """По каждой игре базы: на каком действии взят каждый уровень, эталон, число уровней."""
    b = json.load(open(RUN / "benchmark.json", encoding="utf-8"))["game_runs"]
    out = []
    for r in b:
        f = glob.glob(str(RUN / "artifacts" / f"{r['game_id']}_p0_events.jsonl"))
        marks, k = [], 0
        for line in open(f[0], encoding="utf-8"):
            d = json.loads(line)
            if d.get("type") != "action":
                continue
            k += 1
            if d.get("level_completed"):
                marks.append(k)
        out.append({"lv": marks, "base": json.loads(str(r["base_actions_per_level"])),
                    "n": int(r["number_of_levels"]), "acts": k})
    return out


def score_at(g, acts):
    """Балл игры, получившей столько действий (уровни засчитываются по мере прохождения)."""
    prev, num, den = 0, 0.0, 0.0
    done = [i for i, m in enumerate(g["lv"]) if m <= acts]
    for i in range(g["n"]):
        w = i + 1
        den += w
        if i < len(done) and i < len(g["base"]):
            spent = g["lv"][i] - prev
            prev = g["lv"][i]
            if spent > 0:
                num += min(CAP_HARNESS, (g["base"][i] / spent) ** 2 * 100) * w
    return num / den if den else 0.0


def sim(pool, conc, window, seed, cap_min=None, stall_min=None, stall_acts=None):
    """Поминутная раздача потока.

    cap_min     — потолок на игру в минутах (нынешняя схема: 132);
    stall_min   — снятие после N минут без нового уровня;
    stall_acts  — снятие после N ХОДОВ без нового уровня. Именно этот вид правила
                  переносится между схемами: при 110 играх сразу каждая получает вчетверо
                  меньше ходов в минуту, и порог в минутах снял бы всех подряд.
    """
    rnd = random.Random(seed)
    gs = [dict(g=pool[rnd.randrange(len(pool))], acts=0.0, last_m=0.0, last_a=0.0,
               start=0.0, done=False) for _ in range(110)]
    waiting = list(range(len(gs)))
    active = []
    for minute in range(int(window)):
        while len(active) < conc and waiting:
            i = waiting.pop(0)
            gs[i].update(start=minute, last_m=minute, last_a=0.0)
            active.append(i)
        if not active:
            break
        share = FLOW / len(active)
        for i in list(active):
            s = gs[i]
            before = len([m for m in s["g"]["lv"] if m <= s["acts"]])
            s["acts"] += share
            after = len([m for m in s["g"]["lv"] if m <= s["acts"]])
            if after > before:
                s["last_m"], s["last_a"] = minute, s["acts"]
            out = False
            if cap_min is not None and minute - s["start"] >= cap_min:
                out = True
            if stall_min is not None and minute - s["last_m"] >= stall_min:
                out = True
            if stall_acts is not None and s["acts"] - s["last_a"] >= stall_acts:
                out = True
            if out:
                s["done"] = True
                active.remove(i)
    total = sum(score_at(s["g"], s["acts"]) for s in gs)
    started = sum(1 for s in gs if s["acts"] > 0)
    return total / len(gs), started, statistics.median(s["acts"] for s in gs)


def sim_waves(pool, wave, window, seed, stall_acts=None, backfill=False, early=True):
    """Волнами БЕЗ подсадки: освободившееся место остаётся волне, а не очереди.

    Именно это отличает схему от нынешней. Сейчас снятую игру немедленно заменяет следующая
    из списка, поэтому её доля потока уходит к тому, кто ещё не начинал. Если не подсаживать,
    доля достаётся тем, кто в волне остался, — то есть тем, кто ещё берёт уровни.

    early=True — если волна вымерла раньше срока, следующая начинается сразу, а не ждёт часы.
    """
    rnd = random.Random(seed)
    gs = [dict(g=pool[rnd.randrange(len(pool))], acts=0.0, last_a=0.0) for _ in range(110)]
    order = list(range(len(gs)))
    n_waves = -(-len(gs) // wave)
    wave_len = window / n_waves
    minute = 0.0
    for w in range(n_waves):
        active = order[w * wave:(w + 1) * wave]
        for i in active:
            gs[i]["last_a"] = 0.0
        end = min(window, (w + 1) * wave_len) if not early else min(window, minute + wave_len)
        while minute < end and active:
            share = FLOW / len(active)
            for i in list(active):
                st = gs[i]
                before = len([m for m in st["g"]["lv"] if m <= st["acts"]])
                st["acts"] += share
                after = len([m for m in st["g"]["lv"] if m <= st["acts"]])
                if after > before:
                    st["last_a"] = st["acts"]
                if stall_acts is not None and st["acts"] - st["last_a"] >= stall_acts:
                    active.remove(i)
                    if backfill and order[(w + 1) * wave:]:
                        pass
            minute += 1
        minute = max(minute, end) if not early else minute
        if minute >= window:
            break
    total = sum(score_at(st["g"], st["acts"]) for st in gs)
    started = sum(1 for st in gs if st["acts"] > 0)
    return total / len(gs), started, statistics.median(st["acts"] for st in gs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=530.0)
    ap.add_argument("--stall", type=float, default=45.0)
    ap.add_argument("--trials", type=int, default=30)
    a = ap.parse_args()
    pool = games()
    print("поток сервера %.1f действия/мин; в базе на игру пришлось %d действий за 132 мин\n"
          % (FLOW, statistics.median(g["acts"] for g in pool)))
    print("%-40s %-9s %-11s %s" % ("схема", "RHAE", "сыграло", "ходов на игру"))
    variants = [
        ("СЕЙЧАС: 28 мест, потолок 132 мин", dict(conc=28, cap_min=132)),
        ("28 мест + снятие 45 мин", dict(conc=28, cap_min=132, stall_min=45)),
        ("28 мест + снятие 60 ходов", dict(conc=28, cap_min=132, stall_acts=60)),
        ("все 110 сразу, без снятия", dict(conc=110)),
        ("все 110 сразу + снятие 60 ходов", dict(conc=110, stall_acts=60)),
        ("все 110 сразу + снятие 80 ходов", dict(conc=110, stall_acts=80)),
        ("все 110 сразу + снятие 100 ходов", dict(conc=110, stall_acts=100)),
        ("55 мест + снятие 80 ходов", dict(conc=55, stall_acts=80)),
    ]
    for name, kw in variants:
        res = [sim(pool, window=a.window, seed=t, **kw) for t in range(a.trials)]
        print("%-40s %-9.2f %-11.0f %.0f" % (
            name, statistics.mean(r[0] for r in res), statistics.mean(r[1] for r in res),
            statistics.mean(r[2] for r in res)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

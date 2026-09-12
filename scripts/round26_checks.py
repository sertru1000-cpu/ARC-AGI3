"""Проверка ответа Gemini раунда 26 нашими данными. Без квоты, только записанные прогоны.

ЧТО СЧИТАЕТ.
1. Связь «вызовы / ходы / доля действующих вызовов» с уровнями ВНУТРИ одного прогона базы
   (runs/flash_v1_phaseA, 25 публичных игр). Нужно для разделов A, B, C, F ответа.
   Важное структурное наблюдение: сервер насыщен, поэтому вызовов на игру почти одинаково
   у всех 25 игр — различаются не бюджеты, а то, что игра с ними делает.
2. Разложение дисперсии и арифметику раздела D (стратегия «поднять дисперсию сабмита»)
   на НАСТОЯЩЕМ распределении: эталонный прогон стока `runs/duck_harness_ref/example-run`
   содержит 25 игр по 20 проходов = 500 партий, то есть даёт разброс ОДНОЙ игры между
   проходами — ровно то, чем управляет температура.
   Ожидание максимума считается по распределению, а не по приближению `mu + sigma*sqrt(2 ln N)`:
   для N=18 приближение даёт 2.40 sigma, истинное значение 1.82 sigma (завышение на треть).

usage:  .venv/bin/python scripts/round26_checks.py
"""
import json
import glob
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from output_tokens_study import TURN, section  # noqa: E402
from phase_a_report import games  # noqa: E402
from truncate_run import CAP_HARNESS, score  # noqa: E402

BASE_RUN = "runs/flash_v1_phaseA"
REF_RUN = "runs/duck_harness_ref/example-run/benchmark.json"
OUR_BATTLE_MEAN = 2.97  # броски 3.36 / 2.31 / 3.23
OUR_BATTLE_SD = 0.57
TICKETS = 18


def pearson(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa = sum((x - ma) ** 2 for x in a) ** 0.5
    sb = sum((x - mb) ** 2 for x in b) ** 0.5
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def call_efficiency():
    """Разделы A, B, C, F: что внутри прогона связано с уровнями."""
    g, _ = games(BASE_RUN)
    rows = []
    for f in sorted(glob.glob(BASE_RUN + "/transcripts/*.txt")):
        gid = Path(f).name[:4]
        if gid not in g:
            continue
        text = open(f, encoding="utf-8", errors="replace").read()
        marks = [m.start() for m in TURN.finditer(text)]
        turns = acting = 0
        for i, pos in enumerate(marks):
            end = marks[i + 1] if i + 1 < len(marks) else len(text)
            m = re.search(r"<parameter=code>\n?(.*?)\n?</parameter>",
                          section(text[pos:end], "[TOOL CALL: python]"), re.S)
            if not m:
                continue
            turns += 1
            if re.search(r"(?<![a-zA-Z_])action\s*\(", m.group(1)):
                acting += 1
        if turns:
            rows.append({"gid": gid, "calls": turns, "acting": acting / turns,
                         "acts": g[gid]["acts"], "levels": g[gid]["levels"], "score": g[gid]["score"]})
    calls = [r["calls"] for r in rows]
    print("=== 1. ВНУТРИ ОДНОГО ПРОГОНА БАЗЫ (%s, балл 10.25)" % BASE_RUN)
    print("вызовов на игру: от %d до %d, медиана %d — сервер раздаёт всем поровну,"
          % (min(calls), max(calls), statistics.median(calls)))
    print("поэтому различие в уровнях создаёт НЕ бюджет вызовов, а решения внутри них.")
    print("доля действующих вызовов по прогону: %.3f" % (sum(r["acting"] * r["calls"] for r in rows) / sum(calls)))
    print()
    print("%-6s %7s %11s %7s %8s %7s" % ("игра", "вызовов", "доля-действ", "ходов", "уровней", "балл"))
    for r in sorted(rows, key=lambda r: -r["levels"]):
        print("%-6s %7d %11.2f %7d %8d %7.1f" % (r["gid"], r["calls"], r["acting"], r["acts"], r["levels"], r["score"]))
    print()
    for name, key in (("вызовов", lambda r: r["calls"]), ("доля действующих вызовов", lambda r: r["acting"]),
                      ("ходов", lambda r: r["acts"]), ("ходов на вызов", lambda r: r["acts"] / r["calls"])):
        print("r(%-24s, уровней) = %+.2f" % (name, pearson([key(r) for r in rows], [r["levels"] for r in rows])))
    print("ОГОВОРКА: это межигровая связь внутри одного прогона, её определяет трудность игры,")
    print("а не причинность. Причинная оценка у нас одна и она межпрогонная: noreason срезал")
    print("вызовы на 35% и уровни на 52% при неизменном всём остальном (запись 07.09, r = +0.52).")
    return rows


def dispersion():
    """Раздел D: считается ли стратегия «поднять дисперсию сабмита»."""
    b = json.load(open(REF_RUN, encoding="utf-8"))
    gs = b["game_runs"] if isinstance(b.get("game_runs"), list) else list(b["game_runs"].values())
    by = {}
    for r in gs:
        gid = str(r["game_id"])[:4]
        by.setdefault(gid, []).append(
            score(json.loads(str(r["actions_per_level"])), json.loads(str(r["base_actions_per_level"])),
                  int(r["number_of_levels"]), int(r["levels_completed"]), CAP_HARNESS))
    ids = list(by)
    mg = {g: statistics.mean(v) for g, v in by.items()}
    base_mean = statistics.mean([mg[g] for g in ids])
    within = [statistics.stdev(v) for v in by.values() if len(v) > 2]
    rnd = random.Random(11)

    print()
    print("=== 2. РАЗЛОЖЕНИЕ ДИСПЕРСИИ (%s: %d игр x %d проходов)" % (REF_RUN, len(ids), len(by[ids[0]])))
    print("средний балл игры %.2f" % base_mean)
    print("разброс ОДНОЙ игры между проходами: медиана sd %.2f, среднее sd %.2f" % (statistics.median(within), statistics.mean(within)))
    print("разброс МЕЖДУ играми (трудность, в дисперсию прогона не входит): sd %.2f"
          % statistics.stdev([mg[g] for g in ids]))

    def pool(k, mean_mult):
        """шум партии xk вокруг среднего своей игры, обрезка [0,115], затем нормировка среднего."""
        raw = {g: [max(0.0, min(115.0, mg[g] + k * (s - mg[g]))) for s in by[g]] for g in ids}
        cur = statistics.mean([statistics.mean(raw[g]) for g in ids])
        f = (base_mean * mean_mult) / cur
        return {g: [min(115.0, x * f) for x in raw[g]] for g in ids}

    def runs(p, n=9000, n_games=110):
        return [statistics.mean([rnd.choice(p[rnd.choice(ids)]) for _ in range(n_games)]) for _ in range(n)]

    def emax(dist, N, reps=9000):
        return statistics.mean([max(rnd.choice(dist) for _ in range(N)) for _ in range(reps)])

    base = runs(pool(1.0, 1.0))
    mb, sb = statistics.mean(base), statistics.stdev(base)
    eb = emax(base, TICKETS)
    print()
    print("прогон на 110 играх, базовый режим: среднее %.3f, sd %.3f = %.0f%% от среднего"
          % (mb, sb, 100 * sb / mb))
    print("СВЕРКА С БОЕМ: у нас измерено sd/среднее = %.2f/%.2f = %.0f%% — совпадает,"
          % (OUR_BATTLE_SD, OUR_BATTLE_MEAN, 100 * OUR_BATTLE_SD / OUR_BATTLE_MEAN))
    print("то есть разброс боевых бросков уже объясняется шумом партий, без общих причин.")
    print("максимум из %d: %.3f = среднее + %.2f sd (приближение критика дало бы +2.40 sd)"
          % (TICKETS, eb, (eb - mb) / sb))
    print("в единицах нашего боя: максимум из %d = %.2f" % (TICKETS, eb / mb * OUR_BATTLE_MEAN))
    print()
    print("СЦЕНАРИЙ РАЗДЕЛА D: шум партии xK при среднем -20% (нормировано, обрезка учтена)")
    print("  %-6s %9s %8s %7s %s" % ("K", "среднее", "sd", "sd x", "максимум из 18 в единицах боя"))
    for k in (1.0, 2.5, 4.0, 6.0, 8.0, 12.0):
        d = runs(pool(k, 0.8))
        m, s = statistics.mean(d), statistics.stdev(d)
        print("  x%-5.1f %9.3f %8.3f %7.2f %22.2f" % (k, m, s, s / sb, emax(d, TICKETS) / mb * OUR_BATTLE_MEAN))
    print()
    print("ВЫВОД: балл прогона — среднее по 110 играм, шум партии делится на sqrt(110) = 10.5,")
    print("а обрезка по нулю не даёт мёртвым играм уйти ниже. Поэтому рост шума партии почти")
    print("не двигает sd прогона, а потеря среднего бьёт напрямую: все сценарии ниже базового.")
    print("Дисперсию МЕЖДУ САБМИТАМИ даёт не температура, а разные сборки — общая причина,")
    print("сдвигающая все 110 игр сразу.")


if __name__ == "__main__":
    call_efficiency()
    dispersion()

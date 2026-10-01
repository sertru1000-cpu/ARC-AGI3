"""Когда брались уровни: кривая «уровней взято к минуте t» по benchmark.json (30.09).
Время уровня = wallclock_seconds хода, которым закрыт уровень (накопленные ходы по actions_per_level).
Зачем: проверить, ускоряет ли сборка взятие уровней или поднимает потолок (бой REAP 4.00 против 4.30 при ×2 на поде за 30 мин).
usage: .venv/bin/python scripts/level_times.py runs/A runs/B ... [--marks 10,20,30,60,90,132]
"""
import json, sys
args = [a for a in sys.argv[1:] if not a.startswith("--")]
marks = [10, 20, 30, 45, 60, 90, 132]
for a in sys.argv[1:]:
    if a.startswith("--marks="): marks = [float(x) for x in a.split("=", 1)[1].split(",")]
for d in args:
    runs = json.load(open(d + "/benchmark.json"))["game_runs"]
    times = []; end = []
    for r in runs:
        h = r.get("history") or []; apl = r.get("actions_per_level") or []
        k = int(r.get("levels_completed") or 0); acc = 0
        for i in range(k):
            acc += apl[i]
            if 0 < acc <= len(h):
                times.append(h[acc - 1].get("wallclock_seconds", 0) / 60)
        end.append((r.get("final_wallclock_seconds") or 0) / 60)
    row = " | ".join("%3.0f мин: %2d" % (m, sum(t <= m for t in times)) for m in marks)
    print("%-28s уровней %2d | %s | игра до %.0f мин (медиана)" % (d.split("/")[-1], len(times), row, sorted(end)[len(end) // 2]))

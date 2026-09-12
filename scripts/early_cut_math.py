"""Стоит ли отсекать неперспективные игры рано: считаем ПРЕДЕЛЬНУЮ отдачу от времени.

ПОЧЕМУ ВОПРОС НЕТРИВИАЛЕН. Суммарное число ходов задано пропускной способностью сервера
и от расписания не зависит (см. `scripts/concurrency_math.py`). Значит отсечение игры —
это не создание ходов, а ПЕРЕДАЧА её ходов другим. Выгодно это тогда и только тогда, когда
у отсечённой игры отдача от следующей минуты ниже, чем у той, кому эти минуты достанутся.

КАК СЧИТАЕТСЯ. По полному базовому прогону (25 публичных игр, 132 минуты каждой) строится
кривая балла игры от времени: обрезка на t минутах даёт взятые к этому моменту уровни и
потраченные на них действия, отсюда балл по формуле харнесса. Затем игры делятся по признаку,
наблюдаемому В ИГРЕ на минуте t: взят ли хотя бы один уровень. Для каждой группы считается,
сколько балла она добирает за оставшееся время — это и есть предельная отдача.

ЧЕГО ЗАМЕР НЕ ДАЁТ. В базе никто не играл дольше 132 минут, поэтому прибавка от ДОБАВЛЕННОГО
сверх этого времени не измерена — оценивается сверху по темпу последней трети.

usage:  .venv/bin/python scripts/early_cut_math.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from truncate_run import CAP_HARNESS, replay, score  # noqa: E402

RUN = Path("runs/flash_v1_phaseA")
MARKS = [10, 20, 30, 45, 60, 90, 132]


def recent(run: Path, r: dict) -> list[float]:
    """Минуты, на которых игра брала уровни."""
    import glob
    hist = r["history"]
    f = glob.glob(str(run / "artifacts" / f"{r['game_id']}_p0_events.jsonl"))
    out = []
    for line in open(f[0], encoding="utf-8"):
        d = json.loads(line)
        if d.get("type") == "action" and d.get("level_completed"):
            i = int(d["action_num"]) - 1
            if i < len(hist):
                out.append(float(hist[i].get("wallclock_seconds") or 0.0) / 60.0)
    return sorted(out)


def hot_cold(games, run: Path) -> None:
    """ПРАВИЛЬНЫЙ признак: не «брала ли уровень когда-нибудь», а «брала ли НЕДАВНО».

    Первый признак обманывает: из 21 игры, взявшей хоть один уровень, в последней трети
    прогона добирали балл только пять. Остальные шестнадцать застряли так же, как те,
    что не взяли ничего. Признак «уровень за последние N минут» разделяет будущую отдачу
    примерно в десять раз — это и есть основание отсекать.
    """
    def sc(r, m):
        b = json.loads(str(r["base_actions_per_level"]))
        n = int(r["number_of_levels"])
        per, done = replay(run, r, m * 60.0)
        return score(per, b, n, done, CAP_HARNESS)

    print("\nПРЕДЕЛЬНАЯ ОТДАЧА ПО ПРИЗНАКУ «БРАЛА УРОВЕНЬ НЕДАВНО»")
    print("%-10s %-8s %-22s %s" % ("отметка", "окно", "живые: игр, балла/мин", "остывшие: игр, балла/мин"))
    for t, win in ((60, 30), (60, 45), (90, 45)):
        hot = [r for r in games if any(t - win <= x <= t for x in recent(run, r))]
        cold = [r for r in games if r not in hot]
        gh = sum(sc(r, 132) - sc(r, t) for r in hot)
        gc = sum(sc(r, 132) - sc(r, t) for r in cold)
        rh = gh / (len(hot) * (132 - t)) if hot else 0.0
        rc = gc / (len(cold) * (132 - t)) if cold else 0.0
        print("%-10s %-8s %-22s %s   (разрыв x%.0f)" % (
            "%d мин" % t, "%d мин" % win, "%d, %.3f" % (len(hot), rh),
            "%d, %.3f" % (len(cold), rc), rh / rc if rc else 99))


def main() -> int:
    games = json.load(open(RUN / "benchmark.json", encoding="utf-8"))["game_runs"]
    curves = {}
    for r in games:
        b = json.loads(str(r["base_actions_per_level"]))
        n = int(r["number_of_levels"])
        gid = str(r["game_id"])[:4]
        pts = {}
        for m in MARKS:
            per, done = replay(RUN, r, m * 60.0)
            pts[m] = (score(per, b, n, done, CAP_HARNESS), done)
        curves[gid] = pts

    print("КРИВАЯ БАЛЛА ПО ВРЕМЕНИ (балл / взятых уровней)")
    print("%-6s %s" % ("игра", "  ".join("%9s" % f"{m} мин" for m in MARKS)))
    for gid in sorted(curves):
        row = "  ".join("%9s" % ("%.1f/%d" % curves[gid][m]) for m in MARKS)
        print("%-6s %s" % (gid, row))

    print("\nСРЕДНИЙ БАЛЛ ПО ВСЕМ 25 ИГРАМ")
    for m in MARKS:
        print("  %3d мин: %.2f" % (m, sum(curves[g][m][0] for g in curves) / len(curves)))

    print("\nПРЕДЕЛЬНАЯ ОТДАЧА: сколько балла игра ДОБИРАЕТ после отметки t")
    print("%-8s %-26s %-14s %-14s %s" % ("отметка", "группа", "игр", "балл на t", "добирает до 132"))
    for m in (10, 20, 30, 45, 60):
        for label, sel in (("уже взяла уровень", lambda g: curves[g][m][1] >= 1),
                           ("ещё ни одного", lambda g: curves[g][m][1] == 0)):
            gs = [g for g in curves if sel(g)]
            if not gs:
                continue
            now = sum(curves[g][m][0] for g in gs) / len(gs)
            end = sum(curves[g][132][0] for g in gs) / len(gs)
            print("%-8s %-26s %-14d %-14.2f +%.2f" % ("%d мин" % m, label, len(gs), now, end - now))
        print()
    hot_cold(games, RUN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

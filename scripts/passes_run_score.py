"""Средний балл ОДНОГО запуска в прогоне с несколькими проходами (пункт 3 субботы).

Итоговый балл харнесса при n_passes > 1 берёт по каждой игре лучший проход и потому завышен.
Здесь каждый запуск считается отдельно по записям `game_runs`, которые TAAF кладёт по схеме
`game_runs[проход * число_игр + игра]` (taaf/benchmark.py), штатной формулой балла с потолком 115.

usage:  .venv/bin/python scripts/passes_run_score.py runs/<прогон>
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from truncate_run import CAP_HARNESS, score  # noqa: E402


def run_scores(run_dir):
    b = json.load(open(Path(run_dir) / "benchmark.json", encoding="utf-8"))
    runs = b["game_runs"] if isinstance(b.get("game_runs"), list) else list(b["game_runs"].values())
    n_passes = int(b.get("n_passes") or 1)
    if len(runs) % n_passes:
        raise SystemExit("записей %d не делится на число проходов %d" % (len(runs), n_passes))
    n_games = len(runs) // n_passes
    out = []
    for p in range(n_passes):
        for g in range(n_games):
            r = runs[p * n_games + g]
            s = score(json.loads(str(r["actions_per_level"])), json.loads(str(r["base_actions_per_level"])),
                      int(r["number_of_levels"]), int(r["levels_completed"]), CAP_HARNESS)
            out.append({"pass": p, "game": str(r["game_id"])[:4], "score": s, "levels": int(r["levels_completed"])})
    return out, n_passes, n_games


def main() -> int:
    rows, n_passes, n_games = run_scores(sys.argv[1])
    for p in range(n_passes):
        ps = [r["score"] for r in rows if r["pass"] == p]
        print("проход %d: игр %d, средний балл %.2f" % (p, len(ps), statistics.mean(ps)))
    allm = statistics.mean(r["score"] for r in rows)
    best = statistics.mean(max(r["score"] for r in rows if r["game"] == g) for g in {r["game"] for r in rows})
    print("СРЕДНИЙ БАЛЛ ОДНОГО ЗАПУСКА: %.2f  (запусков %d)" % (allm, len(rows)))
    print("лучший проход по игре (как итог харнесса, завышен): %.2f" % best)
    print("пороги пункта 3: >= 7.5 — разрыв переноса от приватных игр; <= 5.0 — от расписания")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

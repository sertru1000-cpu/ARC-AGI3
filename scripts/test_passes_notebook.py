"""Проверка сборки пункта 3 (2 прохода) и подсчёта среднего балла одного запуска."""
import types, json, sys, statistics, tempfile, os
from pathlib import Path
sys.path.insert(0, "scripts")
ok = lambda b, m: print(("ok   " if b else "СБОЙ ") + m)
cell = Path("kernels/notebooks_stockflash_passes/cell15.py").read_text(encoding="utf-8")
for sub, want in ((False, 2), (True, 1)):
    bm = types.SimpleNamespace(n_passes=1, games=list(range(25)), solver=types.SimpleNamespace(concurrency=28, max_runtime_s_per_game=7920.0))
    exec(compile(cell, "c", "exec"), {"TRUE_SUBMISSION": sub, "bm": bm, "print": lambda *a, **k: None})
    ok(bm.n_passes == want and bm.solver.max_runtime_s_per_game == 7920.0 and bm.solver.concurrency == 28,
       "%s: проходов %d, потолок и места стоковые" % ("бой" if sub else "Фаза A", bm.n_passes))
bm = types.SimpleNamespace(n_passes=1, solver=types.SimpleNamespace(concurrency=28, max_runtime_s_per_game=7920.0))
exec(compile(cell, "c", "exec"), {"TRUE_SUBMISSION": False, "bm": bm, "print": lambda *a, **k: None})
ok(bm.n_passes == 2, "игры ещё не назначены — строка печати не падает")
from passes_run_score import run_scores
from truncate_run import CAP_HARNESS, score
rows, np_, ng = run_scores("runs/flash_v1_phaseA")
b = json.load(open("runs/flash_v1_phaseA/benchmark.json", encoding="utf-8"))["game_runs"]
direct = statistics.mean(score(json.loads(str(r["actions_per_level"])), json.loads(str(r["base_actions_per_level"])),
                               int(r["number_of_levels"]), int(r["levels_completed"]), CAP_HARNESS) for r in b)
ok(np_ == 1 and ng == 25 and abs(statistics.mean(r["score"] for r in rows) - direct) < 1e-9,
   "на базе (1 проход): средний запуск %.2f = обычное среднее %.2f" % (statistics.mean(r["score"] for r in rows), direct))
# синтетика: уровень ровно за эталон = 100 баллов (115 только быстрее эталона)
fake = {"n_passes": 2, "game_runs": [{"game_id": "g%02d" % g, "actions_per_level": "[10]", "base_actions_per_level": "[10]",
         "number_of_levels": 1, "levels_completed": 1 if p == 0 else 0} for p in range(2) for g in range(3)]}
d = tempfile.mkdtemp(); json.dump(fake, open(os.path.join(d, "benchmark.json"), "w"))
rows, np_, ng = run_scores(d)
ok(np_ == 2 and ng == 3 and [r["pass"] for r in rows] == [0, 0, 0, 1, 1, 1] and statistics.mean(r["score"] for r in rows) == 50.0,
   "синтетика 2 прохода x 3 игры: проход 0 — 100, проход 1 — 0, средний запуск 50 (итог «лучший проход» дал бы 100)")

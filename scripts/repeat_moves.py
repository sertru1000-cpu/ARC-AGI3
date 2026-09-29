"""Повторные ходы (29.09): доля ходов, которые повторяют уже сделанную пару «та же доска + тот же ход» на уровне,
и сколько из повторов снова ничего не изменили. По записанной истории на локальном движке.
usage: .venv/bin/python scripts/repeat_moves.py runs/pod_kv8 runs/pod_kv6c_1h ...
"""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule_learner_probe import game_transitions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for run in sys.argv[1:]:
    c = Counter()
    for gr in json.loads((ROOT / run / "benchmark.json").read_text())["game_runs"]:
        seen = Counter()
        for g0, a, data, g1 in game_transitions(gr["game_id"], gr.get("history") or [], str(ROOT / "environment_files")):
            key = (g0.tobytes(), a, json.dumps(data, sort_keys=True))
            c["moves"] += 1
            same = bool((g0 == g1).all())
            c["noop"] += same
            if seen[key]:
                c["repeat"] += 1; c["repeat_noop"] += same
            seen[key] += 1
    m = c["moves"] or 1
    print("%-28s ходов %4d | повторов пары «доска+ход» %4d (%.1f%%), из них снова без изменений %d | пустых ходов всего %.0f%%" % (
        run.split("/")[-1], c["moves"], c["repeat"], 100 * c["repeat"] / m, c["repeat_noop"], 100 * c["noop"] / m))

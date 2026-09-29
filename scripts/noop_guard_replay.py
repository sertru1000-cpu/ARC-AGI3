"""Сколько раз запрет пустого хода МОГ сработать — по записанной истории прогона на локальном движке (27.09).

Отказанный ход в историю не попадает, поэтому отказы напрямую не видны. Считаем обратное: ход, который в ТОМ ЖЕ
состоянии доски (в пределах уровня) уже N>=3 раз ничего не менял и всё же ушёл в движок. У сборки с работающим
запретом таких должно быть ноль; у контроля это число — сколько отказов запрет выдал бы.
Подпись доски — весь последний кадр (у обвязки — без полосы индикатора, так что здесь оценка сверху).
usage: .venv/bin/python scripts/noop_guard_replay.py runs/pod_a0_control runs/pod_v4_a ...
"""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule_learner_probe import game_transitions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for run in sys.argv[1:]:
    tot = Counter()
    for gr in json.loads((ROOT / run / "benchmark.json").read_text())["game_runs"]:
        tr = game_transitions(gr["game_id"], gr.get("history") or [], str(ROOT / "environment_files"))
        seen = Counter()
        for g0, a, data, g1 in tr:
            key = (g0.tobytes(), a, json.dumps(data, sort_keys=True))
            tot["moves"] += 1
            if seen[key] >= 3:
                tot["would_block"] += 1
            if (g0 == g1).all():
                seen[key] += 1; tot["noop"] += 1
            else:
                seen.pop(key, None)
    m = tot["moves"] or 1
    print("%-28s ходов %4d | пустых %4d (%2.0f%%) | повтор пустого 4-й раз и дальше: %3d (%.1f%% ходов)" % (
        run.split("/")[-1], tot["moves"], tot["noop"], 100 * tot["noop"] / m, tot["would_block"], 100 * tot["would_block"] / m))

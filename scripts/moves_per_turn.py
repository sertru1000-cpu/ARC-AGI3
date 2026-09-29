"""Сколько вызовов модели уходит на один ход игры: разбор по транскриптам (27.09).
Ход модели = блок `--- analysis_step` (один или несколько запросов к модели). Блок кончается либо ходом
(ANALYZER STATUS step_executed: True; число ходов — «The code executed N» в начале СЛЕДУЮЩЕГО блока), либо
прерыванием по времени (yield, turn_time_budget 60 с) — тогда ходов 0, а следующий блок ПОВТОРЯЕТ старую строку
«executed N» (из-за этого первая версия скрипта считала ходы вдвое). Сверка: сумма ходов против benchmark.json.
usage: .venv/bin/python scripts/moves_per_turn.py runs/night_nextfork-b1 [...]
"""
import json, re, sys
from collections import Counter
from pathlib import Path
for run in sys.argv[1:]:
    bench = {r["game_id"][:4]: len([h for h in (r.get("history") or []) if (h.get("action") or {}).get("id")])
             for r in json.loads(Path(run, "benchmark.json").read_text())["game_runs"]}
    blocks_all = 0; reqs = 0; executed = []; yielded = 0; total = 0; bench_total = 0
    for f in sorted(Path(run, "transcripts").glob("*.txt")):
        blocks = re.split(r"\n--- analysis_step=", f.read_text(errors="ignore"))[1:]
        game_moves = 0
        for i, b in enumerate(blocks):
            blocks_all += 1
            reqs += len(re.findall(r"^\[TOOL CALL: python\]", b, re.M))
            ex = re.search(r"step_executed: (True|False)", b)
            if ex and ex.group(1) == "True":
                m = re.search(r"The code executed (\d+) action", blocks[i + 1]) if i + 1 < len(blocks) else None
                if m:
                    n = int(m.group(1)); executed.append(n); game_moves += n
            elif "Yielded control" in b:
                yielded += 1
        total += game_moves; bench_total += bench.get(f.name[:4], 0)
    one = sum(1 for n in executed if n == 1)
    dist = Counter(min(n, 10) for n in executed)
    print("%s: блоков %d | запросов к модели %d | блоков с ходом %d, прерваны по времени %d (%.0f%%)" % (
        run, blocks_all, reqs, len(executed), yielded, 100 * yielded / max(blocks_all, 1)))
    print("   ходов игры %d (benchmark %d) | %.2f хода на блок с ходом, %.2f на запрос | блок с ходом = ровно 1 ход: %.0f%% | %s" % (
        total, bench_total, total / max(len(executed), 1), total / max(reqs, 1), 100 * one / max(len(executed), 1), dict(sorted(dist.items()))))

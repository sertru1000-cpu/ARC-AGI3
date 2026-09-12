"""Влияет ли конкурентность на число ходов: счёт по прогонам, а не по рассуждению.

ВОПРОС. Если играть меньшим числом игр одновременно, достанется ли каждой больше ходов —
и вырастет ли их СУММА? Это два разных вопроса, и ответы у них разные.

КАК СЧИТАЕТСЯ. Для каждого прогона берутся: число действий (из событийных журналов),
длительность окна (из `benchmark.json`), а из выгрузки счётчиков сервера
(`vllm-metrics-final.prom`) — успешные запросы, сгенерированные токены и время ожидания
в очереди. Отсюда — ходов в час на весь прогон и на игру, и загрузка сервера.

ЗАЧЕМ ИМЕННО ТАК. Наш агент не считает ходы сам: каждый ход — это один запрос к серверу,
поэтому потолок ходов задаёт пропускная способность сервера, а конкурентность решает
только, между сколькими играми она делится. Эти числа показывают, где проходит граница.

usage:  .venv/bin/python scripts/concurrency_math.py
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

RUNS = ["runs/flash_v1_phaseA", "runs/flash_exploit_v1_phaseA", "runs/flash_noreason_phaseA",
        "runs/flash_brake_phaseA", "runs/flash_avo_phaseA", "runs/flash_vote_phaseA",
        "runs/flash_skeptic_v1_phaseA"]


def prom(path: Path) -> dict:
    """Счётчики сервера: запросы, токены, суммарное ожидание в очереди."""
    out = {}
    if not path.exists():
        return out
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("#"):
            continue
        m = re.match(r"([a-z0-9_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)", line)
        if not m:
            continue
        name, val = m.group(1), float(m.group(3))
        if name.endswith(("_sum", "_total", "_count")):
            out[name] = out.get(name, 0.0) + val
    return out


def actions_of(run: Path) -> int:
    n = 0
    for f in glob.glob(str(run / "artifacts" / "*_events.jsonl")):
        for line in open(f, encoding="utf-8"):
            if '"type": "action"' in line or '"type":"action"' in line:
                n += 1
    return n


def main() -> int:
    print("%-26s %-6s %-7s %-9s %-10s %-11s %-9s" % (
        "прогон", "игр", "часов", "действий", "ходов/ч", "ходов/игру", "очередь, с"))
    rows = []
    for r in RUNS:
        run = Path(r)
        bj = run / "benchmark.json"
        if not bj.exists():
            continue
        b = json.load(open(bj, encoding="utf-8"))
        games = b["game_runs"] if isinstance(b.get("game_runs"), list) else list(b["game_runs"].values())
        hours = 0.0
        try:  # время в benchmark.json записано строкой ISO
            from datetime import datetime
            hours = (datetime.fromisoformat(str(b["end_time"])) -
                     datetime.fromisoformat(str(b["start_time"]))).total_seconds() / 3600.0
        except Exception:
            hours = 0.0
        if hours <= 0:
            spans = [max(float(h.get("wallclock_seconds") or 0) for h in g["history"]) if g.get("history") else 0
                     for g in games]
            hours = max(spans) / 3600.0
        acts = actions_of(run)
        p = prom(run / "vllm-metrics-final.prom")
        req = p.get("vllm:request_success_total", 0.0)
        qsum = p.get("vllm:request_queue_time_seconds_sum", 0.0)
        gen = p.get("vllm:generation_tokens_total", 0.0)
        rows.append((run.name, len(games), hours, acts, acts / hours if hours else 0,
                     acts / len(games) if games else 0, qsum / req if req else 0, req, gen))
        print("%-26s %-6d %-7.2f %-9d %-10.0f %-11.1f %-9.1f" % rows[-1][:7])

    print("\nГЛАВНЫЙ ИНВАРИАНТ: токенов в час (пропускная способность сервера)")
    print("%-26s %-12s %-12s %-14s" % ("прогон", "запросов/ч", "ток/запрос", "токенов/ч"))
    tph = []
    for name, g, hours, acts, aph, apg, q, req, gen in rows:
        if req and hours:
            tph.append(gen / hours)
            print("%-26s %-12.0f %-12.0f %-14.0f" % (name, req / hours, gen / req, gen / hours))
    if tph:
        m = sum(tph) / len(tph)
        print("  среднее %.0f токенов/ч, разброс %.0f..%.0f (+-%.1f%%)" % (
            m, min(tph), max(tph), 100 * max(abs(x - m) for x in tph) / m))

    print("\nСКОЛЬКО ЗАПРОСОВ ЖДЁТ, А СКОЛЬКО СЧИТАЕТСЯ (закон Литтла: N = поток x время)")
    print("%-26s %-10s %-10s %-12s %-12s" % ("прогон", "e2e, с", "очередь, с", "в системе", "считается"))
    for r, name in zip(rows, [x[0] for x in rows]):
        run = Path("runs") / name
        p2 = prom(run / "vllm-metrics-final.prom")
        req = p2.get("vllm:request_success_total", 0.0)
        e2e = p2.get("vllm:e2e_request_latency_seconds_sum", 0.0)
        qs = p2.get("vllm:request_queue_time_seconds_sum", 0.0)
        if not req or not e2e:
            continue
        hours = r[2]
        lam = req / (hours * 3600.0)          # запросов в секунду
        e, qq = e2e / req, qs / req
        print("%-26s %-10.1f %-10.1f %-12.1f %-12.1f" % (name, e, qq, lam * e, lam * (e - qq)))

    print("\nсервер: запросы и токены")
    print("%-26s %-10s %-12s %-12s" % ("прогон", "запросов", "ток/запрос", "запросов/ч"))
    for name, g, hours, acts, aph, apg, q, req, gen in rows:
        if req:
            print("%-26s %-10.0f %-12.0f %-12.0f" % (name, req, gen / req, req / hours))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

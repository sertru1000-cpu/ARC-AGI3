"""Сравнение прогонов по механике сервера и игры + уровни (28.09, проверка кэша vLLM).
Для каждого каталога: балл, уровни, 1-й уровень, ходов/игру, запросов, ходов/запрос, прерываний по времени,
и из метрик vLLM (разность after-before, если есть, иначе итоговые): вход/выход токенов, очередь, вытеснения.
usage: .venv/bin/python scripts/serve_compare.py runs/night_nextfork-v4lite runs/pod_kv8 ...
"""
import json, re, sys, glob, subprocess
from pathlib import Path

def metrics(p):
    out = {}
    if not p or not Path(p).exists():
        return out
    for name, val in re.findall(r"^(vllm:[a-z_]+)(?:\{[^}]*\})? ([0-9.e+]+)", Path(p).read_text(), re.M):
        out[name] = out.get(name, 0.0) + float(val)
    return out

for d in sys.argv[1:]:
    d = Path(d)
    b = json.load(open(d / "benchmark.json"))["game_runs"]
    lv = [r.get("levels_completed") or 0 for r in b]
    after = sorted(glob.glob(str(d / "*metrics_after.prom"))) or sorted(glob.glob(str(d / "vllm-metrics-final.prom")))
    before = sorted(glob.glob(str(d / "*metrics_before.prom")))
    ma = metrics(after[0] if after else None); mb = metrics(before[0] if before else None)
    m = {k: ma.get(k, 0) - mb.get(k, 0) for k in ma}
    req = m.get("vllm:request_success_total", 0) or 1
    yl = sum(open(f, errors="ignore").read().count("Yielded control") for f in glob.glob(str(d / "transcripts/*.txt")))
    moves = sum(len(r.get("history") or []) for r in b)
    print("%-28s балл %.2f | уровней %2d | 1-й ур %2d/25 | 2-й %d | ходов/игру %3.0f | запросов %4.0f | ходов/запрос %.2f | прерван %d (%.0f%%) | вход/запр %5.0f | выход/запр %4.0f | очередь %3.0f с | вытесн %3.0f" % (
        d.name, sum(r.get("final_score") or 0 for r in b) / len(b), sum(lv), sum(x >= 1 for x in lv), sum(x >= 2 for x in lv),
        moves / len(b), req, moves / req, yl, 100 * yl / req, m.get("vllm:prompt_tokens_total", 0) / req,
        m.get("vllm:generation_tokens_total", 0) / req, m.get("vllm:request_queue_time_seconds_sum", 0) / req,
        m.get("vllm:num_preemptions_total", 0)))

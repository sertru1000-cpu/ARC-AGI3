"""Единый отчёт по прогону Фазы A против полной базы: балл, ходы, токены, петли, код.

Сравнение с `runs/flash_v1_phaseA` (25 публичных игр, 2.2 ч, один проход) — парным тестом по играм,
а не по среднему. Считает то, по чему записаны пороги субботних прогонов: генерацию на ответ,
действия на игру, повтор того же хода после возврата в виденное состояние, токены кода на ход
и долю повторных строк кода.

usage:  .venv/bin/python scripts/phase_a_report.py runs/<прогон> [ещё прогоны...]
"""
import glob
import hashlib
import json
import re
import statistics
import sys
from datetime import datetime
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from concurrency_math import prom  # noqa: E402
from output_tokens_study import TURN, section  # noqa: E402
from truncate_run import CAP_HARNESS, score  # noqa: E402

BASE = Path("runs/flash_v1_phaseA")
CH = 2.87


def games(run):
    b = json.load(open(Path(run) / "benchmark.json", encoding="utf-8"))
    gs = b["game_runs"] if isinstance(b.get("game_runs"), list) else list(b["game_runs"].values())
    hours = (datetime.fromisoformat(str(b["end_time"])) - datetime.fromisoformat(str(b["start_time"]))).total_seconds() / 3600
    out = {}
    for r in gs:
        gid = str(r["game_id"])[:4]
        per = json.loads(str(r["actions_per_level"]))
        out[gid] = {"score": score(per, json.loads(str(r["base_actions_per_level"])), int(r["number_of_levels"]),
                                  int(r["levels_completed"]), CAP_HARNESS),
                    "acts": sum(per), "levels": int(r["levels_completed"])}
    return out, hours


def revisit_same(run, gid):
    f = glob.glob(str(Path(run) / "artifacts" / (gid + "*_events.jsonl")))
    if not f:
        return 0, 0
    acts = [json.loads(l) for l in open(f[0], encoding="utf-8")]
    acts = [d for d in acts if d.get("type") == "action"]
    names = [d.get("action_display") or "" for d in acts]
    seen, back, same = {}, 0, 0
    for i, d in enumerate(acts):
        key = (d.get("level"), hashlib.md5(json.dumps(d.get("board")).encode()).hexdigest())
        if key in seen:
            back += 1
            j = seen[key]
            if i + 1 < len(acts) and j + 1 < len(acts) and names[i + 1] == names[j + 1]:
                same += 1
        else:
            seen[key] = i
    return back, same


def code_stats(run):
    tot_code = tot_rep = turns = 0
    for f in sorted(glob.glob(str(Path(run) / "transcripts" / "*.txt"))):
        text = open(f, encoding="utf-8", errors="replace").read()
        marks = [m.start() for m in TURN.finditer(text)]
        seen = set()
        for i, pos in enumerate(marks):
            end = marks[i + 1] if i + 1 < len(marks) else len(text)
            m = re.search(r"<parameter=code>\n?(.*?)\n?</parameter>", section(text[pos:end], "[TOOL CALL: python]"), re.S)
            if not m:
                continue
            turns += 1
            src = m.group(1)
            tot_code += len(src)
            for line in src.splitlines():
                s = line.strip()
                if len(s) < 8:
                    continue
                if s in seen:
                    tot_rep += len(line) + 1
                seen.add(s)
    if not turns:
        return 0, 0, 0
    return tot_code / turns / CH, tot_rep / turns / CH, turns


def report(run):
    ours, hours = games(run)
    base, bhours = games(BASE)
    common = [g for g in ours if g in base]
    w = sum(ours[g]["score"] > base[g]["score"] + 1e-9 for g in common)
    l = sum(ours[g]["score"] < base[g]["score"] - 1e-9 for g in common)
    n = w + l
    p = min(1.0, sum(comb(n, i) for i in range(min(w, l) + 1)) * 2 / 2 ** n) if n else 1.0
    pm = prom(Path(run) / "vllm-metrics-final.prom")
    req = pm.get("vllm:request_success_total", 0) or 1
    code, rep, turns = code_stats(run)
    bk = sm = 0
    for g in ours:
        a, b = revisit_same(run, g)
        bk += a
        sm += b
    print("=== %s (%.2f ч) ===" % (Path(run).name, hours))
    print("  RHAE %.2f против базы %.2f | уровней %d (%d) | первый уровень %d/%d (%d)" % (
        statistics.mean(x["score"] for x in ours.values()), statistics.mean(base[g]["score"] for g in common),
        sum(x["levels"] for x in ours.values()), sum(base[g]["levels"] for g in common),
        sum(1 for x in ours.values() if x["levels"] >= 1), len(ours), sum(1 for g in common if base[g]["levels"] >= 1)))
    print("  парный тест: побед %d, поражений %d, ничьих %d -> p = %.3f" % (w, l, len(common) - n, p))
    print("  действий на игру %.1f (база 175.2) | всего действий %d" % (
        statistics.mean(x["acts"] for x in ours.values()), sum(x["acts"] for x in ours.values())))
    print("  генерация %.0f ток/ответ (база 1438) | промпт %.0f (20250) | запросов %.0f (%.0f/ч) | очередь %.1f с (121.6)" % (
        pm.get("vllm:generation_tokens_total", 0) / req, pm.get("vllm:prompt_tokens_total", 0) / req,
        req, req / hours, pm.get("vllm:request_queue_time_seconds_sum", 0) / req))
    print("  код %.0f ток/ход (база 225), из них повторных строк %.0f (база 83) на %d ходах" % (code, rep, turns))
    print("  возвратов в виденное состояние %d, из них тот же ход %d (%.0f%%; база 29%%)" % (bk, sm, 100 * sm / max(1, bk)))
    print("  проиграли: %s" % sorted(g for g in common if ours[g]["score"] < base[g]["score"] - 1e-9))
    print("  выиграли:  %s" % sorted(g for g in common if ours[g]["score"] > base[g]["score"] + 1e-9))


if __name__ == "__main__":
    for r in sys.argv[1:]:
        report(r)

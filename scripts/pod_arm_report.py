"""Отчёт по варианту, прогнанному на поде (27.09): забирает /root/runs/<вариант> и печатает одну сводку.

Балл — той же формулой, что для Kaggle (truncate_run.py, потолок 1800 с). Сервер — по разнице счётчиков vLLM
до/после варианта (сервер общий для всех вариантов). Трубопровод — audit_plumbing / audit_sandbox.
usage: .venv/bin/python scripts/pod_arm_report.py a0_control [a1_ledger ...]
"""
import json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POD = str(Path.home() / ".ssh/pod.sh")
PY = str(ROOT / ".venv/bin/python")


def prom(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"^(vllm:[a-z_]+)(\{[^}]*\})? ([0-9.eE+-]+)$", line)
        if m and not m.group(1).endswith(("_bucket", "_created")):
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


def report(arm):
    dst = ROOT / "runs" / ("pod_" + arm)
    dst.mkdir(parents=True, exist_ok=True)
    tar = subprocess.run([POD, "cd /root/runs && tar -cz %s %s_metrics_before.prom %s_metrics_after.prom %s_out.ipynb 2>/dev/null" % (arm, arm, arm, arm)],
                         capture_output=True)
    tmp = dst.parent / (".podtar_" + arm + ".tgz"); tmp.write_bytes(tar.stdout)
    subprocess.run(["tar", "-xzf", str(tmp), "-C", str(dst)], capture_output=True)
    tmp.unlink()
    inner = dst / arm
    import shutil
    for p in inner.glob("*"):
        if (dst / p.name).exists():
            shutil.rmtree(dst / p.name) if (dst / p.name).is_dir() else (dst / p.name).unlink()
        p.rename(dst / p.name)
    if inner.exists():
        shutil.rmtree(inner)
    if not (dst / "benchmark.json").exists():
        print("%s: benchmark.json нет — вариант не доигран" % arm); return
    bench = json.loads((dst / "benchmark.json").read_text())["game_runs"]
    # балл = среднее final_score обвязки (потолок уровня 115) по 25 играм; так же считаны 30-мин пробы на Kaggle:
    # Скотт фаза A 3.44, Скотт+v3 фаза A 2.42, F2 1.76 (truncate_run на поде не сверяется — не используем)
    score = sum(r.get("final_score") or 0 for r in bench) / max(len(bench), 1)
    lv = [r.get("levels_completed") or 0 for r in bench]
    levels = sum(lv); l1 = sum(x >= 1 for x in lv); l2 = sum(x >= 2 for x in lv)
    moves = sum(len([h for h in (r.get("history") or []) if (h.get("action") or {}).get("id")]) for r in bench) / max(len(bench), 1)
    a, b = prom((dst / (arm + "_metrics_before.prom")).read_text() if (dst / (arm + "_metrics_before.prom")).exists() else ""), \
           prom((dst / (arm + "_metrics_after.prom")).read_text() if (dst / (arm + "_metrics_after.prom")).exists() else "")
    d = lambda k: b.get(k, 0) - a.get(k, 0)
    req = d("vllm:request_success_total")
    pl = subprocess.run([PY, str(ROOT / "scripts/audit_plumbing.py"), str(dst)], capture_output=True, text=True).stdout
    sb = subprocess.run([PY, str(ROOT / "scripts/audit_sandbox.py"), str(dst)], capture_output=True, text=True).stdout
    lost = (re.search(r"его нет: (\d+)%", pl) or [None, "?"])[1]
    fail = (re.search(r"упали: ([\d.]+)%", sb) or [None, "?"])[1]
    dup = (re.search(r"раньше: (\d+)%", sb) or [None, "?"])[1]
    print("%-12s | балл %.2f | первый ур. %d/25 | второй %d/25 | уровней %d | ходов/игру %.0f | запросов %d | вход/запрос %.0f тыс. | "
          "выход/запрос %.0f | вытеснений %d | очередь %.0f с/запрос | потеря блока модели мира %s%% | упало вызовов %s%% | повтор кода %s%%" % (
              arm, score, l1, l2, levels, moves,
              req, d("vllm:prompt_tokens_total") / max(req, 1) / 1000, d("vllm:generation_tokens_total") / max(req, 1),
              d("vllm:num_preemptions_total"), d("vllm:request_queue_time_seconds_sum") / max(req, 1), lost, fail, dup))


if __name__ == "__main__":
    for arm in sys.argv[1:]:
        report(arm)

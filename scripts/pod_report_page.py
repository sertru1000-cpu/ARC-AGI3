"""Страница отчёта по вариантам пода 27.09: docs/artifacts/pod_2709.html (http://127.0.0.1:8765/page/pod_2709.html).
Берёт runs/pod_<вариант> (скачанные scripts/pod_arm_report.py) и считает: балл (среднее final_score), уровни, запросы,
ходы на запрос, долю ходов модели без хода, вызовы-осмотры, падения, повтор кода.
usage: .venv/bin/python scripts/pod_report_page.py
"""
import html, json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/bin/python")
ARMS = [("a0_control", "контроль №1 (Скотт как есть)"), ("a1_ledger", "контроль №2 (журнал не включился)"),
        ("a0_c2", "контроль №3"), ("a0_c3", "контроль №4"),
        ("a2_dedup", "сжатие повторяющейся инструкции (F11)"), ("a3_noreason", "без старых рассуждений"),
        ("a4_persist", "память функций v1"), ("a4b_persist", "память функций v2 (повтор)"),
        ("a5_helpers", "готовые функции песочницы + difflib/time"), ("a1b_ledger", "журнал эффектов (F7+F2+F5)"),
        ("a6_observe", "наблюдение в сообщении v1"), ("a6b_observe", "наблюдение в сообщении v2"),
        ("a7_zoom", "крупная картинка вокруг игрока ×32"), ("v4_a", "БОЕВОЙ КАНДИДАТ v4 (код: v3 + Скотт + память функций)"),
        ("v4_b", "БОЕВОЙ КАНДИДАТ v4, повтор"), ("a8_buildwm", "достройка симулятора в затянувшихся уровнях"),
        ("a9_persist_observe", "память функций + наблюдение v2"), ("a7_zoom_r2", "крупная картинка ×32, повтор"),
        ("a6b_observe_r2", "наблюдение v2, повтор"), ("v4lite_a", "v4-lite (код: Скотт + память функций, БЕЗ v3)")] + \
       [("v4_" + c, "v4, ночной замер") for c in "cdefghij"] + [("v4lite_" + c, "v4-lite, ночной замер") for c in "bcdefg"] + \
       [("v4_full", "v4 полный, 132 мин на игру (как в бою)"), ("a8b_buildwm_1h", "достройка симулятора v2: порог 8 + заготовка tpl_step, 1 ч")]


def run(script, d):
    return subprocess.run([PY, str(ROOT / "scripts" / script), str(d)], capture_output=True, text=True).stdout


def row(arm):
    d = ROOT / "runs" / ("pod_" + arm)
    if not (d / "benchmark.json").exists():
        return None
    b = json.loads((d / "benchmark.json").read_text())["game_runs"]
    lv = [r.get("levels_completed") or 0 for r in b]
    mt = run("moves_per_turn.py", d); sb = run("audit_sandbox.py", d); ic = run("inspect_calls.py", d)
    g = lambda pat, t: (re.search(pat, t) or [None, "—"])[1]
    return {"score": sum(r.get("final_score") or 0 for r in b) / len(b), "l1": sum(x >= 1 for x in lv), "l2": sum(x >= 2 for x in lv),
            "lev": sum(lv), "req": g(r"запросов к модели (\d+)", mt), "mpr": g(r"([\d.]+) на запрос", mt),
            "idle": g(r"прерваны по времени \d+ \((\d+)%\)", mt), "insp": g(r"вызовов-осмотров (\d+)", ic),
            "fail": g(r"упали: ([\d.]+)%", sb), "dup": g(r"раньше: (\d+)%", sb)}


def main():
    rows = []
    for arm, label in ARMS:
        r = row(arm)
        if r:
            rows.append((arm, label, r))
    ctrl = [r["score"] for a, _, r in rows if a in ("a0_control", "a1_ledger", "a0_c2", "a0_c3")]
    trs = "".join(
        "<tr class='%s'><td><b>%s</b><br><span class='l'>%s</span></td><td>%.2f</td><td>%d</td><td>%d</td><td>%d</td><td>%s</td>"
        "<td>%s</td><td>%s%%</td><td>%s</td><td>%s%%</td><td>%s%%</td></tr>" % (
            "c" if a.startswith(("a0", "a1_")) else ("v" if a.startswith("v4") else ""), html.escape(a), html.escape(l),
            r["score"], r["l1"], r["l2"], r["lev"], r["req"], r["mpr"], r["idle"], r["insp"], r["fail"], r["dup"]) for a, l, r in rows)
    cm = "%.2f (n=%d: %s)" % (sum(ctrl) / len(ctrl), len(ctrl), " · ".join("%.2f" % x for x in ctrl)) if ctrl else "—"
    page = """<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Под 27.09</title>
<style>:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2dfd6;--c:#eef3f8;--v:#f4efe2}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--mut:#9a9a93;--line:#34332f;--c:#1c2530;--v:#2c2618}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px}
table{border-collapse:collapse;width:100%%;font-variant-numeric:tabular-nums}td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right}
td:first-child,th:first-child{text-align:left}tr.c{background:var(--c)}tr.v{background:var(--v)}.l{color:var(--mut);font-size:13px}
h1{font-size:22px}p{color:var(--mut)}</style></head><body>
<h1>Под RTX PRO 6000, 27.09 — варианты поверх обвязки Скотта, 30 мин на игру</h1>
<p>Балл — среднее final_score по 25 играм. Среднее контролей: <b>%s</b>. Одиночный 30-мин прогон шумит сильнее эффектов — судить по механике:
запросы к модели, ходы игры на запрос, доля ходов модели без хода (прервано по времени), вызовы-осмотры, падения, повтор кода.</p>
<table><tr><th>вариант</th><th>балл</th><th>1-й ур.</th><th>2-й ур.</th><th>уровней</th><th>запросов</th><th>ходов/запрос</th>
<th>без хода</th><th>осмотров</th><th>падения</th><th>повтор кода</th></tr>%s</table>
<p>Обновлено: %s</p></body></html>""" % (cm, trs, subprocess.run(["date", "+%d.%m %H:%M"], capture_output=True, text=True, env={"TZ": "Europe/Moscow"}).stdout.strip())
    (ROOT / "docs/artifacts/pod_2709.html").write_text(page, encoding="utf-8")
    print("записано: docs/artifacts/pod_2709.html, вариантов %d" % len(rows))


if __name__ == "__main__":
    main()

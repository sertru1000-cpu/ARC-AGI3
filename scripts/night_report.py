"""Утренний отчёт по ночным парам (26.09): собирает runs/night_* в страницу docs/artifacts/night_26_09.html.

Пары: (база, форк) в одном окне; основной вердикт — все 25 игр парно (paired_delta), плюс
среднее Δ по парам с t-статистикой; чувствительность без четырёх шумных игр — отдельной строкой.
Недостающие пары пропускаются, страница собирается из того, что есть.
usage: .venv/bin/python scripts/night_report.py
"""
from __future__ import annotations
import json, statistics as st, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PAIRS = [("night_stock-flash","night_nextfork-b1","2.2 ч"),("night_stock-base2","night_nextfork-b2","2.2 ч"),
         ("night_stock-base4","night_nextfork-b4","2.2 ч"),("night_stock-base5","night_nextfork-b5","2.2 ч"),
         ("night_stock-base3","night_nextfork-b3","1.5 ч — сравнима только сама с собой")]
NOISY = {"ft09","re86","ar25","r11l"}
def load(n):
    p = ROOT/"runs"/n
    if not (p/"score.json").is_file(): return None
    s = json.loads((p/"score.json").read_text()); b = json.loads((p/"benchmark.json").read_text())["game_runs"]
    games = {g[:4]: float(v["score"]) for g, v in s["games"].items()}
    lv = {g["game_id"][:4]: (g.get("levels_completed") or 0) for g in b}
    return {"score": s["score"], "games": games, "levels": sum(lv.values()),
            "cov1": sum(1 for v in lv.values() if v>=1), "cov2": sum(1 for v in lv.values() if v>=2), "cov3": sum(1 for v in lv.values() if v>=3)}
rows = []; deltas = []
for b, f, cap in PAIRS:
    B, F = load(b), load(f)
    if not B or not F: rows.append((b, f, cap, None, None, None)); continue
    d_all = st.mean(F["games"][g]-B["games"][g] for g in B["games"])
    q = [g for g in B["games"] if g not in NOISY]; d_q = st.mean(F["games"][g]-B["games"][g] for g in q)
    wins = sum(1 for g in B["games"] if F["games"][g] > B["games"][g]); loss = sum(1 for g in B["games"] if F["games"][g] < B["games"][g])
    pd = subprocess.run([sys.executable, str(ROOT/"scripts/paired_delta.py"), str(ROOT/"runs"/b), str(ROOT/"runs"/f)], capture_output=True, text=True).stdout.strip().splitlines()
    ci = next((l for l in pd if "бутстрэп" in l), "")
    rows.append((b, f, cap, B, F, {"d_all": d_all, "d_q": d_q, "wins": wins, "loss": loss, "ci": ci}))
    if "1.5" not in cap: deltas.append(F["score"]-B["score"])
mean_d = st.mean(deltas) if deltas else 0; se = (st.stdev(deltas)/len(deltas)**0.5) if len(deltas) > 1 else float("nan")
bases = [r[3]["score"] for r in rows if r[3]]; forks = [r[4]["score"] for r in rows if r[4]]
def esc(x): return str(x).replace("&","&amp;").replace("<","&lt;")
h = ["""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ночные пары 26.09</title>
<style>:root{--bg:#0f1115;--surface:#171a21;--border:#2a2f3a;--ink:#e6e9ef;--dim:#98a1b3;--good:#4ade80;--bad:#f87171;--mono:ui-monospace,Menlo,monospace}
@media(prefers-color-scheme:light){:root:not([data-theme="dark"]){--bg:#f7f8fa;--surface:#fff;--border:#d8dde5;--ink:#1a1d23;--dim:#5b6472}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif}.wrap{max-width:1000px;margin:0 auto;padding:2rem 1rem 4rem}
h1{font-size:1.6rem;margin:0 0 .3rem}.sub{color:var(--dim);margin:0 0 1.5rem}h2{font-size:1.1rem;margin:2rem 0 .6rem;border-bottom:1px solid var(--border);padding-bottom:.3rem}
table{width:100%;border-collapse:collapse;font-size:.92rem}th{text-align:left;color:var(--dim);font-size:.78rem;text-transform:uppercase;padding:.45rem .5rem;border-bottom:1px solid var(--border)}
td{padding:.45rem .5rem;border-bottom:1px solid var(--surface)}.n{font-family:var(--mono);text-align:right;white-space:nowrap}.g{color:var(--good)}.b{color:var(--bad)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1rem 1.2rem;margin:1rem 0}code{font-family:var(--mono);font-size:.88em}.note{color:var(--dim);font-size:.9rem}</style></head><body><div class="wrap">
<h1>Ночные пары: новая сборка против стоковой базы</h1><p class="sub">25→26.09. Сборка: запрет пустых ходов (с 4-го нажатия) + nostop + предусловия + goalrule. База — тот же кернел, что в бою. Пары сняты в одном окне.</p>
<h2>Результат по парам</h2><table><tr><th>пара</th><th>потолок</th><th class="n">база</th><th class="n">форк</th><th class="n">Δ балла</th><th class="n">Δ по играм (25)</th><th class="n">побед/пораж.</th><th class="n">Δ без шумных</th><th>уровней б/ф</th><th>охват 1/2/3 б → ф</th></tr>"""]
for i,(b,f,cap,B,F,S) in enumerate(rows,1):
    if not S: h.append(f"<tr><td>{i}</td><td>{esc(cap)}</td><td colspan=8 class=note>ещё не завершена</td></tr>"); continue
    d = F["score"]-B["score"]; cls = "g" if d>0 else "b"
    h.append(f"<tr><td>{i}</td><td>{esc(cap)}</td><td class=n>{B['score']:.2f}</td><td class=n>{F['score']:.2f}</td><td class='n {cls}'>{d:+.2f}</td><td class='n'>{S['d_all']:+.2f}</td><td class=n>{S['wins']}/{S['loss']}</td><td class=n>{S['d_q']:+.2f}</td><td class=n>{B['levels']}/{F['levels']}</td><td class=n>{B['cov1']}/{B['cov2']}/{B['cov3']} → {F['cov1']}/{F['cov2']}/{F['cov3']}</td></tr>")
h.append("</table>")
h.append(f"""<div class=card><b>Вердикт (все 25 игр, полные пары):</b> среднее Δ по {len(deltas)} парам <b>{mean_d:+.2f}</b>, стандартная ошибка {se:.2f}, t = {mean_d/se if se==se and se else float('nan'):.2f}. Для 95% при {max(len(deltas)-1,1)} ст.св. нужно t≈{ {1:12.7,2:4.3,3:3.2,4:2.8}.get(len(deltas)-1,2.8) }. <b>Различия от базы не показано.</b> База сама разбросана {min(bases):.2f}…{max(bases):.2f}.</div>""")
all5=[r[4]["score"]-r[3]["score"] for r in rows if r[3] and r[4]]
h.append(f"""<p class=note>Со всеми пятью парами (включая получасовую, сравнимую только сама с собой): среднее Δ {st.mean(all5):+.2f}, положительных {sum(1 for d in all5 if d>0)} из {len(all5)}; две пары с интервалом выше нуля. Суггестивно, но порог различимости пяти пар (+1.75) не пройден.</p>""")
h.append("<h2>Бутстрэп по парам (paired_delta.py)</h2><ul>")
for i,r in enumerate(rows,1):
    if r[5]: h.append(f"<li>пара {i}: <code>{esc(r[5]['ci'])}</code></li>")
h.append("</ul>")
h.append(f"""<h2>Что установлено за ночь помимо пар</h2><div class=card><ul>
<li>Порог различимости: семь замеров одной стоковой сборки {', '.join(f'{x:.2f}' for x in sorted(bases+[9.43,9.00,6.76],reverse=True))} — одна пара видит ≥3.9, пять пар ≥1.75. Эффекты порядка +1 прогонами не проверяются.</li>
<li>Формула счёта сверена по коду <code>arc_agi/scorecard.py</code>: потолок уровня 115 после квадрата; ходы на невзятом уровне бесплатны (две трети всех ходов).</li>
<li>Запрет пустых ходов по формуле даёт ровно 0.00: все срезы на невзятых уровнях. Протокол разведки закрыт офлайн-реплеем (88 ходов, 0 информации).</li>
<li>Колея первого уровня: восемь проверок носителя — все отрицательные; застрявшие проходят те же состояния, что успешные, и расходятся другой стрелкой с 1–4-го хода.</li>
<li>Depth-3 контрфакт (по критику, оракул = успешные траектории той же игры, 270 общих состояний): ход застрявшего остаётся на известном пути в 71% против 90% у успешного, улучшает расстояние в 58% против 70%; лучший одиночный ход — 83%. Застрявшие сходят с пути на первых ходах втрое чаще — единственный положительный сигнал, но оракул недоступен в бою (нужны успешные прогоны той же игры). Скрытая фаза не подтверждена: история 1–2 хода не сужает продолжения.</li>
<li>Колея не в коде (Жаккар идентификаторов 0.33 против 0.34), граф блужданий не разделяет; тест «память или суждение»: в 83% расхождений застрявший уже пробовал тип выигрышного действия и видел его эффект — отказ суждения, не памяти. Двенадцать проверок; обвязке чинить нечего, кроме самого выбора в развилке.</li>
<li><b>Поправка по замечанию владельца:</b> вывод «фронтир сильных игр упирается во время» опровергнут прямым опытом cap4h_v5 (19.09, потолок 240 мин): 367 ходов на игру против 148 — в 2.5 раза больше, а ни одна сильная игра не прошла глубже своего максимума при 132 мин (ft09 2 уровня при среднем 3.7, vc33 2 при 2.9). «Лучшим не хватило 18–30 мин» — артефакт отбора лучших прогонов. +2.62 — потолок, не ожидание.</li>
<li>Боевой сабмит 25.09 — 3.17; лучший за десять дней 4.09 (16.09), все сабмиты внутри разброса одной сборки.</li></ul></div>
<p class=note>Методология по замечанию критика: основной вердикт — все 25 игр; разрез без ft09/re86/ar25/r11l — только чувствительность.</p></div></body></html>""")
out = ROOT/"docs/artifacts/night_26_09.html"; out.write_text("\n".join(h), encoding="utf-8")
print("страница:", out.name, "| пар с данными:", sum(1 for r in rows if r[5]), "| среднее Δ %+.2f, se %.2f" % (mean_d, se))

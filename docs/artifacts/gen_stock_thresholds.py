"""Generate docs/artifacts/stock_thresholds.html — the distribution explainer.

Kept in the repo, not the scratchpad: scratchpad directories get wiped, and a
page source was lost that way once already (see docs/artifacts/README.md).
Re-run after editing, then republish to the SAME artifact url.

    .venv/Scripts/python.exe docs/artifacts/gen_stock_thresholds.py
"""
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent / "stock_thresholds.html"

W, H = 880, 300
PAD_L, PAD_R, PAD_T, PAD_B = 46, 18, 18, 34
RULER_H = 96
X0, X1 = 0.0, 3.0
SIG = 0.45
MU_STOCK = 1.10
MU_OURS, SIG_OURS = 0.90, 0.21
SUBS = [1.19, 0.06, 1.09, 1.14, 0.82, 0.92, 0.65, 0.57, 0.85]


def pdf(x, mu, sig):
    return math.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))


def phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def sx(x):
    return PAD_L + (x - X0) / (X1 - X0) * (W - PAD_L - PAD_R)


YMAX = pdf(0, 0, SIG) * 1.08


def make_sy(height, padb):
    def sy(y, ymax):
        return height - padb - y / ymax * (height - PAD_T - padb)
    return sy


def curve(mu, sig, height, padb, close=False):
    sy = make_sy(height, padb)
    pts = [(sx(X0 + (X1 - X0) * i / 240), sy(pdf(X0 + (X1 - X0) * i / 240, mu, sig), YMAX))
           for i in range(241)]
    d = "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    if close:
        d += f" L {sx(X1):.1f},{sy(0, YMAX):.1f} L {sx(X0):.1f},{sy(0, YMAX):.1f} Z"
    return d


def axis(height, padb):
    out = [f'<line x1="{sx(X0):.1f}" y1="{height-padb:.1f}" x2="{sx(X1):.1f}" '
           f'y2="{height-padb:.1f}" stroke="var(--rule-strong)"/>']
    for t in (0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        out.append(f'<line x1="{sx(t):.1f}" y1="{height-padb:.1f}" x2="{sx(t):.1f}" '
                   f'y2="{height-padb+5:.1f}" stroke="var(--rule-strong)"/>')
        out.append(f'<text x="{sx(t):.1f}" y="{height-padb+19:.1f}" text-anchor="middle" '
                   f'class="tick">{t:.1f}</text>')
    return "".join(out)


# ---------------------------------------------------------------- panel 1
HYPS = [(1.70, "если сток вдвое лучше нас", "var(--h3)"),
        (1.10, "если сток около 1.10", "var(--h2)"),
        (0.85, "если сток равен нам", "var(--h1)")]

p1 = [f'<rect x="{sx(X0):.1f}" y="{PAD_T}" width="{sx(0.60)-sx(X0):.1f}" '
      f'height="{H-PAD_T-PAD_B}" fill="var(--zone-bad)"/>',
      f'<rect x="{sx(1.30):.1f}" y="{PAD_T}" width="{sx(X1)-sx(1.30):.1f}" '
      f'height="{H-PAD_T-PAD_B}" fill="var(--zone-good)"/>']
sy1 = make_sy(H, PAD_B)
for mu, _label, col in HYPS:
    p1.append(f'<path d="{curve(mu, SIG, H, PAD_B, close=True)}" fill="{col}" opacity=".16"/>')
    p1.append(f'<path d="{curve(mu, SIG, H, PAD_B)}" fill="none" stroke="{col}" stroke-width="2"/>')
    p1.append(f'<text x="{sx(mu):.1f}" y="{sy1(pdf(mu,mu,SIG),YMAX)-9:.1f}" text-anchor="middle" '
              f'class="curvelab" fill="{col}">{mu:.2f}</text>')

thresholds = "".join(
    f'<line x1="{sx(v):.1f}" y1="{PAD_T}" x2="{sx(v):.1f}" y2="{H-PAD_B:.1f}" '
    f'stroke="var(--ink-2)" stroke-width="1.5" stroke-dasharray="5 4"/>'
    f'<text x="{sx(v):.1f}" y="{PAD_T-4:.1f}" text-anchor="middle" class="thlab">{lab}</text>'
    for v, lab in ((0.60, "0.60"), (1.30, "1.30")))

rows = []
for mu, label, col in HYPS:
    hi = 1 - phi((1.30 - mu) / SIG)
    lo = phi((0.60 - mu) / SIG)
    rows.append(f'<tr><td><span class="dot" style="background:{col}"></span>{label} '
                f'(медиана {mu:.2f})</td><td class="num">{lo*100:.0f}%</td>'
                f'<td class="num">{(1-hi-lo)*100:.0f}%</td><td class="num">{hi*100:.0f}%</td></tr>')

# ------------------------------------------------- sigma ruler (new block)
def bracket(y, a, b, label, strong=False):
    w = "2" if strong else "1.2"
    return (f'<line x1="{sx(a):.1f}" y1="{y}" x2="{sx(b):.1f}" y2="{y}" '
            f'stroke="var(--ink-2)" stroke-width="{w}"/>'
            f'<line x1="{sx(a):.1f}" y1="{y-6}" x2="{sx(a):.1f}" y2="{y+6}" stroke="var(--ink-2)" stroke-width="{w}"/>'
            f'<line x1="{sx(b):.1f}" y1="{y-6}" x2="{sx(b):.1f}" y2="{y+6}" stroke="var(--ink-2)" stroke-width="{w}"/>'
            f'<text x="{sx((a+b)/2):.1f}" y="{y-11}" text-anchor="middle" class="brlab">{label}</text>')

ruler = [f'<line x1="{sx(MU_STOCK):.1f}" y1="8" x2="{sx(MU_STOCK):.1f}" y2="{RULER_H-22}" '
         f'stroke="var(--h2)" stroke-width="1.5" stroke-dasharray="3 3"/>',
         f'<text x="{sx(MU_STOCK):.1f}" y="{RULER_H-8}" text-anchor="middle" class="brlab" '
         f'fill="var(--h2)">медиана 1.10</text>',
         bracket(34, MU_STOCK - SIG, MU_STOCK + SIG, "±1σ = 0.65…1.55, сюда попадает 68% прогонов", strong=True),
         bracket(70, MU_STOCK - 2*SIG, MU_STOCK + 2*SIG, "±2σ = 0.20…2.00, сюда попадает 95%")]

# ---------------------------------------------------------------- panel 2
H2, PAD_B2 = 230, 34
sy2 = make_sy(H2, PAD_B2)
def overlap_path():
    pts = []
    for i in range(241):
        x = X0 + (X1 - X0) * i / 240
        pts.append((sx(x), sy2(min(pdf(x, 0.85, SIG), pdf(x, MU_STOCK, SIG)), YMAX)))
    return ("M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
            + f" L {sx(X1):.1f},{sy2(0,YMAX):.1f} L {sx(X0):.1f},{sy2(0,YMAX):.1f} Z")

p2 = [f'<path d="{overlap_path()}" fill="var(--ink-3)" opacity=".22"/>',
      f'<path d="{curve(0.85, SIG, H2, PAD_B2)}" fill="none" stroke="var(--h1)" stroke-width="2"/>',
      f'<path d="{curve(MU_STOCK, SIG, H2, PAD_B2)}" fill="none" stroke="var(--h2)" stroke-width="2"/>',
      f'<text x="{sx(0.85):.1f}" y="{sy2(pdf(0.85,0.85,SIG),YMAX)-9:.1f}" text-anchor="middle" '
      f'class="curvelab" fill="var(--h1)">мы, 0.85</text>',
      f'<text x="{sx(MU_STOCK):.1f}" y="{sy2(pdf(MU_STOCK,MU_STOCK,SIG),YMAX)-9:.1f}" '
      f'text-anchor="middle" class="curvelab" fill="var(--h2)">сток, 1.10</text>']
dots = "".join(
    f'<circle cx="{sx(v):.1f}" cy="{H2-PAD_B2-10-(i%3)*11:.1f}" r="5" fill="var(--amber)" '
    f'opacity=".85"><title>сабмит {i+1}: {v:.2f}</title></circle>'
    for i, v in enumerate(SUBS))

d = MU_STOCK - 0.85
overlap = 2 * phi(-d / (2 * SIG))
worse_wins = phi(-d / (SIG * math.sqrt(2)))
real = [v for v in SUBS if v > 0.1]
in1 = sum(1 for v in real if abs(v - MU_OURS) <= SIG_OURS)
in2 = sum(1 for v in real if abs(v - MU_OURS) <= 2 * SIG_OURS)

html = f"""<title>Пороги стока: откуда числа</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@700;800&family=IBM+Plex+Mono:wght@400;500&family=Public+Sans:wght@400;500;600&display=swap">
<style>
:root{{
  --ground:#F2EFE8; --surface:#FBF9F4; --surface-2:#F5F1E9;
  --ink:#232019; --ink-2:#544D42; --ink-3:#7C7466;
  --rule:#DFD8C9; --rule-strong:#C9C0AD;
  --amber:#B4763A;
  --h1:#8FC7C1; --h2:#3E9E96; --h3:#00706A;
  --zone-good:rgba(0,150,140,.09); --zone-bad:rgba(160,59,46,.09);
}}
@media (prefers-color-scheme: dark){{
  :root:not([data-theme="light"]){{
    --ground:#141311; --surface:#1C1B18; --surface-2:#232119;
    --ink:#EFEADF; --ink-2:#B7AF9F; --ink-3:#8B8375;
    --rule:#332F27; --rule-strong:#474137;
    --amber:#C08343;
    --h1:#2E6B66; --h2:#3E9E96; --h3:#5FD3C6;
    --zone-good:rgba(61,191,178,.10); --zone-bad:rgba(210,112,94,.10);
  }}
}}
:root[data-theme="dark"]{{
  --ground:#141311; --surface:#1C1B18; --surface-2:#232119;
  --ink:#EFEADF; --ink-2:#B7AF9F; --ink-3:#8B8375;
  --rule:#332F27; --rule-strong:#474137;
  --amber:#C08343;
  --h1:#2E6B66; --h2:#3E9E96; --h3:#5FD3C6;
  --zone-good:rgba(61,191,178,.10); --zone-bad:rgba(210,112,94,.10);
}}
*{{box-sizing:border-box}}
body{{background:var(--ground); color:var(--ink); margin:0;
  font-family:"Public Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:15px; line-height:1.6; -webkit-font-smoothing:antialiased}}
.wrap{{max-width:960px; margin:0 auto; padding:40px 22px 70px; display:flex; flex-direction:column; gap:30px}}
h1,h2{{margin:0; text-wrap:balance; font-family:"Big Shoulders Display",Impact,"Arial Narrow",sans-serif}}
h1{{font-size:clamp(36px,6vw,58px); font-weight:800; line-height:.95; text-transform:uppercase}}
h2{{font-size:24px; font-weight:700; text-transform:uppercase; letter-spacing:.04em}}
p{{margin:0}}
.eyebrow{{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11.5px; letter-spacing:.15em;
  text-transform:uppercase; color:var(--ink-3)}}
.masthead{{border-bottom:2px solid var(--ink); padding-bottom:20px; display:flex; flex-direction:column; gap:8px}}
.standfirst{{font-size:17px; color:var(--ink-2); max-width:64ch; margin-top:8px}}
.card{{background:var(--surface); border:1px solid var(--rule); border-radius:3px; padding:20px 22px;
  display:flex; flex-direction:column; gap:14px}}
.card p{{font-size:14.5px; color:var(--ink-2); max-width:74ch}}
.plot{{overflow-x:auto}}
svg{{display:block; max-width:100%; height:auto}}
.tick{{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11px; fill:var(--ink-3)}}
.curvelab{{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:12px; font-weight:500}}
.thlab{{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11px; fill:var(--ink-2)}}
.brlab{{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11.5px; fill:var(--ink-2)}}
.axname{{font-size:12px; fill:var(--ink-3)}}
table{{border-collapse:collapse; width:100%; font-size:13.5px}}
th,td{{padding:8px 12px; text-align:left; border-bottom:1px solid var(--rule)}}
thead th{{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:10.5px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--ink-3); border-bottom:1.5px solid var(--rule-strong)}}
td.num{{font-family:"IBM Plex Mono",ui-monospace,monospace; font-variant-numeric:tabular-nums; text-align:right}}
tbody tr:last-child td{{border-bottom:none}}
.dot{{display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:8px; vertical-align:-1px}}
.legend{{display:flex; gap:20px; flex-wrap:wrap; font-size:13px; color:var(--ink-2)}}
.legend span{{display:inline-flex; align-items:center; gap:7px}}
.sw{{width:22px; height:3px; border-radius:2px; display:inline-block}}
.big{{display:flex; gap:26px; flex-wrap:wrap; margin-top:2px}}
.big div{{display:flex; flex-direction:column; gap:2px}}
.big b{{font-family:"Big Shoulders Display",Impact,sans-serif; font-size:38px; font-weight:800;
  line-height:1; color:var(--ink)}}
.big span{{font-size:12.5px; color:var(--ink-3); max-width:26ch}}
footer{{border-top:1px solid var(--rule); padding-top:16px;
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11.5px; color:var(--ink-3)}}
</style>

<div class="wrap">

<header class="masthead">
  <p class="eyebrow">ARC-AGI-3 · почему пороги именно такие</p>
  <h1>Пороги стока: откуда числа</h1>
  <p class="standfirst">Завтрашний сабмит стока даст <b>одно число</b>. Оно не измеряет качество
  стока — оно вытягивает один случайный билет из распределения. Эта страница показывает, какие
  выводы такой билет вообще способен подтвердить, а какие нет.</p>
</header>

<section class="card">
  <h2>Что такое «ширина колокола»</h2>
  <p>Колокол — это нормальное распределение результатов одной и той же сборки. Его ширину задаёт
  <b>сигма (σ)</b> — среднеквадратичное отклонение. У стока она измерена: <b>σ&nbsp;=&nbsp;0.45</b>
  по 20 проходам на 25 публичных играх. Читается она через интервалы вокруг медианы:</p>
  <div class="plot">
    <svg viewBox="0 0 {W} {RULER_H}" role="img" aria-label="Что такое сигма: интервалы вокруг медианы 1.10">
      {"".join(ruler)}
    </svg>
  </div>
  <p>То есть «σ = 0.45» означает: две трети прогонов стока лягут между 0.65 и 1.55, а видимый
  размах колокола на графиках ниже — это примерно ±3σ, почти от нуля до 2.5. Чем шире колокол,
  тем менее осмысленно одно измерение.</p>
</section>

<section class="card">
  <h2>Один сабмит — один бросок из колокола</h2>
  <p>Истинную медиану стока на скрытом наборе мы не знаем — знаем три правдоподобные версии.
  Ниже они как три колокола одинаковой ширины: чем темнее, тем выше предполагаемая медиана.</p>
  <div class="plot">
    <svg viewBox="0 0 {W} {H}" role="img" aria-label="Три гипотезы о медиане стока и два порога решения">
      {"".join(p1)}
      {thresholds}
      {axis(H, PAD_B)}
      <text x="{W-PAD_R}" y="{H-2}" text-anchor="end" class="axname">балл на скрытом наборе</text>
    </svg>
  </div>
  <div class="legend">
    <span><i class="sw" style="background:var(--h3)"></i>сток вдвое лучше нас (1.70)</span>
    <span><i class="sw" style="background:var(--h2)"></i>сток около 1.10</span>
    <span><i class="sw" style="background:var(--h1)"></i>сток равен нам (0.85)</span>
  </div>
  <table>
    <thead><tr><th>если истинная медиана стока…</th><th>≤ 0.60</th><th>0.60 – 1.30</th><th>≥ 1.30</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  <p><b>Как это читать.</b> Порог 1.30 стоит там, где версия «сток вдвое лучше» даёт 81% попаданий,
  а версия «сток равен нам» — всего 16%: впятеро реже. Порог 0.60 работает так же в другую сторону.
  А середина — самая широкая часть всех трёх колоколов сразу, она не отличает ни одну версию
  от других. Поэтому она заранее записана как «неразличимо», а не как «плохо».</p>
</section>

<section class="card">
  <h2>Почему разницу в 0.26 сабмитами не поймать</h2>
  <p>Наша медиана 0.85, оценка стока 1.10. Расстояние между вершинами — <b>0.26, то есть
  {d/SIG:.2f}&nbsp;σ</b>: вершины стоят ближе, чем одна ширина. Серым закрашено перекрытие.</p>
  <div class="plot">
    <svg viewBox="0 0 {W} {H2}" role="img" aria-label="Перекрытие распределений нашей сборки и стока">
      {"".join(p2)}
      {dots}
      {axis(H2, PAD_B2)}
      <text x="{W-PAD_R}" y="{H2-2}" text-anchor="end" class="axname">балл на скрытом наборе</text>
    </svg>
  </div>
  <div class="legend">
    <span><i class="sw" style="background:var(--amber)"></i>наши девять реальных сабмитов</span>
    <span><i class="sw" style="background:var(--ink-3)"></i>перекрытие распределений</span>
  </div>
  <div class="big">
    <div><b>{overlap*100:.0f}%</b><span>площадь перекрытия двух колоколов</span></div>
    <div><b>{worse_wins*100:.0f}%</b><span>вероятность, что один бросок ХУДШЕЙ сборки побьёт один бросок лучшей</span></div>
    <div><b>~50</b><span>сабмитов на каждую сторону, чтобы разницу увидеть надёжно</span></div>
  </div>
  <p>Последнее число и есть приговор: 50 на сторону — это сотня слотов при наших 29. А второе
  означает, что если завтра сток покажет меньше нашего 0.85, то в трети случаев это не значит
  ровно ничего — просто неудачный билет.</p>
</section>

<section class="card">
  <h2>Проверка: описывает ли модель наши настоящие данные</h2>
  <p>Наш собственный колокол: среднее <b>0.90</b>, σ&nbsp;=&nbsp;<b>0.21</b> — вдвое уже стокового.
  Модель предсказывает, что внутри ±1σ (0.69…1.11) должно лежать около 5 сабмитов из 8, внутри
  ±2σ (0.48…1.32) — практически все.</p>
  <p>Факт по нашим восьми отправкам (без аварии 0.06): <b>{in1} внутри одной сигмы, {in2} из 8
  внутри двух.</b> Совпадает с ожиданием — значит расчёты на основе этой модели не выдуманы,
  а описывают реальное поведение наших сабмитов.</p>
</section>

<section class="card">
  <h2>Что отсюда следует практически</h2>
  <p><b>Сабмит стока — не измерение, а проверка на крайности.</b> Он способен подтвердить только
  сильные версии: «форк вредит заметно» или «скрытый набор гораздо тяжелее». Умеренную разницу он
  не различит никогда, сколько бы отправок мы ни сделали.</p>
  <p><b>Середина — ожидаемый исход, а не провал.</b> Она заранее записана как «переходим к
  программе замеров», чтобы завтра не подгонять трактовку под увиденное число. В этом и смысл
  фиксировать пороги до эксперимента.</p>
  <p><b>И поэтому разработку нельзя проверять сабмитами.</b> Любая наша правка даёт эффект порядка
  тех же 0.2–0.7 балла, то есть лежит внутри этого перекрытия. Локальный замер на конверсии даёт
  49 наблюдений за прогон вместо одного и стоит $7, тогда как сабмит стоит сутки и не даёт ничего.</p>
</section>

<footer>
  σ = 0.45 — по 20 проходам стока на 25 публичных играх (runs/duck_harness_ref).
  Оценка медианы стока 1.07–1.15 — двумя путями: обратный счёт от максимума FOYSAL (2.23 за 99
  сабмитов) и наш публичный замер 1.53 с налогом скрытого набора ×0.75.
  Исходник страницы: docs/artifacts/gen_stock_thresholds.py
</footer>

</div>
"""
OUT.write_text(html, encoding="utf-8", newline="\n")
print(f"wrote {OUT} ({len(html)} chars)")

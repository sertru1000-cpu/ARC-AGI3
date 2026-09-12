"""Страница для РУЧНОГО просмотра ответов модели: что мы ей подсунули и что она сделала.

ЗАЧЕМ. Слово владельца: «давай я вручную буду ответы проверять». Поговорить с нашей моделью
по ссылке нельзя — она живёт только внутри кернела Kaggle, без интернета и ровно на время
прогона. Зато её транскрипты приходят в выходных файлах, и вот их удобно читать глазами.

ЧТО ВИДНО В КАЖДОМ ХОДУ:
  * была ли в промпте НАША вставка и какая именно (правило оракула, словарь механик,
    накопленная сводка) — ищется по дословным маркерам этих сборок;
  * рассуждение модели (обрезано, полностью раскрывается);
  * код, который она выполнила;
  * что ответил движок: сколько действий исполнено, какие, и взят ли уровень
    (берётся из промпта СЛЕДУЮЩЕГО хода — там харнесс это и сообщает).

Сверху — таблица по играм: сколько ходов до вставки и после, сколько уровней до и после.
Это и есть тот вопрос, ради которого страница делается: изменила ли выданная подсказка поведение.

usage:
    .venv/bin/python scripts/build_transcript_page.py runs/flash_input_v1 --name transcripts_2a
    .venv/bin/python scripts/build_transcript_page.py runs/<оракул> --name transcripts_oracle --from-action 40
"""
import argparse
import glob
import html
import json
import re
from pathlib import Path

HDR = re.compile(r"^\[[A-Z][A-Z /:_-]+\]\s*$")
STEP = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+)")
CODE = re.compile(r"<parameter=code>\n?(.*?)\n?</parameter>", re.S)

# дословные маркеры наших вставок
# Каждая строка СВЕРЕНА с ячейкой соответствующей сборки, а не написана по памяти:
# первая версия распознавания молчала именно потому, что маркеры были выдуманы.
MARKS = [
    ("правило оракула", "Known mechanic for this game"),          # notebooks_stockflash_oracle
    ("словарь механик", "Cross-game notes"),                      # notebooks_stockflash_xgame
    ("накопленная сводка", "So far in this run"),                 # он же, динамическая часть
    ("разбор перехода", "HARNESS DIFF"),                          # notebooks_stockflash_input, стр. 52
    ("предупреждение о петле", "already been tried and led back"),  # он же
]


def blocks(path: Path):
    """Транскрипт → список ходов; каждый ход это словарь разделов."""
    out, cur, sect, buf = [], None, None, []

    def flush():
        if cur is not None and sect is not None:
            cur.setdefault(sect, []).append("\n".join(buf))

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = STEP.match(line)
        if m:
            flush()
            if cur is not None:
                out.append(cur)
            cur = {"step": int(m.group(1)), "action": int(m.group(2))}
            sect, buf = None, []
            continue
        if HDR.match(line):
            flush()
            sect, buf = line.strip(), []
            continue
        buf.append(line)
    flush()
    if cur is not None:
        out.append(cur)
    return out


def sect(turn, name):
    return "\n".join(turn.get(name, []))


def outcome(nxt):
    """Что харнесс сообщил о ПРЕДЫДУЩЕМ ходе — это и есть ответ движка на наш ход."""
    if nxt is None:
        return "", False
    text = sect(nxt, "[USER PROMPT]")
    line = ""
    m = re.search(r"The code executed (\d+) actions? in the previous sequence\.", text)
    if m:
        line = "исполнено действий: %s" % m.group(1)
    m = re.search(r"Executed actions(?: \(first 10\))?: ([^\n]+)", text)
    if m:
        line += " — " + m.group(1).strip()
    level = "You have progressed to a new level!" in text
    return line, level


def game_rows(run: Path, from_action: int, max_turns: int):
    rows = []
    for f in sorted(glob.glob(str(run / "transcripts" / "*.txt"))):
        gid = Path(f).name[:4]
        turns = blocks(Path(f))
        marked = [i for i, t in enumerate(turns)
                  if any(k in sect(t, "[USER PROMPT]") for _, k in MARKS)]
        first = marked[0] if marked else None
        lv_before = lv_after = 0
        for i, t in enumerate(turns):
            _, lv = outcome(turns[i + 1] if i + 1 < len(turns) else None)
            if lv:
                if first is not None and i >= first:
                    lv_after += 1
                else:
                    lv_before += 1
        # Показываем ходы начиная с первой вставки (там и видно, изменила ли она поведение),
        # но не больше max_turns на игру: в сборках, где вставка идёт КАЖДЫЙ ход, страница
        # иначе распухает до нескольких мегабайт и читать её глазами невозможно.
        start = first if first is not None else 0
        shown = [i for i, t in enumerate(turns)
                 if i >= start and (t["action"] >= from_action or first is not None)]
        rows.append({"gid": gid, "turns": turns, "first": first,
                     "n": len(turns), "lv_before": lv_before, "lv_after": lv_after,
                     "shown": shown[:max_turns]})
    return rows


def render(rows, run_name, from_action):
    e = html.escape
    P = ['<title>Транскрипты: %s</title>' % e(run_name), """
<style>
:root{--bg:#f6f4ee;--surface:#fffdf8;--ink:#1f1d18;--ink-2:#5d5748;--rule:#dcd6c6;--accent:#2b5d8a;--hit:#fff4d6;--hit-rule:#e0c169}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15140f;--surface:#1d1b15;--ink:#ece7da;--ink-2:#aaa392;--rule:#332f26;--accent:#7fb0dc;--hit:#2b2410;--hit-rule:#6b5620}}
body{background:var(--bg);color:var(--ink);font:15px/1.55 "Public Sans",-apple-system,"Segoe UI",sans-serif;margin:0}
main{max-width:1000px;margin:0 auto;padding:28px 18px 60px}
h1{font:700 24px/1.2 sans-serif;margin:0 0 4px} h2{font:700 18px/1.3 sans-serif;margin:28px 0 8px}
p.lead{color:var(--ink-2);margin:0 0 18px}
table{border-collapse:collapse;width:100%;margin:0 0 22px;font-size:14px}
th,td{border:1px solid var(--rule);padding:6px 9px;text-align:left}
th{background:var(--surface)}
td.num{text-align:right;font-variant-numeric:tabular-nums}
details{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:10px 14px;margin:0 0 10px}
details.hit{background:var(--hit);border-color:var(--hit-rule)}
summary{cursor:pointer;font-weight:600}
pre{white-space:pre-wrap;word-wrap:break-word;font:12.5px/1.5 ui-monospace,Consolas,monospace;background:var(--bg);border:1px solid var(--rule);border-radius:6px;padding:9px 11px;overflow-x:auto;margin:8px 0}
.tag{display:inline-block;background:var(--accent);color:#fff;border-radius:4px;padding:1px 7px;font-size:12px;margin-left:6px}
.out{color:var(--ink-2);font-size:13px;margin:6px 0 0}
</style>"""]
    P.append("<main><h1>Транскрипты: %s</h1>" % e(run_name))
    P.append('<p class="lead">Жёлтым отмечены ходы, где в промпте была НАША вставка. '
             'Показаны ходы начиная с действия %d либо с первой вставки. '
             'В каждом ходу: что подсунули, что модель подумала, какой код выполнила '
             'и что ответил движок.</p>' % from_action)
    P.append("<table><tr><th>игра</th><th>ходов</th><th>первая вставка</th>"
             "<th>уровней до</th><th>уровней после</th></tr>")
    for r in rows:
        P.append("<tr><td><a href='#%s'>%s</a></td><td class='num'>%d</td><td class='num'>%s</td>"
                 "<td class='num'>%d</td><td class='num'>%d</td></tr>"
                 % (r["gid"], r["gid"], r["n"],
                    ("ход %d" % (r["first"] + 1)) if r["first"] is not None else "—",
                    r["lv_before"], r["lv_after"]))
    P.append("</table>")

    for r in rows:
        P.append("<h2 id='%s'>%s</h2>" % (r["gid"], r["gid"]))
        turns = r["turns"]
        for i in r["shown"]:
            t = turns[i]
            up = sect(t, "[USER PROMPT]")
            hits = [n for n, k in MARKS if k in up]
            think = sect(t, "[THINKING]").strip()
            code_m = CODE.search(sect(t, "[ASSISTANT]"))
            code = code_m.group(1) if code_m else ""
            line, lv = outcome(turns[i + 1] if i + 1 < len(turns) else None)
            cls = " class='hit'" if hits else ""
            tags = "".join("<span class='tag'>%s</span>" % e(h) for h in hits)
            if lv:
                tags += "<span class='tag'>УРОВЕНЬ ВЗЯТ</span>"
            P.append("<details%s><summary>ход %d (действие %d)%s</summary>" % (cls, i + 1, t["action"], tags))
            if hits:
                for name, key in MARKS:
                    j = up.find(key)
                    if j >= 0:
                        P.append("<pre>%s</pre>" % e(up[j:j + 1200]))
            if think:
                P.append("<pre>%s</pre>" % e(think[:1500] + ("…" if len(think) > 1500 else "")))
            if code:
                P.append("<pre>%s</pre>" % e(code[:2500]))
            if line:
                P.append("<p class='out'>движок: %s%s</p>" % (e(line), " — УРОВЕНЬ ВЗЯТ" if lv else ""))
            P.append("</details>")
    P.append("</main>")
    return "\n".join(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--name", required=True, help="имя страницы без .html")
    ap.add_argument("--from-action", type=int, default=40)
    ap.add_argument("--max-turns", type=int, default=12, help="сколько ходов на игру показывать")
    a = ap.parse_args()
    run = Path(a.run)
    rows = game_rows(run, a.from_action, a.max_turns)
    out = Path("docs/artifacts") / (a.name + ".html")
    out.write_text(render(rows, run.name, a.from_action), encoding="utf-8")
    shown = sum(len(r["shown"]) for r in rows)
    marked = sum(1 for r in rows if r["first"] is not None)
    print("ok   игр %d, показано ходов %d, игр со вставкой %d" % (len(rows), shown, marked))
    print("ok   уровней после первой вставки: %d, до неё: %d"
          % (sum(r["lv_after"] for r in rows), sum(r["lv_before"] for r in rows)))
    print("ok   размер страницы %.0f КБ" % (out.stat().st_size / 1024))
    print("ok   ссылка: http://127.0.0.1:8765/page/%s.html" % a.name)


if __name__ == "__main__":
    main()

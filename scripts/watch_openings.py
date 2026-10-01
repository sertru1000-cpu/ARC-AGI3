"""Сторож раскрытий в ночь этапа 2 (30.09): новые публичные кернелы соревнования и новые репозитории GitHub про ARC-AGI-3.
Печатает только новое (с момента запуска), без однотипного «STEP_adapt_SFT»; раз в 5 минут до 04:00 МСК.
usage: .venv/bin/python scripts/watch_openings.py >> runs/pubwatch/openings.log
"""
import json, subprocess, time, urllib.request, datetime as dt
ROOT = "/Users/sergeimakarov/Projects/ARC-AGI-3"
seen_k, seen_g, first = set(), set(), True
def kernels():
    out = subprocess.run([ROOT + "/.venv/bin/kaggle", "kernels", "list", "--competition", "arc-prize-2026-arc-agi-3",
                          "--sort-by", "dateCreated", "--page-size", "40", "--csv"], capture_output=True, text=True).stdout
    rows = [l.split(",") for l in out.splitlines()[1:] if l.strip()]
    return [(r[0], ",".join(r[1:-3])) for r in rows]
def repos():
    q = "https://api.github.com/search/repositories?q=arc-agi-3+created:%3E=2026-09-30&sort=updated&per_page=30"
    try:
        d = json.load(urllib.request.urlopen(q, timeout=20))
        return [(r["full_name"], r["size"], (r["description"] or "")[:100]) for r in d.get("items", [])]
    except Exception:
        return []
while True:
    now = dt.datetime.utcnow() + dt.timedelta(hours=3)
    new = []
    for ref, title in kernels():
        if ref not in seen_k:
            seen_k.add(ref)
            if not first and "step" not in (ref + title).lower():
                new.append("КЕРНЕЛ %s | %s" % (ref, title))
    for name, size, desc in repos():
        if name not in seen_g:
            seen_g.add(name)
            if not first:
                new.append("GITHUB %s (%s КБ) %s" % (name, size, desc))
    print("%s %s" % (now.strftime("%d.%m %H:%M"), ("НОВОЕ: " + " || ".join(new)) if new else ("тихо (%d кернелов, %d репо)" % (len(seen_k), len(seen_g)))), flush=True)
    first = False
    if now.hour == 4:
        break
    time.sleep(300)

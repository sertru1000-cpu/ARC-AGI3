"""Что модель делает в вызовах-осмотрах (код без action) — по транскриптам (27.09).
Для каждого вызова python БЕЗ action(...) — признаки по коду (несколько сразу возможны):
  объекты     — segmentation / nodes / связные компоненты своими руками (comps, flood, BFS по цвету);
  кусок доски — ascii с срезами строк/столбцов (crop), печать строк доски;
  разница     — previous_frame / last_transition / transitions / сравнение двух кадров;
  история     — обход history (прошлые ходы и доски);
  счёт цветов — Counter / подсчёт клеток по цветам;
  поиск/план  — BFS/deque/heapq/поиск пути без хода;
  анимация    — кадры анимации (frames);
и объём напечатанного (длина результата). Плюс: какой ход модели следовал за осмотром.
usage: .venv/bin/python scripts/inspect_calls.py runs/night_nextfork-b1 [...]
"""
import re, statistics as st, sys
from collections import Counter
from pathlib import Path
SEC = re.compile(r"^\[([A-Z][A-Z :_a-z]*)\]\s*$", re.M)
CODE = re.compile(r"<parameter=code>\n?(.*?)</parameter>", re.S)
FEAT = [("объекты", r"segmentation|\bnodes\b|\bcomps?\(|flood|components?"),
        ("кусок доски", r"\.ascii|split\('\\n'\)|split\(\"\\n\"\)|crop|\brows?\["),
        ("разница кадров", r"previous_frame|last_transition|before_frame|after_frame|\bdiff\("),
        ("история", r"\bhistory\b"),
        ("переходы", r"\btransitions\b"),
        ("счёт цветов", r"Counter\(|\.count\("),
        ("поиск/план", r"deque|heapq|\bbfs\b|dijkstra|\bpath\b"),
        ("анимация", r"\bframes\b")]
for run in sys.argv[1:]:
    n = 0; feats = Counter(); outlen = []; combos = Counter(); only = Counter()
    for f in sorted(Path(run, "transcripts").glob("*.txt")):
        parts = SEC.split(f.read_text(errors="ignore"))
        code = None
        for i in range(1, len(parts) - 1, 2):
            name, body = parts[i], parts[i + 1]
            if name.startswith("TOOL CALL: python"):
                m = CODE.search(body); code = m.group(1) if m else None
            elif name.startswith("TOOL RESULT: python") and code is not None:
                if not re.search(r"\baction\s*\(", code):
                    n += 1; outlen.append(len(body.strip()))
                    hit = [k for k, rx in FEAT if re.search(rx, code)]
                    feats.update(hit); combos[len(hit)] += 1
                    if len(hit) == 1: only[hit[0]] += 1
                code = None
    print("%s: вызовов-осмотров %d | напечатано: медиана %d знаков" % (run, n, st.median(outlen) if outlen else 0))
    print("   " + " | ".join("%s %.0f%%" % (k, 100 * v / max(n, 1)) for k, v in feats.most_common()))
    print("   признаков в одном осмотре: %s | только один признак: %s" % (dict(sorted(combos.items())), dict(only.most_common())))

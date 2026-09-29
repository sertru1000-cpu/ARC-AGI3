"""Аудит трубопровода обвязки по транскриптам прогона: что из ответа модели доходит до памяти и до следующего вызова (26.09).

Правило feedback-audit-harness-plumbing-first: до новых слоёв — проверить вход модели. Считает по каждому ходу модели
(analysis_step):
  1. где модель пишет модель мира: в ответе (ASSISTANT — его разбирает обвязка) или только в рассуждении (THINKING —
     обвязка его НЕ разбирает, _update_summarized_knowledge_from_assistant(content));
  2. дошёл ли блок «Working world model carried from earlier turns» до СЛЕДУЮЩЕГО запроса и какие поля в нём есть;
  3. сколько результатов кода обрезано обвязкой, сколько ходов без вызова инструмента, сколько сбросов окна.
usage: .venv/bin/python scripts/audit_plumbing.py runs/night_pub-scott [runs/night_nextfork-b1 ...]
"""
import re, sys, statistics as st
from pathlib import Path

LABELS = ["World model", "Goal model", "Action model", "Recent findings", "Open questions", "Plan", "Cross-level notes"]
HDR = re.compile(r"(?im)^[\s>*#_-]*\**\s*(world model|goal model|action model|recent findings|open questions|plan|cross-level notes)\b[^:\n]{0,30}:")
SEC = re.compile(r"^\[([A-Z][A-Z :_a-z]*)\]\s*$", re.M)


def turns(text):
    for chunk in re.split(r"\n--- analysis_step=", text)[1:]:
        parts = SEC.split(chunk)
        d = {}
        for i in range(1, len(parts) - 1, 2):
            d.setdefault(parts[i], []).append(parts[i + 1])
        yield d


def audit(run):
    rows = []
    for f in sorted(Path(run, "transcripts").glob("*.txt")):
        prev_had_wm = None
        for d in turns(f.read_text(errors="ignore")):
            think = "\n".join(d.get("THINKING", [])); ans = "\n".join(d.get("ASSISTANT", []))
            up = (d.get("USER PROMPT") or [""])[0]
            carried = "Working world model carried from earlier turns" in up
            fields = [l for l in LABELS if re.search(r"(?m)^- %s:" % re.escape(l), up)]
            res = "\n".join(d.get("TOOL RESULT: python", []))
            newlvl = ("You have progressed to a new level" in up) or ("You have completed the run" in up)
            over = "The game is over." in up
            rows.append(dict(newlvl=newlvl, over=over,
                wm_in_answer=bool(HDR.search(ans)), wm_in_think=bool(HDR.search(think)),
                carried=carried, n_fields=len(fields), prev_wrote=prev_had_wm,
                truncated=("truncated" in res.lower()), no_tool=("You have not acted yet" in "\n".join(d.get("USER PROMPT", [])[1:])),
                overflow=("context_overflow_recovered" in "\n".join(d.get("ANALYZER STATUS", []))),
                ans_len=len(ans), think_len=len(think)))
            prev_had_wm = bool(HDR.search(ans))
    n = len(rows)
    pc = lambda k: 100 * sum(1 for r in rows if r[k]) / max(n, 1)
    only_think = 100 * sum(1 for r in rows if r["wm_in_think"] and not r["wm_in_answer"]) / max(n, 1)
    after_write = [r for r in rows if r["prev_wrote"] and not r["newlvl"]]
    over_lost = sum(1 for r in after_write if r["over"] and not r["carried"])
    lost = 100 * sum(1 for r in after_write if not r["carried"]) / max(len(after_write), 1)
    print("%s: ходов модели %d" % (run, n))
    print("  модель мира в ОТВЕТЕ (разбирается): %.0f%% | только в РАССУЖДЕНИИ (не разбирается): %.0f%%" % (pc("wm_in_answer"), only_think))
    print("  блок модели мира дошёл до запроса: %.0f%% ходов; полей в нём (медиана) %s" % (pc("carried"), st.median([r["n_fields"] for r in rows]) if rows else 0))
    print("  после хода, где модель ПИСАЛА блок (без перехода на новый уровень), в следующем запросе его нет: %.0f%% (из %d); из них после проигрыша: %d" % (lost, len(after_write), over_lost))
    print("  результат кода обрезан: %.0f%% | ходы без вызова инструмента: %.0f%% | сбросы окна: %.0f%%" % (pc("truncated"), pc("no_tool"), pc("overflow")))
    print("  длина ответа (медиана) %d знаков, рассуждения %d" % (st.median([r["ans_len"] for r in rows]), st.median([r["think_len"] for r in rows])))


for run in sys.argv[1:]:
    audit(run)

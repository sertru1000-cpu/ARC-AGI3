"""Аудит песочницы по транскриптам: что происходит с кодом модели после того, как она его написала (27.09).

Правило feedback-audit-harness-plumbing-first. Бесплатно, по записанным прогонам. По каждому вызову python:
  1. упал ли код (ошибка исполнения, таймаут 30 с, запрещённый импорт) — вызов модели потерян;
  2. обрезан ли ответ песочницы (> ~1024 токенов: «[truncated N chars]»);
  3. вызов с ходом (`action(` в коде): увидела ли модель итог хода (board_changed / level_completed / executed) —
     обвязка отдаёт его, только если код НИЧЕГО не напечатал (tool_agent: stdout вытесняет result);
  4. повтор кода: доля строк кода, уже написанных в ЭТОЙ игре раньше (песочница каждый раз с чистого листа);
  5. какие функции модель определяет заново чаще всего — кандидаты в готовые функции песочницы.
usage: .venv/bin/python scripts/audit_sandbox.py runs/night_nextfork-b1 [runs/... ...]
"""
import re, sys, statistics as st
from collections import Counter
from pathlib import Path

SEC = re.compile(r"^\[([A-Z][A-Z :_a-z]*)\]\s*$", re.M)
CODE = re.compile(r"<parameter=code>\n?(.*?)</parameter>", re.S)
ERR = [("таймаут", re.compile(r"timed out after", re.I)),
       ("запрещённый импорт", re.compile(r"not allowed in the sandbox", re.I)),
       ("синтаксис", re.compile(r"syntax error", re.I)),
       ("ошибка исполнения", re.compile(r"Traceback|(^|\n)(\w+Error|Exception)\b|\"error\":", re.I))]
SEEN_RESULT = re.compile(r"board_changed|level_completed|'executed'|\"executed\"|known_noop|stop_reason")


def calls(path):
    text = Path(path).read_text(errors="ignore")
    parts = SEC.split(text)
    pending = None
    for i in range(1, len(parts) - 1, 2):
        name, body = parts[i], parts[i + 1]
        if name.startswith("TOOL CALL: python"):
            m = CODE.search(body)
            pending = m.group(1) if m else body
        elif name.startswith("TOOL RESULT: python") and pending is not None:
            yield pending, body
            pending = None


def norm(line):
    return re.sub(r"\s+", " ", line.strip())


def audit(run):
    n = 0; err = Counter(); trunc = 0; acting = 0; acting_blind = 0; acting_print = 0
    dup_lines = tot_lines = 0; code_chars = []; defs = Counter(); redefs = Counter()
    for f in sorted(Path(run, "transcripts").glob("*.txt")):
        seen = set(); seen_defs = Counter()
        for code, res in calls(f):
            n += 1
            for label, rx in ERR:
                if rx.search(res):
                    err[label] += 1
                    break
            if "[truncated" in res:
                trunc += 1
            if re.search(r"\baction\s*\(", code):
                acting += 1
                prints = bool(re.search(r"\bprint\s*\(", code))
                acting_print += prints
                if not SEEN_RESULT.search(res):
                    acting_blind += 1
            lines = [norm(l) for l in code.splitlines() if norm(l) and not norm(l).startswith("#")]
            tot_lines += len(lines); dup_lines += sum(1 for l in lines if l in seen); seen.update(lines)
            code_chars.append(len(code))
            for d in re.findall(r"(?m)^\s*def\s+(\w+)\s*\(", code):
                defs[d] += 1
                if seen_defs[d]:
                    redefs[d] += 1
                seen_defs[d] += 1
    pct = lambda a, b: 100.0 * a / max(b, 1)
    print("%s: вызовов python %d" % (run, n))
    print("  упали: %.1f%% (%s)" % (pct(sum(err.values()), n), ", ".join("%s %d" % kv for kv in err.most_common())))
    print("  ответ песочницы обрезан: %.1f%%" % pct(trunc, n))
    print("  с ходом: %d (%.0f%%); из них модель НЕ увидела итог хода: %.0f%%; печатали что-то: %.0f%%" % (
        acting, pct(acting, n), pct(acting_blind, acting), pct(acting_print, acting)))
    print("  строк кода, уже написанных в этой игре раньше: %.0f%% (из %d); код на вызов: медиана %d знаков" % (
        pct(dup_lines, tot_lines), tot_lines, st.median(code_chars) if code_chars else 0))
    print("  функции, определённые заново в той же игре (раз): %s" % ", ".join("%s %d" % kv for kv in redefs.most_common(12)))


if __name__ == "__main__":
    for r in sys.argv[1:]:
        audit(r)

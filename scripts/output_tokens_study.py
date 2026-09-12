"""Из чего состоит ответ модели и что в нём повторяется.

ЗАЧЕМ. Токены ВЫХОДА — единственный рычаг на число ходов: один выходной токен стоит как
2317 входных (`scripts/concurrency_math.py`), и срезание 300 токенов из 1438 даёт +26% ходов.
Значит надо знать, из чего эти 1438 состоят и сколько в них повторов, которые обвязка могла
бы хранить у себя вместо того, чтобы модель писала их заново каждый ход.

КАК СЧИТАЕТСЯ. Стенограммы разбираются на ходы; из каждого берутся две части вывода —
рассуждение [THINKING] и код в вызове инструмента [TOOL CALL: python]. Длина считается
в знаках и переводится в токены по калибровке: суммарная длина делится на измеренное
среднее 1438 токенов на ответ, так что курс знаков-на-токен получается из наших же данных.

ПОВТОР считается по строкам: строка кода (или предложение рассуждения) засчитывается как
повторная, если ДОСЛОВНО встречалась в любом более раннем ходе ЭТОЙ ЖЕ игры. Это нижняя
оценка: перефразированное и переставленное не ловится.

usage:  .venv/bin/python scripts/output_tokens_study.py
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import re
import statistics

TURN = re.compile(r"^--- analysis_step=(\d+) \| action=(\S+) \|", re.M)


def turns(path: str):
    """[(шаг, рассуждение, код)] по одной стенограмме."""
    text = open(path, encoding="utf-8", errors="replace").read()
    marks = [(m.start(), int(m.group(1))) for m in TURN.finditer(text)]
    out = []
    for i, (pos, step) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        block = text[pos:end]
        think = section(block, "[THINKING]")
        code = section(block, "[TOOL CALL: python]")
        out.append((step, think, code))
    return out


def section(block: str, tag: str) -> str:
    i = block.find(tag)
    if i < 0:
        return ""
    j = block.find("\n[", i + len(tag))
    return block[i + len(tag):j if j > 0 else len(block)].strip()


def repeat_share(units_by_turn):
    """Доля единиц, дословно встречавшихся раньше в этой же игре."""
    seen, rep, tot = set(), 0, 0
    per_turn = []
    for units in units_by_turn:
        r = t = 0
        for u in units:
            u = u.strip()
            if len(u) < 8:
                continue
            t += 1
            if u in seen:
                r += 1
            else:
                seen.add(u)
        rep += r
        tot += t
        per_turn.append((r, t))
    return (rep / tot if tot else 0.0), per_turn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/flash_v1_phaseA")
    ap.add_argument("--tok-per-answer", type=float, default=1438.0,
                    help="измеренное среднее токенов на ответ — по нему калибруется курс знаков")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.run, "transcripts", "*.txt")))
    all_think, all_code, n_turns = 0, 0, 0
    rows, lines_pool = [], collections.Counter()
    for f in files:
        ts = turns(f)
        if not ts:
            continue
        think_ch = sum(len(t) for _, t, _ in ts)
        code_ch = sum(len(c) for _, _, c in ts)
        all_think += think_ch
        all_code += code_ch
        n_turns += len(ts)
        r_code, _ = repeat_share([c.splitlines() for _, _, c in ts])
        r_think, _ = repeat_share([re.split(r"(?<=[.!?])\s+", t) for _, t, _ in ts])
        rows.append({"game": os.path.basename(f)[:4], "turns": len(ts),
                     "think_ch": think_ch / len(ts), "code_ch": code_ch / len(ts),
                     "rep_code": r_code, "rep_think": r_think})
        for _, _, c in ts:
            for line in c.splitlines():
                s = line.strip()
                if len(s) >= 12:
                    lines_pool[s] += 1

    ch_per_tok = (all_think + all_code) / (n_turns * a.tok_per_answer)
    print("ходов разобрано: %d в %d играх" % (n_turns, len(rows)))
    print("курс по калибровке: %.2f знака на токен\n" % ch_per_tok)

    print("%-6s %-7s %-13s %-13s %-12s %s" % (
        "игра", "ходов", "рассужд., ток", "код, ток", "повтор кода", "повтор рассужд."))
    for r in sorted(rows, key=lambda r: -r["code_ch"]):
        print("%-6s %-7d %-13.0f %-13.0f %-12.0f%% %.0f%%" % (
            r["game"], r["turns"], r["think_ch"] / ch_per_tok, r["code_ch"] / ch_per_tok,
            100 * r["rep_code"], 100 * r["rep_think"]))

    tt = statistics.median(r["think_ch"] for r in rows) / ch_per_tok
    cc = statistics.median(r["code_ch"] for r in rows) / ch_per_tok
    print("\nМЕДИАНА НА ХОД: рассуждение %.0f токенов, код %.0f токенов (всего %.0f)" % (tt, cc, tt + cc))
    print("доля кода в ответе: %.0f%%" % (100 * cc / (tt + cc)))
    print("повтор кода: медиана %.0f%%, повтор рассуждения: медиана %.0f%%" % (
        100 * statistics.median(r["rep_code"] for r in rows),
        100 * statistics.median(r["rep_think"] for r in rows)))

    print("\nСАМЫЕ ПОВТОРЯЕМЫЕ СТРОКИ КОДА (сколько раз написаны заново за весь прогон):")
    for s, c in lines_pool.most_common(15):
        print("  %5d x  %s" % (c, s[:96]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Аудит длины ответов модели — под жёсткий max_tokens (вопрос D ChatGPT, раунд 2, 12.09).

Кап на выход считает ВСЕ выходные токены, включая рассуждение; код с action() идёт ПОСЛЕ
рассуждения. Значит кап K режет ход у каждого ответа длиннее K. Здесь по транскриптам прогона
восстанавливается распределение длины ответа: рассуждение (reasoning_chars из
[MODEL RESPONSE META]) + код инструмента + текст. Перевод знаков в токены — калибровкой по
счётчику сервера: generation_tokens_total / сумма знаков всех ответов прогона.

usage: .venv/bin/python scripts/response_length_audit.py runs/flash_v1_phaseA [ещё прогоны]
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrency_math import prom  # noqa: E402

META_RE = re.compile(r"\[MODEL RESPONSE META\]\nfinish_reason: (\w+)\ntool_call_count: (\d+)\ncontent_chars: (\d+)\nreasoning_chars: (\d+)")


def responses(path):
    text = open(path, encoding="utf-8").read()
    # блоки ответа: [THINKING] … [ASSISTANT] … [MODEL RESPONSE META]; код — внутри <parameter=code>
    out = []
    for m in META_RE.finditer(text):
        finish, ncalls, content, reasoning = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        # код: ближайший блок [ASSISTANT] перед этим META
        a = text.rfind("[ASSISTANT]", 0, m.start())
        seg = text[a:m.start()] if a >= 0 else ""
        code = sum(len(c) for c in re.findall(r"<parameter=code>(.*?)</parameter>", seg, re.S))
        acts = bool(re.search(r"(?<![A-Za-z_])action\s*\(", seg))
        out.append({"finish": finish, "reasoning": reasoning, "content": content, "code": code, "acts": acts})
    return out


def main():
    for run in sys.argv[1:]:
        rs = []
        for p in sorted(glob.glob(os.path.join(run, "transcripts", "*.txt"))):
            rs.extend(responses(p))
        if not rs:
            print(run, "— транскриптов нет")
            continue
        chars = [r["reasoning"] + r["content"] + r["code"] for r in rs]
        total_chars = sum(chars)
        gen_tok = None
        try:
            pm = prom(os.path.join(run, "vllm-metrics-final.prom"))
            gen_tok = sum(v for k, v in pm.items() if k.startswith("vllm:generation_tokens_total")) or None
        except Exception:  # noqa: BLE001
            pass
        ratio = (gen_tok / total_chars) if gen_tok and total_chars else 0.28
        toks = sorted(c * ratio for c in chars)
        n = len(toks)
        q = lambda f: toks[min(n - 1, int(f * n))]  # noqa: E731
        print("%s: ответов %d; калибровка %.3f ток/знак (%s)" % (run, n, ratio, "по счётчику сервера" if gen_tok else "УМОЛЧАНИЕ 0.28"))
        print("  длина ответа, токены: медиана %.0f, p75 %.0f, p90 %.0f, p95 %.0f, макс %.0f" % (q(.5), q(.75), q(.9), q(.95), toks[-1]))
        for cap in (1000, 1100, 1200, 1300, 1500):
            over = sum(t > cap for t in toks)
            print("  кап %d: обрезано %3.0f%% ответов" % (cap, 100 * over / n))
        acts = [r for r in rs if r["acts"]]
        tk = sorted((r["reasoning"] + r["content"] + r["code"]) * ratio for r in acts)
        if tk:
            m = len(tk)
            print("  ответы С ходом (action): %d из %d; их длина медиана %.0f, p90 %.0f; длиннее 1200 — %.0f%%, длиннее 1000 — %.0f%%"
                  % (m, n, tk[m // 2], tk[min(m - 1, int(.9 * m))], 100 * sum(t > 1200 for t in tk) / m, 100 * sum(t > 1000 for t in tk) / m))
        fin = {}
        for r in rs:
            fin[r["finish"]] = fin.get(r["finish"], 0) + 1
        print("  finish_reason:", fin)


if __name__ == "__main__":
    main()

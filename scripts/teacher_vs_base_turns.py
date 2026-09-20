"""Чем ход учителя отличается от хода базы — по расшифровкам, без единого вызова модели (20.09).

Зачем. Учитель (Gemini-3.6-Flash) на той же обвязке берёт уровни там, где база стоит. Прежде чем платить за
обучение, стоит назвать РАЗНИЦУ В ПОВЕДЕНИИ на уровне одного хода: если она в форме (длина кода, сколько раз
смотрит перед тем как сходить, сколько ходов за раз), часть её может быть достижима и без обучения.

Считаем по одним и тем же играм: ходы обвязки, вызовы модели на ход, строки питоновского кода в вызове,
доля ходов-наблюдений (код без action()), сколько ходов движка модель делает за один ход обвязки,
длина рассуждения (символы ответа), обращения к разбору кадра (segmentation) и к истории (transitions/history).

usage:  .venv/bin/python scripts/teacher_vs_base_turns.py
"""
import json, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEP = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+)", re.M)
TAG = re.compile(r"^\[[^\]\n]{2,40}\]", re.M)
SIDES = {
    "учитель (gemini-3.6-flash)": [("runs/teacher_gemini_pilot", None), ("runs/win32k_tu93", None), ("runs/cache_test", None)],
    "база (qwen3.8-flash-next)": [("runs/flash_v1_phaseA", {"lf52", "lp85", "r11l", "tu93"})],
}


def code_of(meta_body: str) -> str:
    i = meta_body.find("raw_tool_calls:")
    if i < 0:
        return ""
    j = meta_body.find("[", i)
    depth = 0
    for k in range(j, len(meta_body)):
        if meta_body[k] == "[":
            depth += 1
        elif meta_body[k] == "]":
            depth -= 1
            if depth == 0:
                try:
                    calls = json.loads(meta_body[j:k + 1])
                except Exception:
                    return ""
                out = []
                for c in calls:
                    try:
                        out.append(json.loads(c["function"]["arguments"]).get("code") or "")
                    except Exception:
                        pass
                return "\n".join(out)
    return ""


def scan(path: Path) -> dict:
    s = path.read_text(encoding="utf-8", errors="replace")
    heads = [m.start() for m in STEP.finditer(s)]
    acts = [int(m.group(2)) for m in STEP.finditer(s)]
    out = defaultdict(list)
    for n, (a, b) in enumerate(zip(heads, heads[1:] + [len(s)])):
        body = s[a:b]
        marks = list(TAG.finditer(body))
        secs = [(m.group(0), body[m.end(): (marks[i + 1].start() if i + 1 < len(marks) else len(body))])
                for i, m in enumerate(marks)]
        codes = [code_of(t) for tag, t in secs if tag == "[MODEL RESPONSE META]"]
        codes = [c for c in codes if c]
        if not codes:
            continue
        joined = "\n".join(codes)
        out["вызовов на ход"].append(len(codes))
        out["строк кода на ход"].append(sum(len(c.splitlines()) for c in codes))
        out["доля ходов без action()"].append(0.0 if "action(" in joined else 1.0)
        out["ходов движка за ход обвязки"].append(max(0, (acts[n + 1] - acts[n]) if n + 1 < len(acts) else 0))
        out["смотрит segmentation"].append(1.0 if "segmentation" in joined else 0.0)
        out["смотрит историю"].append(1.0 if ("transitions" in joined or "history" in joined) else 0.0)
        out["символов ответа"].append(sum(len(c) for c in codes))
    return out


def main() -> None:
    import statistics as st
    res = {}
    for side, runs in SIDES.items():
        agg = defaultdict(list); games = set()
        for rd, only in runs:
            d = ROOT / rd / "transcripts"
            if not d.exists():
                continue
            for tp in sorted(d.glob("*_p0.txt")):
                g = tp.name[:4]
                if only and g not in only:
                    continue
                games.add(g)
                for k, v in scan(tp).items():
                    agg[k].extend(v)
        res[side] = (agg, games)
    keys = ["вызовов на ход", "строк кода на ход", "ходов движка за ход обвязки", "доля ходов без action()",
            "смотрит segmentation", "смотрит историю", "символов ответа"]
    print("%-32s %22s %22s" % ("признак хода", *res.keys()))
    for k in keys:
        row = []
        for side in res:
            v = res[side][0].get(k) or [0]
            row.append("%.2f (медиана %.1f)" % (sum(v) / len(v), st.median(v)))
        print("%-32s %22s %22s" % (k, *row))
    for side, (agg, games) in res.items():
        print("\n%s: ходов с кодом %d, игр %d (%s)" % (side, len(agg.get("вызовов на ход") or []), len(games), ",".join(sorted(games))))


if __name__ == "__main__":
    main()

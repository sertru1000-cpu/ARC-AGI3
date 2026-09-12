"""Повтор троек действий — прокси-метрика «хождения по кругу» (ChatGPT, раунд 1, 12.09).

Для каждой игры: доля троек подряд идущих действий в ПОСЛЕДНЕЙ ТРЕТИ игры, которые уже
встречались раньше в той же игре (MOUSE — вместе с координатами, как в замере 10.09).
Разбивка: игры с 0–1 уровнем против игр с 2+ уровнями. Ноль квоты: только artifacts/*_events.jsonl.

usage: .venv/bin/python scripts/trigram_repeat.py runs/flash_v1_phaseA runs/flash_input_v1
"""
import glob
import json
import os
import sys


def sig(e):
    d = str(e.get("action_display") or e.get("action_name"))
    if "MOUSE" in d.upper() or "CLICK" in d.upper():
        d = d + ":" + str(e.get("x", e.get("col"))) + "," + str(e.get("y", e.get("row")))
    return d


def game(path):
    acts, lv = [], 0
    for line in open(path, encoding="utf-8"):
        try:
            e = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if e.get("type") != "action":
            continue
        acts.append(sig(e))
        lv = max(lv, int(e.get("level") or 0))
    levels = lv - 1 if lv else 0          # уровень нумеруется с 1; взято = текущий − 1
    tri = [tuple(acts[i:i + 3]) for i in range(len(acts) - 2)]
    if len(tri) < 6:
        return os.path.basename(path)[:4], levels, len(acts), None
    cut = 2 * len(tri) // 3
    seen = set(tri[:cut])
    rep = 0
    for t in tri[cut:]:
        rep += t in seen
        seen.add(t)
    return os.path.basename(path)[:4], levels, len(acts), rep / (len(tri) - cut)


def main():
    for d in sys.argv[1:]:
        rows = [game(p) for p in sorted(glob.glob(os.path.join(d, "artifacts", "*_events.jsonl")))]
        rows = [r for r in rows if r[3] is not None]
        low = [r[3] for r in rows if r[1] <= 1]
        high = [r[3] for r in rows if r[1] >= 2]
        print("%s: игр %d; 0–1 уровень: n=%d, повтор троек %.0f%%; 2+ уровня: n=%d, повтор %.0f%%"
              % (d, len(rows), len(low), 100 * sum(low) / max(1, len(low)),
                 len(high), 100 * sum(high) / max(1, len(high))))
        for g, lv, n, r in sorted(rows, key=lambda r: -r[3]):
            print("  %s lv=%d acts=%4d repeat=%3.0f%%" % (g, lv, n, 100 * r))


if __name__ == "__main__":
    main()

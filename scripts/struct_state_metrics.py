"""H_struct v0, PTR и KBAR по журналу переходов (ChatGPT, раунд 3, 12.09). Ноль квоты.

H_struct v0 (его минимальная версия): убрать строки HUD и хешировать остальную доску.
HUD-строки — те, что меняются на >= 80% ходов игры (эвристика из build_evidence_hints.py).
Дополнительно вариант «строгий»: HUD = строки, меняющиеся на >= 50% ходов.

PTR: ход продуктивен, если новый класс состояния, или новый переход (класс, действие -> класс),
или новый уровень. KBAR (Known-Bad Action Rate): ход считается «заведомо плохим», если тот же
(класс состояния, класс действия) уже >= 2 раз давал тот же класс состояния без прогресса.
Класс действия: стрелки/SPACE как есть; MOUSE — с координатами, округлёнными до ячеек 4x4.

Его пороги, записанные до счёта: PTR медиана < 0.8 — насыщение снято; Spearman(KBAR, уровни)
< -0.30 — патология политики подтверждена, > -0.10 — контроллер не стоит прогона; доля
заведомо плохих ходов >= 20% — контроллер оправдан, < 10% — нет.

usage: .venv/bin/python scripts/struct_state_metrics.py runs/flash_v1_phaseA runs/flash_dose_conc13 ...
"""
import glob
import hashlib
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_delta import load  # noqa: E402


def h(s):
    return hashlib.blake2b(s.encode(), digest_size=8).hexdigest()


def action_class(e):
    d = str(e.get("action_display") or e.get("action_name") or "?")
    if "MOUSE" in d.upper() or "CLICK" in d.upper():
        x = e.get("x", e.get("col")); y = e.get("y", e.get("row"))
        try:
            return "MOUSE:%d,%d" % (int(x) // 4, int(y) // 4)
        except Exception:  # noqa: BLE001
            return "MOUSE"
    return d


def load_events(path):
    ev = []
    for line in open(path, encoding="utf-8"):
        try:
            e = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if e.get("type") in ("initial", "action") and e.get("board_ascii"):
            ev.append(e)
    return ev


def hud_rows(ev, share):
    rows = {}
    prev = None; n = 0
    for e in ev:
        cur = e["board_ascii"].split("\n")
        if prev is not None and e.get("type") == "action":
            n += 1
            for i, (a, b) in enumerate(zip(prev, cur)):
                if a != b:
                    rows[i] = rows.get(i, 0) + 1
        prev = cur
    return {i for i, c in rows.items() if n and c >= share * n}


def metrics(ev, hud):
    def cls(e):
        return h("\n".join(r for i, r in enumerate(e["board_ascii"].split("\n")) if i not in hud))
    seen_s, seen_t, bad = set(), set(), {}
    prev = cls(ev[0]) if ev else None
    lvl = ev[0].get("level") if ev else None
    n = prod = kb = 0
    for e in ev[1:]:
        if e.get("type") != "action":
            continue
        n += 1
        cur = cls(e); a = action_class(e)
        progressed = e.get("level") != lvl
        key = (prev, a)
        if bad.get(key, 0) >= 2:
            kb += 1
        p = progressed or cur not in seen_s or (prev, a, cur) not in seen_t
        prod += p
        if not progressed and cur == prev:
            bad[key] = bad.get(key, 0) + 1
        if progressed:
            lvl = e.get("level"); seen_s, seen_t, bad = set(), set(), {}
        seen_s.add(cur); seen_t.add((prev, a, cur)); prev = cur
    return n, prod / max(1, n), kb / max(1, n)


def spearman(x, y):
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            for k in range(i, j + 1):
                r[s[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rk(x), rk(y); mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** .5
    return num / den if den else float("nan")


def main():
    for run in sys.argv[1:]:
        sc = load(run, True)
        for label, share in (("точный (без маски)", 2.0), ("H_struct v0: HUD >= 80% ходов", 0.8), ("H_struct v0 строгий: HUD >= 50%", 0.5)):
            rows = []
            for p in sorted(glob.glob(os.path.join(run, "artifacts", "*_events.jsonl"))):
                g = os.path.basename(p)[:4]
                ev = load_events(p)
                if len(ev) < 5 or g not in sc:
                    continue
                hud = hud_rows(ev, share) if share <= 1 else set()
                n, ptr, kbar = metrics(ev, hud)
                rows.append((g, sc[g][1], n, ptr, kbar, len(hud)))
            lv = [r[1] for r in rows]
            print("%s | %s: игр %d, HUD-строк медиана %d; PTR медиана %.2f (0–1 ур. %.2f, 2+ ур. %.2f), ρ(PTR,ур.) %+.2f; "
                  "KBAR медиана %.2f, доля ходов всего %.1f%%, ρ(KBAR,ур.) %+.2f"
                  % (run[5:], label, len(rows), statistics.median(r[5] for r in rows),
                     statistics.median(r[3] for r in rows),
                     statistics.mean([r[3] for r in rows if r[1] <= 1] or [float("nan")]),
                     statistics.mean([r[3] for r in rows if r[1] >= 2] or [float("nan")]),
                     spearman([r[3] for r in rows], lv),
                     statistics.median(r[4] for r in rows), 100 * sum(r[4] * r[2] for r in rows) / max(1, sum(r[2] for r in rows)),
                     spearman([r[4] for r in rows], lv)))
            if share == 0.8:
                for g, l, n, ptr, kbar, nh in sorted(rows, key=lambda r: -r[4])[:6]:
                    print("      %s lv=%d acts=%4d PTR=%.2f KBAR=%.2f hud=%d" % (g, l, n, ptr, kbar, nh))
        print()


if __name__ == "__main__":
    main()

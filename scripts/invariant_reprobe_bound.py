"""Потолок выгоды пункта 7 критика раунда 10 — перенос ПРОВЕРЕННЫХ правил механики между уровнями (19.09).

Идея критика: после уровня 1 переносить не путь и не модель мира, а проверенные инварианты («стрелка двигает
объект цвета c», «SPACE ничего не делает»), чтобы на уровне 2 не тратить ходы на их повторную проверку.

Потолок: на каждом ВЗЯТОМ уровне k >= 2 ищем «повторные пробы» — первое применение простого действия (ACTION1-5)
на уровне k, если на уровне k-1 у этого действия был устойчивый эффект (>= 80% применений с одной подписью)
и на уровне k подпись та же (правило действительно перенеслось, значит проба была лишней).
Затем пересчитываем балл так, будто этих ходов не было (actions_per_level[k] - число лишних проб).
Это верхняя граница: реальная правка не уберёт больше, чем лишние пробы, и ничего сверх этого не изменит.

Подпись эффекта перехода b0 -> b1: «холостой», если доска не изменилась; иначе — множество цветов, клетки которых
изменились, плюс знак сдвига центра масс каждого такого цвета по строкам и столбцам.

Порог (задан до замера): прирост балла >= +10% относительно базы по среднему трёх прогонов — строить;
меньше — пункт 7 закрыт.

usage:  .venv/bin/python scripts/invariant_reprobe_bound.py
"""
import glob, json, os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["flash_v1_phaseA", "public_flash_keithtyser", "public_flash_tufa"]
SIMPLE = {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"}


def sig(b0, b1):
    if b0.shape != b1.shape:
        return None
    ch = b0 != b1
    if not ch.any():
        return ("noop",)
    out = []
    for c in sorted(set(np.unique(b0[ch]).tolist()) | set(np.unique(b1[ch]).tolist())):
        y0, x0 = np.nonzero(b0 == c); y1, x1 = np.nonzero(b1 == c)
        if len(y0) and len(y1):
            out.append((c, int(np.sign(round(y1.mean() - y0.mean(), 2))), int(np.sign(round(x1.mean() - x0.mean(), 2)))))
        else:
            out.append((c, "появился" if len(y1) else "исчез"))
    return tuple(out)


def score(per, base, n, done, cap=115.0):
    num = den = 0.0
    for i in range(n):
        w = i + 1; den += w
        if i < done and i < len(per) and per[i] and i < len(base):
            num += min(cap, (base[i] / per[i]) ** 2 * 100) * w
    return num / den if den else 0.0


def main():
    tot_before = tot_after = 0.0; games_n = 0; saved_total = 0; lvl2_moves = 0
    rows = []
    for run in RUNS:
        rd = ROOT / "runs" / run
        br = {r["game_id"][:4]: r for r in json.load(open(rd / "benchmark.json"))["game_runs"]}
        run_before = run_after = 0.0
        for g, r in br.items():
            f = rd / "artifacts" / ("%s_p0_events.jsonl" % r["game_id"])
            per = list(r.get("actions_per_level") or []); base = r.get("base_actions_per_level") or []
            n = r.get("number_of_levels") or 0; done = r.get("levels_completed") or 0
            s0 = score(per, base, n, done)
            saved = defaultdict(int)
            if f.exists() and done >= 2:
                ev = [json.loads(l) for l in open(f, encoding="utf-8")]
                ev = [e for e in ev if e.get("type") == "action" and isinstance(e.get("board"), list)]
                eff = defaultdict(lambda: defaultdict(list))      # уровень -> действие -> [подписи]
                first_use = {}                                      # (уровень, действие) -> подпись первого применения
                for e0, e1 in zip(ev, ev[1:]):
                    l0, l1 = int(e0.get("level") or 0), int(e1.get("level") or 0)
                    a = str(e1.get("action_name", ""))
                    if l0 != l1 or a not in SIMPLE:
                        continue
                    s = sig(np.asarray(e0["board"]), np.asarray(e1["board"]))
                    if s is None:
                        continue
                    eff[l1][a].append(s)
                    first_use.setdefault((l1, a), s)
                for (lvl, a), s_first in first_use.items():
                    if lvl < 2 or lvl > done:          # только взятые уровни со второго
                        continue
                    prev = eff.get(lvl - 1, {}).get(a, [])
                    if len(prev) >= 2:
                        top, cnt = Counter(prev).most_common(1)[0]
                        if cnt / len(prev) >= 0.8 and top == s_first:
                            saved[lvl] += 1
            per2 = list(per)
            for lvl, k in saved.items():
                i = lvl - 1
                if i < len(per2) and per2[i] > k:
                    per2[i] -= k
            s1 = score(per2, base, n, done)
            run_before += s0; run_after += s1; saved_total += sum(saved.values())
            lvl2_moves += sum(per[i] for i in range(1, min(done, len(per))))
            rows.append({"run": run, "game": g, "levels": done, "saved": dict(saved), "before": s0, "after": s1})
        k = max(1, len(br))
        print("%-24s балл %.2f -> потолок %.2f (+%.1f%%)" % (run, run_before / k, run_after / k, 100 * (run_after - run_before) / max(1e-9, run_before)))
        tot_before += run_before / k; tot_after += run_after / k; games_n += 1
    print("\nлишних проб на взятых уровнях 2+: %d хода из %d ходов этих уровней (%.1f%%)" % (saved_total, lvl2_moves, 100 * saved_total / max(1, lvl2_moves)))
    gain = 100 * (tot_after - tot_before) / max(1e-9, tot_before)
    print("СРЕДНЕЕ трёх прогонов: %.2f -> потолок %.2f, прирост %+.1f%%; порог >= +10%% -> %s"
          % (tot_before / games_n, tot_after / games_n, gain, "СТРОИТЬ" if gain >= 10 else "ПУНКТ 7 ЗАКРЫТ"))
    json.dump(rows, open(ROOT / "runs/invariant_reprobe_bound_19_09.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

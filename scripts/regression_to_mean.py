"""Рисунок «платит на сильных играх» — свойство надстройки или регрессия к среднему? (25.09)

Откуда вопрос. В прогонах goalrule и combined потери сосредоточены в играх, где база взяла
много: ft09 71->48, re86 42->17, ar25 27->3. Соблазн объяснить это надстройкой и строить
адаптивную защиту — включать слой только на застрявших играх.

Но у этого рисунка есть скучное объяснение. Балл игры сильно шумит, и игра, оказавшаяся
наверху в одном прогоне, на повторе ТОГО ЖЕ КОДА упадёт просто потому, что наверх её
занёс в том числе шум. Если так, адаптивная защита целится в призрак.

Проверка бесплатна: у нас есть три прогона одной и той же сборки (flash_v1_phaseA 9.43,
public_flash_tufa 9.00, public_flash_keithtyser 6.76). Считаем на каждой паре ОДИНАКОВЫХ
прогонов ту же величину, что у надстроек: средняя дельта в четырёх сильнейших играх
исходного прогона. Если у одинаковых сборок она такая же отрицательная — рисунок к
надстройке отношения не имеет.

usage: .venv/bin/python scripts/regression_to_mean.py
"""
from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAME = ["flash_v1_phaseA", "public_flash_tufa", "public_flash_keithtyser"]
LAYERS = [("flash_v1_phaseA", "flash_goalrule"), ("flash_v1_phaseA", "flash_combined")]
TOP = 4


def scores(name: str) -> dict[str, float]:
    raw = json.loads((ROOT / "runs" / name / "score.json").read_text())["games"]
    return {gid[:4]: float(v["score"]) for gid, v in raw.items()}


def top_delta(a: dict[str, float], b: dict[str, float]) -> tuple[float, float, list[str]]:
    common = sorted(set(a) & set(b), key=lambda g: -a[g])
    top = common[:TOP]
    rest = common[TOP:]
    return (
        sum(b[g] - a[g] for g in top) / max(len(top), 1),
        sum(b[g] - a[g] for g in rest) / max(len(rest), 1),
        top,
    )


def main() -> None:
    print("Средняя дельта в %d сильнейших играх исходного прогона и во всех остальных.\n" % TOP)
    print("=== ОДИНАКОВЫЕ сборки (надстройки нет, любой сдвиг — шум) ===")
    print("%-28s %-28s %9s %9s  %s" % ("откуда", "куда", "топ-4", "прочие", "сильнейшие"))
    same_top = []
    for x, y in permutations(SAME, 2):
        a, b = scores(x), scores(y)
        t, r, names = top_delta(a, b)
        same_top.append(t)
        print("%-28s %-28s %+9.1f %+9.1f  %s" % (x, y, t, r, ",".join(names)))

    print("\n=== НАДСТРОЙКИ ===")
    for x, y in LAYERS:
        a, b = scores(x), scores(y)
        t, r, names = top_delta(a, b)
        print("%-28s %-28s %+9.1f %+9.1f  %s" % (x, y, t, r, ",".join(names)))

    lo, hi = min(same_top), max(same_top)
    print("\nу одинаковых сборок дельта топ-4 лежит в [%+.1f, %+.1f]" % (lo, hi))
    print("вывод: если дельта надстройки попадает в этот разброс, рисунок «платит на сильных»")
    print("       объясняется регрессией к среднему и о слое ничего не говорит.")


if __name__ == "__main__":
    main()

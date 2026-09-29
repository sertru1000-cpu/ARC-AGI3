"""Сравнение слоя со СРЕДНИМ по нескольким замерам базы, а не с одной точкой (26.09).

Зачем. 25.09 все четыре слоя мерились против flash_v1_phaseA — точки от 6 сентября. Пять
замеров той же стоковой сборки дают 9.43 / 9.00 / 6.76 / 6.13 / 7.77, отклонение 1.41: сравнение
с одной точкой почти ничего не измеряло. Среднее по пяти базам шумит в корень из пяти раз меньше.

Коридор шума строится тем же способом, что и сравнение, — иначе он не сопоставим: каждый
базовый прогон сравнивается со средним по ОСТАЛЬНЫМ базам. Получается распределение того, что
даёт сборка, в которой ничего не менялось.

Отдельно даётся разрез без четырёх шумных игр (ft09, re86, ar25, r11l): они дают 40% разброса
балла, набор выбран по разбросу базы против самой себя, до всякого сравнения слоёв.

usage: .venv/bin/python scripts/vs_base_mean.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASES = ["flash_v1_phaseA", "public_flash_tufa", "public_flash_keithtyser",
         "night_stock-flash", "night_stock-base2", "night_stock-base4", "night_stock-base5"]
LAYERS = [("precond_v2", "precond"), ("nostop_v2", "nostop"),
          ("flash_combined", "combined"), ("flash_goalrule", "goalrule"),
          ("night_nextfork-b1", "форк b1"), ("night_nextfork-b2", "форк b2"),
          ("night_nextfork-b4", "форк b4"), ("night_nextfork-b5", "форк b5")]
NOISY = {"ft09", "re86", "ar25", "r11l"}


def games(name: str) -> dict[str, float] | None:
    p = ROOT / "runs" / name / "score.json"
    if not p.is_file():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))["games"]
    return {g[:4]: float(v["score"]) for g, v in raw.items()}


def delta(a: dict[str, float], b: dict[str, float], *, quiet: bool) -> float:
    keys = set(a) & set(b)
    if quiet:
        keys -= NOISY
    return sum(b[g] - a[g] for g in keys) / len(keys)


def mean_of(names: list[str]) -> dict[str, float]:
    got = [games(n) for n in names]
    got = [g for g in got if g]
    keys = set.intersection(*(set(g) for g in got))
    return {k: st.mean(g[k] for g in got) for k in keys}


def main() -> None:
    have = [n for n in BASES if games(n)]
    print("замеров базы: %d\n" % len(have))

    # коридор: каждая база против среднего по остальным
    null_all, null_quiet = [], []
    for n in have:
        rest = [m for m in have if m != n]
        if len(rest) < 2:
            continue
        ref = mean_of(rest)
        null_all.append(delta(ref, games(n), quiet=False))
        null_quiet.append(delta(ref, games(n), quiet=True))
    ka = 1.96 * st.stdev(null_all) if len(null_all) > 2 else None
    kq = 1.96 * st.stdev(null_quiet) if len(null_quiet) > 2 else None
    print("коридор (база против среднего остальных, 95%%):")
    print("  по всем играм:  ±%.2f   (точки: %s)" % (ka, ", ".join("%+.2f" % v for v in null_all)))
    print("  без шумных:     ±%.2f   (точки: %s)" % (kq, ", ".join("%+.2f" % v for v in null_quiet)))

    # Методология по замечанию критика 26.09: ОСНОВНОЙ вердикт — по всем 25 играм;
    # разрез без четырёх шумных — только анализ чувствительности, не вердикт. Иначе
    # получается отбор игр после просмотра результата.
    ref = mean_of(have)
    print("\n%-12s %10s %10s | %12s %10s" % ("сборка", "Δ все 25", "ВЕРДИКТ", "Δ без шумных", "(чувств.)"))
    for name, label in LAYERS:
        g = games(name)
        if not g:
            continue
        da, dq = delta(ref, g, quiet=False), delta(ref, g, quiet=True)
        va = "ПОЛЬЗА" if da > ka else "ВРЕД" if da < -ka else "в коридоре"
        vq = "польза" if dq > kq else "вред" if dq < -kq else "в коридоре"
        print("%-12s %+10.2f %10s | %+12.2f %10s" % (label, da, va, dq, vq))


if __name__ == "__main__":
    main()

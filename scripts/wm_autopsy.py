"""Разбор прогона со слоем исполняемой модели мира (arc3-stock-flash-wm).

Первый вопрос не «какой балл», а «жив ли метод у нашей модели»: сколько гипотез
о правилах модель написала и какая доля из них воспроизводит наблюдённые переходы.
Критерий, записанный до пуска: метод жив, если >= 5 игр с лучшей точностью >= 0.5
и >= 2 игр с >= 0.9.

usage:
    python scripts/wm_autopsy.py runs/flash_wm_phaseA
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def main() -> int:
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/flash_wm_phaseA")
    logs = [p for p in run.glob("*.log") if "vllm" not in p.name]
    if not logs:
        print(f"{run}: лога кернела нет"); return 1
    raw = logs[0].read_text(encoding="utf-8", errors="replace")
    text = "".join(json.loads('"' + m + '"') for m in re.findall(r'"data":"((?:[^"\\]|\\.){0,400})"', raw)) or raw

    installed = "wm v2 installed" in text or "wm installed" in text
    rows = re.findall(r"\[WM\] проверка гипотезы: точность ([0-9.]+) на (\d+) переходах \(лучшая за игру ([0-9.]+), вызовов (\d+), с predict (\d+)\)", text)
    print(f"слой установлен: {'да' if installed else 'НЕТ — дальше всё бессмысленно'}")
    print(f"проверок гипотез всего: {len(rows)}")
    if not rows:
        print("модель ни разу не проверила гипотезу — метод не запустился")
        return 0

    accs = [float(r[0]) for r in rows]
    per_game_best: list[float] = []
    best = 0.0
    prev_calls = 0
    for acc, _chk, gbest, calls, _pred in rows:                 # счётчики сбрасываются на новой игре
        if int(calls) < prev_calls:
            per_game_best.append(best); best = 0.0
        prev_calls = int(calls); best = max(best, float(gbest))
    per_game_best.append(best)

    good = sum(1 for b in per_game_best if b >= 0.5)
    great = sum(1 for b in per_game_best if b >= 0.9)
    print(f"игр с проверками: {len(per_game_best)}")
    print(f"лучшая точность по играм: {[round(b, 2) for b in sorted(per_game_best, reverse=True)]}")
    print(f"точность >= 0.5 в {good} играх; >= 0.9 в {great}")
    print(f"медиана всех проверок: {sorted(accs)[len(accs)//2]:.2f}; доля проверок с 0.00: {100*sum(1 for a in accs if a == 0)/len(accs):.0f}%")
    verdict = "МЕТОД ЖИВ" if (good >= 5 and great >= 2) else "метод мёртв для этой модели"
    print(f"\nвердикт по записанному критерию (>=5 игр с 0.5 и >=2 с 0.9): {verdict}")

    bench = run / "benchmark.json"
    if bench.exists():
        runs = json.load(open(bench, encoding="utf-8"))["game_runs"]
        lv = sum(r.get("levels_completed") or 0 for r in runs)
        print(f"для справки: уровней {lv}, игр {len(runs)} (балл считать parred_delta против flash v1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Пункт 4 критика раунда 10: сколько ходов экономит ВЫБОР ХОДА ПО РАЗЛИЧЕНИЮ ГИПОТЕЗ (24.09).

Вопрос. Агент изучает незнакомую механику пробами. Если выбирать пробы так, чтобы каждая максимально
сокращала число оставшихся гипотез («двадцать вопросов»), механика выучивается быстрее, и освободившиеся
ходы идут на решение. Стоит ли строить такой слой?

Как меряем ПОТОЛОК, а не пользу. Локальный движок детерминированно повторяет бой (561/561), поэтому из
любого состояния можно развернуть любой ход бесплатно. Значит можно сравнить две стратегии проб на одних
и тех же уровнях:
  * КАК ИГРАЛА МОДЕЛЬ -- последовательность ходов из записанного решения;
  * ИНФОРМАТИВНО -- на каждом шаге выбираем ход, который сильнее всего сокращает множество гипотез.
Считаем, на каком ходу каждая стратегия доходит до одной гипотезы (механика выучена).

Пространство гипотез. Берём то, что в этих играх и составляет механику простых действий: какое из пяти
действий чем является. Гипотеза -- назначение «действие -> роль», роль наблюдаема по эффекту хода
(ничего не изменилось / сдвиг объекта / смена кадра целиком / смена уровня). Множество гипотез в начале --
все назначения, совместимые с пустым опытом; каждая проба выбрасывает несовместимые.

Порог решения (записан ДО замера): если информативный выбор экономит менее 15% ходов до выученной
механики, слой не строим -- это тот же порядок, что у уже закрытых пунктов (перенос инвариантов дал
потолок 4.4%, анти-петля по смыслу 0%).

ПОПРАВКА 24.09 при первом запуске: сначала я взял пути из runs/bfs_originals.json -- это решения ПОИСКОМ,
а не игра модели (18 ходов против 561 в настоящей партии). Сравнение идеального выбора с идеальным же
решением ничего не говорит. Здесь берутся НАСТОЯЩИЕ истории из runs/flash_v1_phaseA/benchmark.json.

usage:  .venv/bin/python scripts/active_probe_bound.py [--games 25] [--out runs/active_probe_24_09.json]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT / "scripts"))
import logging  # noqa: E402

logging.disable(logging.ERROR)
import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

SIMPLE = ["ACTION%d" % i for i in range(1, 6)]


def frame_of(fr):
    if fr is None or not fr.frame:
        return None
    return np.asarray(fr.frame[-1], dtype=np.int16)


def effect_role(before: np.ndarray, after: np.ndarray, score_changed: bool) -> str:
    """Наблюдаемая роль хода: то, что агент реально видит после пробы."""
    if score_changed:
        return "уровень"
    if after is None:
        return "конец"
    if before.shape != after.shape:
        return "кадр"
    diff = int(np.count_nonzero(before != after))
    if diff == 0:
        return "ничего"
    if diff <= max(8, before.size // 200):
        return "сдвиг"
    return "кадр"


def probe_all(env, base_frame, base_score):
    """Развернуть каждое из пяти действий из текущего состояния -- роль каждого. Бесплатно: локальный движок."""
    out = {}
    for name in SIMPLE:
        e = copy.deepcopy(env)
        fr = e.step(GameAction[name], data=None)
        g = frame_of(fr)
        sc = getattr(fr, "score", base_score) if fr is not None else base_score
        out[name] = effect_role(base_frame, g, sc != base_score)
    return out


def consistent(hyps, observed):
    """Оставить гипотезы, совместимые с наблюдениями {действие: роль}."""
    return [h for h in hyps if all(h.get(a) == r for a, r in observed.items())]


def run_history(arc, gid: str, history: list, max_steps: int) -> dict | None:
    """Прогоняем НАСТОЯЩУЮ партию и считаем, во что обошлось изучение механики."""
    env = arc.make(gid)
    fr = env.reset()
    g = frame_of(fr)
    if g is None:
        return None
    score = getattr(fr, "score", 0)

    first_try = {}        # действие -> номер хода первой пробы
    roles = {}            # действие -> наблюдённая роль
    idle = 0              # ходы, ничего не изменившие
    repeat_idle = 0       # повторные пробы действия, уже известного как бесполезное
    n = 0
    for t, rec in enumerate(history[:max_steps]):
        act = (rec.get("action") or {})
        name = act.get("id")
        data = act.get("data")
        if not name:
            continue
        try:
            fr2 = env.step(GameAction[name] if name != "RESET" else GameAction.RESET, data=data)
        except Exception:
            break
        g2 = frame_of(fr2)
        sc2 = getattr(fr2, "score", score) if fr2 is not None else score
        role = effect_role(g, g2, sc2 != score)
        n += 1
        if role == "ничего":
            idle += 1
            if name in SIMPLE and roles.get(name) == "ничего":
                repeat_idle += 1
        if name in SIMPLE and name not in first_try:
            first_try[name] = t + 1
            roles[name] = role
        if g2 is None:
            break
        g, score = g2, sc2

    learned = max(first_try.values()) if len(first_try) == len(SIMPLE) else None
    return {"game": gid[:4], "ходов": n,
            "ходов_до_знания_ролей": learned,
            "действий_опробовано": len(first_try),
            "холостых": idle, "повторные_бесполезные": repeat_idle}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="runs/flash_v1_phaseA/benchmark.json")
    ap.add_argument("--games", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--out", default="runs/active_probe_24_09.json")
    a = ap.parse_args()

    bench = json.load(open(ROOT / a.bench, encoding="utf-8"))
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL,
                         environments_dir=str(ROOT / "environment_files"))
    rows = []
    for gr in bench["game_runs"][:a.games]:
        gid = gr["game_id"]
        r = run_history(arc, gid, gr.get("history") or [], a.max_steps)
        if r is None:
            continue
        r["уровней"] = gr.get("levels_completed")
        rows.append(r)
        print("  %-5s ходов %3d | уровней %s | опробовано действий %d/5 | роли известны к ходу %s"
              " | холостых %d (%.0f%%) | повторные бесполезные %d"
              % (r["game"], r["ходов"], r["уровней"], r["действий_опробовано"],
                 r["ходов_до_знания_ролей"], r["холостых"],
                 100 * r["холостых"] / max(r["ходов"], 1), r["повторные_бесполезные"]), flush=True)

    if not rows:
        print("партий не нашлось"); return
    full = [r for r in rows if r["ходов_до_знания_ролей"]]
    tot = sum(r["ходов"] for r in rows)
    idle = sum(r["холостых"] for r in rows)
    rep = sum(r["повторные_бесполезные"] for r in rows)
    print("\nИТОГ по %d партиям, %d ходов" % (len(rows), tot))
    if full:
        m = float(np.mean([r["ходов_до_знания_ролей"] for r in full]))
        print("  роли всех пяти действий узнаны в %d партиях, в среднем к ходу %.1f"
              " (информативный минимум 5) -> лишних %.1f хода на партию, это %.1f%% ходов партии"
              % (len(full), m, m - 5, 100 * (m - 5) / (tot / len(rows))))
    else:
        print("  ни в одной партии модель не попробовала все пять простых действий")
    print("  холостых ходов всего: %d (%.1f%% ходов)" % (idle, 100 * idle / max(tot, 1)))
    print("  из них повторные пробы уже известного бесполезного действия: %d (%.1f%% ходов)"
          % (rep, 100 * rep / max(tot, 1)))
    print("\nПОРОГ до замера: слой строить, если он может вернуть >=15%% ходов.")
    Path(ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"строки": rows}, open(ROOT / a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

"""Сколько ходов вернёт правило «не кликай в мёртвую клетку» (24.09).

Откуда вопрос. Замер scripts/active_probe_bound.py показал: холостых ходов (ничего не изменилось) 7.9%
в среднем, но они собраны в застрявших партиях -- у sk48 183 из 284 ходов, 64%, при нуле взятых уровней.
Различение гипотез тут ни при чём: модель просто повторно тычет туда, где уже ничего не произошло.

Что считаем, ровно три числа:
  1. КЛЕТКА УЖЕ БЫЛА МЁРТВОЙ -- клик в координату, которая на этом же уровне уже давала «ничего».
     Это то, что убирается тривиальным списком и никаких гипотез не требует.
  2. КЛЕТКА НОВАЯ, НО МЁРТВАЯ -- первый клик в неё; убрать можно лишь угадав, где пусто.
  3. Сколько ходов вернёт правило и что это даёт по баллу: балл RHAE ~ (базлайн/ходы)^2, поэтому
     экономия доли p ходов на взятом уровне поднимает его вклад в (1/(1-p))^2 раз.

Порог (записан ДО замера, тот же, что у пункта 4): строить, если правило возвращает >= 15% ходов.

usage:  .venv/bin/python scripts/dead_click_bound.py
"""
from __future__ import annotations

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


def frame_of(fr):
    if fr is None or not fr.frame:
        return None
    return np.asarray(fr.frame[-1], dtype=np.int16)


def main() -> None:
    bench = json.load(open(ROOT / "runs/flash_v1_phaseA/benchmark.json", encoding="utf-8"))
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL,
                         environments_dir=str(ROOT / "environment_files"))
    tot = dead_repeat = dead_new = clicks = 0
    rows = []
    for gr in bench["game_runs"]:
        gid = gr["game_id"]
        env = arc.make(gid)
        fr = env.reset()
        g = frame_of(fr)
        if g is None:
            continue
        lv = getattr(fr, "levels_completed", 0) or 0   # ЛОВУШКА 25.09: у кадра НЕТ поля score
        dead: set[tuple[int, int]] = set()     # мёртвые клетки текущего уровня
        n = rep = new = cl = 0
        for rec in (gr.get("history") or []):
            act = (rec.get("action") or {})
            name, data = act.get("id"), act.get("data")
            if not name:
                continue
            try:
                fr2 = env.step(GameAction[name] if name != "RESET" else GameAction.RESET, data=data)
            except Exception:
                break
            g2 = frame_of(fr2)
            lv2 = (getattr(fr2, "levels_completed", lv) or lv) if fr2 is not None else lv
            n += 1
            is_click = name == "ACTION6" and isinstance(data, dict)
            xy = (data.get("x"), data.get("y")) if is_click else None
            changed = (g2 is None) or (g2.shape != g.shape) or bool(np.any(g2 != g)) or lv2 != lv
            if is_click:
                cl += 1
                if not changed:
                    if xy in dead:
                        rep += 1          # повтор в уже мёртвую клетку -- убирается списком
                    else:
                        new += 1          # первый раз сюда -- узнать заранее нельзя
                        dead.add(xy)
            if lv2 != lv:
                dead.clear()              # новый уровень -- прежние выводы недействительны
            if g2 is None:
                break
            g, lv = g2, lv2
        rows.append((gid[:4], n, cl, rep, new, gr.get("levels_completed")))
        tot += n; clicks += cl; dead_repeat += rep; dead_new += new

    print("%-6s %6s %7s %9s %9s %8s" % ("игра", "ходов", "кликов", "мёртв.повтор", "мёртв.новых", "уровней"))
    for r in sorted(rows, key=lambda x: -x[3]):
        print("%-6s %6d %7d %9d %9d %8s" % r)
    print("\nВСЕГО: ходов %d, кликов %d" % (tot, clicks))
    print("  повторные клики в уже мёртвую клетку: %d (%.1f%% всех ходов)" % (dead_repeat, 100 * dead_repeat / max(tot, 1)))
    print("  первые клики в мёртвую клетку:        %d (%.1f%% всех ходов)" % (dead_new, 100 * dead_new / max(tot, 1)))
    p = dead_repeat / max(tot, 1)
    print("\nправило «не кликай в мёртвую клетку» вернуло бы %.1f%% ходов" % (100 * p))
    print("в баллах это множитель примерно (1/(1-p))^2 = %.3f, то есть %+.1f%% к RHAE"
          % ((1 / (1 - p)) ** 2, 100 * ((1 / (1 - p)) ** 2 - 1)))
    print("ПОРОГ 15%% ->", "СТРОИТЬ" if p >= 0.15 else "ЗАКРЫТЬ")


if __name__ == "__main__":
    main()

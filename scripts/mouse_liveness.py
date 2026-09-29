"""Есть ли в игре хоть одна клетка, где клик работает, и как быстро её найти (25.09).

Откуда вопрос. sk48 — игра с худшим охватом (первый уровень взят в 1 прогоне из 7). Полный
перебор всех 4096 клеток в шести состояниях (старт, ходы 20, 50, 84, 120, 168, до и после
взятия уровня) не нашёл НИ ОДНОГО клика, меняющего доску: мышь там мертва целиком. При этом
движок объявляет ACTION6 доступным, и модель потратила на клики 185 ходов из 1290 по семи
прогонам, а в худшем прогоне — 58% всех ходов.

Что здесь считаем. Обвязка могла бы выяснить это дешевле модели — но в бою перебора нет,
каждый клик стоит хода. Вопрос решается выборкой, а её размер зависит от ПЛОТНОСТИ живых
клеток в играх, где мышь работает. Если живых клеток много, десяток случайных проб надёжно
отличает «мышь мертва» от «не попал»; если единицы — выборка бесполезна и затея закрыта.

Меряем по стартовому состоянию каждой игры: сколько клеток из 4096 меняют доску.
Перебор идёт подряд на одном движке, пока клики пустые (состояние не меняется, поэтому пробы
независимы); после первого срабатывания движок пересоздаётся и воспроизводится до той же точки.

usage: .venv/bin/python scripts/mouse_liveness.py [--max-live 40]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
logging.disable(logging.ERROR)
import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402


def grid(fr):
    return None if fr is None or not fr.frame else np.asarray(fr.frame[-1], dtype=np.int16)


def scan(arc, gid: str, max_live: int) -> tuple[int, int]:
    """Возвращает (живых клеток, проверено клеток). Останавливается на max_live."""
    env = arc.make(gid)
    fr = env.reset()
    base = grid(fr)
    if base is None:
        return 0, 0
    live = checked = 0
    for idx in range(64 * 64):
        y, x = divmod(idx, 64)
        r = env.step(GameAction.ACTION6, data={"x": x, "y": y})
        checked += 1
        rg = grid(r)
        if rg is None or rg.shape != base.shape or bool(np.any(rg != base)):
            live += 1
            if live >= max_live:
                break
            env = arc.make(gid)          # доска изменилась — возвращаемся в исходное состояние
            fr = env.reset()
            base = grid(fr)
    return live, checked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-live", type=int, default=40,
                    help="остановиться, найдя столько живых клеток (полный перебор не нужен)")
    a = ap.parse_args()
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL,
                         environments_dir=str(ROOT / "environment_files"))
    bench = json.loads((ROOT / "runs/flash_v1_phaseA/benchmark.json").read_text(encoding="utf-8"))
    print("Живые клетки в СТАРТОВОМ состоянии. Перебор останавливается на %d находках.\n" % a.max_live)
    print("%-6s %10s %10s %s" % ("игра", "живых", "проверено", "вывод"))
    dead = []
    for g in bench["game_runs"]:
        gid = g["game_id"]
        t0 = time.time()
        live, checked = scan(arc, gid, a.max_live)
        if live == 0:
            dead.append(gid[:4])
        note = ("мышь мертва" if live == 0 else
                "редкая: %.1f%% клеток" % (100 * live / checked) if live < a.max_live else
                "частая: не меньше %.0f%%" % (100 * live / checked))
        print("%-6s %10d %10d %s  (%.0f с)" % (gid[:4], live, checked, note, time.time() - t0))
    print("\nигр с полностью мёртвой мышью: %d — %s" % (len(dead), ", ".join(dead) or "нет"))


if __name__ == "__main__":
    main()

"""Есть ли в играх действия, мёртвые ЦЕЛИКОМ, и сколько ходов на них уходит (25.09).

Откуда вопрос. Охват первых уровней — главная цель (решение владельца 25.09): полный охват
стоит +2.04 балла при коридоре шума ±1.74, и ни одна из 25 игр не является непроходимой —
каждая хотя бы раз брала первый уровень.

Разбор sk48 показал механизм потери охвата: мышь в этой игре не сработала НИ РАЗУ за 171 клик
по четырём прогонам, при том что стрелки работают в 82-95% случаев. В единственном прогоне,
где sk48 взял уровень, кликов почти не было; в худшем они составили 58% ходов.

Движок сообщает СПИСОК доступных действий, но не сообщает, что действие бесполезно во всей
игре. Модель обязана выяснить это сама, ценой ходов. Здесь считаем, в скольких играх такое
бывает и во что обходится.

Считается по всем полным прогонам в runs/, объединённо: действие считается мёртвым в игре,
если ни одно его применение (за все прогоны, не меньше порога наблюдений) не изменило доску.

usage: .venv/bin/python scripts/dead_action_scan.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
logging.disable(logging.ERROR)
import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

RUNS = ["flash_v1_phaseA", "public_flash_tufa", "public_flash_keithtyser",
        "precond_v2", "nostop_v2", "flash_combined", "flash_goalrule"]
NAMES = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT",
         "ACTION4": "RIGHT", "ACTION5": "пятая", "ACTION6": "МЫШЬ", "ACTION7": "седьмая"}
MIN_TRIES = 10          # меньше — не вывод, а совпадение


def grid(fr):
    return None if fr is None or not fr.frame else np.asarray(fr.frame[-1], dtype=np.int16)


def main() -> None:
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL,
                         environments_dir=str(ROOT / "environment_files"))
    # (игра, действие) -> [сделано, сработало]; и общее число ходов в игре
    stat: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    moves: dict[str, int] = defaultdict(int)
    for name in RUNS:
        p = ROOT / "runs" / name / "benchmark.json"
        if not p.is_file():
            continue
        for gr in json.loads(p.read_text(encoding="utf-8"))["game_runs"]:
            gid = gr["game_id"][:4]
            env = arc.make(gr["game_id"])
            fr = env.reset()
            g = grid(fr)
            if g is None:
                continue
            lv = getattr(fr, "levels_completed", 0) or 0
            for rec in (gr.get("history") or []):
                a = rec.get("action") or {}
                if not a.get("id"):
                    continue
                try:
                    fr2 = env.step(GameAction[a["id"]] if a["id"] != "RESET" else GameAction.RESET,
                                   data=a.get("data"))
                except Exception:
                    break
                g2 = grid(fr2)
                lv2 = (getattr(fr2, "levels_completed", lv) or lv) if fr2 is not None else lv
                ch = (g2 is None) or (g2.shape != g.shape) or bool(np.any(g2 != g)) or lv2 != lv
                s = stat[(gid, a["id"])]
                s[0] += 1
                s[1] += int(ch)
                moves[gid] += 1
                if g2 is None:
                    break
                g, lv = g2, lv2

    dead = [(gid, act, t) for (gid, act), (t, c) in stat.items() if c == 0 and t >= MIN_TRIES]
    dead.sort(key=lambda x: -x[2])
    wasted = defaultdict(int)
    for gid, act, t in dead:
        wasted[gid] += t
    print("Действия, не сработавшие НИ РАЗУ (не меньше %d попыток), по %d прогонам\n" % (MIN_TRIES, len(RUNS)))
    print("%-6s %-8s %9s %14s" % ("игра", "действие", "попыток", "% ходов игры"))
    for gid, act, t in dead:
        print("%-6s %-8s %9d %13.0f%%" % (gid, NAMES.get(act, act), t, 100 * t / moves[gid]))
    tot_moves = sum(moves.values())
    tot_dead = sum(t for _, _, t in dead)
    print("\nигр с полностью мёртвым действием: %d из %d" % (len(wasted), len(moves)))
    print("ходов, потраченных на заведомо мёртвые действия: %d из %d (%.1f%%)"
          % (tot_dead, tot_moves, 100 * tot_dead / tot_moves))
    print("\nхуже всего задеты:")
    for gid, w in sorted(wasted.items(), key=lambda x: -x[1] / moves[x[0]])[:6]:
        print("   %-6s %4d из %4d ходов (%.0f%%)" % (gid, w, moves[gid], 100 * w / moves[gid]))


if __name__ == "__main__":
    main()

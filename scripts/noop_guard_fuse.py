"""Предохранитель на запрет пустых ходов: при каком пороге он полезен? (25.09)

Что выяснилось. Запрет повторного пустого хода при той же доске срезает 200 ходов из 4380,
но 37 срезов (18%) отнимают работающий ход: доска — не полное состояние игры, у неё есть
скрытая часть, невидимая в кадре. Довод «движок детерминирован, значит ошибок быть не может»
был неверен: проверка детерминизма показывает лишь, что та же ПОСЛЕДОВАТЕЛЬНОСТЬ ходов даёт
ту же траекторию.

Но ошибки распределены крайне неравномерно. 160 срезов из 200 — это sk48, где промахов 4 (2%)
и где агент взял ноль уровней при 183 пустых ходах из 284. В остальных играх срезов единицы,
и почти все ошибочны.

Отсюда предохранитель: включать запрет на уровне только после того, как там накопилось K
пустых ходов. Здесь K подбирается по записи: сколько срезов остаётся и какой ценой.

usage: .venv/bin/python scripts/noop_guard_fuse.py
"""
from __future__ import annotations

import hashlib
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

THRESHOLDS = [0, 5, 10, 20, 30, 50, 80]


def grid_of(fr):
    return None if fr is None or not fr.frame else np.asarray(fr.frame[-1], dtype=np.int16)


def nframes(fr):
    return 0 if fr is None or not fr.frame else len(fr.frame)


def act_key(a):
    d, n = a.get("data"), a.get("id") or "?"
    return "%s(%s,%s)" % (n, d["x"], d["y"]) if isinstance(d, dict) and "x" in d else n


def main() -> None:
    bench = json.load(open(ROOT / "runs/flash_v1_phaseA/benchmark.json", encoding="utf-8"))
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL,
                         environments_dir=str(ROOT / "environment_files"))
    # для каждой игры собираем события «этот ход был бы срезан» с числом пустых ходов
    # на уровне К ЭТОМУ МОМЕНТУ и с тем, сработал ли ход на самом деле
    events: dict[str, list[tuple[int, bool]]] = defaultdict(list)
    total_moves = 0
    for gr in bench["game_runs"]:
        env = arc.make(gr["game_id"])
        fr = env.reset()
        g = grid_of(fr)
        if g is None:
            continue
        lv = getattr(fr, "levels_completed", 0) or 0
        seen: set[tuple[str, str]] = set()
        dead_so_far = 0
        for rec in (gr.get("history") or []):
            a = rec.get("action") or {}
            if not a.get("id"):
                continue
            sig = (hashlib.blake2b(g.tobytes(), digest_size=8).hexdigest(), act_key(a))
            would_cut = sig in seen
            try:
                fr2 = env.step(GameAction[a["id"]] if a["id"] != "RESET" else GameAction.RESET,
                               data=a.get("data"))
            except Exception:
                break
            g2 = grid_of(fr2)
            lv2 = (getattr(fr2, "levels_completed", lv) or lv) if fr2 is not None else lv
            changed = (g2 is None) or (g2.shape != g.shape) or bool(np.any(g2 != g)) or lv2 != lv
            total_moves += 1
            if would_cut:
                events[gr["game_id"][:4]].append((dead_so_far, bool(changed)))
            if not changed:
                dead_so_far += 1
                if nframes(fr2) <= 1:
                    seen.add(sig)
            if lv2 != lv:
                seen.clear()
                dead_so_far = 0          # новый уровень — счётчик застревания с нуля
            if g2 is None:
                break
            g, lv = g2, lv2

    print("Порог K = сколько пустых ходов должно накопиться на уровне, прежде чем запрет включится.\n")
    print("%4s %9s %10s %7s %9s %s" % ("K", "срезали", "ошибочно", "доля", "% ходов", "в каких играх"))
    for k in THRESHOLDS:
        cut = wrong = 0
        per_game = {}
        for gid, evs in events.items():
            c = sum(1 for d, _ in evs if d >= k)
            w = sum(1 for d, ch in evs if d >= k and ch)
            if c:
                per_game[gid] = (c, w)
            cut += c
            wrong += w
        games = ",".join("%s %d/%d" % (g, v[0], v[1]) for g, v in
                         sorted(per_game.items(), key=lambda x: -x[1][0])[:4])
        print("%4d %9d %10d %6.0f%% %8.1f%%  %s"
              % (k, cut, wrong, 100 * wrong / max(cut, 1), 100 * cut / total_moves, games))
    print("\n(в колонке игр: срезали/ошибочно)")


if __name__ == "__main__":
    main()

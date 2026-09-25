"""Сколько вернёт правило «если клики в этой игре не работают — перестань кликать» (25.09).

Откуда. В sk48 модель сделала 284 хода и не взяла ни одного уровня, потратив 165 ходов (58%) на клики,
которые в этой игре ничего не делают: первый уровень решается 14 ходами ОДНИМИ простыми действиями
(runs/bfs_originals.json). Это не «повторяется» и не «ходит по кругу» — это выбор неверной модальности:
агент тычет мышью там, где надо нажимать стрелки.

Правило, которое проверяем (его в обвязке НЕТ): считаем подряд идущие клики, ничего не изменившие.
Как только их стало N, клики в этой игре объявляются бесполезными и запрещаются до конца партии.
Считаем: сколько ходов освободилось бы и не пострадали бы игры, где клики РАБОТАЮТ.

Опасность, которую замер обязан поймать: в большинстве игр клик — главное орудие. Если правило
сработает там, где клики нужны, оно отнимет уровни. Поэтому отдельно считается, сколько ПОЛЕЗНЫХ
кликов (изменивших экран) оказалось бы под запретом.

Порог: строить, если возвращается >= 15% ходов И ни одного полезного клика не запрещено в играх,
где клики работают.

usage: .venv/bin/python scripts/click_modality_bound.py
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

    for N in (8, 15, 30):
        print("\n=== порог: %d бесполезных кликов подряд -> клики запрещены ===" % N)
        print("%-6s %6s %7s %10s %12s %9s" % ("игра", "ходов", "кликов", "сэкономим", "полезных под", "уровней"))
        tot = saved = lost = 0
        rows = []
        for gr in bench["game_runs"]:
            gid = gr["game_id"]
            env = arc.make(gid)
            fr = env.reset()
            g = frame_of(fr)
            if g is None:
                continue
            score = getattr(fr, "score", 0)
            streak = 0
            banned_from = None
            n = clicks = save = lose = 0
            for i, rec in enumerate(gr.get("history") or []):
                act = (rec.get("action") or {})
                name, data = act.get("id"), act.get("data")
                if not name:
                    continue
                try:
                    fr2 = env.step(GameAction[name] if name != "RESET" else GameAction.RESET, data=data)
                except Exception:
                    break
                g2 = frame_of(fr2)
                sc2 = getattr(fr2, "score", score) if fr2 is not None else score
                changed = (g2 is None) or (g2.shape != g.shape) or bool(np.any(g2 != g)) or sc2 != score
                n += 1
                is_click = name == "ACTION6"
                if is_click:
                    clicks += 1
                    if banned_from is not None:          # клик уже был бы запрещён
                        save += 1
                        if changed:
                            lose += 1                    # а он был ПОЛЕЗНЫЙ -- это потеря
                    else:
                        streak = 0 if changed else streak + 1
                        if streak >= N:
                            banned_from = i
                if g2 is None:
                    break
                g, score = g2, sc2
            rows.append((gid[:4], n, clicks, save, lose, gr.get("levels_completed") or 0))
            tot += n; saved += save; lost += lose

        for r in sorted(rows, key=lambda x: -x[3])[:8]:
            print("%-6s %6d %7d %10d %12d %9d" % r)
        print("  ВСЕГО: ходов %d | освободится %d (%.1f%%) | из них ПОЛЕЗНЫХ кликов запрещено %d"
              % (tot, saved, 100 * saved / max(tot, 1), lost))
        p = saved / max(tot, 1)
        print("  потолок по баллу не более %+.1f%% | полезных потеряно %d -> %s"
              % (100 * ((1 / (1 - p)) ** 2 - 1), lost,
                 "СТРОИТЬ" if (p >= 0.15 and lost == 0) else "не проходит порог"))


if __name__ == "__main__":
    main()

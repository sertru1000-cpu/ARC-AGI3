"""Те же 41 «нарушения детерминизма» — это анимация? (25.09)

scripts/same_board_noop_bound.py считает ход пустым, если ПОСЛЕДНИЙ кадр совпал с доской до
хода. По этому признаку 209 ходов — повтор пустого при той же доске, и 41 из них всё-таки
что-то изменил. Для детерминированного движка это невозможно, значит признак «пусто» неверен.

Подозрение: анимация. Движок возвращает на один ход несколько кадров; в анимации отказа первый
и последний кадр совпадают по построению, доска «не меняется», но ход сработал. Ровно ради этого
в форке заведён frame_count, и NoopGuard.observe не записывает многокадровый ход как пустой.

Здесь тот же подсчёт, но с этим правилом: ход считается пустым, только если кадр ОДИН и доска
не изменилась. Если 41 нарушение обратится в ноль — признак верен, а запрет безопасен.

usage: .venv/bin/python scripts/same_board_noop_animated.py
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
logging.disable(logging.ERROR)
import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402


def grid_of(fr):
    return None if fr is None or not fr.frame else np.asarray(fr.frame[-1], dtype=np.int16)


def frames_of(fr):
    return 0 if fr is None or not fr.frame else len(fr.frame)


def act_key(a):
    d = a.get("data")
    n = a.get("id") or "?"
    return "%s(%s,%s)" % (n, d["x"], d["y"]) if isinstance(d, dict) and "x" in d else n


def run(use_animation_rule: bool):
    bench = json.load(open(ROOT / "runs/flash_v1_phaseA/benchmark.json", encoding="utf-8"))
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL,
                         environments_dir=str(ROOT / "environment_files"))
    tot = cut = broke = 0
    for gr in bench["game_runs"]:
        env = arc.make(gr["game_id"])
        fr = env.reset()
        g = grid_of(fr)
        if g is None:
            continue
        lv = getattr(fr, "levels_completed", 0) or 0
        seen: set[tuple[str, str]] = set()
        for rec in (gr.get("history") or []):
            act = rec.get("action") or {}
            if not act.get("id"):
                continue
            sig = (hashlib.blake2b(g.tobytes(), digest_size=8).hexdigest(), act_key(act))
            would_cut = sig in seen
            try:
                fr2 = env.step(GameAction[act["id"]] if act["id"] != "RESET" else GameAction.RESET,
                               data=act.get("data"))
            except Exception:
                break
            g2 = grid_of(fr2)
            lv2 = (getattr(fr2, "levels_completed", lv) or lv) if fr2 is not None else lv
            changed = (g2 is None) or (g2.shape != g.shape) or bool(np.any(g2 != g)) or lv2 != lv
            animated = frames_of(fr2) > 1
            tot += 1
            if would_cut:
                cut += 1
                if changed:
                    broke += 1
            # вот единственное отличие: анимированный ход пустым не считается
            if not changed and not (use_animation_rule and animated):
                seen.add(sig)
            if lv2 != lv:
                seen.clear()
            if g2 is None:
                break
            g, lv = g2, lv2
    return tot, cut, broke


def main() -> None:
    print("%-34s %7s %9s %12s" % ("правило «пусто»", "ходов", "срезали", "нарушений"))
    for rule, label in ((False, "только доска (прежний замер)"), (True, "доска И один кадр (форк)")):
        tot, cut, broke = run(rule)
        print("%-34s %7d %9d %12d   (%.1f%% ходов)" % (label, tot, cut, broke, 100 * cut / max(tot, 1)))


if __name__ == "__main__":
    main()

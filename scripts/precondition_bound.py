"""Как часто действие СНАЧАЛА не работает, а ПОТОМ начинает (25.09, проверка совета критика).

Совет критика: промпт учит связке «действие -> эффект», а в играх встречается «предусловие -> действие ->
эффект» (sk48: тянуть фигуру можно только после того, как её пронзили). Модель, увидев пустой ход,
записывает действие в бесполезные и больше к нему не возвращается — отсюда 58% пустых кликов.

Прежде чем трогать промпт, меряем ПРЕДПОСЫЛКУ: существуют ли в наших записях такие действия вообще.
Для каждого действия (с координатой для кликов) на каждом уровне смотрим последовательность исходов:
  * «мёртвое» — не сработало ни разу;
  * «сразу рабочее» — сработало с первого раза;
  * «С ПРЕДУСЛОВИЕМ» — сначала не работало, потом заработало. Вот это и есть цель совета.

Дополнительно, и это важнее всего: сколько ходов модель потратила ПОСЛЕ последнего пустого исхода
такого действия, но ДО того, как оно заработало — то есть сколько времени она ходила мимо.

Порог (как у прошлых веток): совет стоит внедрять, если действия с предусловием встречаются
достаточно часто, чтобы затрагивать >= 15% ходов.

usage: .venv/bin/python scripts/precondition_bound.py
"""
from __future__ import annotations

import collections
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


def act_key(a, by_class: bool = False):
    """by_class=True: все клики считаются ОДНИМ действием. Предусловие относится к КЛАССУ
    («клик по фигуре не работает, пока она не пронзена»), а не к конкретной координате, —
    строгий ключ с координатами такие случаи просто не видит."""
    name = a.get("id") or "?"
    d = a.get("data")
    if by_class:
        return name
    return "%s(%s,%s)" % (name, d["x"], d["y"]) if isinstance(d, dict) and "x" in d else name


def main(by_class: bool = False) -> None:
    bench = json.load(open(ROOT / "runs/flash_v1_phaseA/benchmark.json", encoding="utf-8"))
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL,
                         environments_dir=str(ROOT / "environment_files"))
    tot = 0
    kinds = collections.Counter()
    wasted = 0          # ходы между первым пустым исходом действия и его первым удачным
    rows = []
    for gr in bench["game_runs"]:
        gid = gr["game_id"]
        env = arc.make(gid)
        fr = env.reset()
        g = frame_of(fr)
        if g is None:
            continue
        lv = getattr(fr, "levels_completed", 0) or 0   # ЛОВУШКА 25.09: у кадра НЕТ поля score
        level = 0
        # действие -> список (номер хода, сработало ли) в пределах уровня
        seq: dict[str, list[tuple[int, bool]]] = collections.defaultdict(list)
        n = 0
        per_game_pre = 0
        for t, rec in enumerate(gr.get("history") or []):
            act = (rec.get("action") or {})
            if not act.get("id"):
                continue
            try:
                fr2 = env.step(GameAction[act["id"]] if act["id"] != "RESET" else GameAction.RESET,
                               data=act.get("data"))
            except Exception:
                break
            g2 = frame_of(fr2)
            lv2 = (getattr(fr2, "levels_completed", lv) or lv) if fr2 is not None else lv
            changed = (g2 is None) or (g2.shape != g.shape) or bool(np.any(g2 != g)) or lv2 != lv
            n += 1
            seq[act_key(act, by_class)].append((t, bool(changed)))
            if lv2 != lv:
                # уровень закончился: разбираем накопленное
                for key, hist in seq.items():
                    outs = [c for _, c in hist]
                    if not any(outs):
                        kinds["мёртвое"] += 1
                    elif outs[0]:
                        kinds["сразу рабочее"] += 1
                    else:
                        kinds["С ПРЕДУСЛОВИЕМ"] += 1
                        per_game_pre += 1
                        first_dead = next(i for i, (_, c) in enumerate(hist) if not c)
                        first_live = next(i for i, (_, c) in enumerate(hist) if c)
                        wasted += hist[first_live][0] - hist[first_dead][0]
                seq.clear()
                level += 1
            if g2 is None:
                break
            g, lv = g2, lv2
        # незавершённый уровень тоже разбираем
        for key, hist in seq.items():
            outs = [c for _, c in hist]
            if not any(outs):
                kinds["мёртвое"] += 1
            elif outs[0]:
                kinds["сразу рабочее"] += 1
            else:
                kinds["С ПРЕДУСЛОВИЕМ"] += 1
                per_game_pre += 1
                first_dead = next(i for i, (_, c) in enumerate(hist) if not c)
                first_live = next(i for i, (_, c) in enumerate(hist) if c)
                wasted += hist[first_live][0] - hist[first_dead][0]
        rows.append((gid[:4], n, per_game_pre, gr.get("levels_completed") or 0))
        tot += n

    print("%-6s %6s %14s %8s" % ("игра", "ходов", "с предусловием", "уровней"))
    for r in sorted(rows, key=lambda x: -x[2])[:8]:
        print("%-6s %6d %14d %8d" % r)
    total_kinds = sum(kinds.values())
    print("\nразличных действий (в пределах уровня): %d" % total_kinds)
    for k in ("сразу рабочее", "мёртвое", "С ПРЕДУСЛОВИЕМ"):
        print("  %-16s %5d (%.1f%%)" % (k, kinds[k], 100 * kinds[k] / max(total_kinds, 1)))
    print("\nходов между первым пустым и первым удачным исходом таких действий: %d (%.1f%% всех ходов)"
          % (wasted, 100 * wasted / max(tot, 1)))
    p = wasted / max(tot, 1)
    print("ПОРОГ 15%% ->", "ЕСТЬ ЧТО ЛЕЧИТЬ" if p >= 0.15 else "предпосылка слабая")


if __name__ == "__main__":
    import sys as _s
    by_class = "--by-class" in _s.argv
    print("=== ключ действия: %s ===" % ("КЛАСС (все клики вместе)" if by_class else "точный, с координатой"))
    main(by_class)

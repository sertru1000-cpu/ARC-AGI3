"""Нужна ли история ГЛУБЖЕ тридцати ходов (25.09, проверка совета про «полный лог + поиск кодом»).

Постановка. Обвязка держит в промпте последние 30 ходов. Совет: вести полный лог и искать по нему
программно. Прежде чем строить поиск, надо узнать, есть ли ЧТО искать — то есть как часто ответ на
вопрос «что произойдёт от этого хода здесь» лежит в истории ДАЛЬШЕ окна в 30 ходов и отсутствует внутри.

Как считаем, по настоящим партиям (runs/flash_v1_phaseA, 25 игр, 4380 ходов). Для каждого хода t:
  * ключ = (номер уровня, хеш доски, действие);
  * ход СЧИТАЕТСЯ «глубоким», если такой же ключ уже встречался раньше, но ТОЛЬКО за пределами
    последних 30 ходов — то есть окно ответа не содержит, а полный лог содержит.
Плюс отдельно: сколько таких ходов оказались ПОВТОРОМ ошибки (прошлый раз ничего не изменилось).

Порог (как у прошлых веток): строить, если глубокая история могла бы подсказать >= 15% ходов.

usage: .venv/bin/python scripts/history_depth_bound.py
"""
from __future__ import annotations

import hashlib
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

WINDOW = 30      # столько ходов обвязка держит в промпте


def frame_of(fr):
    if fr is None or not fr.frame:
        return None
    return np.asarray(fr.frame[-1], dtype=np.int16)


def board_key(g, level):
    return "%d:%s" % (level, hashlib.blake2b(g.tobytes(), digest_size=8).hexdigest())


def act_key(a):
    name = a.get("id") or "?"
    d = a.get("data")
    return "%s(%s,%s)" % (name, d["x"], d["y"]) if isinstance(d, dict) and "x" in d else name


def main() -> None:
    bench = json.load(open(ROOT / "runs/flash_v1_phaseA/benchmark.json", encoding="utf-8"))
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL,
                         environments_dir=str(ROOT / "environment_files"))
    tot = deep = deep_useless = inside = 0
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
        seen: dict[tuple[str, str], list[tuple[int, bool]]] = {}
        n = d = du = ins = 0
        for t, rec in enumerate(gr.get("history") or []):
            act = (rec.get("action") or {})
            if not act.get("id"):
                continue
            key = (board_key(g, level), act_key(act))
            try:
                fr2 = env.step(GameAction[act["id"]] if act["id"] != "RESET" else GameAction.RESET,
                               data=act.get("data"))
            except Exception:
                break
            g2 = frame_of(fr2)
            lv2 = (getattr(fr2, "levels_completed", lv) or lv) if fr2 is not None else lv
            changed = (g2 is None) or (g2.shape != g.shape) or bool(np.any(g2 != g)) or lv2 != lv
            n += 1
            prior = seen.get(key)
            if prior:
                last_t, last_changed = prior[-1]
                if t - last_t > WINDOW:
                    d += 1                       # ответ есть, но ДАЛЬШЕ окна
                    if not last_changed:
                        du += 1                  # и это повтор заведомо пустого хода
                else:
                    ins += 1                     # ответ внутри окна, поиск не нужен
            seen.setdefault(key, []).append((t, bool(changed)))
            if lv2 != lv:
                level += 1
                seen.clear()                     # новый уровень — прежний опыт про другую доску
            if g2 is None:
                break
            g, lv = g2, lv2
        rows.append((gid[:4], n, d, du, ins, gr.get("levels_completed") or 0))
        tot += n; deep += d; deep_useless += du; inside += ins

    print("%-6s %6s %8s %12s %10s %8s" % ("игра", "ходов", "глубже30", "из них пустых", "внутри30", "уровней"))
    for r in sorted(rows, key=lambda x: -x[2])[:8]:
        print("%-6s %6d %8d %12d %10d %8d" % r)
    print("\nВСЕГО ходов %d" % tot)
    print("  ответ был ГЛУБЖЕ окна 30 (полный лог помог бы):  %d (%.1f%%)" % (deep, 100 * deep / max(tot, 1)))
    print("  из них повтор заведомо пустого хода:             %d (%.1f%%)" % (deep_useless, 100 * deep_useless / max(tot, 1)))
    print("  ответ был ВНУТРИ окна (поиск не нужен):          %d (%.1f%%)" % (inside, 100 * inside / max(tot, 1)))
    p = deep / max(tot, 1)
    print("\nПОРОГ 15%% ->", "СТРОИТЬ полный лог с поиском" if p >= 0.15 else "ЗАКРЫТЬ: искать почти нечего")


if __name__ == "__main__":
    main()

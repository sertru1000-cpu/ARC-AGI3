"""Сколько дал бы ЖЁСТКИЙ ЗАПРЕТ заведомо пустых действий — и сколько бы он стоил (25.09).

Чего мы не делали ни разу. Все 22 закрытые ветки — это ПОДСКАЗКИ: обвязка сообщала модели факт
(«ты здесь был», «это сочетание пробовал», «мышь не работает»), а решение оставляла ей. Жёсткого
запрета — «этот ход нельзя, он заведомо пустой» — не пробовали.

Что считаем на настоящих партиях (runs/flash_v1_phaseA, 25 игр, 4380 ходов). Для каждого хода:
  * ключ = (уровень, действие с координатой клика);
  * ход — КАНДИДАТ НА ЗАПРЕТ, если этот же ключ раньше на этом уровне уже давал «ничего»;
  * из кандидатов отдельно считаем те, что ВСЁ-ТАКИ СРАБОТАЛИ — это цена ошибки запрета.

Второй счёт — главный. Сегодняшний перебор (scripts/precondition_search.py) показал, что 40% мёртвых
действий оживают после одного хода. Значит запрет может отнять работающую механику, и надо знать,
как часто. Подсказка этого риска не несёт: модель вправе её проигнорировать.

Порог: строить, если запрет возвращает >= 15% ходов И доля ошибочных запретов близка к нулю.

usage: .venv/bin/python scripts/hard_ban_bound.py
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


def act_key(a):
    name = a.get("id") or "?"
    d = a.get("data")
    return "%s(%s,%s)" % (name, d["x"], d["y"]) if isinstance(d, dict) and "x" in d else name


def main() -> None:
    bench = json.load(open(ROOT / "runs/flash_v1_phaseA/benchmark.json", encoding="utf-8"))
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL,
                         environments_dir=str(ROOT / "environment_files"))
    tot = cand = wrong = 0
    rows = []
    by_kind = collections.Counter()
    for gr in bench["game_runs"]:
        gid = gr["game_id"]
        env = arc.make(gid)
        fr = env.reset()
        g = frame_of(fr)
        if g is None:
            continue
        score = getattr(fr, "score", 0)
        level = 0
        dead: set[str] = set()          # ключи, уже давшие «ничего» на этом уровне
        n = c = w = 0
        for rec in (gr.get("history") or []):
            act = (rec.get("action") or {})
            if not act.get("id"):
                continue
            key = act_key(act)
            banned = key in dead
            try:
                fr2 = env.step(GameAction[act["id"]] if act["id"] != "RESET" else GameAction.RESET,
                               data=act.get("data"))
            except Exception:
                break
            g2 = frame_of(fr2)
            sc2 = getattr(fr2, "score", score) if fr2 is not None else score
            changed = (g2 is None) or (g2.shape != g.shape) or bool(np.any(g2 != g)) or sc2 != score
            n += 1
            if banned:
                c += 1
                if changed:
                    w += 1              # запрет отнял бы РАБОТАЮЩИЙ ход
                    by_kind[act.get("id")] += 1
            if not changed:
                dead.add(key)
            else:
                dead.discard(key)       # сработало — больше не считаем мёртвым
            if sc2 != score:
                level += 1
                dead.clear()
            if g2 is None:
                break
            g, score = g2, sc2
        rows.append((gid[:4], n, c, w, gr.get("levels_completed") or 0))
        tot += n; cand += c; wrong += w

    print("%-6s %6s %10s %14s %8s" % ("игра", "ходов", "запретили", "ошибочно", "уровней"))
    for r in sorted(rows, key=lambda x: -x[2])[:8]:
        print("%-6s %6d %10d %14d %8d" % r)
    print("\nВСЕГО ходов %d" % tot)
    print("  кандидатов на запрет (повтор заведомо пустого): %d (%.1f%%)" % (cand, 100 * cand / max(tot, 1)))
    print("  из них СРАБОТАЛИ бы (запрет = ошибка):          %d (%.1f%% кандидатов)"
          % (wrong, 100 * wrong / max(cand, 1)))
    if by_kind:
        print("  ошибочные запреты по типу действия:", dict(by_kind.most_common(5)))
    p = (cand - wrong) / max(tot, 1)
    print("\nчистый возврат ходов: %.1f%%" % (100 * p))
    print("ПОРОГ 15%% и мало ошибок ->",
          "СТРОИТЬ" if (p >= 0.15 and wrong / max(cand, 1) < 0.05) else "не проходит")


if __name__ == "__main__":
    main()

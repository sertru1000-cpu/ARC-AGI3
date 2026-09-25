"""Повтор пустого действия при ТОЙ ЖЕ доске — сколько это ходов и безопасен ли запрет (25.09).

Идея владельца. Прошлый замер (`scripts/hard_ban_bound.py`) запрещал повтор действия, которое раньше
дало пусто, — и 35% таких запретов оказались ошибкой, потому что доска успевала измениться. Здесь
условие строже: запрещаем только если доска ПОБИТОВО ТА ЖЕ, что и в момент пустого исхода.

Почему это безопасно по построению: движок детерминирован (проверено, 561 повтор из 561). Одинаковое
состояние плюс одинаковое действие обязаны дать одинаковый результат. Значит ошибок быть не может,
и замер это проверяет отдельно — доля «сработавших» запретов должна выйти ровно нулевой.

Считаем на записях базы (runs/flash_v1_phaseA, 25 игр, 4380 ходов):
  * сколько ходов — повтор пустого действия при идентичной доске (их можно срезать);
  * сколько из них всё-таки что-то изменили (проверка детерминизма: ожидаем 0);
  * что это даёт по баллу: RHAE ~ (базлайн/ходы)^2.

usage: .venv/bin/python scripts/same_board_noop_bound.py
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


def frame_of(fr):
    if fr is None or not fr.frame:
        return None
    return np.asarray(fr.frame[-1], dtype=np.int16)


def board_hash(g):
    return hashlib.blake2b(g.tobytes(), digest_size=8).hexdigest()


def act_key(a):
    n = a.get("id") or "?"
    d = a.get("data")
    return "%s(%s,%s)" % (n, d["x"], d["y"]) if isinstance(d, dict) and "x" in d else n


def main() -> None:
    bench = json.load(open(ROOT / "runs/flash_v1_phaseA/benchmark.json", encoding="utf-8"))
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL,
                         environments_dir=str(ROOT / "environment_files"))
    tot = cut = broke = 0
    rows = []
    for gr in bench["game_runs"]:
        gid = gr["game_id"]
        env = arc.make(gid)
        fr = env.reset()
        g = frame_of(fr)
        if g is None:
            continue
        lv = getattr(fr, "levels_completed", 0) or 0
        seen: set[tuple[str, str]] = set()       # (хеш доски, действие) -> дало пусто
        n = c = b = 0
        for rec in (gr.get("history") or []):
            act = (rec.get("action") or {})
            if not act.get("id"):
                continue
            sig = (board_hash(g), act_key(act))
            would_cut = sig in seen
            try:
                fr2 = env.step(GameAction[act["id"]] if act["id"] != "RESET" else GameAction.RESET,
                               data=act.get("data"))
            except Exception:
                break
            g2 = frame_of(fr2)
            lv2 = (getattr(fr2, "levels_completed", lv) or lv) if fr2 is not None else lv
            changed = (g2 is None) or (g2.shape != g.shape) or bool(np.any(g2 != g)) or lv2 != lv
            n += 1
            if would_cut:
                c += 1
                if changed:
                    b += 1                        # детерминизм нарушен — такого быть не должно
            if not changed:
                seen.add(sig)
            if lv2 != lv:
                seen.clear()
            if g2 is None:
                break
            g, lv = g2, lv2
        rows.append((gid[:4], n, c, b, gr.get("levels_completed") or 0))
        tot += n; cut += c; broke += b

    print("%-6s %6s %10s %12s %8s" % ("игра", "ходов", "срезали", "нарушений", "уровней"))
    for r in sorted(rows, key=lambda x: -x[2])[:10]:
        print("%-6s %6d %10d %12d %8d" % r)
    print("\nВСЕГО ходов %d" % tot)
    print("  повтор пустого действия при ТОЙ ЖЕ доске: %d (%.1f%%)" % (cut, 100 * cut / max(tot, 1)))
    print("  из них что-то изменили (нарушение детерминизма): %d" % broke)
    p = cut / max(tot, 1)
    print("\nзапрет вернул бы %.1f%% ходов, ошибок по построению %s"
          % (100 * p, "НЕТ" if broke == 0 else "ЕСТЬ (%d) — проверить движок" % broke))
    print("в баллах не более (1/(1-p))^2 = %+.1f%%" % (100 * ((1 / (1 - p)) ** 2 - 1)))
    print("ПОРОГ 15%% ->", "СТРОИТЬ" if p >= 0.15 else "ниже порога")


if __name__ == "__main__":
    main()

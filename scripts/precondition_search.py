"""Существует ли ПРЕДУСЛОВИЕ, после которого «мёртвое» действие оживает (25.09).

Замер scripts/precondition_bound.py показал: случаи «сначала не работало, потом заработало» есть, но
стоят 4.5% ходов. Он, однако, слеп к главному — к действиям, которые модель отвергла и БОЛЬШЕ НИКОГДА
не проверила. В её статистике они просто «мёртвые», и узнать, было ли у них предусловие, по записи нельзя.

Здесь это проверяется перебором на локальном движке (он детерминированно повторяет бой, 561/561):
берём стартовое состояние игры, находим действия, НЕ меняющие кадр, и смотрим, оживают ли они после
коротких последовательностей других действий (глубина 1..3). Модель не участвует, денег не стоит.

Что считаем:
  * мёртвых действий в стартовом состоянии;
  * из них оживших после предусловия длиной 1, 2, 3;
  * какое именно предусловие потребовалось (для понимания механики).

Это верхняя оценка пользы совета критика: если мёртвые действия не оживают ни после чего, то учить
модель искать предусловия не для чего.

usage: .venv/bin/python scripts/precondition_search.py [--depth 3] [--games sk48,tn36,sp80,g50t]
"""
from __future__ import annotations

import argparse
import copy
import itertools
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

SIMPLE = ["ACTION%d" % i for i in range(1, 6)]


def frame_of(fr):
    if fr is None or not fr.frame:
        return None
    return np.asarray(fr.frame[-1], dtype=np.int16)


def try_action(env, name, base):
    """Развернуть действие на копии среды: изменился ли кадр."""
    e = copy.deepcopy(env)
    fr = e.step(GameAction[name], data=None)
    g = frame_of(fr)
    if g is None:
        return None, False
    return e, (g.shape != base.shape) or bool(np.any(g != base))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--games", default="")
    a = ap.parse_args()

    bench = json.load(open(ROOT / "runs/flash_v1_phaseA/benchmark.json", encoding="utf-8"))
    want = [x.strip() for x in a.games.split(",") if x.strip()]
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL,
                         environments_dir=str(ROOT / "environment_files"))

    total_dead = total_revived = 0
    for gr in bench["game_runs"]:
        gid = gr["game_id"]
        if want and gid[:4] not in want:
            continue
        env = arc.make(gid)
        fr = env.reset()
        base = frame_of(fr)
        if base is None:
            continue

        dead = []
        for name in SIMPLE:
            _, changed = try_action(env, name, base)
            if not changed:
                dead.append(name)
        if not dead:
            print("%-5s мёртвых действий в старте нет" % gid[:4], flush=True)
            continue

        revived = {}
        for d in dead:
            found = None
            for k in range(1, a.depth + 1):
                for combo in itertools.product(SIMPLE, repeat=k):
                    e = copy.deepcopy(env)
                    cur = base
                    ok = True
                    for step in combo:
                        fr2 = e.step(GameAction[step], data=None)
                        g2 = frame_of(fr2)
                        if g2 is None:
                            ok = False
                            break
                        cur = g2
                    if not ok:
                        continue
                    _, changed = try_action(e, d, cur)
                    if changed:
                        found = combo
                        break
                if found:
                    break
            if found:
                revived[d] = found
        total_dead += len(dead)
        total_revived += len(revived)
        print("%-5s мёртвых %d -> ожило %d %s"
              % (gid[:4], len(dead), len(revived),
                 {k: "+".join(v) for k, v in revived.items()} if revived else ""), flush=True)

    print("\nИТОГО: мёртвых действий %d, ожило после предусловия %d (%.0f%%)"
          % (total_dead, total_revived, 100 * total_revived / max(total_dead, 1)))
    print("ВЫВОД:", "предусловия РЕАЛЬНЫ — учить модель их искать имеет смысл"
          if total_revived else "мёртвое действие мертво всегда — совет не о чем")


if __name__ == "__main__":
    main()

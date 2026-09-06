"""Померить, какие игры вообще переносятся случайной раскладкой.

Зачем. Генератор `regen_short_levels.py` строит лабиринт, расставляет по нему
объекты исходного уровня и проверяет перебором. Для игр, где задача -- это
ПУТЬ, это работает: cv01, cl01, cm01, ic01 берут цель почти точно. Для
остальных случайная раскладка не образует нужной конфигурации почти никогда:
sk01 толкает ящик на метку и дала ноль решений из двенадцати, bx01 взрывом
поражает соседние мишени, lz01 требует, чтобы луч прошёл через все зеркала.

Я классифицировал игры по догадке трижды и трижды ошибся, каждый раз узнавая
правду из одиннадцатиминутного прогона. Эта проба отвечает на тот же вопрос
за полминуты: сколько случайных раскладок из N оказались решаемы вообще, без
всякой цели по стоимости.

Читать так:
  доля 0        -- случайной раскладкой игра не переносится, нужна
                   конструктивная сборка с заложенным решением;
  доля до ~0.2  -- переносится, но дорого: цель по стоимости потребует
                   большого бюджета попыток;
  доля выше 0.3 -- ставить в очередь смело.

usage:
    python scripts/probe_regen_fitness.py
    python scripts/probe_regen_fitness.py --tries 30 --game sk01
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import regen_short_levels as R  # noqa: E402
import verify_short_baselines as V  # noqa: E402


def probe(gid: str, tries: int, limit: int, seed: int) -> tuple[int, int, list[int]]:
    cls, _md, mod = V.load(gid)
    name, levels, rows_of, shape = R.levels_of(mod)
    if levels is None:
        return -1, 0, []
    src = rows_of(levels[0])
    syms = Counter(ch for row in src for ch in row if ch not in (R.FLOOR, R.WALL))
    mod.GRID, mod.CELL = R.NEW_GRID, R.NEW_CELL
    rng = random.Random(seed)
    solved, attempted, costs = 0, 0, []
    for _ in range(tries):
        rows = R.place(rng, R.maze(rng, R.NEW_GRID), Counter(syms))
        if rows is None:
            continue
        attempted += 1
        saved = levels[:]
        try:
            levels[0] = dict(levels[0], rows=rows) if shape == "dict" else rows
            opt, _ = V.optimum(cls, mod, 0, limit)
        except Exception:
            opt = None
        finally:
            levels[:] = saved
        if opt:
            solved += 1
            costs.append(opt)
    return solved, attempted, costs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--game", action="append")
    ap.add_argument("--tries", type=int, default=15)
    ap.add_argument("--limit", type=int, default=12_000)
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()

    gids = args.game or sorted(
        p.split(os.sep)[-3] for p in glob.glob(str(ROOT / "our_games" / "*" / "*" / "metadata.json")))
    rows = []
    print(f"{'игра':7}{'решаемых':>10}{'доля':>7}   стоимости найденных")
    for gid in gids:
        try:
            solved, attempted, costs = probe(gid, args.tries, args.limit, args.seed)
        except Exception as exc:
            print(f"{gid:7}{'сбой':>10}          {type(exc).__name__}: {str(exc)[:40]}")
            continue
        if solved < 0:
            print(f"{gid:7}{'—':>10}          уровни не картой, проба не применима")
            continue
        frac = solved / attempted if attempted else 0.0
        shown = ", ".join(str(c) for c in sorted(costs)[:6])
        print(f"{gid:7}{solved:>4}/{attempted:<5}{frac:>7.2f}   {shown}")
        rows.append((gid, frac))

    good = [g for g, f in rows if f >= 0.3]
    slow = [g for g, f in rows if 0 < f < 0.3]
    dead = [g for g, f in rows if f == 0]
    print(f"\nставить в очередь смело ({len(good)}): {' '.join(good)}")
    print(f"переносятся дорого   ({len(slow)}): {' '.join(slow)}")
    print(f"случайной раскладкой НЕ переносятся ({len(dead)}): {' '.join(dead)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

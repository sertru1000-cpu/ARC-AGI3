"""Обрезка прогона по времени: «что было бы, останови мы его на N секундах».

Зачем. Пробы идут с потолком 1800 с на игру, а полная база — 7920 с. Сравнивать
их напрямую нельзя, поэтому точка сравнения для проб строится ОБРЕЗКОЙ полной
базы. Файл `docs/base30_flash_v1.json` такую обрезку уже содержал, но считал балл
по ОФИЦИАЛЬНОЙ формуле (потолок 100), тогда как `final_score` в `benchmark.json`,
по которому меряются наши прогоны, считает харнесс Duck с потолком 115. Разница
ровно 15% и она била в нашу сторону: 3.06 против сопоставимых 3.32.

Как считает. Порядок действий берётся из `artifacts/<игра>_p0_events.jsonl`
(записи type=action: номер действия и флаг level_completed), время каждого
действия — из `history[i].wallclock_seconds` в `benchmark.json`. Действие,
взявшее уровень, засчитывается ТЕКУЩЕМУ уровню, а не следующему (в событии там
уже стоит номер нового уровня — эта ловушка стоила первой неверной сверки).

Проверка встроена: без обрезки восстановление обязано совпасть с
`actions_per_level` и `levels_completed` из `benchmark.json` по каждой игре.
Скрипт печатает результат сверки и отказывается писать файл, если она не прошла.

usage:
    .venv/bin/python scripts/truncate_run.py runs/flash_v1_phaseA --seconds 1800
    .venv/bin/python scripts/truncate_run.py runs/flash_v1_phaseA --seconds 1800 \
        --out docs/base30_flash_v1_h115.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

CAP_HARNESS = 115.0   # taaf/game.py: min(115, (base/actions)**2 * 100) — так считает benchmark.json
CAP_OFFICIAL = 100.0  # официальное описание датасета: отношение режется до 1.0 ДО квадрата


def actions(run: Path, gid: str) -> list[tuple[int, bool]]:
    f = glob.glob(str(run / "artifacts" / f"{gid}_p0_events.jsonl"))
    if not f:
        return []
    out = []
    for line in open(f[0], encoding="utf-8"):
        d = json.loads(line)
        if d.get("type") == "action":
            out.append((int(d["action_num"]), bool(d.get("level_completed"))))
    return out


def replay(run: Path, r: dict, cutoff: float) -> tuple[list[int], int]:
    """Действия по уровням и число взятых уровней к моменту cutoff секунд."""
    hist = r["history"]
    n = int(r["number_of_levels"])
    per = [0] * n
    lvl = done = 0
    for num, completed in actions(run, str(r["game_id"])):
        i = num - 1
        if i >= len(hist):
            break
        if float(hist[i].get("wallclock_seconds") or 0.0) > cutoff:
            break
        if lvl < n:
            per[lvl] += 1
        if completed:
            done += 1
            lvl += 1
    return per, done


def score(per: list[int], base: list[int], n: int, done: int, cap: float) -> float:
    num = den = 0.0
    for i in range(n):
        w = i + 1
        den += w
        if i < done and i < len(per) and per[i] and i < len(base):
            num += min(cap, (base[i] / per[i]) ** 2 * 100) * w
    return num / den if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--seconds", type=float, default=1800.0)
    ap.add_argument("--cap", choices=["harness", "official"], default="harness",
                    help="harness=115 (как benchmark.json, сопоставимо с нашими прогонами); official=100")
    ap.add_argument("--out", help="куда записать пер-игровые баллы (JSON)")
    a = ap.parse_args()
    run = Path(a.run)
    cap = CAP_HARNESS if a.cap == "harness" else CAP_OFFICIAL
    games = json.load(open(run / "benchmark.json", encoding="utf-8"))["game_runs"]

    mismatch = []
    for r in games:
        per, done = replay(run, r, float("inf"))
        if per != json.loads(str(r["actions_per_level"])) or done != int(r["levels_completed"]):
            mismatch.append(str(r["game_id"])[:4])
    print(f"сверка без обрезки: расхождений {len(mismatch)}/{len(games)}" + (f" {mismatch}" if mismatch else ""))
    if mismatch:
        print("восстановление не совпало с benchmark.json — обрезке доверять нельзя, файл не пишется")
        return 1

    rows = []
    for r in games:
        base = json.loads(str(r["base_actions_per_level"]))
        n = int(r["number_of_levels"])
        per, done = replay(run, r, a.seconds)
        rows.append((str(r["game_id"])[:4], score(per, base, n, done, cap), done, sum(per), n))

    n = len(rows)
    print(f"\nОБРЕЗКА {run.name} на {a.seconds:.0f} с (потолок уровня {cap:.0f})")
    print(f"  RHAE                    {sum(x[1] for x in rows)/n:.2f}")
    print(f"  первый уровень взят в   {sum(1 for x in rows if x[2] >= 1)}/{n} играх")
    print(f"  второй уровень взят в   {sum(1 for x in rows if x[2] >= 2)}/{n} играх")
    print(f"  действий на игру        {sum(x[3] for x in rows)/n:.1f}")
    for gid, s, done, acts, tot in sorted(rows):
        print(f"    {gid}  балл {s:7.3f}  уровней {done}/{tot}  действий {acts}")

    if a.out:
        json.dump({g: round(s, 3) for g, s, _, _, _ in rows}, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\nзаписано: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

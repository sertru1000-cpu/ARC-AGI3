"""Парное сравнение двух прогонов по играм: знаковый тест + парный бутстрэп + ROPE.

История. Знаковый тест (Gemini р.22, 06.09): дельта по играм, «мёртвые» (0 уровней в обоих)
и «потолочные» (все уровни в обоих) выброшены, вердикт по числу побед/поражений и медиане.
12.09 второй критик (ChatGPT р.1) указал, что знаковый тест выбрасывает величину: +0.01 и +5.0
весят одинаково. Добавлен ПАРНЫЙ БУТСТРЭП по играм и ROPE (область практической
эквивалентности): среднее Δ по ВСЕМ играм (балл прогона — среднее по всем, мёртвые входят),
10 000 пересборок игр с возвратом, 95% интервал и P(Δ > 0). Вердикт по ROPE:
интервал целиком выше +rope — ПОЛЬЗА; целиком ниже −rope — ВРЕД; целиком внутри ±rope —
НЕРАЗЛИЧИМО; иначе — НЕОПРЕДЕЛЁННО (не хватает данных, а не «шум»).

Ширина ROPE по умолчанию 1.0 балла на публичных 25: разброс той же сборки 6.76 / 9.00 / 9.43 /
10.25 даёт sd ≈ 1.5, ROPE ≈ 0.7 sd — та же пропорция, что предложенные 0.35 при боевом sd 0.50.

Два счётчика балла: по умолчанию `final_score` харнесса из benchmark.json (так судились прогоны
до 11.09); `--h115` — наш пересчёт с потолком 115 из actions_per_level (так судились 2а/2б,
база 10.25). Смешивать нельзя: у одного и того же прогона 9.43 и 10.25.

usage:
    .venv/bin/python scripts/paired_delta.py runs/flash_v1_phaseA runs/flash_avo_phaseA
    .venv/bin/python scripts/paired_delta.py --h115 runs/flash_v1_phaseA runs/flash_input_v1 runs/flash_carry_v1
    (несколько новых прогонов — сводная таблица в конце)
"""
from __future__ import annotations
import argparse
import json
import random
import statistics
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load(d: str, h115: bool) -> dict[str, tuple[float, int, int]]:
    b = json.load(open(Path(d) / "benchmark.json", encoding="utf-8"))
    runs = b["game_runs"] if isinstance(b.get("game_runs"), list) else list(b["game_runs"].values())
    if int(b.get("n_passes") or 1) > 1:
        # несколько проходов: берём ПЕРВЫЙ проход каждой игры, иначе сравнение нечестное
        n_games = len(runs) // int(b["n_passes"])
        runs = runs[:n_games]
    out = {}
    for r in runs:
        if h115:
            from truncate_run import CAP_HARNESS, score
            s = score(json.loads(str(r["actions_per_level"])), json.loads(str(r["base_actions_per_level"])),
                      int(r["number_of_levels"]), int(r["levels_completed"]), CAP_HARNESS)
        else:
            s = float(r.get("final_score") or 0.0)
        out[str(r["game_id"])[:4]] = (float(s), int(r.get("levels_completed") or 0), int(r.get("number_of_levels") or 0))
    return out


def sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap(deltas: list[float], n: int = 10000, seed: int = 12) -> tuple[float, float, float, float]:
    rng = random.Random(seed)
    m = len(deltas)
    means = []
    for _ in range(n):
        means.append(sum(deltas[rng.randrange(m)] for _ in range(m)) / m)
    means.sort()
    return (statistics.mean(deltas), means[int(0.025 * n)], means[int(0.975 * n) - 1],
            sum(x > 0 for x in means) / n)


def compare(base: dict, new: dict, rope: float, verbose: bool = True) -> dict:
    games = sorted(set(base) & set(new))
    dead = [g for g in games if base[g][1] == 0 and new[g][1] == 0]
    ceil = [g for g in games if base[g][2] and base[g][1] >= base[g][2] and new[g][1] >= new[g][2]]
    keep = [g for g in games if g not in dead and g not in ceil]
    rows = [(g, base[g][0], new[g][0], new[g][0] - base[g][0], base[g][1], new[g][1], new[g][1] - base[g][1]) for g in keep]
    wins = [r for r in rows if r[3] > 0.5]; losses = [r for r in rows if r[3] < -0.5]
    lw = [r for r in rows if r[6] > 0]; ll = [r for r in rows if r[6] < 0]
    nz = sorted(r[3] for r in rows if abs(r[3]) > 0.5); med = nz[len(nz) // 2] if nz else 0.0
    p = sign_p(len(wins), len(losses))
    all_d = [new[g][0] - base[g][0] for g in games]
    mean_d, lo, hi, p_pos = bootstrap(all_d)
    if lo > rope:
        verdict = "ПОЛЬЗА"
    elif hi < -rope:
        verdict = "ВРЕД"
    elif -rope <= lo and hi <= rope:
        verdict = "НЕРАЗЛИЧИМО"
    else:
        verdict = "НЕОПРЕДЕЛЁННО"
    old = "WIN" if (len(wins) >= 4 and len(losses) <= 1 and med > 10) else ("LOSS" if len(losses) >= 4 and len(wins) <= 1 else "NOISE")
    if verbose:
        print(f"games {len(games)}, dead {len(dead)} {dead}, ceiling {len(ceil)} {ceil}, compared {len(keep)}")
        print(f"{'game':5s} {'base':>6s} {'new':>6s} {'delta':>7s}  {'lv base':>7s} {'lv new':>6s}")
        for r in sorted(rows, key=lambda r: -r[3]):
            print(f"{r[0]:5s} {r[1]:6.1f} {r[2]:6.1f} {r[3]:+7.1f}  {r[4]:7d} {r[5]:6d}")
        print(f"знаковый тест: побед {len(wins)}, поражений {len(losses)}, p = {p:.3f}, медиана ненулевых Δ {med:+.1f}; "
              f"уровни: побед {len(lw)}, поражений {len(ll)} (старый вердикт {old})")
        print(f"бутстрэп по {len(games)} играм: средняя Δ {mean_d:+.2f}, 95% [{lo:+.2f}, {hi:+.2f}], P(Δ>0) = {p_pos:.2f}; "
              f"ROPE ±{rope:.2f} → {verdict}")
    return {"mean_base": statistics.mean(v[0] for v in base.values()), "mean_new": statistics.mean(v[0] for v in new.values()),
            "wins": len(wins), "losses": len(losses), "p": p, "mean_d": mean_d, "lo": lo, "hi": hi,
            "p_pos": p_pos, "verdict": verdict, "old": old,
            "levels_base": sum(v[1] for v in base.values()), "levels_new": sum(v[1] for v in new.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new", nargs="+")
    ap.add_argument("--h115", action="store_true", help="наш счётчик с потолком 115 (как у 2а/2б), а не final_score харнесса")
    ap.add_argument("--rope", type=float, default=1.0, help="полуширина области практической эквивалентности, баллов")
    a = ap.parse_args()
    base = load(a.base, a.h115)
    summary = []
    for nd in a.new:
        print(f"\n=== {a.base} -> {nd} ({'h115' if a.h115 else 'final_score'})")
        r = compare(base, load(nd, a.h115), a.rope, verbose=len(a.new) == 1)
        summary.append((nd, r))
    if len(a.new) > 1:
        print(f"\n{'прогон':28s} {'база':>6s} {'новый':>6s} {'ур.':>7s} {'знак':>7s} {'p':>6s} {'Δ':>6s} {'95% интервал':>16s} {'P>0':>5s}  вердикт")
        for nd, r in summary:
            print(f"{nd[5:]:28s} {r['mean_base']:6.2f} {r['mean_new']:6.2f} {r['levels_base']:3d}/{r['levels_new']:<3d} "
                  f"{r['wins']:3d}/{r['losses']:<3d} {r['p']:6.3f} {r['mean_d']:+6.2f} [{r['lo']:+6.2f}, {r['hi']:+6.2f}] {r['p_pos']:5.2f}  {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

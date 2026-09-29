"""Сколько баллов стоит идея — до того, как тратить на неё прогон (25.09).

ЗАЧЕМ. Весь сентябрь мы мерили надстройки прогонами по 4.4 часа и получали «неопределённо».
25.09 выяснилось почему: три прогона ОДНОЙ И ТОЙ ЖЕ сборки дают средние разности -0.43, -2.24,
-2.67 при интервалах около ±4.5. Одна пара не видит эффект меньше пяти баллов при итоге 9.43.
То есть половина потраченной квоты ушла на замеры, исход которых был предрешён арифметикой.

Этот скрипт считает арифметику ЗАРАНЕЕ. На входе — что именно улучшится, на выходе — прибавка
к баллу и сравнение с коридором шума. Если прибавка ниже коридора, прогон покупать нечего:
он вернёт «неразличимо» при любом качестве идеи.

ФОРМУЛА. Взята из исходников боевой базы (taaf/game.py, _compute_final_score, зеркалит
arc_agi.scorecard.EnvironmentScoreCalculator v0.9.8), а не восстановлена по догадке:

    балл уровня = min(115, (базлайн / ходы)^2 * 100), если уровень взят хотя бы за один ход,
    вес уровня  = его номер, считая с единицы,
    балл игры   = SUM(балл_i * вес_i) / SUM(вес_i по ВСЕМ уровням),
    и сверху ограничен max_weights / total_weights * 100, где max_weights — сумма весов
    только тех уровней, что набрали больше нуля.

Две тонкости, которых не было в первой версии этого скрипта и из-за которых он расходился
с официальным счётом на 2%: потолок уровня 115, а не 100, и ограничение сверху по max_weights.

КОРИДОР ШУМА считается здесь же, по всем прогонам одной и той же сборки, какие есть в runs/.
Он сужается сам по себе по мере накопления замеров — заглядывать в скрипт и править константу
не нужно. Отдельно даётся коридор «без шумных игр»: четыре игры (ft09, re86, ar25, r11l) дают
40% всего разброса, и их исключение — единственный приём, который 25.09 позволил хоть что-то
различить. Набор выбран по разбросу базы против самой себя, а не по результатам слоёв.

usage:
    .venv/bin/python scripts/price_idea.py --save-moves 5      # экономим 5% ходов
    .venv/bin/python scripts/price_idea.py --extra-level 3     # в 3 играх берём уровень сверх
    .venv/bin/python scripts/price_idea.py --first-level 2     # 2 мёртвые игры берут первый уровень
    .venv/bin/python scripts/price_idea.py --save-moves 5 --extra-level 1
"""
from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = "flash_v1_phaseA"
# Прогоны ОДНОЙ И ТОЙ ЖЕ сборки. Пополняется по мере накопления — ночь 25/26.09 добавит ещё.
SAME_BUILD = ["flash_v1_phaseA", "public_flash_tufa", "public_flash_keithtyser",
              "night_stock-flash", "night_stock-base2", "night_stock-base3",
              "night_stock-base4", "night_stock-base5"]
NOISY = {"ft09", "re86", "ar25", "r11l"}

# Вероятность дойти до уровня i при условии, что первый взят. Измерено на 146 наблюдениях
# (7 прогонов x 25 игр, взявшие хотя бы уровень): 46% доходят до второго, 21% до третьего,
# 8% до четвёртого; средняя глубина 1.79.
#
# Зачем это нужно. Первый счёт «разблокировать мёртвую игру» давал +0.59 за все четыре и
# выглядел пренебрежимо. В нём было скрытое допущение: игра возьмёт первый уровень и там
# остановится. Владелец возразил — без первых уровней нет и вторых. Данные подтвердили:
# первый уровень самый дорогой (медиана эффективности 1.10 против 1.20 и 1.42 дальше),
# он и есть вход на лестницу. С ожидаемой траекторией те же четыре игры стоят +1.85.
LADDER = {1: 1.00, 2: 0.46, 3: 0.21, 4: 0.08, 5: 0.03}
LADDER_TAIL = 0.01


def load(name: str):
    d = ROOT / "runs" / name
    if not (d / "benchmark.json").is_file():
        return None
    return json.loads((d / "benchmark.json").read_text(encoding="utf-8"))["game_runs"]


def game_score(g, *, save: float = 0.0, extra: int = 0) -> float:
    """Балл одной игры по формуле из taaf/game.py. save — доля срезанных ходов,
    extra — сколько уровней добавить сверх взятых."""
    base = g.get("base_actions_per_level") or []
    act = g.get("actions_per_level") or []
    nlev = int(g.get("number_of_levels") or 1)
    done = int(g.get("levels_completed") or 0)
    total_score = 0.0
    total_weights = max_weights = 0
    for i in range(nlev):
        w = i + 1
        total_weights += w
        level_score = 0.0
        if i < done + extra and i < len(base) and i < len(act):
            spent = act[i]
            if i >= done:
                # Добавленный уровень. act[i] — ходы, уже потраченные на него без завершения.
                # Считаем, что он закончит ровно на этом числе: оценка ОПТИМИСТИЧНАЯ.
                # Если ходов не потрачено вовсе, берём базлайн, то есть полные 100 очков.
                spent = spent or base[i]
            spent *= (1.0 - save)
            if spent > 0:
                level_score = min(115.0, (base[i] / spent) ** 2 * 100)
        if level_score > 0:
            max_weights += w
        total_score += level_score * w
    if not total_weights:
        return 0.0
    return min(total_score / total_weights, max_weights / total_weights * 100)


def total(runs, **kw) -> float:
    return sum(game_score(g, **kw) for g in runs) / len(runs)


def noise_corridor():
    """Разброс между прогонами одной сборки: по всем играм и без шумных."""
    scores = {}
    for n in SAME_BUILD:
        p = ROOT / "runs" / n / "score.json"
        if p.is_file():
            raw = json.loads(p.read_text(encoding="utf-8"))["games"]
            scores[n] = {g[:4]: float(v["score"]) for g, v in raw.items()}
    out = {"runs": list(scores), "all": [], "quiet": []}
    for x, y in permutations(scores, 2):
        a, b = scores[x], scores[y]
        common = set(a) & set(b)
        out["all"].append(sum(b[g] - a[g] for g in common) / len(common))
        q = common - NOISY
        out["quiet"].append(sum(b[g] - a[g] for g in q) / len(q))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-moves", type=float, default=0.0, help="процент срезанных ходов")
    ap.add_argument("--extra-level", type=int, default=0,
                    help="в скольких играх взять уровень сверх текущего (берутся самые дешёвые)")
    ap.add_argument("--first-level", type=int, default=0,
                    help="сколько мёртвых игр возьмут первый уровень И ОСТАНОВЯТСЯ (нижняя оценка)")
    ap.add_argument("--unlock", type=int, default=0,
                    help="сколько мёртвых игр войдут на лестницу как обычные (ожидаемая траектория)")
    ap.add_argument("--ref", default=REF)
    a = ap.parse_args()

    runs = load(a.ref)
    if runs is None:
        raise SystemExit("нет прогона %s" % a.ref)
    save = max(0.0, a.save_moves) / 100.0
    base_total = total(runs)

    # прибавка от экономии ходов
    after = total(runs, save=save)

    # прибавка от лишних уровней: выбираем игры, где это дешевле всего по вкладу
    gains = []
    for g in runs:
        done = int(g.get("levels_completed") or 0)
        if (done == 0 and a.first_level) or (done > 0 and a.extra_level):
            d = game_score(g, save=save, extra=1) - game_score(g, save=save)
            gains.append((d, g["game_id"][:4], done))
    gains.sort(reverse=True)
    picked_extra = [x for x in gains if x[2] > 0][:a.extra_level]
    picked_first = [x for x in gains if x[2] == 0][:a.first_level]
    after += sum(d for d, _, _ in picked_extra + picked_first) / len(runs)

    # разблокировка: мёртвая игра переходит в население живых, глубина по LADDER
    unlocked = []
    if a.unlock:
        cand = []
        for g in runs:
            if (g.get("levels_completed") or 0):
                continue
            n = int(g.get("number_of_levels") or 1)
            den = 100 * n * (n + 1) / 2
            val = sum(100 * 100 * i * LADDER.get(i, LADDER_TAIL) / den for i in range(1, n + 1))
            cand.append((val, g["game_id"][:4]))
        cand.sort(reverse=True)
        unlocked = cand[:a.unlock]
        after += sum(v for v, _ in unlocked) / len(runs)

    delta = after - base_total
    n = noise_corridor()
    lo_a, hi_a = (min(n["all"]), max(n["all"])) if n["all"] else (0, 0)
    lo_q, hi_q = (min(n["quiet"]), max(n["quiet"])) if n["quiet"] else (0, 0)

    print("точка отсчёта %s: восстановленный балл %.2f" % (a.ref, base_total))
    if save:
        print("  экономия ходов %.1f%%" % (100 * save))
    for d, gid, done in picked_extra:
        print("  +1 уровень в %s (было %d): %+.2f к баллу игры" % (gid, done, d))
    for d, gid, _ in picked_first:
        print("  первый уровень в %s (и остановка): %+.2f к баллу игры" % (gid, d))
    for v, gid in unlocked:
        print("  %s входит на лестницу: %+.2f к баллу игры (ожидаемая глубина 1.79)" % (gid, v))
    print("\nитог %.2f, прибавка %+.2f" % (after, delta))
    print("\nкоридор шума (прогонов одной сборки: %d)" % len(n["runs"]))
    print("  по всем играм:  от %+.2f до %+.2f" % (lo_a, hi_a))
    print("  без шумных:     от %+.2f до %+.2f   (исключены %s)" % (lo_q, hi_q, ", ".join(sorted(NOISY))))
    verdict = ("ВИДНО одной парой" if delta > hi_a else
               "видно только без шумных игр" if delta > hi_q else
               "НИЖЕ ШУМА — прогон вернёт «неопределённо» при любом качестве идеи")
    print("\nвердикт: %s" % verdict)
    if delta <= hi_q:
        need = hi_q / max(delta, 1e-9)
        print("чтобы стало различимо, эффект должен быть примерно в %.0f раз больше" % need)


if __name__ == "__main__":
    main()

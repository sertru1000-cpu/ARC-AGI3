"""Путь 2 (27.09): сколько запросов к модели можно было сэкономить, если бы предсказуемые ходы шли пачкой одним вызовом.

Ход «предсказуем», если шаблоны правил (rule_learner_probe.Learner, обученный только на прошлых переходах того же
уровня) угадали доску после хода (без полосы индикатора) — то есть исход был выводим из уже увиденного, нового
модель из него не узнала. Блок (ход модели в транскрипте) «предсказуем», если предсказуемы все его ходы.
Подряд идущие предсказуемые блоки можно слить в один вызов: запросы всех блоков серии, кроме первого, — экономия.
Блоки, прерванные по времени (0 ходов), внутри серии тоже сливаются (это чистое раздумье без новой информации).
Отдельно: повтор того же хода отдельным блоком (RIGHT, потом снова RIGHT) — самая простая пачка.
Это ВЕРХНЯЯ оценка по предсказуемости и НИЖНЯЯ по знанию модели: модель могла знать больше шаблонов (или меньше).
usage: .venv/bin/python scripts/batchable_moves.py runs/pod_a4b_persist ...
"""
import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule_learner_probe import Learner, hud_mask, eq  # noqa: E402
import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def moves_with_flags(game_id, history):
    """[(action_id, predictable)] по порядку ходов истории (сброс уровня/переход уровня — непредсказуемы)."""
    env = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT / "environment_files")).make(game_id)
    fr = env.reset(); g = np.asarray(fr.frame[-1], dtype=np.int16); lv = fr.levels_completed or 0
    out = []; L = Learner(); tr = []
    for rec in history:
        a = rec.get("action") or {}
        if not a.get("id"):
            continue
        if a["id"] == "RESET":
            out.append((a["id"], False)); fr = env.step(GameAction["RESET"])
            if fr is not None and fr.frame:
                g = np.asarray(fr.frame[-1], dtype=np.int16); lv = fr.levels_completed or 0
            continue
        data = a.get("data") or {}
        fr = env.step(GameAction[a["id"]], data=data)
        if fr is None or not fr.frame:
            break
        g1 = np.asarray(fr.frame[-1], dtype=np.int16); lv1 = fr.levels_completed or 0
        ok = False
        if lv1 == lv and g1.shape == g.shape:
            p, _ = L.predict(g, a["id"], data)
            ok = eq(p, g1, hud_mask(tr, *g.shape))
            L.observe(g, a["id"], data, g1); tr.append((g, a["id"], data, g1))
        else:
            L = Learner(); tr = []                                   # новый уровень — правила учатся заново
        out.append((a["id"], ok)); g, lv = g1, lv1
    return out


def blocks_of(text):
    """[(запросов, ходов)] по блокам транскрипта (логика moves_per_turn.py)."""
    bl = re.split(r"\n--- analysis_step=", text)[1:]; out = []
    for i, b in enumerate(bl):
        reqs = len(re.findall(r"^\[TOOL CALL: python\]", b, re.M)); n = 0
        ex = re.search(r"step_executed: (True|False)", b)
        if ex and ex.group(1) == "True" and i + 1 < len(bl):
            m = re.search(r"The code executed (\d+) action", bl[i + 1]); n = int(m.group(1)) if m else 0
        out.append((max(reqs, 1), n))
    return out


for run in sys.argv[1:]:
    runs = {r["game_id"][:4]: r for r in json.loads(Path(ROOT / run, "benchmark.json").read_text())["game_runs"]}
    c = Counter()
    for f in sorted(Path(ROOT / run, "transcripts").glob("*.txt")):
        gr = runs.get(f.name[:4])
        if not gr:
            continue
        mv = moves_with_flags(gr["game_id"], gr.get("history") or [])
        bl = blocks_of(f.read_text(errors="ignore"))
        if sum(n for _, n in bl) != len(mv):                          # выравнивание не сошлось — игру пропускаем
            c["skipped"] += 1; continue
        k = 0; prev_pred = False; prev_act = None
        for reqs, n in bl:
            ms = mv[k:k + n]; k += n
            pred = all(p for _, p in ms)                               # блок без ходов — «предсказуем» (нового не принёс)
            c["reqs"] += reqs; c["moves"] += n; c["blocks"] += 1
            if prev_pred and pred:
                c["saved"] += reqs                                    # слился бы с предыдущим вызовом
            if n == 1 and prev_act == ms[0][0]:
                c["repeat_single"] += 1
            c["pred_moves"] += sum(p for _, p in ms)
            prev_pred = pred and n > 0 or (prev_pred and n == 0)
            prev_act = ms[-1][0] if ms else prev_act
        c["games"] += 1
    R, M = c["reqs"], c["moves"]
    print("%-24s игр %d (пропущено %d) | ходов %d, запросов %d, %.2f хода на запрос | предсказуемых ходов %.0f%%" % (
        run.split("/")[-1], c["games"], c["skipped"], M, R, M / max(R, 1), 100 * c["pred_moves"] / max(M, 1)))
    print("   слить серии предсказуемых блоков: запросов %d -> %d (−%.0f%%), %.2f хода на запрос; повтор того же хода отдельным блоком: %d" % (
        R, R - c["saved"], 100 * c["saved"] / max(R, 1), M / max(R - c["saved"], 1), c["repeat_single"]))

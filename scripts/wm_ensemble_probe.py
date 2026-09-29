"""Ансамбль алгоритмических моделей мира: на каждом ходу игры берётся предсказатель, который до сих пор чаще угадывал (27.09).
Кандидаты: «ничего не меняется»; шаблоны правил (rule_learner_probe.Learner); локальные правила клетки 5x5→3x3
(local_rule_probe.Local) в осторожном режиме — клетка меняется, только если окрестность видели >=2 раз и исход один в >=90%.
Сверка ход за ходом, обучение только на прошлом, индикатор у края исключён. Итог — верхняя граница «лучшего из трёх
задним числом» и честный онлайн-выбор.
usage: .venv/bin/python scripts/wm_ensemble_probe.py [runs/...]
"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rule_learner_probe import game_transitions, hud_mask, eq, Learner  # noqa: E402
from local_rule_probe import Local  # noqa: E402


def careful(L, g, a, data):
    k = L.keys(g, a, data).ravel(); out = g.ravel().copy(); hit = np.zeros(out.shape, dtype=bool)
    for i, kk in enumerate(k.tolist()):
        c = L.t.get(kk)
        if c:
            v, n = c.most_common(1)[0]; tot = sum(c.values())
            if tot >= 2 and n >= 0.9 * tot:
                out[i] = v; hit[i] = True
    return out.reshape(g.shape), hit.reshape(g.shape)


for run in sys.argv[1:] or ["runs/flash_v1_phaseA"]:
    tot = Counter(); games = 0
    for gr in json.loads((ROOT / run / "benchmark.json").read_text())["game_runs"]:
        tr = game_transitions(gr["game_id"], gr.get("history") or [], str(ROOT / "environment_files"))
        if len(tr) < 5:
            continue
        games += 1
        T, L1, L2 = Learner(), Local(1), Local(2); score = Counter(); c = Counter()
        for t, (g0, a, d, g1) in enumerate(tr):
            m = hud_mask(tr[:t], *g0.shape)
            pt, _ = T.predict(g0, a, d)
            p1, _ = careful(L1, g0, a, d); p2, h2 = careful(L2, g0, a, d); pl = np.where(h2, p2, p1)
            preds = {"ident": g0, "templ": pt, "local": pl}
            best = max(preds, key=lambda k: (score[k], k == "ident"))          # онлайн: кто чаще угадывал до сих пор
            ok = {k: eq(v, g1, m) for k, v in preds.items()}
            c["online"] += ok[best]; c["oracle"] += any(ok.values())
            for k in ok:
                c[k] += ok[k]; score[k] += ok[k]
            T.score_modes(a, g1, m); T.observe(g0, a, d, g1); L1.observe(g0, a, d, g1); L2.observe(g0, a, d, g1)
        for k, v in c.items():
            tot[k] += v
        tot["n"] += len(tr)
    N = tot["n"]
    print("%s: игр %d, переходов %d | «ничего не меняется» %.0f%% | шаблоны %.0f%% | локальные (осторожно) %.0f%% | "
          "ОНЛАЙН-выбор %.0f%% | лучший задним числом %.0f%%" % (run, games, N, 100 * tot["ident"] / N, 100 * tot["templ"] / N,
                                                             100 * tot["local"] / N, 100 * tot["online"] / N, 100 * tot["oracle"] / N))

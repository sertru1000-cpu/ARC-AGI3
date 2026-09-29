"""Точность заготовки a8b base_step на записанных играх — офлайн, без модели (27.09).
Для каждого перехода уровня: «увиденные» = прошлые переходы того же уровня; предсказать доску после хода и сверить
с настоящей (локальный движок) без полосы индикатора, как в rule_learner_probe. Сравнение с заглушкой «ничего не меняется». Отдельно — только новые (не
повтор уже увиденного) и только те, где доска изменилась.
usage: .venv/bin/python scripts/base_step_probe.py [base_step|tpl_step] runs/pod_a0_control ...
"""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule_learner_probe import game_transitions, hud_mask, eq  # noqa: E402
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NS = {}
exec(open(ROOT / "nextfork/src/ARC3-Inference/inference/agent/buildwm_sandbox/base_step.py").read(), NS)
exec(open(ROOT / "nextfork/src/ARC3-Inference/inference/agent/buildwm_sandbox/tpl_step.py").read(), NS)
FN = sys.argv.pop(1) if len(sys.argv) > 1 and sys.argv[1] in ("base_step", "tpl_step") else "base_step"
CH = "0123456789abcdefghij"
rows = lambda g: ["".join(CH[int(v)] for v in r) for r in g]
for run in sys.argv[1:]:
    c = Counter()
    for gr in json.loads((ROOT / run / "benchmark.json").read_text())["game_runs"]:
        tr = game_transitions(gr["game_id"], gr.get("history") or [], str(ROOT / "environment_files"))
        seen = []
        for t, (g0, a, data, g1) in enumerate(tr):
            mask = hud_mask(tr[:t], *g0.shape)                     # полоса индикатора исключена, как в прежних зондах
            act = ("MOUSE(row=%d, col=%d)" % (data["y"], data["x"])) if a == "ACTION6" and data else {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT"}.get(a, a)
            b, af = rows(g0), rows(g1)
            NS["level_transitions"] = lambda s=list(seen): s
            p = NS[FN](b, act)
            P = np.array([[CH.index(ch) for ch in r] for r in p]) if len(p) == g1.shape[0] else None
            ok, ident = eq(P, g1, mask), eq(g0, g1, mask)
            new = not any(x[1] == act and x[0] == b for x in seen)
            c["n"] += 1; c["ok"] += ok; c["id"] += ident
            if new:
                c["new"] += 1; c["new_ok"] += ok; c["new_id"] += ident
                if not ident:
                    c["newchg"] += 1; c["newchg_ok"] += ok
            seen.append((b, act, af))
    pc = lambda x, y: 100 * c[x] / max(c[y], 1)
    print(FN, "%-22s переходов %4d | base_step %2.0f%% против «ничего не меняется» %2.0f%% | новые %d: %2.0f%% против %2.0f%% | новые с изменением %d: угадано %2.0f%%" % (
        run.split("/")[-1], c["n"], pc("ok", "n"), pc("id", "n"), c["new"], pc("new_ok", "new"), pc("new_id", "new"), c["newchg"], pc("newchg_ok", "newchg")))

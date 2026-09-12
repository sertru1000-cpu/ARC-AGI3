"""Многошаговое обобщение программ модели (вопрос ChatGPT, раунд 1, 12.09).

Одношаговый тест (`llm_program_generalization.py`) дал 86% на изменившихся состояниях. Критик
возразил: одношаговое предсказание — слишком лёгкое свойство; для планирования нужна цепочка.
Здесь та же принятая программа гоняется ЦЕПОЧКОЙ: state_of(before_1) → predict(·, a_1) →
predict(·, a_2) → … → predict(·, a_k), и сравнивается ТОЛЬКО конечное состояние с
state_of(after_k). Промежуточные наблюдения программе не показываются. Цепочки берутся из
переходов N+1..конец (которых при приёме не было), скользящим окном, только внутри одного
уровня. Считаются отдельно цепочки, где конечное состояние отличается от начального
(«нетривиальные»), как и в одношаговом тесте. Ноль квоты, ноль вызовов модели.

usage: .venv/bin/python scripts/llm_program_multistep.py /tmp/v5_admitted_programs.json runs/flash_wm_v5 scripts/wm_helpers_v9.py 1 3 5
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_program_generalization import game_state, pts  # noqa: E402

CHECK = r'''
_pairs = _wm_pairs(10000)
_new = _pairs[%d:]
_K = %d
_res = {"chains": 0, "checked": 0, "ok": 0, "nontriv": 0, "nontriv_ok": 0, "skipped": 0}
for _i in range(0, len(_new) - _K + 1):
    _win = _new[_i:_i + _K]
    # цепочка должна быть непрерывной: after шага j == before шага j+1 (по ascii)
    if any(_win[_j][2].ascii != _win[_j + 1][0].ascii for _j in range(_K - 1)):
        continue
    _res["chains"] += 1
    try:
        _s0 = state_of(_win[0][0]); _s = _s0
        for _b, _a, _af in _win:
            _s = predict(_s, _a)
            if _s is None: break
        _target = state_of(_win[-1][2])
    except Exception:
        _res["skipped"] += 1; continue
    if _s is None:
        _res["skipped"] += 1; continue
    _res["checked"] += 1
    _hit = _wm_norm(_s) == _wm_norm(_target)
    _res["ok"] += _hit
    if _wm_norm(_target) != _wm_norm(_s0):
        _res["nontriv"] += 1; _res["nontriv_ok"] += _hit
print("MULTI " + " ".join("%%s=%%d" %% kv for kv in sorted(_res.items())))
'''


def main(progs_path, run_dir, helpers_path, *ks):
    ns = {}; exec(open(helpers_path).read(), ns); H = ns["WM_HELPERS"]
    progs = json.load(open(progs_path))
    events = {os.path.basename(f).split("_p0")[0]: f
              for f in glob.glob(os.path.join(run_dir, "artifacts", "*_events.jsonl"))}
    ks = [int(k) for k in ks] or [1, 3, 5]
    print("%-3s %7s %8s %6s %7s %10s %7s" % ("k", "цепочек", "взялась", "верно", "нетрив.", "нетрив.вер", "%нетр"))
    per_game = {}
    for k in ks:
        tot = {"chains": 0, "checked": 0, "ok": 0, "nontriv": 0, "nontriv_ok": 0, "skipped": 0}
        for g, n, code in progs:
            st = game_state(events.get(g, ""))
            if st is None:
                continue
            out = pts.run_sandboxed_python(
                code=H + "\n" + code + "\n" + CHECK % (n, k), timeout_seconds=120, initial_state=st,
                action_handler=lambda a: (_ for _ in ()).throw(RuntimeError("no actions")))
            m = re.search(r"MULTI (.*)", str(out.get("stdout", "") or ""))
            if not m:
                continue
            r = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in m.group(1).split()}
            for key in tot:
                tot[key] += r[key]
            per_game.setdefault(g[:4], {})[k] = (r["nontriv"], r["nontriv_ok"])
        pct = 100 * tot["nontriv_ok"] / tot["nontriv"] if tot["nontriv"] else float("nan")
        print("%-3d %7d %8d %6d %7d %10d %6.0f%%" % (k, tot["chains"], tot["checked"], tot["ok"],
                                                   tot["nontriv"], tot["nontriv_ok"], pct))
    print("\nпо играм (нетривиальные: верно/всего) при k =", ks)
    for g, d in sorted(per_game.items()):
        print("  %s  " % g + "  ".join("k%d %d/%d" % (k, d[k][1], d[k][0]) for k in ks if k in d))


if __name__ == "__main__":
    main(*sys.argv[1:])

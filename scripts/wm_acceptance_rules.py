"""Какое правило приёма отбирает программы, работающие на ЧУЖОМ пути, а не на своём?

ЗАЧЕМ. Замер 09.09 (scripts/wm_plan_offline.py) показал две цифры рядом. На своей траектории,
где программа и была принята, поиск внутри неё находил путь к взятию уровня в 4 играх из 4.
На чужой траектории той же игры — в 5 из 19, а доля верных предсказаний падала с 86% до 56%.
ВЫВОД, который отсюда следует: точный приём на коротком буфере отбирает программы, описывающие
пройденный путь, а не правила игры. Это переобучение на траекторию, и лечится оно так же, как
в машинном обучении, — отложенной выборкой.

ЧТО МЕРЯЕТСЯ. Пул из всех программ, которые модель когда-либо написала (включая отвергнутые:
78 из них в прогоне v9 не были приняты только из-за дефекта харнесса). Каждое правило — фильтр,
пропускающий часть пула по ПЕРВЫМ K переходам своей траектории. Затем принятые программы
меряются на том, чего они не видели:
  * хвост своей траектории (переходы K+1 и дальше);
  * ЦЕЛИКОМ чужая траектория той же игры из базового прогона;
  * находит ли поиск внутри программы путь к состоянию, из которого агент взял уровень.

ПОРОГИ, ЗАПИСАННЫЕ ДО РАСЧЁТА. Нынешнее правило даёт 56% верных на чужом пути и планирование
в 5 играх из 19. Правило считается лучше, если поднимает планирование до 9 игр из 19 и выше,
не уменьшив числа принятых программ. Если ни одно не поднимает — дело не в правиле, а в том,
что программы этой модели описывают траекторию, и линию можно закрывать с этим выводом.

Ни одного запроса к модели и ни секунды GPU: всё считается на записях.

usage:
    .venv/bin/python scripts/wm_acceptance_rules.py
    .venv/bin/python scripts/wm_acceptance_rules.py --k 12 --runs runs/flash_wm_v5 runs/flash_wm_v9 runs/flash_wm_v10
"""

from __future__ import annotations

import argparse
import ast
import glob
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARC3 = ROOT / "atlas_src" / "src" / "ARC3-Inference"
sys.path.insert(0, str(ARC3))
spec = importlib.util.spec_from_file_location("pts", ARC3 / "inference/agent/python_tool_sandbox.py")
pts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pts)

NS: dict = {}
exec(open(ROOT / "scripts" / "wm_helpers_v9.py", encoding="utf-8").read(), NS)
HELPERS = NS["WM_HELPERS"]

CODE_RE = re.compile(r'"arguments":\s*"(.*?)"\s*\n\s*\}', re.S)


def definitions_only(code: str) -> str:
    try:
        tree = ast.parse(code)
    except Exception:
        return ""
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            seg = ast.get_source_segment(code, node)
            if seg and "action(" not in seg:
                keep.append(seg)
    src = "\n".join(keep)
    return src if ("def predict" in src and "def state_of" in src) else ""


def candidates(runs: list[str]) -> list[tuple[str, str, str]]:
    """Все программы, написанные моделью: (прогон, игра, код). Отвергнутые тоже — они и есть
    материал для сравнения правил."""
    out, seen = [], set()
    for run in runs:
        for f in sorted(glob.glob(os.path.join(run, "transcripts", "*.txt"))):
            game = os.path.basename(f)[:4]
            text = open(f, encoding="utf-8", errors="replace").read()
            for m in CODE_RE.finditer(text):
                try:
                    code = json.loads(json.loads('"' + m.group(1) + '"')).get("code", "")
                except Exception:
                    continue
                if "def predict" not in code or "def state_of" not in code:
                    continue
                defs = definitions_only(code)
                key = (game, defs)
                if not defs or key in seen:
                    continue
                seen.add(key)
                out.append((os.path.basename(run.rstrip("/")), game, defs))
    return out


def trajectory(events_path: str) -> dict | None:
    hist = []
    for line in open(events_path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") not in ("initial", "action") or not e.get("board"):
            continue
        hist.append({"action": str(e.get("action_display") or "") if e.get("type") == "action" else "",
                     "frame": {"ascii": e["board_ascii"], "step": int(e.get("action_num") or 0),
                               "level": int(e.get("level") or 0), "shape": [64, 64], "grid": e["board"]}})
    if not hist:
        return None
    return {"current_frame": hist[-1]["frame"], "history": hist,
            "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT", "SPACE"], "last_action_result": {}}


def counts(name: str) -> str:
    """Покрытие и верность на отрезке переходов: сколько программа взялась предсказать и сколько верно."""
    return f'''
_cov = _cor = 0
for _b, _a, _f in {name}:
    try:
        _s = state_of(_b); _p = predict(_s, _a); _t = state_of(_f)
    except Exception:
        continue
    if _p is None: continue
    _cov += 1
    if _wm_norm(_p) == _wm_norm(_t): _cor += 1
'''


OWN = r'''
_all = _wm_pairs(10000, level=_LEVEL)
_pre = _all[:_K]; _tail = _all[_K:]
_half = len(_pre) // 2
_out = []
for _name, _seg in (("A", _pre[:_half]), ("B", _pre[_half:]), ("T", _tail)):
''' + '''
    _cov = _cor = 0
    for _b, _a, _f in _seg:
        try:
            _s = state_of(_b); _p = predict(_s, _a); _t = state_of(_f)
        except Exception:
            continue
        if _p is None: continue
        _cov += 1
        if _wm_norm(_p) == _wm_norm(_t): _cor += 1
    _out.append("%s:%d/%d/%d" % (_name, len(_seg), _cov, _cor))
print("OWN " + " ".join(_out))
'''

FOREIGN = r'''
_all = _wm_pairs(10000, level=_LEVEL)
_win = None
for _i, (_b, _a, _f) in enumerate(_all):
    if wm_level(_f) != wm_level(_b):
        _win = _i; break
_cov = _cor = 0
for _b, _a, _f in (_all if _win is None else _all[:_win]):
    try:
        _s = state_of(_b); _p = predict(_s, _a); _t = state_of(_f)
    except Exception:
        continue
    if _p is None: continue
    _cov += 1
    if _wm_norm(_p) == _wm_norm(_t): _cor += 1
_found = 0; _plen = -1
if _win is not None:
    _entry = state_of(_all[0][0]); _target = _wm_norm(state_of(_all[_win][0]))
    _acts = []
    for _b, _a, _f in _all:
        if _a not in _acts: _acts.append(_a)
    _seen = {_wm_norm(_entry)}; _q = _wm_deque([(_entry, 0)]); _nodes = 0
    _paths = {_wm_norm(_entry): 0}
    while _q and _nodes < _NODES:
        _s, _d = _q.popleft()
        if _d >= _DEPTH: continue
        for _a in _acts:
            _nodes += 1
            try: _n = predict(_s, _a)
            except Exception: _n = None
            if _n is None: continue
            _k = _wm_norm(_n)
            if _k == _target:
                _found = 1; _plen = _d + 1; break
            if _k in _seen: continue
            _seen.add(_k); _q.append((_n, _d + 1))
        if _found: break
print("FOREIGN n=%d cov=%d cor=%d win=%s found=%d plen=%d" %
      (len(_all) if _win is None else _win, _cov, _cor,
       "-" if _win is None else _win, _found, _plen))
'''


def run_sandbox(code: str, state: dict, level: int, k: int, depth: int, nodes: int) -> str:
    head = HELPERS + f"\n_LEVEL = {level}\n_K = {k}\n_DEPTH = {depth}\n_NODES = {nodes}\n"
    out = pts.run_sandboxed_python(code=head + code, timeout_seconds=180, initial_state=state,
                                   action_handler=lambda a: (_ for _ in ()).throw(RuntimeError("нельзя")))
    return str(out.get("stdout", "") or "") + str(out.get("error", "") or "")


RULES = {
    "нынешнее (точный приём)":      lambda a, b, t: a[1] == a[0] and b[1] == b[0] and a[2] == a[0] and b[2] == b[0],
    "частичное, покрытие ≥60%":     lambda a, b, t: (a[1] + b[1]) >= 0.6 * (a[0] + b[0]) and (a[2] + b[2]) == (a[1] + b[1]),
    "с отложенной половиной":       lambda a, b, t: a[1] == a[0] and a[2] == a[0] and b[1] == b[0] and b[2] == b[0] and b[0] >= 3,
    "точность ≥90% при покрытии ≥50%": lambda a, b, t: (a[1] + b[1]) >= 0.5 * (a[0] + b[0]) and (a[2] + b[2]) >= 0.9 * (a[1] + b[1]) and (a[1] + b[1]) > 0,
    "всё подряд (контроль)":        lambda a, b, t: True,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12, help="сколько первых переходов считается буфером приёма")
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--nodes", type=int, default=60000)
    ap.add_argument("--runs", nargs="*", default=["runs/flash_wm_v5", "runs/flash_wm_v9", "runs/flash_wm_v10"])
    ap.add_argument("--foreign", default="runs/flash_v1_phaseA")
    ap.add_argument("--out", default="docs/wm_acceptance_rows.json",
                    help="куда сложить посчитанное, чтобы не пересчитывать при смене правил")
    a = ap.parse_args()

    pool = candidates(a.runs)
    print(f"кандидатов в пуле: {len(pool)} (без повторов), игр: {len({g for _, g, _ in pool})}")
    print(f"буфер приёма: первые {a.k} переходов своей траектории; поиск: глубина {a.depth}, узлов {a.nodes}\n")

    own_ev = {os.path.basename(f).split('_p0')[0][:4]: f
              for r in a.runs for f in glob.glob(os.path.join(r, "artifacts", "*_events.jsonl"))}
    frn_ev = {os.path.basename(f).split('_p0')[0][:4]: f
              for f in glob.glob(os.path.join(a.foreign, "artifacts", "*_events.jsonl"))}

    rows = []
    for i, (run, game, code) in enumerate(pool, 1):
        own_state = trajectory(own_ev[game]) if game in own_ev else None
        frn_state = trajectory(frn_ev[game]) if game in frn_ev else None
        if own_state is None or frn_state is None:
            continue
        lvl_own = own_state["history"][0]["frame"]["level"]
        lvl_frn = frn_state["history"][0]["frame"]["level"]
        so = run_sandbox(code + "\n" + OWN, own_state, lvl_own, a.k, a.depth, a.nodes)
        m = re.search(r"OWN A:(\d+)/(\d+)/(\d+) B:(\d+)/(\d+)/(\d+) T:(\d+)/(\d+)/(\d+)", so)
        if not m:
            continue
        v = list(map(int, m.groups()))
        A, B, T = tuple(v[0:3]), tuple(v[3:6]), tuple(v[6:9])
        sf = run_sandbox(code + "\n" + FOREIGN, frn_state, lvl_frn, a.k, a.depth, a.nodes)
        mf = re.search(r"FOREIGN n=(\d+) cov=(\d+) cor=(\d+) win=(\S+) found=(\d) plen=(-?\d+)", sf)
        if not mf:
            continue
        n_f, cov_f, cor_f, win, found, plen = (int(mf.group(1)), int(mf.group(2)), int(mf.group(3)),
                                               mf.group(4), int(mf.group(5)), int(mf.group(6)))
        rows.append({"run": run, "game": game, "A": A, "B": B, "T": T,
                     "f_n": n_f, "f_cov": cov_f, "f_cor": cor_f, "win": win != "-",
                     "found": found, "plen": plen})
        if i % 25 == 0:
            print(f"  ...обработано {i} из {len(pool)}", flush=True)

    if a.out:
        json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"\nсырые числа сохранены: {a.out}")
    print(f"программ с обеими траекториями: {len(rows)}\n")
    print("%-34s %-9s %-11s %-13s %-14s" % ("правило приёма", "принято", "игр", "верно на чужом", "план найден"))
    for name, rule in RULES.items():
        acc = [r for r in rows if rule(r["A"], r["B"], r["T"])]
        if not acc:
            print("%-34s %-9s %-11s %-13s %-14s" % (name, 0, 0, "-", "-"))
            continue
        games = {r["game"] for r in acc}
        cov = sum(r["f_cov"] for r in acc); cor = sum(r["f_cor"] for r in acc)
        win_games = {r["game"] for r in acc if r["win"]}
        found_games = {r["game"] for r in acc if r["found"]}
        print("%-34s %-9d %-11s %-13s %-14s" % (
            name, len(acc), f"{len(games)}",
            f"{100*cor/cov:.0f}% ({cor}/{cov})" if cov else "-",
            f"{len(found_games)} из {len(win_games)}" if win_games else "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

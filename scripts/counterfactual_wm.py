"""Контрфактический контроль модели мира (раунд 4, результат 3): в записанных боевых состояниях применяем
АЛЬТЕРНАТИВНЫЕ действия на локальном движке (он воспроизводит бой 561/561) и проверяем оффлайн-предсказатель
(state_of/predict из docs/wm_offline_flash_s1*.json) на настоящих следующих состояниях.
usage: counterfactual_wm.py <game4> [--k 20] [--run runs/flash_v1_phaseA]"""
import argparse, glob, json, os, random, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from replay_battle_local import load_actions, play
from agent.harness.perception import segment

def letter_map(run, g):
    p = glob.glob(f"{run}/artifacts/{g}-*_p0_events.jsonl")[0]; m = {}
    for l in open(p, encoding="utf-8"):
        e = json.loads(l)
        if e.get("board") and e.get("board_ascii"):
            rows = e["board_ascii"].split("\n")
            for r, row in enumerate(e["board"]):
                for c, v in enumerate(row):
                    if r < len(rows) and c < len(rows[r]): m[int(v)] = rows[r][c]
            if len(m) >= 10: break
    return m
def to_rows(board, m):
    return ["".join(m.get(int(v), "?") for v in row) for row in board]
def best_code(g):
    best = None
    for f in ("docs/wm_offline_flash_s1.json", "docs/wm_offline_flash_s1_retry.json"):
        for x in json.load(open(f)):
            if x.get("game") == g and x.get("code") and x.get("ok") is not None:
                if best is None or x["ok"] > best["ok"]: best = x
    return best
def _norm(x):
    if isinstance(x, dict): return tuple(sorted((str(k), _norm(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)): return tuple(_norm(v) for v in x)
    if isinstance(x, set): return tuple(sorted(_norm(v) for v in x))
    return x
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("game"); ap.add_argument("--k", type=int, default=20); ap.add_argument("--run", default="runs/flash_v1_phaseA"); a = ap.parse_args()
    import logging; logging.disable(logging.WARNING)
    import arc_agi
    from arc_agi import OperationMode
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    gid = [e.game_id.split("-")[0] for e in arc.get_environments() if e.game_id.startswith(a.game)][0]
    acts = load_actions(a.run, a.game); m = letter_map(a.run, a.game); rec = best_code(a.game)
    ns = {}; exec(rec["code"], ns); state_of, predict = ns["state_of"], ns["predict"]
    names = sorted(set(x["name"] for x in acts if x["name"] != "ACTION6")); has_click = any(x["name"] == "ACTION6" for x in acts)
    rng = random.Random(0); idxs = sorted(rng.sample(range(1, len(acts)), min(a.k, len(acts) - 1)))
    stats = {}; t0 = time.time(); moves = 0
    for i in idxs:
        before = acts[i - 1]["board"]
        alts = [{"name": n, "payload": None, "cls": "arrow"} for n in names]
        if has_click:
            objs = segment(before).non_background()[:8]
            alts += [{"name": "ACTION6", "payload": {"x": int(round(o.centroid[1])), "y": int(round(o.centroid[0]))}, "cls": "click_obj"} for o in objs]
            alts += [{"name": "ACTION6", "payload": {"x": rng.randint(0, 63), "y": rng.randint(0, 63)}, "cls": "click_rand"} for _ in range(2)]
        for alt in alts:
            tr = play(arc, gid, acts[:i] + [alt]); moves += len(tr)
            after = tr[-1][0]
            if tr[-2][0].shape != before.shape or not (tr[-2][0] == before).all(): continue   # повтор разошёлся
            disp = alt["name"] if alt["name"] != "ACTION6" else "MOUSE(row=%d, col=%d)" % (alt["payload"]["y"], alt["payload"]["x"])
            disp = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT", "ACTION5": "SPACE"}.get(disp, disp)
            changed = not (after == before).all()
            st = stats.setdefault(alt["cls"], {"n": 0, "changed": 0, "nontriv": 0, "covered": 0, "ok": 0, "err": 0, "pred_change_when_none": 0})
            st["n"] += 1; st["changed"] += int(changed)
            try:
                sb = state_of(to_rows(before, m)); sa = state_of(to_rows(after, m))
            except Exception:
                st["err"] += 1; continue
            if _norm(sb) == _norm(sa):
                # доска/состояние не изменилось: предсказатель прав, если вернул None или то же состояние
                try: p = predict(sb, disp)
                except Exception: st["err"] += 1; continue
                if p is not None and _norm(p) != _norm(sb): st["pred_change_when_none"] += 1
                continue
            st["nontriv"] += 1
            try: p = predict(sb, disp)
            except Exception: st["err"] += 1; continue
            if p is None: continue
            st["covered"] += 1; st["ok"] += int(_norm(p) == _norm(sa))
    print(f"{a.game}: предсказатель из записи ok={rec['ok']}/{rec['n']} (в распределении); контрфактических состояний {len(idxs)}, ходов на движке {moves} за {time.time()-t0:.0f} с")
    for cls, st in stats.items():
        print(f"   {cls:10s} n={st['n']:3d} доска менялась {st['changed']:3d} | нетривиальных {st['nontriv']:3d}, покрыто {st['covered']:3d}, верно {st['ok']:3d} | ложное предсказание изменения при неизменном состоянии {st['pred_change_when_none']:3d} | ошибок {st['err']}")

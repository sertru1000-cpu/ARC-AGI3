"""Набор данных политики из решений BFS на вариантах публичных игр (15.09): для каждого решённого варианта повторяем
путь на движке и пишем (доска ДО хода, ход, клетка клика, игра-источник).
usage: imitation_dataset_pub.py --env-dir <variants dir> --solutions a.json[,b.json,...] --out dataset.npz"""
import argparse, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from replay_battle_local import play

ID = {"ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4, "ACTION5": 5, "ACTION6": 6}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--env-dir", required=True); ap.add_argument("--solutions", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    import logging; logging.disable(logging.WARNING)
    import arc_agi
    from arc_agi import OperationMode
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=a.env_dir)
    sols = {}
    for f in a.solutions.split(","):
        sols.update(json.load(open(f)))
    X, Y, C, G = [], [], [], []
    def _src(gid):
        m = re.match(r"^([a-z0-9]{4})v\d{3}", gid); return m.group(1) if m else gid.split("-")[0]
    games = sorted({_src(gid) for gid in sols}); gidx = {g: i for i, g in enumerate(games)}
    n_ok = 0
    for gid, r in sols.items():
        if not r.get("solved"):
            continue
        script = [{"name": "RESET", "payload": None}] + [{"name": n, "payload": p} for n, p in r["path"]]
        try:
            tr = play(arc, gid, script)
        except Exception as exc:
            print(gid, "повтор упал:", repr(exc)[:100]); continue
        if tr[-1][1] < 1:
            print(gid, "повтор пути не взял уровень"); continue
        n_ok += 1
        for j, (n, p) in enumerate(r["path"]):
            if j + 1 >= len(tr):
                break
            X.append(tr[j][0].astype(np.int8)); Y.append(ID[n]); C.append((int(p["y"]), int(p["x"])) if p else (-1, -1)); G.append(gidx[_src(gid)])
    np.savez_compressed(a.out, X=np.stack(X) if X else np.zeros((0, 64, 64), np.int8), y=np.array(Y, np.int64), mech=np.array(G, np.int64), level=np.zeros(len(Y), np.int64), click=np.array(C, np.int64), mechs=np.array(games))
    print(f"решённых вариантов {n_ok}, примеров {len(Y)}, кликов {sum(1 for y in Y if y == 6)}, игр {games} -> {a.out}")


if __name__ == "__main__":
    main()

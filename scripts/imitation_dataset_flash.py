"""Набор данных из БОЕВЫХ решений flash: для каждого взятого уровня в записанных прогонах -- пары (доска до хода, ход)
на отрезке ходов этого уровня (15.09). Настоящие игры и настоящие решения, включая игры с движением по полю.
usage: imitation_dataset_flash.py --runs runs/flash_v1_phaseA,runs/public_flash_tufa,runs/public_flash_keithtyser --out ds.npz [--exclude sp80,sk48,...]"""
import argparse, glob, json, re
from pathlib import Path
import numpy as np

ID = {"ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4, "ACTION5": 5, "ACTION6": 6}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--exclude", default=""); a = ap.parse_args()
    excl = set(x for x in a.exclude.split(",") if x)
    X, Y, C, G = [], [], [], []; games = []; n_levels = 0
    for run in a.runs.split(","):
        for p in sorted(glob.glob(f"{run}/artifacts/*_p0_events.jsonl")):
            g = Path(p).name[:4]
            if g in excl:
                continue
            seg = []   # (board_before, action, click)
            prev_board = None
            for l in open(p, encoding="utf-8"):
                e = json.loads(l)
                if e.get("type") not in ("initial", "action"):
                    continue
                if e.get("type") == "action":
                    name = str(e.get("action_name")); disp = str(e.get("action_display") or "")
                    if name == "RESET":
                        seg = []
                    else:
                        m = re.match(r"MOUSE\(row=(\d+), col=(\d+)\)", disp)
                        if prev_board is not None and name in ID:
                            seg.append((prev_board, ID[name], (int(m.group(1)), int(m.group(2))) if m else (-1, -1)))
                        if e.get("level_completed") in (True, "True"):
                            if g not in games:
                                games.append(g)
                            gi = games.index(g); n_levels += 1
                            for b, y, c in seg:
                                X.append(np.asarray(b, dtype=np.int8)); Y.append(y); C.append(c); G.append(gi)
                            seg = []
                prev_board = e.get("board")
    np.savez_compressed(a.out, X=np.stack(X) if X else np.zeros((0, 64, 64), np.int8), y=np.array(Y, np.int64), mech=np.array(G, np.int64), level=np.zeros(len(Y), np.int64), click=np.array(C, np.int64), mechs=np.array(games))
    print(f"уровней flash {n_levels}, примеров {len(Y)}, кликов {sum(1 for y in Y if y == 6)}, игр {len(games)} -> {a.out}")


if __name__ == "__main__":
    main()

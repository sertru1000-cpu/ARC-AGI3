"""Набор переходов (S, A, S') с меткой игры и варианта для few-shot проверки переноса (критик, раунд 6, тест №1).
Источники: решения BFS на вариантах (повтор пути на движке), решения BFS на оригиналах, боевые взятые уровни flash
(доска до/после хода есть в журнале событий). usage: fewshot_dataset.py --var-dir <pubvar_div> --var-solutions a.json,b.json --out ds.npz"""
import argparse, glob, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from replay_battle_local import play

ID = {"ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4, "ACTION5": 5, "ACTION6": 6}
SRC = {"var": 0, "orig": 1, "flash": 2}


class Acc:
    def __init__(self):
        self.X, self.Xn, self.y, self.c, self.g, self.v, self.s = [], [], [], [], [], [], []
        self.games = []; self.vids = {}

    def add(self, before, after, y, click, game, vid, src):
        if game not in self.games:
            self.games.append(game)
        self.X.append(np.asarray(before, np.int8)); self.Xn.append(np.asarray(after, np.int8)); self.y.append(y); self.c.append(click)
        self.g.append(self.games.index(game)); self.v.append(self.vids.setdefault(vid, len(self.vids))); self.s.append(SRC[src])


def replay_solutions(acc, env_dir, sols, src, game_of):
    import arc_agi
    from arc_agi import OperationMode
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=env_dir)
    avail = {e.game_id for e in arc.get_environments()}
    n = 0
    for gid, r in sols.items():
        if not r.get("solved") or gid not in avail:
            continue
        script = [{"name": "RESET", "payload": None}] + [{"name": a, "payload": p} for a, p in r["path"]]
        try:
            tr = play(arc, gid, script)
        except Exception as exc:
            print(gid, "повтор упал:", repr(exc)[:80]); continue
        if tr[-1][1] < 1:
            continue
        n += 1
        for j, (a, p) in enumerate(r["path"]):
            if j + 1 >= len(tr):
                break
            acc.add(tr[j][0], tr[j + 1][0], ID[a], (int(p["y"]), int(p["x"])) if p else (-1, -1), game_of(gid), f"{src}:{gid}", src)
    return n


def flash_events(acc, runs):
    n = 0
    for run in runs:
        for p in sorted(glob.glob(f"{run}/artifacts/*_p0_events.jsonl")):
            g = Path(p).name[:4]; seg = []; prev = None; lvl = 1
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
                        if prev is not None and name in ID and e.get("board") is not None:
                            seg.append((prev, e["board"], ID[name], (int(m.group(1)), int(m.group(2))) if m else (-1, -1)))
                        if e.get("level_completed") in (True, "True"):
                            for b, b2, y, c in seg:
                                acc.add(b, b2, y, c, g, f"flash:{run}:{g}:L{lvl}", "flash")
                            n += 1; lvl += 1; seg = []
                prev = e.get("board")
    return n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--var-dir", required=True); ap.add_argument("--var-solutions", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default="runs/flash_v1_phaseA,runs/public_flash_tufa,runs/public_flash_keithtyser"); a = ap.parse_args()
    import logging; logging.disable(logging.ERROR)
    acc = Acc()
    sols = {}
    for f in a.var_solutions.split(","):
        sols.update(json.load(open(f)))
    def _src(gid):
        m = re.match(r"^([a-z0-9]{4})v\d{3}", gid); return m.group(1) if m else gid[:4]
    nv = replay_solutions(acc, a.var_dir, sols, "var", _src)
    no = replay_solutions(acc, "environment_files", json.load(open(ROOT / "runs" / "bfs_originals.json")), "orig", lambda gid: gid[:4])
    nf = flash_events(acc, a.runs.split(","))
    np.savez_compressed(a.out, X=np.stack(acc.X), Xn=np.stack(acc.Xn), y=np.array(acc.y, np.int64), click=np.array(acc.c, np.int64),
                        game=np.array(acc.g, np.int64), vid=np.array(acc.v, np.int64), src=np.array(acc.s, np.int64), games=np.array(acc.games))
    per = {g: int((np.array(acc.g) == i).sum()) for i, g in enumerate(acc.games)}
    print(f"вариантов {nv}, оригиналов {no}, уровней flash {nf}; переходов {len(acc.y)}, вариантов-источников {len(acc.vids)}, игр {len(acc.games)}: {per} -> {a.out}")


if __name__ == "__main__":
    main()

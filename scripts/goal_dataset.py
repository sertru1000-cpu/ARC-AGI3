"""Набор для детектора цели/прогресса (критик раунда 6, №3): для каждого шага решённого пути -- положительный переход
(S -> следующее состояние пути) и отрицательные (S -> дети от других ходов алфавита, меняющих доску и не проигрывающих).
Источники: решения перебора на вариантах, оригиналы (bfs_originals), боевые пути flash (повторяются на движке).
Снимки среды (copy.deepcopy) -- дети без повтора пути. usage: goal_dataset.py --var-dir <dir> --var-solutions a.json,b.json --out goal.npz [--neg 12]"""
import argparse, copy, glob, json, random, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import engine_bfs2 as B2
from arcengine import GameAction

ID = {"ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4, "ACTION5": 5, "ACTION6": 6}


class Acc:
    def __init__(self):
        self.X, self.Xn, self.y, self.g, self.v, self.grp, self.dist, self.act = [], [], [], [], [], [], [], []
        self.games = []; self.vids = {}; self.ngrp = 0

    def add(self, s, sn, y, game, vid, dist, act):
        if game not in self.games:
            self.games.append(game)
        self.X.append(s.astype(np.int8)); self.Xn.append(sn.astype(np.int8)); self.y.append(y); self.g.append(self.games.index(game))
        self.v.append(self.vids.setdefault(vid, len(self.vids))); self.grp.append(self.ngrp); self.dist.append(dist); self.act.append(act)


def walk_path(arc, gid, path, acc, game, vid, n_neg, rng):
    cnt = B2.Counter(10**9, 1e18)
    root = B2.root_snap(arc, gid, cnt)
    clicks = B2.active_clicks(root, cnt)
    cur = root; lvl0 = root.lvl; n = len(path)
    for t, (name, payload) in enumerate(path):
        act = (name, payload)
        nxt = B2.step(cur, act, cnt)
        if nxt is None:
            return False
        if nxt.lvl > cur.lvl or t == n - 1:
            pass
        # братья: все ходы алфавита, кроме выбранного
        sib = [a for a in B2.simple_actions(cur.avail) + clicks if a != act]
        rng.shuffle(sib); negs = []
        for a in sib:
            if len(negs) >= n_neg:
                break
            s = B2.step(cur, a, cnt)
            if s is None or s.state == "GAME_OVER" or s.lvl > cur.lvl:
                continue
            if not (s.grid != cur.grid).any() or (s.grid == nxt.grid).all():
                continue
            negs.append((s.grid, a))
        if negs:
            acc.ngrp += 1
            acc.add(cur.grid, nxt.grid, 1, game, vid, n - t, ID.get(name, 0))
            for g2, a in negs:
                acc.add(cur.grid, g2, 0, game, vid, n - t, ID.get(a[0], 0))
        cur = nxt
        if cur.lvl > lvl0:
            return True
        if cur.state == "GAME_OVER":
            return False
    return cur.lvl > lvl0


def flash_paths(runs):
    """{(game4, run): [path]} -- отрезок ходов до первого взятия уровня 1 (без RESET внутри)."""
    out = {}
    for run in runs:
        for p in glob.glob(f"{run}/artifacts/*_p0_events.jsonl"):
            g = Path(p).name[:4]; seg = []
            for l in open(p, encoding="utf-8"):
                e = json.loads(l)
                if e.get("type") != "action":
                    continue
                name = str(e.get("action_name")); disp = str(e.get("action_display") or "")
                if name == "RESET":
                    seg = []; continue
                m = re.match(r"MOUSE\(row=(\d+), col=(\d+)\)", disp)
                seg.append(("ACTION6", {"x": int(m.group(2)), "y": int(m.group(1))}) if m else (name, None))
                if e.get("level_completed") in (True, "True"):
                    out[(g, run)] = seg; break
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--var-dir", required=True); ap.add_argument("--var-solutions", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--neg", type=int, default=12); ap.add_argument("--runs", default="runs/flash_v1_phaseA,runs/public_flash_tufa,runs/public_flash_keithtyser"); a = ap.parse_args()
    import logging; logging.disable(logging.ERROR)
    import arc_agi
    from arc_agi import OperationMode
    rng = random.Random(0); acc = Acc(); stats = {"var": 0, "orig": 0, "flash": 0}
    # варианты
    sols = {}
    for f in a.var_solutions.split(","):
        sols.update(json.load(open(f)))
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=a.var_dir)
    avail = {e.game_id for e in arc.get_environments()}
    for gid, r in sols.items():
        if not r.get("solved") or gid not in avail:
            continue
        m = re.match(r"^([a-z0-9]{4})v\d{3}", gid); game = m.group(1) if m else gid[:4]
        try:
            if walk_path(arc, gid, [tuple(x) for x in r["path"]], acc, game, "var:" + gid, a.neg, rng):
                stats["var"] += 1
        except Exception as exc:
            print(gid, "упал:", repr(exc)[:80])
    # оригиналы: перебор + flash
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir="environment_files")
    envs = {e.game_id[:4]: e.game_id for e in arc.get_environments()}
    for f in sorted(glob.glob("runs/bfs2_16_09/bfs2_orig_exact*.json")):
        for gid, r in json.load(open(f)).items():
            if r.get("solved") and gid[:4] in envs:
                try:
                    if walk_path(arc, envs[gid[:4]], [tuple(x) for x in r["path"]], acc, gid[:4], "orig:" + gid[:4], a.neg, rng):
                        stats["orig"] += 1
                except Exception as exc:
                    print(gid, "упал:", repr(exc)[:80])
    for (g, run), path in flash_paths(a.runs.split(",")).items():
        if g in envs:
            try:
                if walk_path(arc, envs[g], path, acc, g, f"flash:{run}:{g}", a.neg, rng):
                    stats["flash"] += 1
            except Exception as exc:
                print(g, run, "упал:", repr(exc)[:80])
    np.savez_compressed(a.out, X=np.stack(acc.X), Xn=np.stack(acc.Xn), y=np.array(acc.y, np.int64), game=np.array(acc.g, np.int64), vid=np.array(acc.v, np.int64),
                        grp=np.array(acc.grp, np.int64), dist=np.array(acc.dist, np.int64), act=np.array(acc.act, np.int64), games=np.array(acc.games))
    per = {g: int((np.array(acc.g) == i).sum()) for i, g in enumerate(acc.games)}
    print(f"путей {stats}; примеров {len(acc.y)} (положительных {sum(acc.y)}), групп {acc.ngrp}, игр {len(acc.games)}: {per} -> {a.out}")


if __name__ == "__main__":
    main()

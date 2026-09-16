"""Дорешивание вариантов переносом известного решения (15.09, слово владельца «нерешённых 824 — давай их решим»).
Источники решений для (игра, уровень): (а) боевые партии flash из runs/flash_v1_phaseA (отрезок ходов между взятиями
уровней), (б) решения BFS на оригиналах (bfs_originals.json, уровень 1). Для варианта: повторить путь источника
(зеркальные варианты -- с зеркалированием ходов: lr: LEFT<->RIGHT, x -> 63-x; ud: UP<->DOWN, y -> 63-y); если уровень
не взят -- короткий BFS от последнего состояния пути (и от старта), с масками часов.
usage: solve_variants_by_transfer.py --env-dir <variants> --unsolved a.json,b.json --out solutions.json [--bfs-moves 20000]"""
import argparse, glob, json, re, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import engine_bfs as B

NAME2ID = {"UP": "ACTION1", "DOWN": "ACTION2", "LEFT": "ACTION3", "RIGHT": "ACTION4", "SPACE": "ACTION5"}


def flash_solutions(runs=("runs/flash_v1_phaseA", "runs/public_flash_tufa", "runs/public_flash_keithtyser")):
    """{(game4, level_index_from_1): [actions]} -- отрезки боевых партий между взятиями уровней (первое найденное)."""
    out = {}
    for run in runs:
        for p in glob.glob(f"{run}/artifacts/*_p0_events.jsonl"):
            g = Path(p).name[:4]; seg = []; lvl = 1
            for l in open(p, encoding="utf-8"):
                e = json.loads(l)
                if e.get("type") != "action":
                    continue
                name = str(e.get("action_name")); disp = str(e.get("action_display") or "")
                m = re.match(r"MOUSE\(row=(\d+), col=(\d+)\)", disp)
                act = ("ACTION6", {"x": int(m.group(2)), "y": int(m.group(1))}) if m else (name, None)
                if name == "RESET":
                    seg = []; continue
                seg.append(act)
                if e.get("level_completed") in (True, "True"):
                    out.setdefault((g, lvl), seg); lvl += 1; seg = []
    return out


def mirror_actions(path, axis):
    out = []
    for name, payload in path:
        if name == "ACTION6" and payload:
            x, y = payload["x"], payload["y"]
            out.append(("ACTION6", {"x": 63 - x if axis == "lr" else x, "y": 63 - y if axis == "ud" else y}))
        elif axis == "lr" and name in ("ACTION3", "ACTION4"):
            out.append(("ACTION4" if name == "ACTION3" else "ACTION3", None))
        elif axis == "ud" and name in ("ACTION1", "ACTION2"):
            out.append(("ACTION2" if name == "ACTION1" else "ACTION1", None))
        else:
            out.append((name, payload))
    return out


def variant_info(env_dir, gid):
    d = next(Path(env_dir).glob(f"{gid.split('-')[0]}/*/"))
    meta = json.loads((d / "metadata.json").read_text())
    src = next(d.glob("*.py")).read_text(encoding="utf-8")
    axis = "lr" if "set_mirror_lr(True)" in src else ("ud" if "set_mirror_ud(True)" in src else None)
    return meta.get("source_game"), int(meta.get("source_level", 1)), axis


def try_path(env, path):
    fr = env.reset_and_replay([])
    lvl0 = B.frame_info(fr)[1]
    for i, act in enumerate(path):
        fr = env.do(act)
        g, lvl, st, av = B.frame_info(fr)
        if lvl > lvl0:
            return path[: i + 1]
        if st == "GAME_OVER":
            return None
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--env-dir", required=True); ap.add_argument("--unsolved", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--bfs-moves", type=int, default=20000); ap.add_argument("--bfs-states", type=int, default=1500); a = ap.parse_args()
    import logging; logging.disable(logging.WARNING)
    import arc_agi
    from arc_agi import OperationMode
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=a.env_dir)
    avail = {e.game_id: e.game_id for e in arc.get_environments()}
    todo = {}
    for f in a.unsolved.split(","):
        for gid, r in json.load(open(f)).items():
            if not r.get("solved") and gid in avail:
                todo[gid] = r
    fs = flash_solutions()
    bfs_orig = {}
    p = ROOT / "runs" / "bfs_originals.json"
    if p.exists():
        for gid, r in json.load(open(p)).items():
            if r.get("solved"):
                bfs_orig[(gid[:4], 1)] = [tuple(x) if isinstance(x, list) else x for x in r["path"]]
                bfs_orig[(gid[:4], 1)] = [(n, pl) for n, pl in bfs_orig[(gid[:4], 1)]]
    print(f"нерешённых {len(todo)}; решений flash {len(fs)}, BFS-оригиналов {len(bfs_orig)}", flush=True)
    res = {}; n_solved = 0; t0 = time.time(); how = {"transfer": 0, "transfer+bfs": 0, "bfs": 0}
    for i, gid in enumerate(sorted(todo)):
        try:
            g, lv, axis = variant_info(a.env_dir, gid)
            cands = []
            for key, path in ((("flash", fs.get((g, lv)))), ("bfs", bfs_orig.get((g, lv)))):
                if path:
                    cands.append(mirror_actions(path, axis) if axis else list(path))
            env = B.Env(arc, gid); found = None; way = None
            for path in cands:
                found = try_path(env, path)
                if found:
                    way = "transfer"; break
            if not found and cands and a.bfs_moves > 0:
                # короткий BFS от конца перенесённого пути (уровень не взят, но, возможно, близко)
                pref = cands[0]
                env.reset_and_replay([]); prefix = []
                for act in pref:
                    fr = env.do(act); g2, lvl, st, av = B.frame_info(fr)
                    if st == "GAME_OVER":
                        break
                    prefix.append(act)
                r = B.solve_from(arc, gid, prefix, a.bfs_states, a.bfs_moves) if hasattr(B, "solve_from") else None
                if r and r.get("solved"):
                    found = r["path"]; way = "transfer+bfs"
            if not found and a.bfs_moves > 0:
                r = B.solve(arc, gid, a.bfs_states, a.bfs_moves, 24)
                if r.get("solved"):
                    found = r["path"]; way = "bfs"
            if found:
                n_solved += 1; how[way] += 1
                res[gid] = {"solved": True, "path": [list(x) if isinstance(x, tuple) else x for x in found], "how": way, "source": [g, lv, axis]}
            else:
                res[gid] = {"solved": False, "path": None, "source": [g, lv, axis]}
        except Exception as exc:
            res[gid] = {"solved": False, "path": None, "error": repr(exc)[:160]}
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(todo)}: решено {n_solved} {how}, {time.time() - t0:.0f} с", flush=True)
            Path(a.out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"итого решено {n_solved} из {len(todo)}: {how}; {time.time() - t0:.0f} с -> {a.out}")


if __name__ == "__main__":
    main()

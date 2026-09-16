"""BFS по движку со снимками состояния (16.09): copy.deepcopy(env) даёт точную копию за ~1 мс, поэтому раскрытие узла
стоит один ход, а не RESET + повтор пути (engine_bfs.py). Два режима рёбер:
  exact  -- каждое простое действие/клик = ребро (как engine_bfs.py);
  both   -- и одиночный шаг, и макроход (полнота exact, ускорение macro);
  macro  -- стрелка повторяется, пока доска меняется «одинаково» (перенос того же набора клеток/цветов); остановка при
            отсутствии изменений (стена), при изменении характера сдвига (взаимодействие), при уровне/проигрыше, при 24 ходах.
            Клики -- по-прежнему одиночные рёбра. Проверка критика (раунд 6, тест №2): падает ли ветвление >= 5x.
Ключ состояния: доска с маской часов (как в engine_bfs.py). Метрики: states, expansions, depth (рёбер / ходов), moves,
seconds, solved, path (примитивные ходы).
usage: engine_bfs2.py --env-dir <dir> --games a,b [--mode exact|macro] [--max-states 3000] [--max-moves 60000] [--timeout 300] [--max-levels 1] --out res.json"""
import argparse, copy, json, sys, time
from collections import deque
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from arcengine import GameAction
from engine_bfs import frame_info, alphabet

SIMPLE = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")


class Snap:
    """Снимок среды: env (deepcopy-able LocalEnvironmentWrapper), доска, уровень, доступные действия."""
    __slots__ = ("env", "grid", "lvl", "state", "avail")

    def __init__(self, env, fr):
        self.env = env
        self.grid, self.lvl, self.state, self.avail = frame_info(fr)


class Counter:
    def __init__(self, max_moves, deadline):
        self.moves = 0; self.max_moves = max_moves; self.deadline = deadline

    def exhausted(self):
        return self.moves >= self.max_moves or time.time() > self.deadline


def step(snap, act, cnt):
    """Копия снимка + один ход. Возвращает новый Snap (env копии) или None при исключении."""
    env = copy.deepcopy(snap.env)
    name, payload = act
    a = GameAction.RESET if name == "RESET" else GameAction[name]
    try:
        fr = env.step(a, data=dict(payload) if payload else None)
    except Exception:
        return None
    if fr is None:
        return None
    cnt.moves += 1
    return Snap(env, fr)


def root_snap(arc, gid, cnt):
    env = arc.make(gid)
    if env is None:
        raise RuntimeError("env creation failed")
    fr = env.reset(); cnt.moves += 1
    if fr is None:
        fr = env.step(GameAction.RESET)
    return Snap(env, fr)


def simple_actions(avail):
    return [("ACTION" + a[-1], None) for a in avail if a in ("1", "2", "3", "4", "5") + SIMPLE]


def active_clicks(root, cnt, step_px=4):
    if not any(a in ("6", "ACTION6") for a in root.avail):
        return []
    from agent.harness.perception import segment
    pts = [(x, y) for y in range(step_px // 2, 64, step_px) for x in range(step_px // 2, 64, step_px)]
    for o in segment(root.grid).non_background()[:60]:
        pts.append((int(round(o.centroid[1])), int(round(o.centroid[0]))))
    active = []; seen = set()
    for x, y in pts:
        if (x, y) in seen:
            continue
        seen.add((x, y))
        s = step(root, ("ACTION6", {"x": x, "y": y}), cnt)
        if s is None:
            continue
        if (s.grid != root.grid).any() or s.lvl > root.lvl:
            active.append(("ACTION6", {"x": x, "y": y}))
    return active


def clock_mask(root, cnt, n_clicks=24, warm=60, thr=0.8):
    import random as _r
    rng = _r.Random(0); acts = alphabet(root.grid, root.avail, n_clicks)
    if not acts:
        return np.zeros_like(root.grid, dtype=bool)
    changes = np.zeros_like(root.grid, dtype=np.int32); n = 0; prev = root.grid
    row_ch = np.zeros(root.grid.shape[0], dtype=np.int32); col_ch = np.zeros(root.grid.shape[1], dtype=np.int32)
    cur = Snap(copy.deepcopy(root.env), None); cur.grid, cur.lvl, cur.state, cur.avail = root.grid, root.lvl, root.state, root.avail
    for _ in range(warm):
        nxt = step(cur, rng.choice(acts), cnt)
        if nxt is None:
            break
        if nxt.state == "GAME_OVER" or nxt.lvl > root.lvl:
            cur = Snap(copy.deepcopy(root.env), None); cur.grid, cur.lvl, cur.state, cur.avail = root.grid, root.lvl, root.state, root.avail; prev = root.grid; continue
        g = nxt.grid
        if g.shape == prev.shape:
            d = (g != prev)
            if d.any():
                changes += d; n += 1; row_ch += d.any(axis=1); col_ch += d.any(axis=0)
        prev = g; cur = nxt
    mask = (changes >= thr * max(1, n)) & (changes >= 3)
    for r in np.where(row_ch >= thr * max(1, n))[0]:
        if n >= 5: mask[r, :] = True
    for c in np.where(col_ch >= thr * max(1, n))[0]:
        if n >= 5: mask[:, c] = True
    return mask


def macro_step(snap, act, cnt, mask, lvl0, cap=24):
    """Повтор стрелки до стены / взаимодействия / конца. Возвращает (Snap, список примитивных ходов, причина) или None."""
    cur = snap; prev = snap.grid; steps = []; last_sig = None
    for _ in range(cap):
        nxt = step(cur, act, cnt)
        if nxt is None:
            return None
        if nxt.state == "GAME_OVER" or nxt.lvl > lvl0:
            return nxt, steps + [act], "end"
        d = (nxt.grid != prev) & ~mask
        if not d.any():
            return (cur, steps, "stuck") if steps else None
        sig = (int(d.sum()), tuple(sorted(set(prev[d].tolist()) | set(nxt.grid[d].tolist()))))
        steps.append(act); cur = nxt; prev = nxt.grid
        if last_sig is not None and sig != last_sig:
            return cur, steps, "interaction"
        last_sig = sig
    return cur, steps, "cap"


def solve_level(root, cnt, mode, max_states, key_of, clicks):
    """BFS одного уровня от снимка root. Возвращает (found_path | None, stats)."""
    lvl0 = root.lvl; mask = key_of.mask
    seen = {key_of(root.grid): 0}; q = deque([(root, [], 0)]); expansions = 0; max_depth = 0; max_moves_depth = 0; found = None
    while q and len(seen) < max_states and not cnt.exhausted() and found is None:
        snap, path, depth = q.popleft(); expansions += 1
        edges = []
        for act in simple_actions(snap.avail):
            if mode in ("macro", "both") and act[0] != "ACTION5":
                r = macro_step(snap, act, cnt, mask, lvl0)
                if r is not None:
                    edges.append((r[0], r[1]))
            if mode != "macro":
                s = step(snap, act, cnt)
                if s is not None:
                    edges.append((s, [act]))
            if cnt.exhausted():
                break
        for act in clicks:
            if cnt.exhausted():
                break
            s = step(snap, act, cnt)
            if s is not None:
                edges.append((s, [act]))
        for s, acts in edges:
            if s.lvl > lvl0:
                found = path + acts; break
            if s.state == "GAME_OVER":
                continue
            k = key_of(s.grid)
            if k in seen:
                continue
            seen[k] = depth + 1; q.append((s, path + acts, depth + 1))
            max_depth = max(max_depth, depth + 1); max_moves_depth = max(max_moves_depth, len(path) + len(acts))
    return found, {"states": len(seen), "expansions": expansions, "depth_edges": max_depth, "depth_moves": max_moves_depth}


class KeyOf:
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, g):
        k = g.copy(); k[self.mask] = -1; return k.tobytes()


def solve(arc, gid, mode, max_states, max_moves, timeout, max_levels=1):
    t0 = time.time(); cnt = Counter(max_moves, t0 + timeout)
    root = root_snap(arc, gid, cnt); prefix = []; levels = 0; per_level = []; stats_all = []
    while levels < max_levels and not cnt.exhausted():
        clicks = active_clicks(root, cnt)
        key_of = KeyOf(clock_mask(root, cnt))
        found, st = solve_level(root, cnt, mode, max_states, key_of, clicks)
        st["clicks"] = len(clicks); stats_all.append(st)
        if found is None:
            break
        prefix = prefix + found; levels += 1; per_level.append(len(found))
        # новый корень: повторяем найденный путь от текущего корня (снимок)
        cur = root
        for act in found:
            cur = step(cur, act, cnt)
        root = cur
    agg = {"states": sum(s["states"] for s in stats_all), "expansions": sum(s["expansions"] for s in stats_all),
           "depth_edges": max([s["depth_edges"] for s in stats_all] or [0]), "depth_moves": max([s["depth_moves"] for s in stats_all] or [0]),
           "clicks": stats_all[0]["clicks"] if stats_all else 0}
    return {"solved": levels > 0, "levels": levels, "path": prefix if levels else None, "per_level": per_level, "mode": mode,
            "moves": cnt.moves, "seconds": round(time.time() - t0, 1), "timeout": time.time() > cnt.deadline, **agg}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--env-dir", required=True); ap.add_argument("--games", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="exact", choices=["exact", "macro", "both"]); ap.add_argument("--max-states", type=int, default=3000)
    ap.add_argument("--max-moves", type=int, default=60000); ap.add_argument("--timeout", type=int, default=300); ap.add_argument("--max-levels", type=int, default=1)
    a = ap.parse_args()
    import logging; logging.disable(logging.ERROR)
    import arc_agi
    from arc_agi import OperationMode
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=a.env_dir)
    envs = {e.game_id: e.game_id for e in arc.get_environments()}
    res = {}
    for g in a.games.split(","):
        gid = next((k for k in envs if k.startswith(g)), None)
        if gid is None:
            res[g] = {"solved": False, "error": "no env"}; continue
        try:
            r = solve(arc, gid, a.mode, a.max_states, a.max_moves, a.timeout, a.max_levels)
        except Exception as exc:
            r = {"solved": False, "error": repr(exc)[:200]}
        res[gid] = r
        print(f"{gid[:4]} {a.mode}: {'РЕШЕНО ' + str(r.get('levels')) + ' ур, путь ' + str(len(r['path'])) if r.get('solved') else 'нет'}; "
              f"состояний {r.get('states')}, раскрытий {r.get('expansions')}, глубина {r.get('depth_edges')} рёбер / {r.get('depth_moves')} ходов, "
              f"ходов {r.get('moves')}, {r.get('seconds')} с{' (таймаут)' if r.get('timeout') else ''}{' ' + r['error'] if r.get('error') else ''}", flush=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

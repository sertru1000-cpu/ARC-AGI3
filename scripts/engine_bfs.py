"""BFS по настоящему движку: кратчайший путь до взятия уровня 1 (или до цели) без знания правил (15.09).
Состояние = точная доска; расширение узла = RESET + повтор пути + действие (движок детерминирован, 100–480 ходов/с).
Алфавит: доступные простые действия + клики по разнообразным целям (малые объекты первыми) для игр с MOUSE.
usage: engine_bfs.py --env-dir <dir> [--games a,b] [--max-states 3000] [--max-moves 80000] [--clicks 24] [--out solutions.json]
Пишет для каждой игры: solved, path (список действий), states, moves, seconds."""
import argparse, json, sys, time
from collections import deque
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from replay_battle_local import Scripted
from agent.harness.perception import spread_click_targets
from arcengine import GameAction, GameState


class Env:
    """Один экземпляр среды с повтором пути: cheap step-by-step через Agent.take_action."""
    def __init__(self, arc, gid):
        self.arc, self.gid = arc, gid
        self.env = arc.make(gid)
        self.ag = Scripted(card_id="bfs", game_id=gid, agent_name="bfs." + gid, ROOT_URL="http://localhost", record=False, arc_env=self.env, tags=["bfs"], script=[])
        self.moves = 0

    def do(self, act):
        name, payload = act
        a = GameAction.RESET if name == "RESET" else GameAction[name]
        if payload:
            a.set_data(payload)
        fr = self.ag.take_action(a); self.moves += 1
        if fr is None:
            return None
        return fr

    def reset_and_replay(self, path):
        fr = self.do(("RESET", None))
        for act in path:
            fr = self.do(act)
        return fr


def frame_info(fr):
    grid = np.asarray(fr.frame[-1], dtype=np.int16)
    avail = [str(a).split(".")[-1] for a in (fr.available_actions or [])]
    return grid, int(fr.levels_completed or 0), str(fr.state).split(".")[-1], avail


def alphabet(grid, avail, n_clicks):
    acts = []
    for a in avail:
        if a in ("1", "2", "3", "4", "5", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"):
            acts.append(("ACTION" + a[-1], None))
    if any(a in ("6", "ACTION6") for a in avail):
        seen_xy = set()
        for x, y in spread_click_targets(grid, n_clicks):
            seen_xy.add((x, y)); acts.append(("ACTION6", {"x": x, "y": y}))
        from agent.harness.perception import segment
        for o in sorted(segment(grid).non_background(), key=lambda o: -o.cells)[:16]:   # крупные объекты тоже
            x, y = int(round(o.centroid[1])), int(round(o.centroid[0]))
            if (x, y) not in seen_xy:
                seen_xy.add((x, y)); acts.append(("ACTION6", {"x": x, "y": y}))
    return acts


def clock_mask(env, grid0, avail, n_clicks, warm=60, thr=0.8):
    """Клетки-часы: меняются в >= thr доле ходов разминки (счётчики, таймеры) -- исключаются из ключа состояния."""
    import random as _r
    rng = _r.Random(0); acts = alphabet(grid0, avail, n_clicks)
    if not acts:
        return np.zeros_like(grid0, dtype=bool)
    changes = np.zeros_like(grid0, dtype=np.int32); n = 0; prev = grid0; path = []
    row_ch = np.zeros(grid0.shape[0], dtype=np.int32); col_ch = np.zeros(grid0.shape[1], dtype=np.int32)
    for _ in range(warm):
        act = rng.choice(acts)
        fr = env.do(act)
        if fr is None:
            break
        g, lvl, st, av = frame_info(fr)
        if st == "GAME_OVER":
            fr = env.reset_and_replay([]); prev = frame_info(fr)[0]; continue
        if g.shape == prev.shape:
            d = (g != prev)
            if d.any():
                changes += d; n += 1
                row_ch += d.any(axis=1); col_ch += d.any(axis=0)
        prev = g
    env.reset_and_replay([])
    mask = (changes >= thr * max(1, n)) & (changes >= 3)
    # полосы-счётчики (cn04: строка 0 укорачивается на клетку за ход; re86: строка 63; tn36: строка 1): строка/столбец,
    # менявшиеся в >= thr доле ходов с изменениями, маскируются целиком
    for r in np.where(row_ch >= thr * max(1, n))[0]:
        if n >= 5: mask[r, :] = True
    for c in np.where(col_ch >= thr * max(1, n))[0]:
        if n >= 5: mask[:, c] = True
    return mask


def active_clicks(env, grid0, avail, step=4):
    """Клики, которые меняют доску из стартового состояния: сетка с шагом step + центры объектов (ft09: блоки
    сливаются с фоном при сегментации, центры объектов их не задевают). Проверка -- по одному ходу на точку."""
    if not any(a in ("6", "ACTION6") for a in avail):
        return []
    from agent.harness.perception import segment
    pts = [(x, y) for y in range(step // 2, 64, step) for x in range(step // 2, 64, step)]
    for o in segment(grid0).non_background()[:60]:
        pts.append((int(round(o.centroid[1])), int(round(o.centroid[0]))))
    active = []; seen = set()
    for x, y in pts:
        if (x, y) in seen:
            continue
        seen.add((x, y))
        fr = env.reset_and_replay([("ACTION6", {"x": x, "y": y})])
        if fr is None:
            continue
        g, lvl, st, av = frame_info(fr)
        if (g != grid0).any() or lvl > 0:
            active.append(("ACTION6", {"x": x, "y": y}))
    return active


def solve(arc, gid, max_states, max_moves, n_clicks, target_levels=1):
    t0 = time.time(); env = Env(arc, gid)
    fr = env.reset_and_replay([]); grid, lvl0, state, avail = frame_info(fr)
    clicks = active_clicks(env, grid, avail)
    env.moves = 0
    mask = clock_mask(env, grid, avail, n_clicks)
    def key_of(g):
        k = g.copy(); k[mask] = -1; return k.tobytes()
    start_key = key_of(grid)
    seen = {start_key: []}; q = deque([(start_key, [], grid, avail)])
    while q and len(seen) < max_states and env.moves < max_moves:
        key, path, grid, avail = q.popleft()
        simple = [("ACTION" + a[-1], None) for a in avail if a in ("1", "2", "3", "4", "5", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")]
        for act in simple + clicks:
            fr = env.reset_and_replay(path + [act])
            if fr is None:
                continue
            g2, lvl, st, av2 = frame_info(fr)
            if lvl >= lvl0 + target_levels:
                return {"solved": True, "path": path + [act], "states": len(seen), "moves": env.moves, "seconds": round(time.time() - t0, 1), "clock_cells": int(mask.sum())}
            if st == "GAME_OVER":
                continue
            k2 = key_of(g2)
            if k2 in seen:
                continue
            seen[k2] = path + [act]; q.append((k2, path + [act], g2, av2))
            if env.moves >= max_moves:
                break
    return {"solved": False, "path": None, "states": len(seen), "moves": env.moves, "seconds": round(time.time() - t0, 1), "clock_cells": int(mask.sum()), "max_depth": max((len(p) for p in seen.values()), default=0), "clicks": len(clicks)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--env-dir", required=True); ap.add_argument("--games", default=None)
    ap.add_argument("--max-states", type=int, default=3000); ap.add_argument("--max-moves", type=int, default=80000); ap.add_argument("--clicks", type=int, default=24)
    ap.add_argument("--out", default=None); a = ap.parse_args()
    import logging; logging.disable(logging.WARNING)
    import arc_agi
    from arc_agi import OperationMode
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=a.env_dir)
    gids = [e.game_id for e in arc.get_environments()]
    if a.games:
        want = set(a.games.split(",")); gids = [g for g in gids if g.split("-")[0] in want]
    res = {}
    for gid in gids:
        r = solve(arc, gid, a.max_states, a.max_moves, a.clicks); res[gid] = r
        print(f"{gid}: {'РЕШЕНО' if r['solved'] else 'нет'} | путь {len(r['path']) if r['path'] else '-'} | состояний {r['states']} | глубина {r.get('max_depth', '-')} | часов {r.get('clock_cells')} | ходов {r['moves']} | {r['seconds']} с", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("решено", sum(1 for r in res.values() if r["solved"]), "из", len(res))


if __name__ == "__main__":
    main()

"""Граф-исследователь без модели (18.09, разбор направления по arXiv 2512.24156 и механике счёта).

Замысел. Наш счёт прощает дорогое исследование ровно в одном случае: игра взята ЦЕЛИКОМ. Тогда движок делает полный
сброс с новыми счётчиками (`handle_reset` при WIN), а балл игры = МАКСИМУМ по прогонам, то есть можно повторить
найденный путь оптимально (память arc-agi-3-scoring-mechanics, проверено по исходникам 16.08). Частичное продвижение
перебором балла не даёт — это мы уже измерили трижды. Значит, вопрос один: доходит ли безмодельное исследование
до КОНЦА игры в отведённое время.

Бюджет. В бою ход через шлюз стоит 20-50 мс, то есть за 2 часа игры доступно 150-350 тысяч ходов; наша модель делает
26-30. Здесь бюджет задаётся в ходах (--moves), чтобы числа переносились на бой.

Устройство (как в работе, но на нашем движке):
  * состояние = доска с маской клеток-часов (счётчики/таймеры не входят в ключ);
  * граф: узел -> испробованные действия -> узлы-наследники; помним, откуда пришли (для возврата кратчайшим путём);
  * приоритет действий: сначала непроверенные пары «состояние-действие», клики — по заметным точкам (центры малых
    объектов и сетка с шагом), стрелки/SPACE всегда;
  * когда в текущем узле всё испробовано, идём кратчайшим путём (по графу) к ближайшему узлу с непроверенными парами;
    если путь недоступен -- RESET уровня и повтор пути от старта (как в бою);
  * уровень взят -> граф обнуляется (новая доска), путь до него запоминается.
Возврат по графу считается ходами, как в бою.

usage:  .venv/bin/python scripts/graph_explorer.py --games all --moves 150000 [--out runs/graph_explorer.json]
"""
import argparse, json, random, sys, time
from collections import deque
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import logging; logging.disable(logging.ERROR)
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
from engine_bfs import Env, frame_info
from agent.harness.perception import segment

SIMPLE = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")


def salient_clicks(grid, cap=40, step=8):
    """точки для кликов: центры небольших объектов (заметность) + редкая сетка."""
    pts = []
    try:
        for o in sorted(segment(grid).non_background(), key=lambda o: o.cells)[:cap]:
            pts.append((int(round(o.centroid[0])), int(round(o.centroid[1]))))
    except Exception:
        pass
    for y in range(step // 2, grid.shape[0], step):
        for x in range(step // 2, grid.shape[1], step):
            pts.append((y, x))
    seen = set(); out = []
    for p in pts:
        if p not in seen:
            seen.add(p); out.append(("ACTION6", {"x": int(p[1]), "y": int(p[0])}))
    return out[:cap + 64]


def alphabet_for(grid, avail):
    """движок отдаёт доступные действия и как «1..6», и как «ACTION1..ACTION6» -- принимаем оба вида (ловушка 18.09)."""
    simple = [("ACTION" + a[-1], None) for a in avail if a in ("1", "2", "3", "4", "5") + SIMPLE]
    clicks = salient_clicks(grid) if any(a in ("6", "ACTION6") for a in avail) else []
    return simple + clicks


def clock_mask(env, grid0, acts, budget=60):
    rng = random.Random(0); ch = np.zeros_like(grid0, dtype=np.int32); n = 0; prev = grid0
    for _ in range(budget):
        fr = env.do(rng.choice(acts))
        g, lvl, st, _ = frame_info(fr)
        if st == "GAME_OVER" or lvl > 0:
            fr = env.reset_and_replay([]); prev = frame_info(fr)[0]; continue
        if g.shape == prev.shape and (g != prev).any():
            ch += (g != prev); n += 1
        prev = g
    m = (ch >= 0.8 * max(1, n)) & (ch >= 3)
    if m.sum() > 0.5 * m.size:   # предохранитель: маска не должна съедать доску, иначе все состояния схлопнутся
        m[:] = False
    return m


def explore_game(arc, gid, max_moves, timeout, verbose=False):
    t0 = time.time()
    env = Env(arc, gid)
    fr = env.reset_and_replay([])
    grid, lvl0, state, avail = frame_info(fr)
    acts = alphabet_for(grid, avail)
    mask = clock_mask(env, grid, acts)
    def key(g):
        k = np.asarray(g, dtype=np.int16).copy()
        if mask.shape == k.shape:
            k[mask] = -1
        return k.tobytes()
    levels = lvl0; level_paths = []
    fr = env.reset_and_replay([]); grid, lvl, state, avail = frame_info(fr)
    nodes = {key(grid): {"path": [], "untried": list(acts)}}
    cur = key(grid); cur_path = []
    stats = {"game": gid[:4], "levels": 0, "moves": 0, "nodes": 0, "returns": 0, "resets": 0, "seconds": 0.0}
    while env.moves < max_moves and time.time() - t0 < timeout:
        node = nodes.get(cur)
        if node is None:
            nodes[cur] = node = {"path": list(cur_path), "untried": list(acts)}
        if not node["untried"] and not node.get("succ"):
            node["untried"] = list(acts)   # узел, полученный возвратом: действия ещё не пробовались отсюда
        if not node["untried"]:
            # кратчайший путь по графу к ближайшему узлу с непроверенными действиями
            target = None
            q = deque([cur]); seen = {cur}
            while q:
                k = q.popleft()
                if nodes[k]["untried"]:
                    target = k; break
                for kk in nodes[k].get("succ", []):
                    if kk in nodes and kk not in seen:
                        seen.add(kk); q.append(kk)
            if target is None:
                target = next((k for k, v in nodes.items() if v["untried"]), None)
            if target is None:
                break
            stats["returns"] += 1; stats["resets"] += 1
            fr = env.reset_and_replay(nodes[target]["path"])
            g, lvl, state, avail = frame_info(fr)
            if lvl > levels:
                levels = lvl
            cur = key(g); cur_path = list(nodes[target]["path"])
            continue
        act = node["untried"].pop(0)
        fr = env.do(act)
        g, lvl, state, avail = frame_info(fr)
        if lvl > levels:   # уровень взят: граф обнуляется, путь запоминается
            levels = lvl; level_paths.append(list(cur_path) + [act])
            stats["levels"] = levels - lvl0
            if verbose:
                print("   %s: уровень %d на ходу %d" % (gid[:4], levels, env.moves), flush=True)
            nodes = {}; cur_path = []
            cur = key(g); nodes[cur] = {"path": [], "untried": list(acts), "succ": []}
            continue
        if state == "GAME_OVER":
            fr = env.reset_and_replay(cur_path); g, lvl, state, avail = frame_info(fr)
            stats["resets"] += 1
            continue
        k2 = key(g)
        node.setdefault("succ", []).append(k2)
        if k2 not in nodes:
            nodes[k2] = {"path": list(cur_path) + [act], "untried": list(acts), "succ": []}
        if k2 != cur:
            cur = k2; cur_path = list(nodes[k2]["path"])
    stats.update({"moves": env.moves, "nodes": len(nodes), "seconds": round(time.time() - t0, 1),
                  "levels": levels - lvl0, "won": state == "WIN"})
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="all"); ap.add_argument("--moves", type=int, default=150000)
    ap.add_argument("--timeout", type=float, default=900.0); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT / "environment_files"))
    ids = [e.game_id for e in arc.get_environments()]
    if a.games != "all":
        want = set(a.games.split(",")); ids = [g for g in ids if g[:4] in want]
    rows = []
    for gid in ids:
        try:
            r = explore_game(arc, gid, a.moves, a.timeout, verbose=True)
        except Exception as exc:
            r = {"game": gid[:4], "error": repr(exc)[:200], "levels": 0}
        rows.append(r); print(json.dumps(r, ensure_ascii=False), flush=True)
    lv = [r.get("levels", 0) for r in rows]
    print("\nИТОГ: игр %d; уровней всего %d; медиана на игру %.1f; игр с уровнем %d; ПОЛНЫХ ПОБЕД %d"
          % (len(rows), sum(lv), float(np.median(lv)) if lv else 0, sum(1 for x in lv if x >= 1), sum(1 for r in rows if r.get("won"))))
    if a.out:
        json.dump(rows, open(ROOT / a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print("записано:", a.out)


if __name__ == "__main__":
    main()

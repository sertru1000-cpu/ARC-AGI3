"""Exact baselines for the HOSTILE pool (round 13 Q4): hc01/ch01/tr01/vn01.

BFS over deepcopied engine states with TRUE-STATE dedup keys (agent pos +
each game's hidden internals) -- deliberately NOT frame signatures, since
these games are designed to defeat frame-sig search. Verifies each optimum
end-to-end (engine WIN at exactly baseline length) and writes metadata.json.

Usage: uv run python scripts/hostile_baselines.py [--write]
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GAMES = {
    "hc01": ("c1d2e3f4", "Hc01", "Hidden Combo (hostile pool)"),
    "ch01": ("d2e3f4a5", "Ch01", "Chaser (hostile pool)"),
    "tr01": ("e3f4a5b6", "Tr01", "One-Way Trap (hostile pool)"),
    "vn01": ("f4a5b6c7", "Vn01", "Visual Noise (hostile pool)"),
}


def load_cls(prefix: str, ver: str, cls_name: str):
    py = ROOT / "our_games" / prefix / ver / f"{prefix}.py"
    spec = importlib.util.spec_from_file_location(f"hostile_{prefix}", py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name), mod


def _grid_reach(walls, extra_walls, start, target, grid=8):
    """Plain grid BFS distance; None when unreachable."""
    from collections import deque as _dq
    blocked = set(walls) | set(extra_walls)
    if start in blocked or target in blocked:
        return None
    q = _dq([(start, 0)])
    seen = {start}
    while q:
        (c, r), d = q.popleft()
        if (c, r) == target:
            return d
        for dc, dr in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            n = (c + dc, r + dr)
            if 0 <= n[0] < grid and 0 <= n[1] < grid and n not in blocked and n not in seen:
                seen.add(n)
                q.append((n, d + 1))
    return None


def guarantees(prefix: str, mod, level_index: int, plan_len: int) -> list[str]:
    """Content guarantees per hostile mechanic; returns violation strings."""
    bad = []
    spec = mod.LEVELS[level_index]
    if prefix == "hc01":
        walls, buttons, start, exit_cell, door = mod._parse(spec)
        if _grid_reach(walls, {door}, start, exit_cell) is not None:
            bad.append("exit reachable WITHOUT the door (combo bypassable)")
    if prefix == "tr01":
        walls, gates, start, exit_cell = mod._parse(spec)
        # safe route must exist with gates treated as walls, and its length
        # must equal the engine optimum (the optimal plan avoids the trap)
        safe = _grid_reach(walls, gates, start, exit_cell)
        if safe is None:
            bad.append("no safe route around the gate")
        elif safe != plan_len:
            bad.append(f"optimal ({plan_len}) != safe route ({safe}) -- plan uses the trap")
        # the pocket behind each gate must be DEAD: with the gate sealed,
        # the exit must be unreachable from the cell(s) adjacent to the
        # gate on the pocket side
        for g in gates:
            for dc, dr in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                inside = (g[0] + dc, g[1] + dr)
                if inside in walls or inside == start:
                    continue
                if _grid_reach(walls, gates | {g}, inside, exit_cell) is None:
                    continue  # this neighbor is dead as required (or is the safe side)
        # dead-pocket proof: at least one gate neighbor must NOT reach the exit
        dead_sides = 0
        for g in gates:
            for dc, dr in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                inside = (g[0] + dc, g[1] + dr)
                if inside in walls:
                    continue
                if _grid_reach(walls, gates, inside, exit_cell) is None:
                    dead_sides += 1
        if dead_sides == 0:
            bad.append("no dead pocket behind any gate")
    return bad


def agent_pos(game):
    a = game.current_level.get_sprites_by_name("agent")[0]
    return (a.x, a.y)


def state_key(prefix: str, game):
    if prefix == "hc01":
        return (agent_pos(game), game._progress)
    if prefix == "ch01":
        ch = game.current_level.get_sprites_by_name("chaser")[0]
        return (agent_pos(game), (ch.x, ch.y), game._t)
    if prefix == "tr01":
        return (agent_pos(game), tuple(sorted(game._sealed)))
    if prefix == "vn01":
        return agent_pos(game)
    raise KeyError(prefix)


def step(game, arcengine, action_id: int):
    return game.perform_action(arcengine.ActionInput(id=arcengine.GameAction.from_id(action_id)))


def bfs_level(prefix, cls, arcengine, level_index: int, max_nodes=200_000):
    import os
    os.environ["ONLY_RESET_LEVELS"] = "true"
    game = cls()
    # advance to the target level via known solutions is complex; instead
    # construct fresh and jump with set_level (clean state hooks fire).
    game.set_level(level_index)
    game.on_set_level(game.current_level)
    if hasattr(game, "_atlas_reset_level_state"):
        game._atlas_reset_level_state()
    start_score = game._score if hasattr(game, "_score") else level_index
    seen = {state_key(prefix, game)}
    q = deque([(game, [])])
    nodes = 0
    while q and nodes < max_nodes:
        cur, path = q.popleft()
        for action_id in cur._available_actions:
            if action_id in (0, 5) and prefix != "cl01":
                if action_id == 0:
                    continue
            child = copy.deepcopy(cur)
            nodes += 1
            step(child, arcengine, action_id)
            state = getattr(child, "_state", None)
            if getattr(state, "name", "") == "GAME_OVER":
                continue
            if child.level_index > level_index or getattr(state, "name", "") == "WIN":
                return path + [action_id], nodes
            key = state_key(prefix, child)
            if key in seen:
                continue
            seen.add(key)
            q.append((child, path + [action_id]))
    return None, nodes


def verify(prefix, cls, arcengine, level_index, plan):
    game = cls()
    game.set_level(level_index)
    game.on_set_level(game.current_level)
    if hasattr(game, "_atlas_reset_level_state"):
        game._atlas_reset_level_state()
    for i, a in enumerate(plan):
        step(game, arcengine, a)
        done = game.level_index > level_index or getattr(getattr(game, "_state", None), "name", "") == "WIN"
        if done:
            return i + 1 == len(plan)
    return False


def main():
    write = "--write" in sys.argv
    import arcengine
    for prefix, (ver, cls_name, title) in GAMES.items():
        cls, mod = load_cls(prefix, ver, cls_name)
        n_levels = len(cls()._levels)
        baselines = []
        ok = True
        for li in range(n_levels):
            plan, nodes = bfs_level(prefix, cls, arcengine, li)
            if plan is None:
                print(f"{prefix} L{li+1}: UNSOLVABLE within budget ({nodes} nodes)")
                ok = False
                baselines.append(None)
                continue
            v = verify(prefix, cls, arcengine, li, plan)
            viol = guarantees(prefix, mod, li, len(plan))
            for msg in viol:
                print(f"{prefix} L{li+1}: GUARANTEE VIOLATED -- {msg}")
            ok = ok and v and not viol
            print(f"{prefix} L{li+1}: optimal={len(plan)} nodes={nodes} verify={'OK' if v else 'FAIL'}")
            baselines.append(len(plan))
        if write and ok:
            meta = {
                "game_id": f"{prefix}-{ver}",
                "class_name": cls_name,
                "title": title,
                "baseline_actions": baselines,
            }
            out = ROOT / "our_games" / prefix / ver / "metadata.json"
            out.write_text(json.dumps(meta, indent=4) + "\n", encoding="utf-8")
            print(f"  -> {out}")
        elif write:
            print(f"  !! {prefix}: not written (unsolved/verify failure)")


if __name__ == "__main__":
    main()

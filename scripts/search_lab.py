"""Search lab over the testbed corpus (backlog 19, items 1-2).

Pure-engine, zero-LLM experiments on our validated levels:

  bench    -- BFS coverage vs node budget (250/750/1500/2500): how many of
              the corpus levels engine search cracks, at what cost. The
              empirical check of the round-10b cap raise. Also harvests
              (state-features -> remaining-steps) training pairs from every
              solved level for the A* heuristic.
  train    -- fit a light h(s) regressor on the harvested pairs (sklearn
              GradientBoosting, holdout packs excluded) and report MAE.
  astar    -- rerun the search on HOLDOUT levels as best-first (g + w*h)
              vs plain BFS: nodes-to-solve comparison. The go/no-go number
              for wiring A* into plan_real.

Games with MOUSE controls (mr01, rg01) are out of scope for v1.

Usage:
  python scripts/search_lab.py bench
  python scripts/search_lab.py train
  python scripts/search_lab.py astar
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "search_lab"
GAME_DIRS = [ROOT / "our_games", ROOT / "our_games_gen"]
SKIP_PREFIXES = ("mr01", "rg01")          # MOUSE-gated mechanics
HOLDOUT = ("fg04", "fg05", "pg04", "pg05", "fl01", "ph01")  # never in training
BUDGETS = (250, 750, 1500, 2500)
PER_LEVEL_WALL_S = 45.0
MAX_ACTION_ID = 5


def iter_games():
    for base in GAME_DIRS:
        for mf in sorted(base.rglob("metadata.json")):
            meta = json.loads(mf.read_text(encoding="utf-8"))
            gid = meta["game_id"]
            prefix = gid.split("-")[0]
            if prefix.startswith(SKIP_PREFIXES):
                continue
            yield prefix, mf.parent, meta


def load_game(pack_dir: Path, meta: dict):
    cls = meta["class_name"]
    py = pack_dir / f"{cls.lower()}.py"
    spec = importlib.util.spec_from_file_location(f"lab_{pack_dir.parent.name}_{pack_dir.name}", py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls)


def make_engine():
    import arcengine
    return arcengine


def grid_of(game, arcengine):
    frame = game.camera.render(game.current_level.get_sprites())
    return np.asarray(frame, dtype=np.uint8)


def sig_of(grid):
    return grid.tobytes()


def actions_of(game):
    return [a for a in game._available_actions if 1 <= a <= MAX_ACTION_ID]


def step(game, arcengine, action_id):
    return game.perform_action(arcengine.ActionInput(id=arcengine.GameAction.from_id(action_id)))


def features(grid: np.ndarray, level_idx: int) -> np.ndarray:
    hist = np.bincount(grid.ravel(), minlength=16)[:16].astype(np.float64)
    total = hist.sum() or 1.0
    hist /= total
    nz = grid.nonzero()
    if len(nz[0]):
        spread = [nz[0].std(), nz[1].std(), nz[0].mean(), nz[1].mean()]
    else:
        spread = [0.0, 0.0, 0.0, 0.0]
    return np.concatenate([hist, np.array(spread) / grid.shape[0],
                           [float((hist > 0).sum()), float(level_idx)]])


def search_level(game, arcengine, budget, mode="bfs", model=None, h_weight=2.0):
    """Search from the game's current state until level_index increases.
    Returns (plan | None, nodes_expanded, seconds). `game` is left UNTOUCHED
    (all stepping happens on deepcopies)."""
    t0 = time.monotonic()
    start_level = game.level_index
    root_grid = grid_of(game, arcengine)
    seen = {sig_of(root_grid)}
    counter = 0
    if mode == "bfs":
        frontier = deque([(game, [])])
        pop = frontier.popleft
        push = lambda item, prio: frontier.append(item)
        empty = lambda: not frontier
    else:
        import heapq
        heap = []
        tie = [0]

        def push(item, prio):
            tie[0] += 1
            heapq.heappush(heap, (prio, tie[0], item))

        def pop():
            return heapq.heappop(heap)[2]

        empty = lambda: not heap
        push((game, []), 0.0)
        if mode == "bfs":
            pass
    if mode == "bfs":
        pass
    nodes = 0
    while not empty():
        if nodes >= budget or time.monotonic() - t0 > PER_LEVEL_WALL_S:
            return None, nodes, time.monotonic() - t0
        cur, path = pop()
        nodes += 1
        for a in actions_of(cur):
            child = copy.deepcopy(cur)
            try:
                frame = step(child, arcengine, a)
            except Exception:
                continue
            if str(getattr(frame, "state", "")).find("GAME_OVER") >= 0:
                continue
            if child.level_index > start_level or "WIN" in str(getattr(frame, "state", "")).upper():
                return path + [a], nodes, time.monotonic() - t0
            g = grid_of(child, arcengine)
            s = sig_of(g)
            if s in seen:
                continue
            seen.add(s)
            if mode == "bfs":
                push((child, path + [a]), 0.0)
            else:
                h = float(model.predict(features(g, child.level_index)[None, :])[0])
                push((child, path + [a]), len(path) + 1 + h_weight * max(h, 0.0))
    return None, nodes, time.monotonic() - t0


def chain_game(prefix, pack_dir, meta, budget, mode="bfs", model=None, harvest=None):
    arcengine = make_engine()
    cls = load_game(pack_dir, meta)
    game = cls()
    baselines = meta.get("baseline_actions") or []
    rows = []
    for lv in range(len(baselines)):
        plan, nodes, secs = search_level(game, arcengine, budget, mode=mode, model=model)
        solved = plan is not None
        rows.append({
            "game": prefix, "level": lv + 1, "budget": budget, "mode": mode,
            "solved": solved, "nodes": nodes, "secs": round(secs, 2),
            "plan_len": len(plan) if plan else None, "baseline": baselines[lv],
        })
        if not solved:
            break
        if harvest is not None and prefix not in HOLDOUT:
            replay = copy.deepcopy(game)
            remaining = len(plan)
            for a in plan:
                harvest["X"].append(features(grid_of(replay, arcengine), replay.level_index))
                harvest["y"].append(float(remaining))
                step(replay, arcengine, a)
                remaining -= 1
        for a in plan:
            step(game, arcengine, a)
    return rows


def cmd_bench() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    harvest = {"X": [], "y": []}
    games = list(iter_games())
    print(f"corpus: {len(games)} games (mouse mechanics excluded)")
    for budget in BUDGETS:
        for prefix, pack_dir, meta in games:
            do_harvest = harvest if budget == BUDGETS[-1] else None
            rows = chain_game(prefix, pack_dir, meta, budget, harvest=do_harvest)
            all_rows.extend(rows)
        solved = sum(1 for r in all_rows if r["budget"] == budget and r["solved"])
        attempted = sum(1 for r in all_rows if r["budget"] == budget)
        print(f"budget {budget:>5}: solved {solved}/{attempted} attempted levels")
    (OUT / "bench.json").write_text(json.dumps(all_rows, indent=1), encoding="utf-8")
    if harvest["X"]:
        np.savez(OUT / "h_dataset.npz", X=np.stack(harvest["X"]), y=np.array(harvest["y"]))
        print(f"harvested {len(harvest['y'])} (state -> remaining) pairs -> h_dataset.npz")
    # summary table
    print("\nper-budget coverage (levels solved of attempted; attempts stop at first failure):")
    for budget in BUDGETS:
        rows = [r for r in all_rows if r["budget"] == budget]
        solved = [r for r in rows if r["solved"]]
        ratios = [r["plan_len"] / r["baseline"] for r in solved if r["baseline"]]
        mean_ratio = sum(ratios) / len(ratios) if ratios else float("nan")
        print(f"  {budget:>5} nodes: {len(solved):>3}/{len(rows):>3} solved, "
              f"plan/baseline mean {mean_ratio:.2f}")


def _load_model():
    data = np.load(OUT / "h_dataset.npz")
    X, y = data["X"], data["y"]
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
    except ImportError:
        from sklearn.linear_model import Ridge  # type: ignore
        model = Ridge()
    return model, X, y


def cmd_train() -> None:
    model, X, y = _load_model()
    n = len(y)
    idx = np.random.RandomState(0).permutation(n)
    cut = int(n * 0.85)
    tr, te = idx[:cut], idx[cut:]
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    mae = float(np.mean(np.abs(pred - y[te])))
    naive = float(np.mean(np.abs(y[te].mean() - y[te])))
    print(f"h(s) train: {cut} pairs, test {n - cut}; MAE {mae:.2f} vs naive {naive:.2f}")
    import pickle
    (OUT / "h_model.pkl").write_bytes(pickle.dumps(model))
    print(f"model -> {OUT / 'h_model.pkl'}")


def cmd_astar() -> None:
    import pickle
    model = pickle.loads((OUT / "h_model.pkl").read_bytes())
    games = [(p, d, m) for p, d, m in iter_games() if p in HOLDOUT]
    print(f"holdout: {[p for p, _, _ in games]}")
    budget = 2500
    for mode in ("bfs", "astar"):
        total_nodes, solved, attempted = 0, 0, 0
        for prefix, pack_dir, meta in games:
            rows = chain_game(prefix, pack_dir, meta, budget,
                              mode=mode, model=model if mode == "astar" else None)
            for r in rows:
                attempted += 1
                total_nodes += r["nodes"]
                if r["solved"]:
                    solved += 1
        print(f"  {mode:>5}: solved {solved}/{attempted}, total nodes {total_nodes}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "train", "astar", "harvest"])
    args = ap.parse_args()
    if args.cmd == "harvest":
        # harvest-only: single top-budget pass over the whole corpus
        global BUDGETS
        BUDGETS = (2500,)
        cmd_bench()
    elif args.cmd == "bench":
        cmd_bench()
    elif args.cmd == "train":
        cmd_train()
    else:
        cmd_astar()


if __name__ == "__main__":
    main()

"""Round 13 Q5.2/Q5.3: mine duck-harness example-run (25 games x 20 passes).

harvest  -- (real frame, remaining-actions-to-level-up) pairs from every
            successful level segment -> runs/search_lab/h_dataset_real.npz
train    -- GBT on real pairs (game-level holdout), compare against the
            synthetic-trained h_model.pkl on the SAME real holdout, plus a
            mixed model -> runs/search_lab/h_model_real.pkl / h_model_mixed.pkl
taxonomy -- classify all 500 plays by failure mode (looping / death-spiral /
            wandering / starved / progressed) -> table + noop-guard notes

Features MUST stay in lockstep with tool_agent._atlas_astar_features
(16-color hist, spread/centroid over H, distinct colors, level idx).

Usage: uv run python scripts/duck_real_h.py harvest|train|taxonomy|all
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "duck_harness_ref" / "example-run"
OUT = ROOT / "runs" / "search_lab"

MAX_SEGMENT = 120        # junk-heavy level segments are dropped whole
MAX_REMAINING = 80       # pairs further than this from the level-up are noise
HOLDOUT_GAMES = ("vc33", "lp85", "ft09", "sb26", "tu93")  # eval-only games


def features(grid, level_idx: int) -> np.ndarray:
    g = np.asarray(grid, dtype=np.uint8)
    if g.ndim != 2 or g.size == 0:
        g = np.zeros((1, 1), dtype=np.uint8)
    hist = np.bincount(g.ravel(), minlength=16)[:16].astype(np.float64)
    total = hist.sum() or 1.0
    hist /= total
    nz = g.nonzero()
    if len(nz[0]):
        spread = [nz[0].std(), nz[1].std(), nz[0].mean(), nz[1].mean()]
    else:
        spread = [0.0, 0.0, 0.0, 0.0]
    return np.concatenate([hist, np.array(spread) / g.shape[0],
                           [float((hist > 0).sum()), float(level_idx)]])


def _iter_action_rows(path: Path):
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") == "action":
            yield row


def _plays(rows):
    """Split one events file into plays on score drops."""
    play, prev = [], None
    for r in rows:
        s = int(r.get("score", 0) or 0)
        if prev is not None and s < prev:
            yield play
            play = []
        play.append(r)
        prev = s
    if play:
        yield play


def cmd_harvest() -> None:
    X, y, seg_count, files = [], [], 0, 0
    for f in sorted(RUN.glob("artifacts/*_events.jsonl")):
        gid = f.name.split("-")[0]
        if gid in HOLDOUT_GAMES:
            continue
        files += 1
        for play in _plays(_iter_action_rows(f)):
            seg_start = 0
            prev_score = int(play[0].get("score", 0) or 0)
            for i, r in enumerate(play):
                s = int(r.get("score", 0) or 0)
                if s > prev_score:  # action i completed level prev_score+1
                    seg = play[seg_start:i + 1]
                    if 0 < len(seg) <= MAX_SEGMENT:
                        seg_count += 1
                        for j, rr in enumerate(seg):
                            remaining = len(seg) - 1 - j
                            if 0 < remaining <= MAX_REMAINING:
                                X.append(features(rr["board"], prev_score + 1))
                                y.append(float(remaining))
                    seg_start = i + 1
                    prev_score = s
    np.savez(OUT / "h_dataset_real.npz", X=np.stack(X), y=np.array(y))
    print(f"harvest: {files} train-game files, {seg_count} level segments, "
          f"{len(y)} pairs -> h_dataset_real.npz "
          f"(remaining: median={np.median(y):.0f} p90={np.percentile(y, 90):.0f})")


def _holdout_pairs():
    X, y = [], []
    for f in sorted(RUN.glob("artifacts/*_events.jsonl")):
        gid = f.name.split("-")[0]
        if gid not in HOLDOUT_GAMES:
            continue
        for play in _plays(_iter_action_rows(f)):
            seg_start = 0
            prev_score = int(play[0].get("score", 0) or 0)
            for i, r in enumerate(play):
                s = int(r.get("score", 0) or 0)
                if s > prev_score:
                    seg = play[seg_start:i + 1]
                    if 0 < len(seg) <= MAX_SEGMENT:
                        for j, rr in enumerate(seg):
                            remaining = len(seg) - 1 - j
                            if 0 < remaining <= MAX_REMAINING:
                                X.append(features(rr["board"], prev_score + 1))
                                y.append(float(remaining))
                    seg_start = i + 1
                    prev_score = s
    return np.stack(X), np.array(y)


def cmd_train() -> None:
    from sklearn.ensemble import GradientBoostingRegressor

    data = np.load(OUT / "h_dataset_real.npz")
    X, y = data["X"], data["y"]
    Xh, yh = _holdout_pairs()
    naive = float(np.mean(np.abs(yh.mean() - yh)))
    print(f"train pairs={len(y)}, holdout pairs={len(yh)} "
          f"({', '.join(HOLDOUT_GAMES)}); naive MAE={naive:.2f}")

    results = {}
    real = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
    real.fit(X, y)
    results["real"] = float(np.mean(np.abs(real.predict(Xh) - yh)))
    (OUT / "h_model_real.pkl").write_bytes(pickle.dumps(real))

    synth_model = pickle.loads((OUT / "h_model.pkl").read_bytes())
    results["synthetic (battle)"] = float(np.mean(np.abs(synth_model.predict(Xh) - yh)))

    sd = np.load(OUT / "h_dataset.npz")
    Xm = np.concatenate([X, sd["X"]])
    ym = np.concatenate([y, sd["y"]])
    mixed = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
    mixed.fit(Xm, ym)
    results["mixed"] = float(np.mean(np.abs(mixed.predict(Xh) - yh)))
    (OUT / "h_model_mixed.pkl").write_bytes(pickle.dumps(mixed))

    print("MAE on REAL unseen games (lower is better):")
    for k, v in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {k:20} {v:6.2f}")
    print(f"  {'naive-mean':20} {naive:6.2f}")


def cmd_taxonomy() -> None:
    cats = Counter()
    per_game = defaultdict(Counter)
    loop_examples = Counter()
    for f in sorted(RUN.glob("artifacts/*_events.jsonl")):
        gid = f.name.split("_p")[0].split("-")[0]
        for play in _plays(_iter_action_rows(f)):
            n = len(play)
            best = max(int(r.get("score", 0) or 0) for r in play)
            deaths = sum(1 for r in play if r.get("game_over"))
            sigs = len({r.get("board_ascii") or json.dumps(r.get("board"))[:2000] for r in play})
            ratio = sigs / n if n else 1.0
            if best >= 1:
                cat = "progressed"
            elif n < 20:
                cat = "starved(<20 actions)"
            elif deaths >= 3:
                cat = "death-spiral(3+ game_overs)"
            elif ratio < 0.35:
                cat = "looping(distinct<35%)"
                loop_examples[gid] += 1
            else:
                cat = "wandering(wrong hypothesis)"
            cats[cat] += 1
            per_game[gid][cat] += 1
    total = sum(cats.values())
    print(f"plays classified: {total}")
    for cat, nn in cats.most_common():
        print(f"  {cat:32} {nn:4} ({nn/total:.0%})")
    print("\nworst looping games:", dict(loop_examples.most_common(6)))
    print("\nper-game failure profile (top offenders, non-progressed):")
    scored = sorted(per_game.items(),
                    key=lambda kv: -(sum(v for k, v in kv[1].items() if k != "progressed")))
    for gid, c in scored[:8]:
        fails = {k: v for k, v in c.items() if k != "progressed"}
        print(f"  {gid:6} progressed={c.get('progressed', 0):2}  {fails}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("harvest", "all"):
        cmd_harvest()
    if cmd in ("train", "all"):
        cmd_train()
    if cmd in ("taxonomy", "all"):
        cmd_taxonomy()

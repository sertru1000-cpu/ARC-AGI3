"""Where do the actions go? Per-game waste profile from a run's artifacts/*_events.jsonl.

For every game: actions, share with no board change, share returning to an already
seen board (within the level), time and action index of the LAST level gain, and the
actions/time spent after it (the "wall" tail). Aggregates at the end.

usage: python scripts/event_waste.py runs/flash_v1_phaseA [more runs...]
"""
from __future__ import annotations
import glob, hashlib, json, os, sys
from datetime import datetime

def h(s: str) -> str: return hashlib.blake2b(s.encode(), digest_size=8).hexdigest()

def game_profile(path: str) -> dict:
    seen: set[str] = set(); prev = None; lvl = None
    a = n = r = 0; last_gain_a = 0; last_gain_t = None; t0 = None; t_last = None
    for line in open(path, encoding="utf-8"):
        try: e = json.loads(line)
        except Exception: continue
        ts = e.get("timestamp") or e.get("time") or e.get("wallclock")
        if e.get("type") == "initial":
            lvl = e.get("level"); bh = h(e.get("board_ascii", "")); seen.add(bh); prev = bh; continue
        if e.get("type") != "action": continue
        a += 1
        if e.get("level") != lvl:
            lvl = e.get("level"); seen = set(); last_gain_a = a; last_gain_t = ts
        bh = h(e.get("board_ascii", ""))
        if str(e.get("board_changed")) == "False" or bh == prev: n += 1
        elif bh in seen: r += 1
        seen.add(bh); prev = bh
    return {"game": os.path.basename(path)[:4], "actions": a, "nochange": n, "returns": r,
            "last_gain_action": last_gain_a, "tail": a - last_gain_a}

def main() -> None:
    for d in sys.argv[1:]:
        files = sorted(glob.glob(os.path.join(d, "artifacts", "*_events.jsonl")))
        bench = {}
        try:
            b = json.load(open(os.path.join(d, "benchmark.json"), encoding="utf-8"))
            for run in b["game_runs"]:
                bench[run["game_id"][:4]] = (run.get("levels_completed") or 0, run.get("number_of_levels") or 0, len(run.get("history") or []))
        except Exception: pass
        rows = [game_profile(f) for f in files]
        print(f"== {d}: {len(rows)} games")
        print(f"{'game':5s} {'lv':>5s} {'act':>5s} {'nochg':>6s} {'return':>7s} {'lastgain@':>10s} {'tail':>5s}  {'tail%':>5s}")
        A = N = R = T = 0
        for p in sorted(rows, key=lambda p: -p["tail"]):
            lv = bench.get(p["game"], (0, 0, 0))
            A += p["actions"]; N += p["nochange"]; R += p["returns"]; T += p["tail"]
            print(f"{p['game']:5s} {lv[0]:>2d}/{lv[1]:<2d} {p['actions']:5d} {p['nochange']:6d} {p['returns']:7d} {p['last_gain_action']:10d} {p['tail']:5d}  {100*p['tail']/max(1,p['actions']):4.0f}%")
        print(f"TOTAL actions={A} no-change={N} ({100*N/max(1,A):.0f}%) returns={R} ({100*R/max(1,A):.0f}%) tail-after-last-gain={T} ({100*T/max(1,A):.0f}%)")

if __name__ == "__main__":
    main()

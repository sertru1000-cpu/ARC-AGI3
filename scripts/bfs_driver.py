"""Разметка вариантов по одной игре в отдельном процессе с жёстким таймаутом (зависшие варианты убиваются).
usage: bfs_driver.py --env-dir <dir> --games a,b,c --out res.json [--timeout 60] [--max-moves 15000] [--max-states 1500]"""
import argparse, json, subprocess, sys, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--env-dir", required=True); ap.add_argument("--games", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=60); ap.add_argument("--max-moves", type=int, default=15000); ap.add_argument("--max-states", type=int, default=1500); a = ap.parse_args()
    res = {}; t0 = time.time()
    games = [g for g in a.games.split(",") if g]
    for i, g in enumerate(games):
        tmp = Path(tempfile.mkstemp(suffix=".json")[1])
        cmd = [sys.executable, str(ROOT / "scripts" / "engine_bfs.py"), "--env-dir", a.env_dir, "--games", g, "--max-states", str(a.max_states), "--max-moves", str(a.max_moves), "--out", str(tmp)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            if tmp.exists() and tmp.stat().st_size > 0:
                res.update(json.load(open(tmp)))
            else:
                res[g] = {"solved": False, "path": None, "error": "no output"}
            line = [l for l in p.stdout.splitlines() if l.startswith(g)]
            print(line[-1][:120] if line else f"{g}: ?", flush=True)
        except subprocess.TimeoutExpired:
            res[g] = {"solved": False, "path": None, "error": "timeout"}; print(f"{g}: TIMEOUT {a.timeout}s", flush=True)
        finally:
            tmp.unlink(missing_ok=True)
        if (i + 1) % 20 == 0:
            Path(a.out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"решено {sum(1 for r in res.values() if r.get('solved'))} из {len(res)}, {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()

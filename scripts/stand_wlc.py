"""Метрика стенда в токенах: уровни, взятые при равном числе сгенерированных токенов на игру.

WLC@T = сумма номеров уровней, взятых игрой не позже T сгенерированных токенов, делённая на сумму номеров всех
уровней всех игр (взвешенная доля взятых уровней). Считается на порогах 5k … 100k; игра, которая не дожила до порога
и не пройдена целиком, на этом пороге помечается «оборвана» (её взятые уровни засчитываются, но порог ненадёжен).
Источники: <run>/*_p0_requests.jsonl (usage по запросам), <run>/artifacts/*_p0_events.jsonl (ходы), <run>/summary.txt.
usage: .venv/bin/python scripts/stand_wlc.py <run> [<run2> ...]     # несколько прогонов — таблица рядом
       .venv/bin/python scripts/stand_wlc.py <run> --csv out.csv    # строки (игра, уровень) в файл
"""
import csv, glob, json, re, sys
from pathlib import Path

CHECKPOINTS = [5_000, 10_000, 20_000, 40_000, 60_000, 80_000, 100_000]


def load(run):
    run = Path(run)
    n_levels = {}
    summary = run / "summary.txt"
    if summary.is_file():
        for m in re.finditer(r"^\s+(\w{4})-\w+: .*levels=[\d.]+/(\d+)", summary.read_text(), re.M):
            n_levels[m.group(1)] = int(m.group(2))
    games = []
    for ev in sorted(glob.glob(str(run / "artifacts" / "*_p0_events.jsonl"))):
        g = Path(ev).name[:4]
        rq = glob.glob(str(run / f"{g}-*_p0_requests.jsonl"))
        tok_by_step, total = {}, 0
        if rq:
            for line in open(rq[0]):
                r = json.loads(line)
                u = r.get("usage")
                if isinstance(u, dict) and u.get("completion_tokens"):
                    step = r.get("analysis_step") or 0
                    tok_by_step[step] = tok_by_step.get(step, 0) + u["completion_tokens"]
                    total += u["completion_tokens"]
        cum, acc = {}, 0
        for step in sorted(tok_by_step):
            acc += tok_by_step[step]; cum[step] = acc

        def tokens_at(step):
            keys = [k for k in cum if k <= (step or 0)]
            return cum[max(keys)] if keys else 0

        levels, undo, resets, moves = [], 0, 0, 0
        for line in open(ev):
            e = json.loads(line)
            if e.get("type") != "action":
                continue
            moves += 1
            name = str(e.get("action_name"))
            undo += name == "ACTION7"; resets += name == "RESET"
            if e.get("level_completed") in (True, "True"):   # k-е взятие = уровень k: поле level в событии уже указывает на следующий
                levels.append({"level": len(levels) + 1, "tokens": tokens_at(e.get("analysis_step")), "moves": int(e.get("action_num") or moves)})
        games.append({"game": g, "n_levels": n_levels.get(g), "total_tokens": total, "moves": moves, "undo": undo,
                      "resets": resets, "levels": levels})
    return games


def wlc(games, T):
    num = den = 0; censored = 0; taken = 0
    for g in games:
        n = g["n_levels"] or max([l["level"] for l in g["levels"]] + [1])
        den += n * (n + 1) // 2
        done = [l for l in g["levels"] if l["tokens"] <= T]
        num += sum(l["level"] for l in done); taken += len(done)
        if g["total_tokens"] < T and len(g["levels"]) < n:
            censored += 1
    return (100 * num / den if den else 0.0), taken, censored


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
    runs = [a for a in args if a != out_csv]
    data = {r: load(r) for r in runs}
    for r, games in data.items():
        tot = sum(g["total_tokens"] for g in games)
        print(f"\n== {r} ==  игр {len(games)}, сгенерировано {tot:,} токенов (медиана на игру "
              f"{sorted(g['total_tokens'] for g in games)[len(games)//2]:,}), уровней взято {sum(len(g['levels']) for g in games)}, "
              f"UNDO {sum(g['undo'] for g in games)}, RESET {sum(g['resets'] for g in games)}")
        print(f"  {'игра':5s} {'уровней':>9s} {'токенов':>9s} {'ходов':>6s} {'UNDO':>5s}  уровень@токены(ходы)")
        for g in games:
            lv = "  ".join(f"L{l['level']}@{l['tokens']:,}({l['moves']})" for l in g["levels"])
            print(f"  {g['game']:5s} {len(g['levels']):>4d}/{g['n_levels'] or '?':<4} {g['total_tokens']:>9,d} {g['moves']:>6d} {g['undo']:>5d}  {lv}")
    print("\n== WLC@T: взвешенная доля взятых уровней, % (в скобках: уровней взято / игр оборвано до порога) ==")
    print(f"  {'порог':>8s} " + " ".join(f"{Path(r).name[:26]:>28s}" for r in runs))
    for T in CHECKPOINTS:
        cells = []
        for r in runs:
            v, taken, cens = wlc(data[r], T)
            cells.append(f"{v:6.2f} ({taken:3d} / {cens:2d})".rjust(28))
        print(f"  {T:>8,d} " + " ".join(cells))
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["run", "game", "n_levels", "level", "tokens_at_completion", "moves_at_completion", "game_total_tokens", "undo", "resets"])
            for r, games in data.items():
                for g in games:
                    for l in g["levels"] or [{"level": "", "tokens": "", "moves": ""}]:
                        w.writerow([Path(r).name, g["game"], g["n_levels"], l["level"], l["tokens"], l["moves"], g["total_tokens"], g["undo"], g["resets"]])
        print(f"\nстроки (игра, уровень) записаны в {out_csv}")

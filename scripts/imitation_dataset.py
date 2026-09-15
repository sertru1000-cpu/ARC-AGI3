"""Набор данных для политики подражания (15.09, слово владельца «делай, обучать будем локально»):
генерируем пакеты уровней 10 механик с решателями (scripts/gen_testbed_levels.py), проигрываем оптимальные
пути на настоящем движке и записываем (доска ДО хода, ход, механика, уровень).
usage: imitation_dataset.py --per-game 20 --out <dir> [--seed 1]"""
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import gen_testbed_levels as G
from replay_battle_local import play

ACT = {1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5"}


def solve_level(mech, spec):
    """Оптимальный путь уровня решателем механики: список действий (id или (6, row, col))."""
    return G.SOLVERS[mech](spec)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-game", type=int, default=10); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mechs", default=",".join(G.SOLVERS.keys())); a = ap.parse_args()
    out = Path(a.out); packs = out / "packs"; packs.mkdir(parents=True, exist_ok=True)
    import logging; logging.disable(logging.WARNING)
    X, Y, M, L, CL = [], [], [], [], []
    specs_by_game = {}
    mechs = [m for m in a.mechs.split(",") if m in G.SOLVERS]
    t0 = time.time(); n_levels = 0; n_ok = 0
    for mi, mech in enumerate(mechs):
        cfg = G.MECHANICS[mech]; source = cfg["src"].read_text(encoding="utf-8")
        for i in range(a.per_game):
            levels = baselines = None
            for attempt in range(8):
                rng = random.Random(a.seed * 1_000_003 + hash(mech) % 10_000 + i * 131 + attempt * 777_777)
                try:
                    levels, baselines = cfg["gen"](rng); break
                except RuntimeError:
                    continue
            if levels is None:
                continue
            body = G.LEVELS_RE.sub("\n" + cfg["fmt"](levels), source, count=1)
            ver = hashlib.md5(body.encode()).hexdigest()[:8]
            prefix = f"{cfg['prefix']}{i:03d}"; game_id = f"{prefix}-{ver}"
            pack = packs / prefix / ver; pack.mkdir(parents=True, exist_ok=True)
            (pack / f"{cfg['class_name'].lower()}.py").write_text(body, encoding="utf-8", newline="\n")
            (pack / "metadata.json").write_text(json.dumps({"game_id": game_id, "class_name": cfg["class_name"], "title": mech, "baseline_actions": baselines}) + "\n", encoding="utf-8")
            specs_by_game[game_id] = (mech, mi, levels)
    import arc_agi
    from arc_agi import OperationMode
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(packs))
    avail = {e.game_id.split("-")[0]: e.game_id for e in arc.get_environments()}
    for game_id, (mech, mi, levels) in specs_by_game.items():
        gid = avail.get(game_id.split("-")[0])
        if gid is None:
            continue
        script = [{"name": "RESET", "payload": None}]; acts = []
        try:
            for spec in levels:
                sol = solve_level(mech, spec)
                for s in sol:
                    if isinstance(s, (tuple, list)):
                        script.append({"name": "ACTION6", "payload": {"x": int(s[2]), "y": int(s[1])}}); acts.append((6, int(s[1]), int(s[2])))
                    else:
                        script.append({"name": ACT[int(s)], "payload": None}); acts.append((int(s), -1, -1))
        except Exception as exc:
            print(f"{game_id}: решатель упал: {exc!r}"); continue
        tr = play(arc, gid, script)
        # tr[k] = кадр после k-го элемента script (0 = RESET); доска ДО действия j = tr[j][0]
        lvl_final = tr[-1][1]; n_levels += len(levels)
        if lvl_final < len(levels):
            print(f"{game_id} ({mech}): решатель прошёл {lvl_final}/{len(levels)} уровней -- беру только пройденные")
        for j, act in enumerate(acts):
            if j + 1 >= len(tr):
                break
            before = tr[j][0]; lvl_before = tr[j][1]
            if lvl_before >= lvl_final:
                break
            X.append(before.astype(np.int8)); Y.append(act[0]); M.append(mi); L.append(lvl_before); CL.append((act[1], act[2]))
        n_ok += lvl_final
    X = np.stack(X) if X else np.zeros((0, 64, 64), np.int8)
    np.savez_compressed(out / "dataset.npz", X=X, y=np.array(Y, np.int64), mech=np.array(M, np.int64), level=np.array(L, np.int64), click=np.array(CL, np.int64), mechs=np.array(mechs))
    print(f"пакетов {len(specs_by_game)}, уровней {n_levels} (пройдено решателем {n_ok}), примеров {len(Y)}, "
          f"кликов {sum(1 for y in Y if y == 6)}, {time.time() - t0:.0f} с -> {out / 'dataset.npz'}")


if __name__ == "__main__":
    main()

"""Контроль детектора цели: на отложенных играх сравнить top-1 детектора с эвристиками без обучения --
«самое большое изменение доски», «самое маленькое ненулевое изменение», случайный выбор; отдельно -- на «трудных»
группах, где все братья меняют >= 8 клеток (клики-промахи с одним счётчиком исключены).
usage: goal_eval.py --data goal.npz --model goal_model.pt [--hold sp80,sk48,ls20,cd82,lf52]"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from goal_train import Goal, load

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--model", required=True); ap.add_argument("--hold", default="sp80,sk48,ls20,cd82,lf52"); a = ap.parse_args()
d, X, Xn, y, game, grp, dist, games = load(a.data)
model = Goal(); model.load_state_dict(torch.load(a.model, map_location="cpu", weights_only=False)["state"]); model.eval()
hold_g = [i for i, g in enumerate(games) if g in set(a.hold.split(","))]
idxs = np.array([i for i in range(len(y)) if game[i] in hold_g])
scores = np.zeros(len(y), np.float32)
with torch.no_grad():
    for i in range(0, len(idxs), 256):
        ii = idxs[i:i + 256]; scores[ii] = model(X[ii], Xn[ii]).numpy()
ndiff = (X != Xn).flatten(1).sum(1).numpy()
by = {}
for i in idxs:
    by.setdefault((game[i], grp[i]), []).append(i)
rng = np.random.default_rng(0)
print(f"{'игра':5s} {'групп':>5s} {'трудн':>5s} | {'детектор':>8s} {'max-diff':>8s} {'min-diff':>8s} {'случ':>6s} | трудные: {'детектор':>8s} {'max-diff':>8s} {'min-diff':>8s} {'случ':>6s}")
for g in hold_g:
    res = {"all": {"det": 0, "max": 0, "min": 0, "rnd": 0.0, "n": 0}, "hard": {"det": 0, "max": 0, "min": 0, "rnd": 0.0, "n": 0}}
    for (gg, gr), ii in by.items():
        if gg != g: continue
        pos = [i for i in ii if y[i] == 1]
        if not pos: continue
        p = pos[0]; hard = all(ndiff[i] >= 8 for i in ii)
        for key in (["all", "hard"] if hard else ["all"]):
            r = res[key]; r["n"] += 1; r["rnd"] += 1.0 / len(ii)
            r["det"] += int(scores[p] >= max(scores[i] for i in ii))
            r["max"] += int(ndiff[p] >= max(ndiff[i] for i in ii))
            r["min"] += int(ndiff[p] <= min(ndiff[i] for i in ii))
    A, H = res["all"], res["hard"]
    f = lambda r, k: (r[k] / r["n"]) if r["n"] else float("nan")
    print(f"{games[g]:5s} {A['n']:5d} {H['n']:5d} | {f(A,'det'):8.2f} {f(A,'max'):8.2f} {f(A,'min'):8.2f} {f(A,'rnd'):6.2f} |          {f(H,'det'):8.2f} {f(H,'max'):8.2f} {f(H,'min'):8.2f} {f(H,'rnd'):6.2f}")

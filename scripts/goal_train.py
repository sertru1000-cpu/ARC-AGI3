"""Детектор цели/прогресса (критик раунда 6, №3): парный кодировщик «доска была / стала» -> оценка «переход ведёт к уровню».
Обучение: softmax по группе братьев (положительный шаг пути против других ходов из того же состояния).
Проверка на отложенных играх: top-1 (настоящий шаг оценён выше всех братьев) против случайного 1/размер группы, AUC,
и корреляция оценки с остатком пути (прогресс). usage: goal_train.py --data goal.npz --hold sp80,sk48,ls20,cd82,lf52 --out m.pt"""
import argparse, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

D = 256


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [16, 32, 64, 96, 128]; layers = []
        for i in range(4):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3, padding=1), nn.BatchNorm2d(ch[i + 1]), nn.ReLU(), nn.MaxPool2d(2)]
        self.body = nn.Sequential(*layers); self.out = nn.Linear(128 * 16, D)

    def forward(self, x):
        oh = F.one_hot(x.long().clamp(0, 15), 16).permute(0, 3, 1, 2).float()
        return self.out(self.body(oh).flatten(1))


class Goal(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Enc()
        # разность досок как отдельный вход: где и что изменилось
        self.diff = nn.Sequential(nn.Conv2d(33, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64 * 16, D))
        self.head = nn.Sequential(nn.Linear(4 * D, D), nn.ReLU(), nn.Dropout(0.2), nn.Linear(D, 1))

    def forward(self, s, sn):
        es, en = self.enc(s), self.enc(sn)
        oh = F.one_hot(s.long().clamp(0, 15), 16).permute(0, 3, 1, 2).float(); ohn = F.one_hot(sn.long().clamp(0, 15), 16).permute(0, 3, 1, 2).float()
        d = self.diff(torch.cat([oh, ohn, (s != sn).float().unsqueeze(1)], 1))
        return self.head(torch.cat([es, en, en - es, d], -1)).squeeze(-1)


def load(p):
    d = np.load(p, allow_pickle=True)
    return d, torch.from_numpy(d["X"].astype(np.int64)), torch.from_numpy(d["Xn"].astype(np.int64)), d["y"], d["game"], d["grp"], d["dist"], list(d["games"])


@torch.no_grad()
def evaluate(model, X, Xn, y, grp, dist, idxs, dev):
    model.eval(); scores = np.zeros(len(y), np.float32)
    for i in range(0, len(idxs), 512):
        ii = idxs[i:i + 512]; scores[ii] = model(X[ii].to(dev), Xn[ii].to(dev)).cpu().numpy()
    groups = {}
    for i in idxs:
        groups.setdefault(grp[i], []).append(i)
    top1 = 0; rand = 0.0; n = 0
    for g, ii in groups.items():
        pos = [i for i in ii if y[i] == 1]
        if not pos:
            continue
        n += 1; rand += 1.0 / len(ii)
        top1 += int(scores[pos[0]] >= max(scores[i] for i in ii))
    # AUC
    pi = [i for i in idxs if y[i] == 1]; ni = [i for i in idxs if y[i] == 0]
    auc = float("nan")
    if pi and ni:
        ps = scores[pi]; ns = scores[ni]
        auc = float((ps[:, None] > ns[None, :]).mean() + 0.5 * (ps[:, None] == ns[None, :]).mean()) if len(pi) * len(ni) < 4e7 else float(np.mean([(p > ns).mean() for p in ps[:2000]]))
    # прогресс: ранговая корреляция оценки положительных с остатком пути (ожидается отрицательная: ближе к цели -- выше)
    rho = float("nan")
    if len(pi) > 5:
        a = scores[pi]; b = dist[pi].astype(np.float64)
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); rho = float(np.corrcoef(ra, rb)[0, 1])
    return top1 / max(1, n), rand / max(1, n), n, auc, rho


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--hold", default="sp80,sk48,ls20,cd82,lf52"); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--groups", type=int, default=32); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    d, X, Xn, y, game, grp, dist, games = load(a.data); hold = set(a.hold.split(","))
    hold_g = [i for i, g in enumerate(games) if g in hold]
    by_grp = {}
    for i in range(len(y)):
        by_grp.setdefault(grp[i], []).append(i)
    train_grps = [g for g, ii in by_grp.items() if game[ii[0]] not in hold_g and any(y[i] == 1 for i in ii)]
    # 10% групп train на проверку
    rng.shuffle(train_grps); nv = len(train_grps) // 10; val_grps, train_grps = train_grps[:nv], train_grps[nv:]
    hold_idx = {g: np.array([i for i in range(len(y)) if game[i] == g]) for g in hold_g}
    tv_idx = np.array([i for g in val_grps for i in by_grp[g]])
    print(f"групп train {len(train_grps)}, val {len(val_grps)}; отложенные {[games[g] for g in hold_g]}; device {dev}", flush=True)
    model = Goal().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.steps); t0 = time.time()
    for s in range(1, a.steps + 1):
        model.train(); gs = rng.choice(len(train_grps), a.groups, replace=False)
        ii = []; owner = []
        for k, gi in enumerate(gs):
            for i in by_grp[train_grps[gi]]:
                ii.append(i); owner.append(k)
        ii = np.array(ii); owner = torch.tensor(owner, device=dev)
        sc = model(X[ii].to(dev), Xn[ii].to(dev)); yy = torch.from_numpy(y[ii]).to(dev)
        # softmax по группе: -log p(положительный)
        loss = 0.0
        for k in range(len(gs)):
            m = owner == k
            loss = loss + F.cross_entropy(sc[m].unsqueeze(0), yy[m].float().argmax().unsqueeze(0))
        loss = loss / len(gs)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if s % 500 == 0 or s == a.steps:
            t1, r1, n1, auc1, rho1 = evaluate(model, X, Xn, y, grp, dist, tv_idx, dev)
            line = f"шаг {s}: loss {loss.item():.3f}, {time.time() - t0:.0f} с | train-val top1 {t1:.2f} (случ {r1:.2f}, n={n1}) AUC {auc1:.2f} |"
            for g in hold_g:
                t, r, n, auc, rho = evaluate(model, X, Xn, y, grp, dist, hold_idx[g], dev); line += f" {games[g]} top1 {t:.2f} (случ {r:.2f}, n={n}) AUC {auc:.2f} rho {rho:+.2f} |"
            print(line, flush=True)
    torch.save({"state": model.state_dict(), "games": games}, a.out); print("сохранено", a.out)


if __name__ == "__main__":
    main()

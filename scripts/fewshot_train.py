"""Few-shot перенос (критик, раунд 6, тест №1): «k решённых переходов новой игры в контексте -> ход в новом состоянии».
Модель: свёрточный кодировщик доски -> токены переходов (S, A, S') и токен запроса -> трансформер -> ход (+ карта клика).
Обучение на играх train (контекст = переходы той же игры из ДРУГИХ вариантов), проверка на отложенных играх при
k = 0 / 5 / 20 и контроль «контекст чужой игры» при k = 20. Доза-отклик по k -- главный критерий.
usage: fewshot_train.py --data fewshot.npz --hold sp80,sk48,ls20,cd82,lf52 --out model.pt [--steps 3000] [--bs 32]"""
import argparse, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

N_ACT = 7; K_MAX = 20; D = 256


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [16, 32, 64, 96, 128]; layers = []
        for i in range(4):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3, padding=1), nn.BatchNorm2d(ch[i + 1]), nn.ReLU(), nn.MaxPool2d(2)]
        self.body = nn.Sequential(*layers); self.out = nn.Linear(128 * 16, D)

    def forward(self, x):                        # x: [N,64,64] int -> [N,D]
        oh = F.one_hot(x.long().clamp(0, 15), 16).permute(0, 3, 1, 2).float()
        return self.out(self.body(oh).flatten(1))


class FewShot(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Enc(); self.act = nn.Embedding(N_ACT, 32); self.clk = nn.Linear(2, 32)
        self.ctx = nn.Sequential(nn.Linear(3 * D + 64, D), nn.ReLU(), nn.Linear(D, D))
        self.q = nn.Linear(D, D); self.ttype = nn.Embedding(2, D)
        layer = nn.TransformerEncoderLayer(D, 4, D * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, 2); self.norm = nn.LayerNorm(D); self.head = nn.Linear(D, N_ACT)
        self.film = nn.Linear(D, 8)
        self.click = nn.Sequential(nn.Conv2d(24, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 5, padding=2), nn.ReLU(), nn.Conv2d(32, 1, 1))

    def forward(self, Sq, Sc, Snc, Ac, Cc, valid):
        """Sq [B,64,64]; Sc,Snc [B,K,64,64]; Ac [B,K] int; Cc [B,K,2] float (row/63, col/63; -1 без клика); valid [B,K] bool."""
        B, K = Ac.shape
        eq = self.enc(Sq)                                                     # [B,D]
        if K > 0:
            es = self.enc(Sc.reshape(B * K, 64, 64)).view(B, K, D); en = self.enc(Snc.reshape(B * K, 64, 64)).view(B, K, D)
            tc = self.ctx(torch.cat([es, en, en - es, self.act(Ac), self.clk(Cc)], -1)) + self.ttype.weight[0]
        tq = (self.q(eq) + self.ttype.weight[1]).unsqueeze(1)
        h = torch.cat([tc, tq], 1) if K > 0 else tq
        pad = torch.cat([~valid, torch.zeros(B, 1, dtype=torch.bool, device=Sq.device)], 1) if K > 0 else None
        h = self.norm(self.tr(h, src_key_padding_mask=pad))[:, -1]              # токен запроса
        logits = self.head(h)
        oh = F.one_hot(Sq.long().clamp(0, 15), 16).permute(0, 3, 1, 2).float()
        f = self.film(h).view(B, 8, 1, 1).expand(B, 8, 64, 64)
        cmap = self.click(torch.cat([oh, f], 1)).flatten(1)
        return logits, cmap


class Data:
    def __init__(self, path):
        d = np.load(path, allow_pickle=True)
        self.X = torch.from_numpy(d["X"].astype(np.int64)); self.Xn = torch.from_numpy(d["Xn"].astype(np.int64)); self.y = torch.from_numpy(d["y"])
        c = d["click"]; self.cidx = torch.from_numpy(np.where(c[:, 0] >= 0, c[:, 0] * 64 + c[:, 1], -1).astype(np.int64))
        self.cf = torch.from_numpy(np.where(c >= 0, c / 63.0, -1.0).astype(np.float32))
        self.game = d["game"]; self.vid = d["vid"]; self.games = list(d["games"]); self.src = d["src"]
        self.by_game = {g: np.where(self.game == g)[0] for g in range(len(self.games))}

    mode = "variants"
    shuffle = False

    def contexts(self, i, k, rng, wrong_game=False):
        """k индексов переходов той же игры (или чужой при wrong_game): режим variants -- из других вариантов, чем у i;
        режим online (критик р.7) -- предыдущие шаги ТОЙ ЖЕ партии (индексы того же vid меньше i), т.е. первые
        реальные взаимодействия именно с этой игрой; при wrong_game -- префикс партии чужой игры той же длины."""
        g = int(self.game[i])
        if self.mode == "online":
            if k == 0:
                return np.zeros(0, np.int64)
            if wrong_game:
                others = [x for x in self.by_game if x != g]
                g2 = rng.choice(others); pool = self.by_game[g2]
                j = int(rng.choice(pool)); prev = pool[(self.vid[pool] == self.vid[j]) & (pool < j)]
            else:
                pool = self.by_game[g]; prev = pool[(self.vid[pool] == self.vid[i]) & (pool < i)]
            prev = prev[-k:] if len(prev) > k else prev
            return prev.astype(np.int64)
        if wrong_game:
            others = [x for x in self.by_game if x != g and len(self.by_game[x]) >= k]
            g = rng.choice(others)
        pool = self.by_game[g]; pool = pool[self.vid[pool] != self.vid[i]]
        if len(pool) < k:
            pool = self.by_game[g]; pool = pool[pool != i]
        if len(pool) == 0 or k == 0:
            return np.zeros(0, np.int64)
        return rng.choice(pool, size=min(k, len(pool)), replace=len(pool) < k)


def batch(data, idx, ks, rng, dev, wrong_game=False):
    B = len(idx); K = max(ks) if ks else 0
    Sc = torch.zeros(B, K, 64, 64, dtype=torch.long); Snc = torch.zeros_like(Sc); Ac = torch.zeros(B, K, dtype=torch.long)
    Cc = torch.full((B, K, 2), -1.0); valid = torch.zeros(B, K, dtype=torch.bool)
    for b, (i, k) in enumerate(zip(idx, ks)):
        cx = data.contexts(i, k, rng, wrong_game)
        n = len(cx)
        if n:
            cx = torch.from_numpy(cx); Sc[b, :n] = data.X[cx]; Snc[b, :n] = data.Xn[cx]; Ac[b, :n] = data.y[cx]; Cc[b, :n] = data.cf[cx]; valid[b, :n] = True
    idx_t = torch.from_numpy(np.asarray(idx))
    return (data.X[idx_t].to(dev), Sc.to(dev), Snc.to(dev), Ac.to(dev), Cc.to(dev), valid.to(dev), data.y[idx_t].to(dev), data.cidx[idx_t].to(dev))


@torch.no_grad()
def evaluate(model, data, idxs, k, dev, seed=0, wrong_game=False, bs=64):
    model.eval(); rng = np.random.default_rng(seed); ok = 0; n = 0; chit = 0; cn = 0
    for i in range(0, len(idxs), bs):
        idx = idxs[i:i + bs]
        Sq, Sc, Snc, Ac, Cc, valid, y, cidx = batch(data, idx, [k] * len(idx), rng, dev, wrong_game)
        lg, cm = model(Sq, Sc, Snc, Ac, Cc, valid); pred = lg.argmax(1)
        ok += (pred == y).sum().item(); n += len(idx)
        m = cidx >= 0
        if m.any():
            pc = cm.argmax(1)[m]; tc = cidx[m]
            chit += ((pc // 64 - tc // 64).abs().le(2) & (pc % 64 - tc % 64).abs().le(2)).sum().item(); cn += int(m.sum())
    return ok / max(1, n), (chit / cn if cn else float("nan")), n, cn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--hold", default="sp80,sk48,ls20,cd82,lf52"); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--context", default="variants", choices=["variants", "online"]); ap.add_argument("--shuffle-labels", action="store_true"); a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); random.seed(a.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = Data(a.data); hold = set(a.hold.split(",")); data.mode = a.context
    if a.shuffle_labels:
        # контроль (критик р.7): метки перемешаны внутри обучения -- ожидается точность на уровне частоты класса
        perm = torch.randperm(len(data.y)); data.y = data.y[perm]; data.cidx = data.cidx[perm]; data.cf = data.cf[perm]
    hold_g = [i for i, g in enumerate(data.games) if g in hold]; train_g = [i for i in range(len(data.games)) if i not in hold_g]
    train_idx = np.concatenate([data.by_game[g] for g in train_g])
    # внутри train: 10% вариантов каждой игры отложены для проверки «та же игра, новый вариант»
    tv = []; tr = []
    for g in train_g:
        vids = np.unique(data.vid[data.by_game[g]]); rng.shuffle(vids); nv = max(1, len(vids) // 10) if len(vids) > 1 else 0
        hv = set(vids[:nv].tolist())
        for i in data.by_game[g]:
            (tv if data.vid[i] in hv else tr).append(i)
    tr = np.array(tr); tv = np.array(tv)
    print(f"игр train {len(train_g)}, hold {[data.games[g] for g in hold_g]}; переходов train {len(tr)}, train-val {len(tv)}, hold {sum(len(data.by_game[g]) for g in hold_g)}; device {dev}", flush=True)
    model = FewShot().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.steps)
    KS = [0, 1, 2, 5, 10, 20]; t0 = time.time()
    for s in range(1, a.steps + 1):
        model.train()
        idx = rng.choice(tr, size=a.bs, replace=False); ks = [int(rng.choice(KS)) for _ in idx]
        Sq, Sc, Snc, Ac, Cc, valid, y, cidx = batch(data, idx, ks, rng, dev)
        lg, cm = model(Sq, Sc, Snc, Ac, Cc, valid)
        loss = F.cross_entropy(lg, y); m = cidx >= 0
        if m.any():
            loss = loss + F.cross_entropy(cm[m], cidx[m])
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if s % 250 == 0 or s == a.steps:
            line = f"шаг {s}: loss {loss.item():.3f}, {time.time() - t0:.0f} с |"
            for k in (0, 20):
                acc, ch, n, cn = evaluate(model, data, tv, k, dev); line += f" train-val k={k}: ход {acc:.3f} клик±2 {ch:.2f} |"
            for k in (0, 5, 20):
                acc, ch, n, cn = evaluate(model, data, np.concatenate([data.by_game[g] for g in hold_g]), k, dev); line += f" hold k={k}: ход {acc:.3f} клик±2 {ch:.2f} |"
            print(line, flush=True)
    torch.save({"state": model.state_dict(), "games": data.games}, a.out)
    print("== итог по отложенным играм (ход = точность действия; клик±2 = попадание карты клика в окрестность ±2)")
    def with_prefix(idxs, k):
        if data.mode != "online" or k == 0:
            return idxs
        pool = idxs; return np.array([i for i in idxs if ((data.vid[pool] == data.vid[i]) & (pool < i)).sum() >= k], np.int64)
    for g in hold_g:
        idxs = data.by_game[g]; row = f"{data.games[g]:5s} n={len(idxs):4d} вариантов {len(np.unique(data.vid[idxs])):3d}:"
        for k in (0, 5, 20):
            sub = with_prefix(idxs, k)
            if len(sub) == 0:
                row += f"  k={k}: —(n=0)"; continue
            acc, ch, n, cn = evaluate(model, data, sub, k, dev, seed=1); row += f"  k={k}: {acc:.3f}/{ch:.2f}(n={n})"
        acc, ch, n, cn = evaluate(model, data, idxs, 20, dev, seed=1, wrong_game=True); row += f"  чужой k=20: {acc:.3f}/{ch:.2f}"
        print(row, flush=True)
    for name, idxs in (("train-val", tv),):
        row = f"{name}:"
        for k in (0, 5, 20):
            acc, ch, n, cn = evaluate(model, data, idxs, k, dev, seed=1); row += f"  k={k}: {acc:.3f}/{ch:.2f}"
        acc, ch, n, cn = evaluate(model, data, idxs, 20, dev, seed=1, wrong_game=True); row += f"  чужой k=20: {acc:.3f}/{ch:.2f}"
        print(row, flush=True)


if __name__ == "__main__":
    main()

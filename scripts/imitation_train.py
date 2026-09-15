"""Политика подражания: свёрточная сеть «доска 64x64 -> ход» на данных решателей наших механик (15.09).
usage: imitation_train.py --train <dir>/dataset.npz --val <dir>/dataset.npz --out <model.pt> [--epochs 8]"""
import argparse, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

N_ACT = 7   # 0 запас, 1..5 стрелки/SPACE, 6 клик (клик в этой версии не обучается)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [16, 32, 64, 96, 128]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3, padding=1), nn.BatchNorm2d(ch[i + 1]), nn.ReLU(), nn.MaxPool2d(2)]   # 64 -> 4
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, N_ACT))
        # голова кликов: полносвёрточная карта 64x64 (для игр с MOUSE)
        self.click = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(32, 32, 5, padding=2), nn.ReLU(), nn.Conv2d(32, 1, 1))

    def forward(self, x, with_click=False):          # x: [B,64,64] int -> one-hot 16
        x = F.one_hot(x.long().clamp(0, 15), 16).permute(0, 3, 1, 2).float()
        logits = self.head(self.body(x))
        if with_click:
            return logits, self.click(x).flatten(1)   # [B, 4096]
        return logits


class PatchTransformer(nn.Module):
    """Трансформер над патчами 4x4 (256 токенов): вторая архитектура для сравнения с CNN на тех же данных."""
    def __init__(self, dim=128, depth=4, heads=4, patch=4):
        super().__init__()
        self.patch = patch; n = (64 // patch) ** 2
        self.embed = nn.Linear(16 * patch * patch, dim)
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim)); self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, N_ACT)
        self.click_tok = nn.Linear(dim, patch * patch)   # логит клика на каждую клетку патча
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x, with_click=False):
        B = x.shape[0]; p = self.patch
        oh = F.one_hot(x.long().clamp(0, 15), 16).float()                   # [B,64,64,16]
        t = oh.view(B, 64 // p, p, 64 // p, p, 16).permute(0, 1, 3, 2, 4, 5).reshape(B, (64 // p) ** 2, p * p * 16)
        h = torch.cat([self.cls.expand(B, -1, -1), self.embed(t)], 1) + self.pos
        h = self.norm(self.enc(h))
        logits = self.head(h[:, 0])
        if with_click:
            ct = self.click_tok(h[:, 1:])                                     # [B, n, p*p]
            cmap = ct.view(B, 64 // p, 64 // p, p, p).permute(0, 1, 3, 2, 4).reshape(B, 4096)
            return logits, cmap
        return logits


def make_model(arch: str):
    return PatchTransformer() if arch == "vit" else Policy()


def load(p):
    d = np.load(p, allow_pickle=True)
    click = d["click"] if "click" in d else np.full((len(d["y"]), 2), -1)
    cidx = torch.from_numpy(np.where(click[:, 0] >= 0, click[:, 0] * 64 + click[:, 1], -1).astype(np.int64))   # row*64+col
    return torch.from_numpy(d["X"].astype(np.int64)), torch.from_numpy(d["y"]), torch.from_numpy(d["mech"]), list(d["mechs"]), cidx


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--val", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--arch", default="cnn", choices=["cnn", "vit"]); a = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    Xt, yt, mt, mechs, ct = load(a.train); Xv, yv, mv, mechs_v, cv = load(a.val)
    print(f"train {len(yt)} val {len(yv)} device {dev}; классы train: {np.bincount(yt.numpy(), minlength=N_ACT).tolist()}")
    model = make_model(a.arch).to(dev); lr = 2e-3 if a.arch == "cnn" else 5e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=a.epochs * ((len(yt) + a.bs - 1) // a.bs))
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(yt)); tl = 0.0; n = 0
        for i in range(0, len(yt), a.bs):
            idx = perm[i:i + a.bs]; xb = Xt[idx].to(dev); yb = yt[idx].to(dev); cb = ct[idx].to(dev)
            logits, cmap = model(xb, with_click=True)
            loss = F.cross_entropy(logits, yb)
            m = cb >= 0
            if m.any():
                loss = loss + F.cross_entropy(cmap[m], cb[m])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(idx); n += len(idx)
        model.eval(); correct = 0; per = {}; chit = 0; cn = 0
        with torch.no_grad():
            for i in range(0, len(yv), 512):
                xb = Xv[i:i + 512].to(dev); lg, cm = model(xb, with_click=True); pred = lg.argmax(1).cpu()
                correct += (pred == yv[i:i + 512]).sum().item()
                cb = cv[i:i + 512]; mm = cb >= 0
                if mm.any():
                    pc = cm.argmax(1).cpu()[mm]; tc = cb[mm]
                    chit += ((pc // 64 - tc // 64).abs().le(2) & (pc % 64 - tc % 64).abs().le(2)).sum().item(); cn += int(mm.sum())
                for m, ok in zip(mv[i:i + 512].tolist(), (pred == yv[i:i + 512]).tolist()):
                    per.setdefault(m, [0, 0]); per[m][0] += ok; per[m][1] += 1
        acc = correct / max(1, len(yv))
        print(f"эпоха {ep + 1}: loss {tl / n:.3f}, val acc {acc:.3f}, клик±2 {chit / max(1, cn):.2f} ({cn}), по механикам " + ", ".join(f"{mechs_v[m] if m < len(mechs_v) else m} {v[0] / v[1]:.2f}" for m, v in sorted(per.items())) + f", {time.time() - t0:.0f} с", flush=True)
    torch.save({"state": model.state_dict(), "mechs": mechs, "arch": a.arch}, a.out); print("сохранено", a.out)


if __name__ == "__main__":
    main()

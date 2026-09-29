"""Модель мира из локальных правил клетки — без языковой модели и без заданных типов механик (27.09, офлайн).

Идея: новый цвет клетки = функция её окрестности и хода. Таблица «(ход, окрестность R) -> Counter(новый цвет)»
учится по ВСЕМ клеткам всех уже увиденных переходов игры и применяется к каждой клетке новой доски; правило,
выученное в одном месте доски, работает в любом другом. Для кликов вместо хода в ключ идёт положение клетки
относительно точки клика (сдвиг по строке и столбцу, обрезанный до ±3, дальше — «далеко»).
Невиданная окрестность -> клетка не меняется. Окрестности: 3x3 (R=1), 5x5 (R=2) и сочетание «5x5, если видели, иначе 3x3».
Сверка — как в rule_learner_probe.py: ход за ходом, обучение только на прошлых переходах игры, индикатор у края исключён.
usage: .venv/bin/python scripts/local_rule_probe.py [runs/flash_v1_phaseA ...]
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from rule_learner_probe import game_transitions, hud_mask, eq  # noqa: E402

M61 = (1 << 61) - 1


def patch_keys(g, R):
    """Хеш окрестности (2R+1)^2 для каждой клетки, за краем — 16."""
    H, W = g.shape
    p = np.full((H + 2 * R, W + 2 * R), 16, dtype=np.int64); p[R:R + H, R:R + W] = g
    k = np.zeros((H, W), dtype=np.int64)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            k = (k * 1000003 + p[R + dy:R + dy + H, R + dx:R + dx + W] + 1) & M61
    return k


def act_keys(a, data, H, W):
    """Ключ хода для каждой клетки: для клавиш — номер хода, для клика — сдвиг от точки клика."""
    base = {"ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4, "ACTION5": 5, "ACTION7": 7}.get(a)
    if base is not None:
        return np.full((H, W), base * 1000, dtype=np.int64)
    y, x = int((data or {}).get("y", -100)), int((data or {}).get("x", -100))
    rr = np.clip(np.arange(H)[:, None] - y, -4, 4); cc = np.clip(np.arange(W)[None, :] - x, -4, 4)
    return (60000 + (rr + 4) * 10 + (cc + 4)).astype(np.int64) + np.zeros((H, W), dtype=np.int64)


class Local:
    def __init__(self, R):
        self.R = R; self.t = defaultdict(Counter)

    def keys(self, g, a, data):
        return (patch_keys(g, self.R) * 131 + act_keys(a, data, *g.shape)) & M61

    def observe(self, g0, a, data, g1):
        k = self.keys(g0, a, data).ravel(); v = g1.ravel()
        for kk, vv in zip(k.tolist(), v.tolist()):
            self.t[kk][vv] += 1

    def predict(self, g, a, data, known=None):
        k = self.keys(g, a, data).ravel(); out = g.ravel().copy(); hit = np.zeros(out.shape, dtype=bool)
        for i, kk in enumerate(k.tolist()):
            c = self.t.get(kk)
            if c:
                out[i] = c.most_common(1)[0][0]; hit[i] = True
        return out.reshape(g.shape), hit.reshape(g.shape)


def main():
    runs = sys.argv[1:] or ["runs/flash_v1_phaseA"]
    env_dir = str(ROOT / "environment_files")
    for run in runs:
        tot = Counter(); rows = []
        for gr in json.loads((ROOT / run / "benchmark.json").read_text())["game_runs"]:
            tr = game_transitions(gr["game_id"], gr.get("history") or [], env_dir)
            if len(tr) < 5:
                continue
            L1, L2 = Local(1), Local(2); n = len(tr); c = Counter(); changed_ok = 0; changed_n = 0
            for t, (g0, a, data, g1) in enumerate(tr):
                mask = hud_mask(tr[:t], *g0.shape)
                p1, _ = L1.predict(g0, a, data); p2, h2 = L2.predict(g0, a, data)
                pc = np.where(h2, p2, p1)                               # 5x5, если видели, иначе 3x3
                c["ident"] += eq(g0, g1, mask); c["r1"] += eq(p1, g1, mask); c["r2"] += eq(p2, g1, mask); c["comb"] += eq(pc, g1, mask)
                if not eq(g0, g1, mask):                                 # переходы, где что-то менялось
                    changed_n += 1; changed_ok += eq(pc, g1, mask)
                L1.observe(g0, a, data, g1); L2.observe(g0, a, data, g1)
            rows.append((gr["game_id"][:4], n, c, changed_ok, changed_n))
            for k in ("ident", "r1", "r2", "comb"):
                tot[k] += c[k]
            tot["n"] += n; tot["chg_ok"] += changed_ok; tot["chg_n"] += changed_n
        print("== %s" % run)
        for g, n, c, co, cn in sorted(rows):
            print("  %s переходов %4d | «ничего не меняется» %3d%% | 3x3 %3d%% | 5x5 %3d%% | 5x5→3x3 %3d%% | из изменившихся угадано %d/%d" % (
                g, n, 100 * c["ident"] // n, 100 * c["r1"] // n, 100 * c["r2"] // n, 100 * c["comb"] // n, co, cn))
        N = tot["n"]
        print("  ИТОГО переходов %d | «ничего не меняется» %.0f%% | 3x3 %.0f%% | 5x5 %.0f%% | 5x5→3x3 %.0f%% | из изменившихся угадано %.0f%% (%d/%d)" % (
            N, 100 * tot["ident"] / N, 100 * tot["r1"] / N, 100 * tot["r2"] / N, 100 * tot["comb"] / N,
            100 * tot["chg_ok"] / max(tot["chg_n"], 1), tot["chg_ok"], tot["chg_n"]))


if __name__ == "__main__":
    main()

"""Сколько «точности» симулятора съедает индикатор (счётчик ходов, полоса энергии) — без модели (26.09).

Зонд CEGIS требует точного совпадения доски. Если в игре есть клетки, меняющиеся почти на каждом ходу независимо
от действия (счётчик, полоса), модель обязана их воспроизвести, иначе «не точно». Считаем по тем же 6 обучающим и
16 отложенным переходам первого уровня (runs/flash_v1_phaseA): клетки-индикатор = меняются в >=60% обучающих
переходов И лежат в строках/столбцах у края (<=3 от края). Затем: сколько переходов отличаются ТОЛЬКО индикатором.
usage: .venv/bin/python scripts/wm_hud_share.py
"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("WM_INFER_SRC", "/tmp/nf_kaggle/src/ARC3-Inference")
import wm_cegis_probe as w

def changed(b, a):
    return {(r, c) for r in range(len(b)) for c in range(len(b[0])) if b[r][c] != a[r][c]}

bench = json.loads((ROOT / "runs/flash_v1_phaseA/benchmark.json").read_text())["game_runs"]
probe = {x["game"]: x for x in json.loads((ROOT / "runs/wm_cegis_probe2.json").read_text())}
tot_hud_only = tot = 0
for g in bench:
    t = w.level1_transitions(g["game_id"], g.get("history") or [])
    if len(t) < 6: continue
    train, test = t[:6], t[6:22]
    cnt = {}
    for b, a, af in train:
        for rc in changed(b, af): cnt[rc] = cnt.get(rc, 0) + 1
    H, W = len(train[0][0]), len(train[0][0][0])
    edge = lambda r, c: r <= 3 or c <= 3 or r >= H - 4 or c >= W - 4
    hud = {rc for rc, k in cnt.items() if k >= 0.6 * len(train) and edge(*rc)}
    sizes = [len(changed(b, af)) for b, a, af in t[:22]]
    only = sum(1 for b, a, af in t[:22] if changed(b, af) and changed(b, af) <= hud)
    noop = sum(1 for s in sizes if s == 0)
    tot_hud_only += only; tot += len(t[:22])
    p = probe.get(g["game_id"][:4], {})
    print("%s  hud-клеток %3d | переходов %2d: пустых %2d, только индикатор %2d | медиана изменённых %4d | зонд: %s %s/%s" % (
        g["game_id"][:4], len(hud), len(t[:22]), noop, only, sorted(sizes)[len(sizes)//2],
        "ТОЧНО" if p.get("exact") else "-    ", p.get("test_ok"), p.get("test_n")))
print("переходов, отличающихся только индикатором: %d из %d" % (tot_hud_only, tot))

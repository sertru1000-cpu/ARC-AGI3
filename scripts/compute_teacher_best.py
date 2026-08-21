"""One-off: compute per-game MAX level reached by the teacher (Gemini Pro),
split into 'pure' (no human hint) vs 'with hints' trace files."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEACHER_DIR = ROOT / "data" / "teacher"

pure_best = defaultdict(int)
pure_denom = {}
hint_best = defaultdict(int)
hint_denom = {}

for f in TEACHER_DIR.glob("*/*.jsonl"):
    game = f.stem
    has_hint = False
    max_level = 0
    win_levels = None
    for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        cross = rec.get("cross_note") or ""
        if "human hint" in cross:
            has_hint = True
        lvl = rec.get("level")
        if isinstance(lvl, int):
            max_level = max(max_level, lvl)
        wl = rec.get("win_levels")
        if isinstance(wl, int) and wl:
            win_levels = wl

    if has_hint:
        hint_best[game] = max(hint_best[game], max_level)
        if win_levels:
            hint_denom[game] = win_levels
    else:
        pure_best[game] = max(pure_best[game], max_level)
        if win_levels:
            pure_denom[game] = win_levels

games = sorted(set(pure_best) | set(hint_best))
print(f"{'game':6} {'pure_best':10} {'hint_best':10}")
for g in games:
    p = pure_best.get(g, "-")
    h = hint_best.get(g, "-") if g in hint_best else "-"
    print(f"{g:6} {str(p):10} {str(h):10}")

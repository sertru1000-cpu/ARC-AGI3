"""Per-game Gemini-3.1-Pro stats table (pure vs with-hint), split by TIMESTAMP
of the hint batch launch (21.08 fix -- the "human hint" substring search used
by compute_teacher_best.py missed several hint-batch runs whose cross_note
didn't literally contain that string, e.g. bp35/sp80/dc22/g50t; verified live
with the user on 20-21.08). Hint batches launched 18.08 21:37 MSK = 18:37 UTC;
any teacher dir timestamped after that for one of the 14 hint-target games is
treated as a hint attempt regardless of whether cross_note echoed the text.

Writes docs/gemini_pro_stats.md (also prints to stdout).
"""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEACHER_DIR = ROOT / "data" / "teacher"
OUT = ROOT / "docs" / "gemini_pro_stats.md"

HINT_CUTOFF = datetime(2026, 8, 18, 18, 37, 0)
HINT_GAMES = {"ar25", "tu93", "tn36", "tr87", "sk48", "dc22", "sc25", "sp80",
              "bp35", "g50t", "cn04", "cd82", "r11l"}

# base-nodistill (raw 35B-A3B, no LoRA) stand, 20.08: 8x3=24 episodes over
# lp85,sb26,wa30,vc33,ft09,m0r0,tn36,ls20 (serve_and_run.sh default GAMES) --
# only lp85 (3/3 reps) and sb26 (1/3) ever reached level 1; every other game
# tested, and every game NOT in this default 8-game set, scored 0/untested.
QWEN_35B_PURE = {"lp85": 1, "sb26": 1}

# Only real Gemini-3.1-Pro dirs (exclude the flash-preview control dirs).
DIR_PREFIXES = ("google_gemini-3.1-pro-preview_", "models_gemini-3.1-pro-preview_")


def dir_timestamp(dirname: str) -> datetime | None:
    for pfx in DIR_PREFIXES:
        if dirname.startswith(pfx):
            stamp = dirname[len(pfx):]
            try:
                return datetime.strptime(stamp, "%Y%m%d_%H%M%S")
            except ValueError:
                return None
    return None


def main() -> None:
    # game -> list of (timestamp, max_level)
    records: dict[str, list[tuple[datetime, int]]] = defaultdict(list)

    for d in sorted(TEACHER_DIR.iterdir()):
        if not d.is_dir():
            continue
        ts = dir_timestamp(d.name)
        if ts is None:
            continue
        for f in d.glob("*.jsonl"):
            game = f.stem.lower()
            max_level = 0
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lvl = rec.get("level")
                if isinstance(lvl, int):
                    max_level = max(max_level, lvl)
            records[game].append((ts, max_level))

    rows = []
    for game, entries in sorted(records.items()):
        pure = [lvl for ts, lvl in entries if not (game in HINT_GAMES and ts > HINT_CUTOFF)]
        hint = [lvl for ts, lvl in entries if game in HINT_GAMES and ts > HINT_CUTOFF]
        pure_best = max(pure) if pure else None
        # Absolute value, not a "not applicable" dash: games never hinted
        # simply carry their pure best forward here (no hint attempt changed it).
        hint_best = max(hint) if hint else pure_best
        rows.append((game, pure_best, hint_best))

    lines = []
    lines.append("# Gemini-3.1-Pro (teacher) per-game stats\n")
    lines.append(
        "Best level reached per game: **pure** (no human hint) vs **+hint best** "
        "(absolute best level reached -- equals pure best for the games that "
        "were never hinted; higher than pure only for the 13 games that got a "
        "human hint after 18.08 21:37 MSK, see `scripts/compute_gemini_stats.py` "
        "for the exact method). `qwen_35b_pure` is left empty here for the raw "
        "35B-A3B base-model comparison (backlog).\n"
    )
    lines.append("| game | pure best | +hint best | qwen_35b_pure |")
    lines.append("|---|---|---|---|")
    for game, pure_best, hint_best in rows:
        pb = str(pure_best) if pure_best is not None else "—"
        hb = str(hint_best) if hint_best is not None else "—"
        qwen = QWEN_35B_PURE.get(game, 0)
        lines.append(f"| {game} | {pb} | {hb} | {qwen} |")

    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()

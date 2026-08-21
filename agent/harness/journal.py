"""Cross-game journal: knowledge that survives from game to game in one run.

Phase B plays ~30 hidden games in a single process — this file carries two
kinds of experience across them:

  1. Statistics (computed, no LLM): how often moves/clicks/interacts actually
     changed the board, how often edge-strips turned out to be HUD timers,
     death counts. Injected as priors into every new game's first message.
  2. Lessons (LLM reflection, only after games where we completed levels):
     short transferable notes, deduplicated, capped.

Guards against negative transfer: lessons are capped, phrased as hints, and
the whole feature sits behind MY_AGENT_CROSS_MEMORY=0.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

# Phase B runs ~30 game threads in one process; every agent touches the same
# journal file. One process-wide lock + atomic replace keeps it uncorrupted
# (a torn read used to silently reset the journal to empty).
_JOURNAL_LOCK = threading.Lock()

import numpy as np

from .perception import grid_diff

MAX_LESSONS = 10
MAX_LESSON_CHARS = 220


class CrossGameJournal:
    def __init__(self, path: str | None = None):
        raw = path or os.getenv("MY_AGENT_CROSS_MEMORY_PATH", "cross_memory.json")
        self.path = Path(raw)
        self.data: dict = {"games": 0, "levels": 0, "stats": {}, "lessons": []}
        try:
            with _JOURNAL_LOCK:
                if self.path.exists():
                    self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # ── recording ─────────────────────────────────────────────────────────
    def record_game(self, replay_log: list[dict], levels: int, lessons: list[str]) -> None:
        # Concurrent agents each hold their own stale copy of the journal;
        # merge against the FRESH on-disk state under the lock, otherwise the
        # last writer silently erases everyone else's updates.
        with _JOURNAL_LOCK:
            try:
                if self.path.exists():
                    self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
            self._record_game_locked(replay_log, levels, lessons)

    def _record_game_locked(self, replay_log: list[dict], levels: int, lessons: list[str]) -> None:
        s = self.data["stats"]

        def bump(key: str, flag: bool) -> None:
            k_num, k_den = f"{key}_yes", f"{key}_of"
            s[k_den] = s.get(k_den, 0) + 1
            if flag:
                s[k_num] = s.get(k_num, 0) + 1

        move_eff = click_eff = interact_eff = hud_seen = False
        deaths = 0
        prev_grid = None
        for e in replay_log:
            grid = e.get("grid")
            if prev_grid is not None and grid is not None and prev_grid.shape == grid.shape:
                d = grid_diff(prev_grid, grid)
                real = d.changed and not d.border_only
                if d.changed and d.border_only:
                    hud_seen = True
                if real:
                    if e["id"] in (1, 2, 3, 4):
                        move_eff = True
                    elif e["id"] == 6:
                        click_eff = True
                    elif e["id"] in (5, 7):
                        interact_eff = True
            if str(e.get("state", "")).endswith("GAME_OVER"):
                deaths += 1
            prev_grid = grid

        bump("moves_changed_board", move_eff)
        bump("clicks_changed_board", click_eff)
        bump("interact_changed_board", interact_eff)
        bump("hud_timer_seen", hud_seen)
        s["deaths"] = s.get("deaths", 0) + deaths

        self.data["games"] += 1
        self.data["levels"] += int(levels)

        for lesson in lessons:
            lesson = " ".join(str(lesson).split())[:MAX_LESSON_CHARS]
            if lesson and lesson not in self.data["lessons"]:
                self.data["lessons"].insert(0, lesson)
        self.data["lessons"] = self.data["lessons"][:MAX_LESSONS]
        self._save_locked()

    def save(self) -> None:
        with _JOURNAL_LOCK:
            self._save_locked()

    def _save_locked(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception:
            pass  # journal must never break a run

    # ── injection ─────────────────────────────────────────────────────────
    def summary_text(self) -> str:
        n = self.data.get("games", 0)
        if n == 0:
            return ""
        s = self.data.get("stats", {})

        def frac(key: str) -> str:
            return f"{s.get(f'{key}_yes', 0)}/{s.get(f'{key}_of', 0)}"

        lines = [
            f"Experience from {n} game(s) already played this run "
            f"({self.data.get('levels', 0)} levels completed):",
            f"- direction actions changed the board in {frac('moves_changed_board')} games",
            f"- CLICK mattered in {frac('clicks_changed_board')} games",
            f"- SPACE/UNDO mattered in {frac('interact_changed_board')} games",
            f"- an edge strip turned out to be a HUD timer in {frac('hud_timer_seen')} games",
        ]
        if self.data.get("lessons"):
            lines.append("Lessons from earlier games (hints only — this game may differ):")
            lines += [f"  * {t}" for t in self.data["lessons"][:5]]
        return "\n".join(lines)


REFLECTION_PROMPT = (
    "You just finished playing an unknown grid game and completed {levels} level(s). "
    "Your final world model was: {world_model}\n\n"
    "Write 1-2 SHORT transferable lessons that could help in OTHER, different "
    "grid games. Rules: generic strategy/process only; do NOT mention specific "
    "colors, coordinates, shapes or this game's mechanics. One lesson per line, "
    "plain text, no numbering."
)

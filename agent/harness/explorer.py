"""Heuristic exploration policy — the no-LLM baseline brain.

Strategy per game:
  1. RESET when the game hasn't started or after GAME_OVER.
  2. Probe: try every available simple action a few times, tracking whether it
     actually changes the board (HUD-only changes count as weaker evidence).
  3. Exploit what probing showed: prefer actions that change the board,
     avoid actions that repeatedly do nothing in the current state.
  4. ACTION6 games: click centers of salient objects (small, rare-colored),
     cycling through targets instead of clicking randomly.
  5. Anti-loop: if recent frames repeat, force the least-used action/target.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from .perception import frame_hash, grid_diff, latest_grid, salient_click_targets


@dataclass
class ActionStats:
    tried: int = 0
    changed_board: int = 0
    hud_only: int = 0
    caused_game_over: int = 0

    @property
    def change_rate(self) -> float:
        return self.changed_board / self.tried if self.tried else 0.0


@dataclass
class ExplorerState:
    """Per-game knowledge the explorer accumulates."""

    stats: dict[str, ActionStats] = field(default_factory=lambda: defaultdict(ActionStats))
    # (frame_hash, action_key) pairs that produced no change — dead in that state.
    dead: set[tuple[str, str]] = field(default_factory=set)
    recent_hashes: deque = field(default_factory=lambda: deque(maxlen=12))
    click_queue: list[tuple[int, int]] = field(default_factory=list)
    clicks_done: Counter = field(default_factory=Counter)
    last_level: int = 0
    prev_grid: np.ndarray | None = None
    last_action_key: str | None = None


class HeuristicExplorer:
    """Chooses the next (action_name, payload|None) given frame history."""

    PROBE_ROUNDS = 2  # try each simple action this many times before exploiting
    EPSILON = 0.15  # residual randomness to escape local ruts

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.state = ExplorerState()

    # ── bookkeeping ────────────────────────────────────────────────────────
    def observe(self, latest_frame) -> None:
        """Update knowledge with the result of the previous action."""
        st = self.state
        grid = latest_grid(latest_frame)
        if grid is None:
            return
        if st.prev_grid is not None and st.last_action_key is not None:
            diff = grid_diff(st.prev_grid, grid)
            a = st.stats[st.last_action_key]
            a.tried += 1
            if diff.changed and not diff.border_only:
                a.changed_board += 1
            elif diff.changed:
                a.hud_only += 1
            else:
                st.dead.add((frame_hash(st.prev_grid), st.last_action_key))
            if str(getattr(latest_frame, "state", "")) .endswith("GAME_OVER"):
                a.caused_game_over += 1

        level = int(getattr(latest_frame, "levels_completed", 0) or 0)
        if level != st.last_level:
            # New level: layouts change — refresh click targets and loop memory.
            st.last_level = level
            st.click_queue.clear()
            st.recent_hashes.clear()
        st.prev_grid = grid
        st.recent_hashes.append(frame_hash(grid))

    # ── decision ───────────────────────────────────────────────────────────
    def decide(self, latest_frame, simple_actions: list[str], has_click: bool) -> tuple[str, dict | None]:
        st = self.state
        grid = latest_grid(latest_frame)
        cur_hash = frame_hash(grid) if grid is not None else ""

        candidates: list[tuple[str, dict | None]] = []
        for name in simple_actions:
            if (cur_hash, name) not in st.dead:
                candidates.append((name, None))
        if has_click and grid is not None:
            if not st.click_queue:
                st.click_queue = salient_click_targets(grid)
            for x, y in st.click_queue[:6]:
                key = f"ACTION6@{x},{y}"
                if (cur_hash, key) not in st.dead:
                    candidates.append(("ACTION6", {"x": x, "y": y}))

        if not candidates:
            # Everything looks dead in this state — reprobe with anything.
            candidates = [(n, None) for n in simple_actions] or [
                ("ACTION6", {"x": self.rng.randint(0, 63), "y": self.rng.randint(0, 63)})
            ]

        looping = len(st.recent_hashes) >= 8 and len(set(st.recent_hashes)) <= 2

        def key_of(name: str, payload: dict | None) -> str:
            return f"{name}@{payload['x']},{payload['y']}" if payload else name

        def score(name: str, payload: dict | None) -> float:
            s = st.stats[key_of(name, payload)]
            if s.tried < self.PROBE_ROUNDS:
                return 10.0 - s.tried  # unexplored first
            base = s.change_rate - 2.0 * (s.caused_game_over / max(s.tried, 1))
            if looping:
                base -= 0.5 * s.tried  # break the cycle: prefer least-used
            return base + self.rng.random() * 0.01

        if self.rng.random() < self.EPSILON:
            choice = self.rng.choice(candidates)
        else:
            choice = max(candidates, key=lambda c: score(*c))

        name, payload = choice
        st.last_action_key = key_of(name, payload)
        if payload:
            st.clicks_done[(payload["x"], payload["y"])] += 1
            # Rotate used click targets to the back of the queue.
            if (payload["x"], payload["y"]) in st.click_queue:
                st.click_queue.remove((payload["x"], payload["y"]))
                st.click_queue.append((payload["x"], payload["y"]))
        return name, payload

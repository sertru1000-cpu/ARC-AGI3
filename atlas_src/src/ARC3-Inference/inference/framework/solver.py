"""TAAF solver adapter for the existing tool-using harness."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import functools
import html
import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import arcengine
import taaf.game
from taaf.solver import Solver

from inference.agent.action_names import (
    to_engine_action,
    to_model_action,
    to_model_actions,
)
from inference.agent.runtime_state import (
    Frame,
    HistoryEntry,
    RUNTIME_STATE_FILENAME,
    write_runtime_state,
)
from inference.agent.tool_agent import ToolAgent
from inference.framework.kaggle import (
    DEFAULT_QWEN_MODEL_DATASET_SOURCE,
    DEFAULT_SERVED_MODEL_NAME,
    DEFAULT_VLLM_MAX_MODEL_LEN,
    DEFAULT_VLLM_PORT,
    DEFAULT_VLLM_TENSOR_PARALLEL_SIZE,
    DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE,
    DEFAULT_WHEELHOUSE_STAMP_TEXT,
    DuckKaggleVllmConfig,
    duck_kaggle_dataset_sources,
    duck_kaggle_setup_command,
    duck_kaggle_teardown_command,
)
from inference.utils.animation import normalize_frames, pick_animation, summarize_animation
from inference.utils.viewer_artifacts import (
    append_raw_events_sidecar,
    reset_raw_events_sidecar,
)

AnalyzerFactory = Callable[[taaf.game.Game, int], Any]

# One model turn can batch several actions; keeping the last few animated ones
# lets the agent look back at any of them without pinning whole runs in memory.
ANIMATION_HISTORY_DEPTH = 4

ANALYZER_RETRY_BACKOFF_SECONDS = 1.0
DEFAULT_CANCEL_DRAIN_TIMEOUT_SECONDS = 120.0

# atlas: free replay-after-win ("speedrun"). See _speedrun_after_win below.
_ATLAS_SPEEDRUN_ENABLED = os.environ.get("ATLAS_SPEEDRUN_AFTER_WIN", "1") != "0"

# atlas 25.08: shared time bank across concurrently running games. Every game
# gets the SAME max_runtime_s_per_game today, uniformly, regardless of how it's
# actually doing -- a game stuck at level 0 burns its whole allocation for zero
# score, while a game that's already proven it can clear levels (RHAE's own
# per-run weighting rewards LATER levels more) could use more time productively.
# Rather than live cross-game preemption (a bigger, riskier change to the
# ThreadPoolExecutor+Semaphore scheduler), this is a simple deposit/draw pool:
# a session that's genuinely stalled (see _ATLAS_STALL_WINDOW/_FRACTION below)
# ends itself early and returns its unused time to the bank; a session that
# just reached a NEW level it hadn't seen before may draw from the bank to
# extend its OWN deadline, bounded by _ATLAS_TIME_BANK_MAX_EXTRA_MULTIPLIER so
# one lucky game can't starve everything else (including games still queued
# behind the concurrency semaphore).
_ATLAS_TIME_BANK_ENABLED = os.environ.get("ATLAS_TIME_BANK_ENABLED", "1") != "0"
# Only allow the early-stop-and-deposit path once this fraction of the base
# per-game cap has elapsed with ZERO level progress -- guards against cutting
# off a game that's just a slow, legitimate starter. Raised 0.4->0.5 in v19
# alongside dropping the old literal-no-op requirement (see
# _atlas_stall_decision's docstring) -- the trigger is more permissive now
# (fires on busy-but-unproductive play too, not just frozen play), so a
# slightly larger margin against cutting off a near-success stays cheap
# insurance.
_ATLAS_STALL_EARLY_STOP_FRACTION = 0.5
# A session may draw at most this fraction of its base cap in ONE grant, and
# never accumulate more than _MAX_EXTRA_MULTIPLIER x its base cap in total
# extensions -- keeps any single game from monopolizing the shared bank.
_ATLAS_TIME_BANK_DRAW_FRACTION = 0.5
_ATLAS_TIME_BANK_MAX_EXTRA_MULTIPLIER = 1.0

# atlas 25.08: retry-storm backstop, found live on v17 -- 3 of 25 games
# (cn04, lp85, re86) spent 15-30 CONSECUTIVE MINUTES retrying the exact same
# analysis_step after repeated analyzer request timeouts (the shared local
# LLM backend under concurrency=14 load), never producing a single python
# tool call for that whole stretch. Neither the force-act circuit breaker
# nor the stall detector above can see this -- both only observe what
# happens INSIDE a python tool call, and this failure happens at the
# analyzer/request level, before any tool call exists. `play()`'s retry loop
# (ANALYZER_RETRY_BACKOFF_SECONDS = 1.0s backoff, no cap) will otherwise
# retry forever until the per-game deadline itself is reached, burning the
# whole allocation on zero real progress. Reuses the same shared time bank:
# once a session hits this many CONSECUTIVE retryable analyzer failures, it
# ends itself early and returns its unused remainder, exactly like a stalled
# session -- just with a different (infra, not gameplay) trigger.
_ATLAS_RETRY_STORM_THRESHOLD = 5
_ATLAS_MOUSE_DISPLAY_RE = re.compile(r"^MOUSE\(row=(\d+), col=(\d+)\)$")
_LOCAL_SERVER_PROCESS_ENV_KEYS = (
    "LOCAL_ANALYZER_API_KEY",
    "OPENAI_API_KEY",
    "LOCAL_ANALYZER_BASE_URL",
    "OPENAI_BASE_URL",
    "LOCAL_ANALYZER_PROVIDER",
    "OPENAI_PROVIDER",
)


@dataclass
class _LocalServerRuntime:
    index: int
    repo_dir: Path
    api_key_file: Path
    env_overrides: dict[str, str]
    base_url: str
    api_key: str = ""


def _analyzer_reported_tokens(analyzer: Any) -> int:
    value = (
        getattr(analyzer, "generated_tokens", None)
        if hasattr(analyzer, "generated_tokens")
        else getattr(analyzer, "total_tokens", 0)
    )
    return max(0, int(value or 0))


def artifact_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _grid_from_state(state: taaf.game.GameState | None) -> tuple[tuple[int, ...], ...]:
    if state is None:
        return ()
    data = state.frame.data
    rows = data.tolist() if hasattr(data, "tolist") else data
    return tuple(tuple(int(cell) for cell in row) for row in rows)


def _raw_frames(state: taaf.game.GameState | None) -> list[tuple[tuple[int, ...], ...]]:
    """Every frame the environment returned for one action.

    ``arcengine`` renders a frame after every internal ``step()``, so a single
    action can come back as a short animation. ``state.frame`` is only the
    last of those (``raw.frame[-1]``), which is why the visible board can look
    untouched even though the action clearly did something -- the whole signal
    then lives in the intermediate frames (measured: ft09, sb26).
    """
    if state is None:
        return []
    return normalize_frames(getattr(state.raw, "frame", None))


def _frame_count(state: taaf.game.GameState | None) -> int:
    """Number of frames the environment returned for one action."""
    if state is None:
        return 0
    return len(getattr(state.raw, "frame", None) or ())


def _level_number(game: taaf.game.Game) -> int:
    state = game.current_state
    completed = int(state.levels_completed)
    if state.won:
        return max(1, int(game.number_of_levels))
    return max(1, min(int(game.number_of_levels), completed + 1))


def _engine_action_names(game: taaf.game.Game) -> list[str]:
    names: list[str] = []
    for action_id in game.current_state.available_actions:
        try:
            name = arcengine.GameAction.from_id(int(action_id)).name
        except Exception:
            continue
        if name == "RESET":
            continue
        if name not in names:
            names.append(name)
    return names


def _model_mouse_action_data(
    action_data: dict[str, Any] | None = None,
) -> dict[str, int]:
    data = action_data or {}
    return {"row": int(data.get("y", 0)), "col": int(data.get("x", 0))}


def _format_action_display(
    action_name: str, action_data: dict[str, Any] | None = None
) -> str:
    if action_name == "ACTION6":
        data = _model_mouse_action_data(action_data)
        return f"MOUSE(row={data['row']}, col={data['col']})"
    return to_model_action(action_name)


def _is_engine_game_over(game: taaf.game.Game) -> bool:
    return game.current_state.raw.state == arcengine.GameState.GAME_OVER


def _is_run_complete(game: taaf.game.Game) -> bool:
    return game.current_state.raw.state == arcengine.GameState.WIN


def _atlas_stall_decision(
    *,
    elapsed: float,
    cap: float,
    start_level: int,
    current_level: int,
    already_deposited: bool,
) -> tuple[bool, float]:
    """Should this session stop itself now and return unused time to the
    bank? Returns (should_stop_now, seconds_to_deposit). Never fires for a
    session that has progressed past its starting level, or has already
    deposited once (one deposit per session, not a recurring drain).

    25.08 v19: dropped the earlier "6 consecutive actions with a literally
    unchanged grid" requirement (see _atlas_recent_stalled, removed) after
    v18's real logs showed it never fires in practice -- 0 deposits, 0 draws
    across a full 4h/25-game run, even though 11 of 25 games spent nearly
    their whole allocation stuck at level 0. Real unproductive play still
    changes the board constantly (wrong clicks, movement that doesn't help,
    etc.) -- it just never advances the LEVEL, which is the only thing RHAE
    actually scores. Zero level progress past a large fraction of the cap is
    itself sufficient evidence; the literal-no-op signal was catching a
    different (and apparently much rarer) failure mode."""
    if current_level > start_level or already_deposited:
        return False, 0.0
    if cap <= 0 or elapsed < cap * _ATLAS_STALL_EARLY_STOP_FRACTION:
        return False, 0.0
    return True, max(0.0, cap - elapsed)


def _atlas_extension_request(
    *,
    cap: float,
    current_extra: float,
    current_level: int,
    last_credited_level: int,
) -> float:
    """How much extra time (if any) this session should TRY to draw from the
    bank, having just reached a level it hadn't been credited for yet. The
    actual grant is capped again by whatever the bank actually holds (see
    HarnessSolver.atlas_draw_time) -- this only bounds the ASK, so one game
    can never accumulate more than _MAX_EXTRA_MULTIPLIER x its own base cap
    even if the bank is flush."""
    if cap <= 0 or current_level <= last_credited_level:
        return 0.0
    max_extra = cap * _ATLAS_TIME_BANK_MAX_EXTRA_MULTIPLIER
    room = max(0.0, max_extra - current_extra)
    if room <= 0:
        return 0.0
    return min(room, cap * _ATLAS_TIME_BANK_DRAW_FRACTION)


def _atlas_retry_storm_decision(
    *, consecutive_failures: int, threshold: int, already_deposited: bool
) -> bool:
    """Should this session end itself now due to a run of consecutive
    analyzer-level request failures (backend timeouts under load, not a
    gameplay stall)? Pure/testable: no session or game state, just counters."""
    if already_deposited:
        return False
    return consecutive_failures >= threshold


def _write_transcript_html(transcript_path: Path, html_path: Path, title: str) -> None:
    if not transcript_path.exists():
        return
    html_path.parent.mkdir(parents=True, exist_ok=True)
    text = transcript_path.read_text(encoding="utf-8")
    body = (
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        "<style>"
        "body{background:#1e1e1e;color:#e0e0e0;font-family:-apple-system,system-ui,sans-serif;"
        "padding:20px;max-width:1100px;margin:0 auto;line-height:1.4;}"
        "h1{color:#fff;}pre{white-space:pre-wrap;background:#111;padding:16px;border-radius:6px;"
        "border:1px solid #333;overflow:auto;}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1><pre>{html.escape(text)}</pre>"
        "</body></html>\n"
    )
    html_path.write_text(body, encoding="utf-8")


@dataclass
class _HarnessGameSession:
    solver: "HarnessSolver"
    game: taaf.game.Game
    analyzer: Any
    game_index: int
    pass_index: int
    state_path: Path
    transcript_path: Path
    analysis_html_relpath: str
    stop_event: threading.Event
    viewer_data_path: Path
    started_at: float = field(default_factory=time.monotonic)
    history_entries: list[HistoryEntry] = field(default_factory=list)
    viewer_events: list[dict[str, Any]] = field(default_factory=list)
    analysis_step: int = 0
    last_engine_action: str | None = None
    token_baseline: int = 0
    # Experiment 3: all three animation-awareness stages ship in one arm, so
    # they have to be separable by log analysis afterwards. Stage 1 counts
    # here (the metadata is produced here); stages 2 and 3 count inside the
    # ToolAgent and are merged in at teardown by _append_experiment_event.
    experiment_counters: dict[str, int] = field(default_factory=dict)
    # Stage 2: raw frames of the most recent animated actions, kept in memory
    # only. They are far too large for the runtime-state file (42 frames of
    # 64x64 for one sb26 action) and are never serialized -- the agent asks
    # for them through step_env and only the compacted view crosses the
    # sandbox boundary.
    animation_history: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=ANIMATION_HISTORY_DEPTH)
    )
    _viewer_events_flushed: int = field(default=0, init=False, repr=False)
    # atlas 25.08: time-bank state, see _ATLAS_TIME_BANK_ENABLED above.
    _atlas_extra_time_s: float = field(default=0.0, init=False, repr=False)
    _atlas_stall_deposited: bool = field(default=False, init=False, repr=False)
    _atlas_start_level: int | None = field(default=None, init=False, repr=False)
    _atlas_last_credited_level: int = field(default=-1, init=False, repr=False)
    _atlas_consecutive_retry_failures: int = field(default=0, init=False, repr=False)
    _atlas_retry_storm_deposited: bool = field(default=False, init=False, repr=False)

    def current_frame(self) -> Frame:
        return Frame(
            grid=_grid_from_state(self.game.current_state),
            step=self.action_count,
            level=_level_number(self.game),
        )

    def write_runtime_state(self) -> None:
        write_runtime_state(
            self.state_path,
            current_frame=self.current_frame(),
            history=self.history_entries,
        )

    def atlas_snapshot_env(self) -> dict[str, Any] | None:
        """Deep-copy enough of the real engine to later restore an exact
        mid-episode point (save_checkpoint/rollback, 26.08). Only possible
        in OFFLINE mode: the raw ARCBaseGame object (`env._game`) is a plain
        Python object with no network/lock handles, confirmed deepcopy-safe
        (it already implements `.clone()` used internally on every reset).
        ONLINE mode has no local engine object at all (server-authoritative,
        HTTP only) -- returns None there, and callers must treat that as
        "checkpoint/rollback unavailable this session", not retry.
        `game_run` (action-count/history bookkeeping, separate from the raw
        engine and from the scorecard) is snapshotted alongside it so
        `action_count`/`current_frame()` -- both derived properties, not
        stored fields -- report the reverted state with no further syncing
        needed once restored. Deliberately does NOT touch the scorecard:
        confirmed scoring is a monotonic max() over already-recorded
        history, independent of live engine state, so rollback cannot
        un-record an already-achieved level -- nothing to snapshot there.
        """
        env = getattr(self.game, "env", None)
        raw_game = getattr(env, "_game", None) if env is not None else None
        if raw_game is None:
            return None
        try:
            return {
                "game": copy.deepcopy(raw_game),
                "game_run": copy.deepcopy(self.game.game_run),
                "history_entries": [
                    HistoryEntry(action=e.action, frame=e.frame) for e in self.history_entries
                ],
            }
        except Exception:
            return None

    def atlas_restore_env(self, snapshot: dict[str, Any] | None) -> bool:
        if not snapshot:
            return False
        env = getattr(self.game, "env", None)
        if env is None:
            return False
        try:
            env._game = copy.deepcopy(snapshot["game"])
            self.game.game_run = copy.deepcopy(snapshot["game_run"])
        except Exception:
            return False
        self.history_entries = [
            HistoryEntry(action=e.action, frame=e.frame) for e in snapshot["history_entries"]
        ]
        self.write_runtime_state()
        return True

    def seed_initial_history(self) -> None:
        if not self.history_entries:
            self.history_entries.append(
                HistoryEntry(action="", frame=self.current_frame())
            )

    @property
    def action_count(self) -> int:
        run = self.game.game_run
        return len(run.history) if run is not None else 0

    def _atlas_effective_runtime_cap(self) -> float | None:
        """Base max_runtime_s_per_game plus whatever this session has drawn
        from the shared time bank (see _ATLAS_TIME_BANK_ENABLED)."""
        base = self.solver.max_runtime_s_per_game
        if base is None:
            return None
        return base + self._atlas_extra_time_s

    def runtime_limit_reached(self) -> bool:
        cap = self._atlas_effective_runtime_cap()
        if cap is None:
            return False
        return (time.monotonic() - self.started_at) >= cap

    def timing_payload(self) -> dict[str, float | None]:
        elapsed = max(0.0, time.monotonic() - self.started_at)
        cap = self._atlas_effective_runtime_cap()
        if cap is None:
            remaining = None
        else:
            remaining = max(0.0, cap - elapsed)
        return {"run_elapsed_seconds": elapsed, "time_remaining_seconds": remaining}

    def _atlas_check_time_bank(self) -> bool:
        """Poll the time bank each should_stop() check: draw extra time if a
        new level was just reached, or end this session early (returning
        unused time to the bank) if it's genuinely stalled. Returns True if
        this session should stop NOW as a result. No-op, returns False
        immediately, if _ATLAS_TIME_BANK_ENABLED is off or there's no cap."""
        if not _ATLAS_TIME_BANK_ENABLED:
            return False
        cap = self.solver.max_runtime_s_per_game
        if cap is None or cap <= 0:
            return False
        current_level = self.current_frame().level
        if self._atlas_start_level is None:
            self._atlas_start_level = current_level
            # The STARTING level is not "new progress" -- without this, the
            # default -1 sentinel makes every session try to draw on its very
            # first poll (harmlessly returns 0 while the bank is still empty,
            # but is not what "just reached a level it hadn't seen" means).
            self._atlas_last_credited_level = current_level

        requested = _atlas_extension_request(
            cap=cap,
            current_extra=self._atlas_extra_time_s,
            current_level=current_level,
            last_credited_level=self._atlas_last_credited_level,
        )
        if current_level > self._atlas_last_credited_level:
            # Mark this level "tried" regardless of whether a grant follows --
            # a new level with no bank room left must not re-ask every poll.
            self._atlas_last_credited_level = current_level
        if requested > 0:
            granted = self.solver.atlas_draw_time(requested)
            if granted > 0:
                self._atlas_extra_time_s += granted
                print(
                    f"atlas: game {self.game_index} drew {granted:.0f}s from the time bank "
                    f"after reaching level {current_level} (total extra {self._atlas_extra_time_s:.0f}s)",
                    flush=True,
                )

        should_stop, deposit = _atlas_stall_decision(
            elapsed=time.monotonic() - self.started_at,
            cap=cap,
            start_level=self._atlas_start_level,
            current_level=current_level,
            already_deposited=self._atlas_stall_deposited,
        )
        if should_stop:
            self.solver.atlas_deposit_time(deposit)
            self._atlas_stall_deposited = True
            print(
                f"atlas: game {self.game_index} stalled with 0 level progress -- "
                f"ending early, returning {deposit:.0f}s to the time bank",
                flush=True,
            )
        return should_stop

    def _atlas_check_retry_storm(self) -> bool:
        """Backstop for repeated ANALYZER-level request failures (e.g. the
        shared LLM backend timing out under concurrent load) -- a failure
        mode _atlas_check_time_bank cannot see, since it happens before any
        python tool call is ever produced. Called from play()'s retry
        branch, not should_stop(), since play() needs to `break` its own
        loop directly rather than wait for the next should_stop() poll."""
        if not _ATLAS_TIME_BANK_ENABLED:
            return False
        if not _atlas_retry_storm_decision(
            consecutive_failures=self._atlas_consecutive_retry_failures,
            threshold=_ATLAS_RETRY_STORM_THRESHOLD,
            already_deposited=self._atlas_retry_storm_deposited,
        ):
            return False
        cap = self._atlas_effective_runtime_cap()
        deposit = max(0.0, cap - (time.monotonic() - self.started_at)) if cap is not None else 0.0
        self.solver.atlas_deposit_time(deposit)
        self._atlas_retry_storm_deposited = True
        run = self.game.game_run
        if run is not None and run.solver_note is None:
            run.solver_note = (
                f"atlas: ended early after {self._atlas_consecutive_retry_failures} "
                "consecutive analyzer request failures"
            )
        print(
            f"atlas: game {self.game_index} hit {self._atlas_consecutive_retry_failures} "
            f"consecutive analyzer failures -- ending early, returning {deposit:.0f}s to the time bank",
            flush=True,
        )
        return True

    def request_timeout_seconds(self) -> float | None:
        candidates: list[float] = []
        configured = getattr(self.analyzer, "_timeout", None)
        try:
            if configured is not None:
                candidates.append(float(configured))
        except (TypeError, ValueError):
            pass
        if self.solver.max_runtime_s_per_game is not None:
            remaining = self.timing_payload()["time_remaining_seconds"]
            if remaining is not None:
                candidates.append(float(remaining))
        soft_remaining = self.solver.soft_time_remaining_seconds()
        if soft_remaining is not None:
            candidates.append(soft_remaining)
        if not candidates:
            return None
        return max(0.1, min(candidates))

    def should_stop(self) -> bool:
        run = self.game.game_run
        if run is None or run.state != "playing":
            return True
        if self.stop_event.is_set():
            return True
        if _is_run_complete(self.game):
            return True
        if self.runtime_limit_reached():
            return True
        if self._atlas_check_time_bank():
            return True
        if (
            self.solver.max_actions_per_game is not None
            and self.action_count >= self.solver.max_actions_per_game
        ):
            return True
        return False

    def play(self) -> None:
        run = self.game.game_run
        assert run is not None, "TAAF starts games before invoking the solver."
        run.solver_analysis_html = self.analysis_html_relpath
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        self.transcript_path.touch(exist_ok=True)
        self.token_baseline = _analyzer_reported_tokens(self.analyzer)
        self.seed_initial_history()
        self.write_runtime_state()
        self._append_initial_viewer_event()
        self.write_viewer_payload()
        try:
            retry_analysis_step: int | None = None
            while not self.should_stop():
                if (
                    _is_engine_game_over(self.game)
                    and self.last_engine_action != "RESET"
                ):
                    self._execute_auto_reset()
                    continue

                if retry_analysis_step is None:
                    self.analysis_step += 1
                    analysis_step = self.analysis_step
                else:
                    analysis_step = retry_analysis_step

                self.write_runtime_state()
                transcript_before = self._read_transcript_bytes()
                try:
                    result = self.analyzer.analyze(
                        self.state_path,
                        self.action_count,
                        valid_actions=_engine_action_names(self.game),
                        step_env=self.step_env,
                        checkpoint_env=self.atlas_snapshot_env,
                        restore_env=self.atlas_restore_env,
                        transcript_path=self.transcript_path,
                        analysis_step=analysis_step,
                        request_timeout_seconds=self.request_timeout_seconds(),
                        should_stop=self.should_stop,
                    )
                finally:
                    transcript_delta = self._transcript_delta_since(transcript_before)
                    if transcript_delta.strip():
                        self._append_analysis_viewer_event(
                            analysis_step, transcript_delta
                        )
                        self.write_viewer_payload()
                if result is None:
                    raise RuntimeError("Analyzer did not return a result.")
                if result.retryable_failure:
                    retry_analysis_step = analysis_step
                    self._atlas_consecutive_retry_failures += 1
                    if self._atlas_check_retry_storm():
                        break
                    if self.should_stop():
                        break
                    time.sleep(ANALYZER_RETRY_BACKOFF_SECONDS)
                    continue

                retry_analysis_step = None
                self._atlas_consecutive_retry_failures = 0
                if getattr(result, "yielded_control", False):
                    retry_analysis_step = analysis_step
                    continue
                if not result.step_executed:
                    continue
            if _ATLAS_SPEEDRUN_ENABLED and _is_run_complete(self.game):
                self._atlas_speedrun_after_win()
        except Exception as exc:
            if run.final_score is None:
                run.solver_note = f"error: {type(exc).__name__}: {exc}"
                if run.state == "playing":
                    run.state = "crashed"
                self._finish_if_needed()
        finally:
            total_tokens = _analyzer_reported_tokens(self.analyzer)
            if run.solver_note is None:
                run.solver_note = f"tokens={total_tokens}"
            self._finish_if_needed()
            self.state_path.unlink(missing_ok=True)
            self._write_analysis_html()
            self._append_experiment_event()
            self.write_viewer_payload()

    def _atlas_parse_action_display(self, display: str) -> tuple[str, dict[str, Any]] | None:
        """Reverse ``_format_action_display``: 'UP' / 'MOUSE(row=4, col=7)' ->
        (engine action name, data dict for ``arcengine.ActionInput``)."""
        display = display.strip()
        if not display:
            return None
        match = _ATLAS_MOUSE_DISPLAY_RE.match(display)
        if match:
            row, col = int(match.group(1)), int(match.group(2))
            return "ACTION6", {"x": col, "y": row}
        engine_name = to_engine_action(display)
        if engine_name is None or engine_name == "RESET":
            return None
        return engine_name, {}

    def _atlas_pruned_winning_actions(self) -> list[tuple[str, dict[str, Any]]]:
        """The winning action sequence with no-op actions dropped.

        An action is a no-op if it changed neither the grid nor the level --
        the same signal ``_execute_action``'s own ``board_changed`` uses.
        """
        pruned: list[tuple[str, dict[str, Any]]] = []
        if not self.history_entries:
            return pruned
        previous_grid = self.history_entries[0].frame.grid
        previous_level = self.history_entries[0].frame.level
        for entry in self.history_entries[1:]:  # [0] is the seeded action="" entry
            parsed = self._atlas_parse_action_display(entry.action)
            if parsed is not None and (
                entry.frame.grid != previous_grid or entry.frame.level != previous_level
            ):
                pruned.append(parsed)
            previous_grid = entry.frame.grid
            previous_level = entry.frame.level
        return pruned

    def _atlas_speedrun_after_win(self) -> None:
        """After a real win, replay the pruned winning sequence for free.

        Score is MAX-over-plays (arc_agi.scorecard.EnvironmentScoreList), and
        a RESET issued while the engine state is WIN triggers a full reset
        (registers a new play in the SAME scorecard card) regardless of
        ONLY_RESET_LEVELS -- verified in arcengine/base_game.py. The catch:
        taaf.game.Game.execute_action() hard-asserts game_run.state ==
        "playing", and WIN sets it to "won" immediately, so a RESET through
        that wrapper is refused. Calling self.game.env.step() directly
        bypasses ONLY that wrapper assertion -- it is the exact same call
        _execute_action() makes internally, and scorecard bookkeeping
        happens server-side inside env.step() (arc_agi.wrapper), keyed by
        scorecard_id/guid, independent of which caller invoked it. So this
        is scored identically to a normal action.

        Trade-off: taaf's OWN GameRun.actions_per_level/history (and so our
        own summary.txt) do NOT see this replay -- only the real scorecard
        does. Any failure here is swallowed and logged, never raised: the
        original win is already a separately recorded play, so a broken
        replay wastes a few free actions and nothing else.
        """
        label = f"{self.game.game_id}#{self.game_index}.{self.pass_index}"
        pruned = self._atlas_pruned_winning_actions()
        if not pruned:
            print(f"atlas speedrun: nothing to replay for {label}", flush=True)
            return
        if self.solver.max_actions_per_game is not None:
            remaining = self.solver.max_actions_per_game - self.action_count
            if remaining < len(pruned) + 1:
                print(
                    f"atlas speedrun: skipped for {label} -- {remaining} actions left, "
                    f"need {len(pruned) + 1} (reset + {len(pruned)} replayed)",
                    flush=True,
                )
                return
        try:
            reset_resp = self.game.env.step(arcengine.GameAction.RESET, data={})
            if reset_resp is None or not getattr(reset_resp, "full_reset", False):
                print(
                    f"atlas speedrun: RESET after WIN did not report full_reset for "
                    f"{label}; aborting (original win is unaffected)",
                    flush=True,
                )
                return
            for step_index, (engine_name, data) in enumerate(pruned, start=1):
                action_id = arcengine.GameAction.from_name(engine_name)
                resp = self.game.env.step(action_id, data=data)
                if resp is None or not resp.frame:
                    print(
                        f"atlas speedrun: replay step {step_index}/{len(pruned)} got no "
                        f"frame for {label}; aborting (original win is unaffected)",
                        flush=True,
                    )
                    return
                if resp.state == arcengine.GameState.WIN:
                    print(
                        f"atlas speedrun: WON in {step_index} actions (pruned from "
                        f"{len(self.history_entries) - 1}) for {label}",
                        flush=True,
                    )
                    return
                if resp.state == arcengine.GameState.GAME_OVER:
                    print(
                        f"atlas speedrun: replay hit GAME_OVER at step {step_index}/"
                        f"{len(pruned)} for {label}; aborting (original win is unaffected)",
                        flush=True,
                    )
                    return
            print(
                f"atlas speedrun: replayed all {len(pruned)} pruned actions without a WIN "
                f"for {label} -- pruning was too aggressive or the game is nondeterministic",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 -- must never break the original win
            print(
                f"atlas speedrun: replay raised {type(exc).__name__}: {exc} for {label}; "
                "aborting (original win is unaffected)",
                flush=True,
            )

    def _finish_if_needed(self) -> None:
        run = self.game.game_run
        if run is not None and run.final_score is None:
            if self.stop_event.is_set() and run.state == "playing":
                run.state = "cancelled"
            self.game.finish_game()

    def _write_analysis_html(self) -> None:
        if self.solver.job_dir is None:
            return
        _write_transcript_html(
            self.transcript_path,
            self.solver.job_dir / self.analysis_html_relpath,
            f"{self.game.game_run.game_id if self.game.game_run else self.game_index} analysis",
        )

    def _read_transcript_bytes(self) -> bytes:
        try:
            return self.transcript_path.read_bytes()
        except OSError:
            return b""

    def _transcript_delta_since(self, previous_transcript: bytes) -> str:
        try:
            current_size = self.transcript_path.stat().st_size
            previous_size = len(previous_transcript)
            with self.transcript_path.open("rb") as file:
                if current_size >= previous_size:
                    current_prefix = file.read(previous_size)
                    if current_prefix == previous_transcript:
                        return file.read().decode("utf-8", errors="replace").strip()
                    file.seek(0)
                return file.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _base_viewer_event(self, frame: Frame) -> dict[str, Any]:
        run = self.game.game_run
        raw_state = self.game.current_state.raw.state
        return {
            "board": [list(row) for row in frame.grid],
            "board_ascii": frame.ascii,
            "score": int(self.game.current_state.levels_completed),
            "state": raw_state.name,
            "level": frame.level,
            "run_status": run.state if run is not None else "playing",
        }

    def _append_initial_viewer_event(self) -> None:
        if self.viewer_events:
            return
        frame = self.current_frame()
        self.viewer_events.append(
            {
                **self._base_viewer_event(frame),
                "type": "initial",
                "title": "Initial State",
                "action_num": self.action_count,
                "analysis_step": None,
                "action_display": "RESET",
                "reward": 0.0,
            }
        )

    def _append_analysis_viewer_event(
        self, analysis_step: int, transcript: str
    ) -> None:
        frame = self.current_frame()
        self.viewer_events.append(
            {
                **self._base_viewer_event(frame),
                "type": "analysis",
                "title": f"Analysis Step {analysis_step}",
                "action_num": self.action_count,
                "analysis_step": analysis_step,
                "transcript": transcript,
            }
        )

    def bump_counter(self, key: str, amount: int = 1) -> None:
        self.experiment_counters[key] = self.experiment_counters.get(key, 0) + amount

    def _append_experiment_event(self) -> None:
        """One per-attempt roll-up of the Experiment 3 counters.

        Written into the existing ``*_events.jsonl`` stream rather than a new
        log format, so the usual artifact tooling picks it up unchanged.
        """
        counters = dict(self.experiment_counters)
        agent_counters = getattr(self.analyzer, "animation_counters", None)
        if isinstance(agent_counters, dict):
            for key, value in agent_counters.items():
                counters[key] = counters.get(key, 0) + int(value or 0)
        if not counters:
            return
        self.viewer_events.append(
            {
                "type": "experiment",
                "experiment": "animation_awareness",
                "title": "Animation awareness counters",
                "action_num": self.action_count,
                "analysis_step": self.analysis_step,
                "animation_awareness": bool(self.solver.animation_awareness),
                "counters": counters,
            }
        )

    def _append_action_viewer_event(
        self, payload: dict[str, Any], frame: Frame
    ) -> None:
        animation = payload.get("animation")
        if isinstance(animation, dict) and animation:
            self.bump_counter("stage1_animation_reported")
            if animation.get("board_unchanged"):
                self.bump_counter("stage1_animation_board_unchanged")
        self.viewer_events.append(
            {
                **self._base_viewer_event(frame),
                "type": "action",
                "title": f"Action {int(payload.get('action_num') or self.action_count)}",
                "action_num": int(payload.get("action_num") or self.action_count),
                "analysis_step": self.analysis_step,
                "action_name": payload.get("action_name"),
                "action_display": payload.get("action_display"),
                "reward": payload.get("reward"),
                "board_changed": payload.get("board_changed"),
                "done": payload.get("done"),
                "level_completed": payload.get("level_completed"),
                "game_over": payload.get("game_over"),
                "run_complete": payload.get("run_complete"),
                "batch_index": payload.get("batch_index"),
                "batch_size": payload.get("batch_size"),
                "animation": animation if isinstance(animation, dict) and animation else None,
            }
        )

    def write_viewer_payload(self) -> None:
        if self.solver.job_dir is None:
            return
        self.viewer_data_path.parent.mkdir(parents=True, exist_ok=True)
        run = self.game.game_run
        last_event_source = next(
            (
                event
                for event in reversed(self.viewer_events)
                if event.get("type") == "action"
            ),
            self.viewer_events[-1] if self.viewer_events else {},
        )
        last_event = dict(last_event_source)
        last_event.pop("board", None)
        last_event.pop("board_ascii", None)
        last_event.pop("transcript", None)
        payload = {
            "game_id": run.game_id if run is not None else str(self.game_index),
            "agent_name": self.solver.label,
            "status": run.state if run is not None else "playing",
            "pass_index": self.pass_index,
            "pass_label": str(self.pass_index),
            "eventCount": len(self.viewer_events),
            "lastEvent": last_event,
            "viewer_steps": [],
            "replay_url": self.analysis_html_relpath,
        }
        if run is not None:
            payload.update(
                {
                    "levels_completed": run.levels_completed,
                    "total_levels": run.number_of_levels,
                    "actions_per_level": list(run.actions_per_level),
                    "final_score": run.final_score,
                }
            )
        if self._viewer_events_flushed == 0:
            reset_raw_events_sidecar(self.viewer_data_path)
        append_raw_events_sidecar(
            self.viewer_data_path, self.viewer_events[self._viewer_events_flushed :]
        )
        self._viewer_events_flushed = len(self.viewer_events)
        tmp_path = self.viewer_data_path.with_suffix(
            f"{self.viewer_data_path.suffix}.tmp"
        )
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self.viewer_data_path)

    def _normalize_actions(
        self, arguments: dict[str, Any]
    ) -> tuple[list[arcengine.ActionInput] | None, str | None]:
        has_single = bool(str(arguments.get("action", "")).strip())
        has_batch = arguments.get("actions") is not None
        if has_single and has_batch:
            return None, "Use either `action` or `actions`, not both."

        if has_batch:
            raw_actions = arguments.get("actions")
            if not isinstance(raw_actions, list):
                return None, "`actions` must be a JSON array of action objects."
            if not raw_actions:
                return None, "`actions` must contain at least one action."
        else:
            if not has_single:
                return None, "step_env requires `action` or `actions`."
            raw_actions = [
                {
                    "action": arguments.get("action"),
                    "row": arguments.get("row"),
                    "col": arguments.get("col"),
                }
            ]

        actions: list[arcengine.ActionInput] = []
        for index, raw_action in enumerate(raw_actions, start=1):
            if not isinstance(raw_action, dict):
                return None, f"Action {index} must be a JSON object."
            action_name = to_engine_action(raw_action.get("action"))
            if not action_name:
                return (
                    None,
                    f"Unknown action at index {index}: {raw_action.get('action')!r}",
                )
            action_id = arcengine.GameAction.from_name(action_name)
            data: dict[str, Any] = {}
            if action_id == arcengine.GameAction.ACTION6:
                try:
                    row = max(0, min(63, int(raw_action["row"])))
                    column = max(0, min(63, int(raw_action["col"])))
                    data = {
                        "x": column,
                        "y": row,
                    }
                except (KeyError, TypeError, ValueError):
                    return (
                        None,
                        f"MOUSE action at index {index} requires integer row and col arguments.",
                    )
            actions.append(arcengine.ActionInput(id=action_id, data=data))
        return actions, None

    def _error_payload(self, message: str) -> dict[str, Any]:
        return {
            "executed": False,
            "error": message,
            "valid_actions": to_model_actions(_engine_action_names(self.game)),
            **self.timing_payload(),
        }

    def _terminal_payload(
        self, requested_actions: list[arcengine.ActionInput]
    ) -> dict[str, Any]:
        raw_state = self.game.current_state.raw.state
        is_game_over = raw_state == arcengine.GameState.GAME_OVER
        is_win = raw_state == arcengine.GameState.WIN
        requested = [
            _format_action_display(action.id.name, dict(action.data))
            for action in requested_actions
        ]
        stop_reason = (
            "run_complete" if is_win else "game_over" if is_game_over else "stopped"
        )
        return {
            "executed": False,
            "error": "No action was executed because the current game state is terminal or stopping.",
            "action_num": self.action_count,
            "level": _level_number(self.game),
            "score": int(self.game.current_state.levels_completed),
            "state": raw_state.name,
            "valid_actions": [],
            "board_changed": False,
            "done": is_win,
            "level_completed": False,
            "game_over": is_game_over,
            "run_complete": is_win,
            "batched": len(requested_actions) > 1,
            "requested_count": len(requested_actions),
            "executed_count": 0,
            "requested_actions": requested,
            "executed_actions": [],
            "stopped_early": True,
            "stop_reason": stop_reason,
            **self.timing_payload(),
        }

    def animation_record(self, action_num: Any = None) -> dict[str, Any] | None:
        """Stored frames for one animated action, newest by default.

        Routed through ``step_env`` rather than a new analyzer callback so no
        signature on the analyzer protocol has to change; this executes
        nothing and spends no action budget.
        """
        if not self.animation_history:
            return None
        if action_num is None:
            return self.animation_history[-1]
        try:
            wanted = int(action_num)
        except (TypeError, ValueError):
            return None
        for record in reversed(self.animation_history):
            if int(record.get("action_num") or 0) == wanted:
                return record
        return None

    def step_env(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if str(arguments.get("query") or "").strip() == "animation":
            if not self.solver.animation_awareness:
                return {"executed": False, "query": "animation", "record": None}
            return {
                "executed": False,
                "query": "animation",
                "record": self.animation_record(arguments.get("action_num")),
            }
        requested_actions, error = self._normalize_actions(arguments)
        if error is not None or requested_actions is None:
            return self._error_payload(error or "Could not parse action request.")
        if self.should_stop() or _is_engine_game_over(self.game):
            return self._terminal_payload(requested_actions)

        executed_payloads: list[dict[str, Any]] = []
        total_reward = 0.0
        stop_reason: str | None = None
        batch_size = len(requested_actions)
        requested_displays = [
            _format_action_display(action.id.name, dict(action.data))
            for action in requested_actions
        ]

        for batch_index, action in enumerate(requested_actions, start=1):
            if self.should_stop():
                stop_reason = "stopped"
                break
            if action.id.value not in self.game.current_state.available_actions:
                message = f"{_format_action_display(action.id.name, dict(action.data))} is not valid right now."
                if executed_payloads:
                    stop_reason = "invalid_action"
                    break
                return self._error_payload(message)

            try:
                payload = self._execute_action(
                    action,
                    batch_index=batch_index,
                    batch_size=batch_size,
                    flush_viewer_payload=False,
                )
            except Exception as exc:
                if executed_payloads:
                    stop_reason = "action_error"
                    break
                return self._error_payload(f"{type(exc).__name__}: {exc}")
            executed_payloads.append(payload)
            total_reward += float(payload.get("reward", 0.0) or 0.0)

            if payload.get("run_complete"):
                stop_reason = "run_complete"
                break
            if payload.get("game_over"):
                stop_reason = "game_over"
                break
            if payload.get("level_completed"):
                stop_reason = "level_completed"
                break

        if not executed_payloads:
            return self._error_payload("No action was executed.")

        final_payload = dict(executed_payloads[-1])
        final_payload["reward"] = total_reward
        final_payload["last_reward"] = executed_payloads[-1].get("reward", 0.0)
        final_payload["batched"] = batch_size > 1
        final_payload["requested_count"] = batch_size
        final_payload["executed_count"] = len(executed_payloads)
        final_payload["requested_actions"] = requested_displays
        final_payload["executed_actions"] = [
            str(item.get("action_display") or item.get("action_name") or "")
            for item in executed_payloads
        ]
        final_payload["board_changed"] = any(
            bool(item.get("board_changed")) for item in executed_payloads
        )
        # Max, not sum: the question this answers is "did any single action in
        # this batch animate", and summing per-action frames would make three
        # plain one-frame actions look like an animation.
        final_payload["frame_count"] = max(
            (int(item.get("frame_count") or 0) for item in executed_payloads),
            default=0,
        )
        batch_animation = pick_animation([item.get("animation") for item in executed_payloads])
        if batch_animation is not None:
            final_payload["animation"] = batch_animation
        else:
            final_payload.pop("animation", None)
        final_payload["stopped_early"] = len(executed_payloads) < batch_size
        if stop_reason is not None:
            final_payload["stop_reason"] = stop_reason
        self.write_viewer_payload()
        return final_payload

    def _execute_auto_reset(self) -> None:
        action = arcengine.ActionInput(id=arcengine.GameAction.RESET, data={})
        self._execute_action(action, batch_index=1, batch_size=1, generated_tokens=0)

    def _execute_action(
        self,
        action: arcengine.ActionInput,
        *,
        batch_index: int,
        batch_size: int,
        generated_tokens: int | None = None,
        flush_viewer_payload: bool = True,
    ) -> dict[str, Any]:
        previous_grid = _grid_from_state(self.game.current_state)
        previous_completed = int(self.game.current_state.levels_completed)
        if generated_tokens is None:
            current_tokens = _analyzer_reported_tokens(self.analyzer)
            generated_tokens = max(0, current_tokens - self.token_baseline)
            self.token_baseline = current_tokens

        new_state = self.game.execute_action(
            action, generated_tokens=generated_tokens, uncached_input_tokens=0
        )
        self.last_engine_action = action.id.name
        action_display = _format_action_display(action.id.name, dict(action.data))
        current_frame = Frame(
            grid=_grid_from_state(new_state),
            step=self.action_count,
            level=_level_number(self.game),
        )
        self.history_entries.append(
            HistoryEntry(action=action_display, frame=current_frame)
        )
        self.write_runtime_state()

        completed = int(new_state.levels_completed)
        reward = float(completed - previous_completed) / max(
            1.0, float(self.game.number_of_levels)
        )
        raw_state = new_state.raw.state
        board_changed = previous_grid != _grid_from_state(new_state)
        frames = _raw_frames(new_state)
        frame_count = len(frames)
        animation = (
            summarize_animation(frames, board_changed=board_changed)
            if self.solver.animation_awareness
            else None
        )
        level_completed = bool(
            new_state.just_won_level and raw_state != arcengine.GameState.WIN
        )
        payload = {
            "executed": True,
            "action_num": self.action_count,
            "level": _level_number(self.game),
            "score": completed,
            "reward": reward,
            "state": raw_state.name,
            "valid_actions": to_model_actions(_engine_action_names(self.game)),
            "board_changed": board_changed,
            # Kept separate from board_changed on purpose: board_changed keeps
            # meaning "the visible board differs", while frame_count carries
            # the independent "this action produced an animation" evidence the
            # no-op guard needs (see _frame_count).
            "frame_count": frame_count,
            "done": raw_state == arcengine.GameState.WIN,
            "level_completed": level_completed,
            "game_over": raw_state == arcengine.GameState.GAME_OVER,
            "run_complete": raw_state == arcengine.GameState.WIN,
            "action_name": action.id.name,
            "action_data": (
                _model_mouse_action_data(action.data)
                if action.id == arcengine.GameAction.ACTION6
                else dict(action.data)
            ),
            "action_display": action_display,
            "batch_index": batch_index,
            "batch_size": batch_size,
            **self.timing_payload(),
        }
        if animation is not None:
            payload["animation"] = animation
            self.animation_history.append(
                {
                    "action_num": self.action_count,
                    "action_display": action_display,
                    "before": previous_grid,
                    "frames": frames,
                    "summary": animation,
                }
            )
        self._append_action_viewer_event(payload, current_frame)
        if flush_viewer_payload:
            self.write_viewer_payload()
        return payload


@dataclass
class HarnessSolver(Solver):
    """Run the existing tool-using harness as a TAAF ``Solver``."""

    label: str = "HarnessSolver"
    model: str = ""
    analyzer_timeout: float | None = 120.0
    max_actions_per_game: int | None = None
    max_runtime_s_per_game: float | None = None
    concurrency: int = 16
    save_request_logs: bool = False
    # Experiment 2 (Known-Noop-Guard, 2026-07-18): captured from the *local*
    # environment when the solver is built, pickled with the benchmark, and
    # passed explicitly to every ToolAgent -- shell env vars do not reach the
    # Kaggle kernel, but this field does.
    hard_noop_guard: bool = field(
        default_factory=lambda: os.environ.get("ARC3_HARD_NOOP_GUARD", "").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    # Experiment 3 (Animation awareness, 2026-08-07): same pickle-the-local-env
    # trick as hard_noop_guard above -- without this field the flag never
    # reaches the Kaggle kernel.
    animation_awareness: bool = field(
        default_factory=lambda: os.environ.get("ARC3_ANIMATION_AWARENESS", "").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    start_local_server: bool = False
    local_server_config: str = ""
    local_server_api_key_file: str = ""
    local_server_repo_dir: str = ""
    local_server_port: int | None = None
    local_server_tensor_parallel_size: int | None = None
    local_server_count: int = 1
    kaggle_enable_vllm: bool = field(default=True, repr=False)
    kaggle_wheelhouse_dataset_source: str = field(
        default=DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE, repr=False
    )
    kaggle_model_dataset_source: str = field(
        default=DEFAULT_QWEN_MODEL_DATASET_SOURCE, repr=False
    )
    kaggle_served_model_name: str = field(default=DEFAULT_SERVED_MODEL_NAME, repr=False)
    kaggle_vllm_port: int = field(default=DEFAULT_VLLM_PORT, repr=False)
    kaggle_vllm_max_model_len: int = field(
        default=DEFAULT_VLLM_MAX_MODEL_LEN, repr=False
    )
    kaggle_vllm_tensor_parallel_size: int = field(
        default=DEFAULT_VLLM_TENSOR_PARALLEL_SIZE, repr=False
    )
    kaggle_wheelhouse_stamp_text: str = field(
        default=DEFAULT_WHEELHOUSE_STAMP_TEXT, repr=False
    )
    cancel_drain_timeout_s: float = DEFAULT_CANCEL_DRAIN_TIMEOUT_SECONDS
    analyzer_factory: AnalyzerFactory | None = field(
        default=None, repr=False, compare=False
    )
    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _local_server_started: bool = field(
        default=False, init=False, repr=False, compare=False
    )
    _local_server_env_overrides: dict[str, str] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _local_server_cwd: str = field(default="", init=False, repr=False, compare=False)
    _local_server_api_key: str = field(
        default="", init=False, repr=False, compare=False
    )
    _local_server_base_url: str = field(
        default="", init=False, repr=False, compare=False
    )
    _local_servers: list[_LocalServerRuntime] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    _local_server_original_env: dict[str, str | None] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    # Custom pool sized to self.concurrency: asyncio.to_thread routes onto
    # Python's default executor, capped at min(32, cpu+4) — which would
    # silently cap real concurrency below self.concurrency.
    _worker_pool: ThreadPoolExecutor | None = field(default=None, init=False, repr=False, compare=False)
    # atlas 25.08: shared time bank, see _ATLAS_TIME_BANK_ENABLED above.
    _atlas_time_bank_s: float = field(default=0.0, init=False, repr=False, compare=False)
    _atlas_time_bank_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def atlas_deposit_time(self, seconds: float) -> None:
        """Return unused per-game time to the shared bank (thread-safe --
        called concurrently from any of the running game sessions)."""
        if seconds <= 0:
            return
        with self._atlas_time_bank_lock:
            self._atlas_time_bank_s += seconds

    def atlas_draw_time(self, requested: float) -> float:
        """Draw up to `requested` seconds from the shared bank; returns the
        amount actually granted (0 if the bank is empty), never more than
        what's available. Thread-safe."""
        if requested <= 0:
            return 0.0
        with self._atlas_time_bank_lock:
            granted = min(requested, self._atlas_time_bank_s)
            self._atlas_time_bank_s -= granted
            return granted

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["analyzer_factory"] = None
        state.pop("_stop_event", None)
        state.pop("_local_server_started", None)
        state.pop("_local_server_env_overrides", None)
        state.pop("_local_server_cwd", None)
        state.pop("_local_server_api_key", None)
        state.pop("_local_server_base_url", None)
        state.pop("_local_servers", None)
        state.pop("_local_server_original_env", None)
        state.pop("_worker_pool", None)
        state.pop("_atlas_time_bank_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._stop_event = threading.Event()
        self._local_server_started = False
        self._local_server_env_overrides = {}
        self._local_server_cwd = ""
        self._local_server_api_key = ""
        self._local_server_base_url = ""
        self._local_servers = []
        self._local_server_original_env = {}
        self._worker_pool = None
        self._atlas_time_bank_lock = threading.Lock()

    def __deepcopy__(self, memo: dict[int, Any]) -> "HarnessSolver":
        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if key == "_stop_event":
                object.__setattr__(new, key, threading.Event())
            elif key == "analyzer_factory":
                object.__setattr__(new, key, value)
            elif key == "_local_servers":
                object.__setattr__(new, key, [])
            elif key == "_local_server_original_env":
                object.__setattr__(new, key, {})
            elif key == "_worker_pool":
                object.__setattr__(new, key, None)
            elif key == "_atlas_time_bank_lock":
                object.__setattr__(new, key, threading.Lock())
            else:
                object.__setattr__(new, key, copy.deepcopy(value, memo))
        return new

    @property
    def kaggle_dataset_sources(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return duck_kaggle_dataset_sources(self._kaggle_vllm_config())

    @property
    def kaggle_setup_commands(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return [duck_kaggle_setup_command(self._kaggle_vllm_config())]

    @property
    def kaggle_teardown_commands(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return [duck_kaggle_teardown_command()]

    def _kaggle_vllm_config(self) -> DuckKaggleVllmConfig:
        return DuckKaggleVllmConfig(
            wheelhouse_dataset_source=self.kaggle_wheelhouse_dataset_source,
            model_dataset_source=self.kaggle_model_dataset_source,
            served_model_name=self.kaggle_served_model_name,
            vllm_port=self.kaggle_vllm_port,
            max_model_len=self.kaggle_vllm_max_model_len,
            tensor_parallel_size=self.kaggle_vllm_tensor_parallel_size,
            wheelhouse_stamp_text=self.kaggle_wheelhouse_stamp_text,
        )

    def _setup(self) -> None:
        print(
            f"HarnessSolver hard_noop_guard={self.hard_noop_guard} "
            f"animation_awareness={self.animation_awareness}",
            flush=True,
        )
        if self.start_local_server:
            self._start_local_servers()
        self._worker_pool = ThreadPoolExecutor(
            max_workers=max(1, int(self.concurrency)),
            thread_name_prefix="harness-game",
        )

    def _teardown(self) -> None:
        if self._local_server_started:
            self._stop_local_servers()
        if self._worker_pool is not None:
            self._worker_pool.shutdown(wait=False)
            self._worker_pool = None

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        self._stop_event.clear()
        semaphore = asyncio.Semaphore(max(1, int(self.concurrency)))
        pass_indices_by_game_id: dict[str, int] = {}
        loop = asyncio.get_running_loop()
        pool = self._worker_pool

        async def run_one(index: int, pass_index: int, game: taaf.game.Game) -> None:
            async with semaphore:
                args = (game, index, pass_index, self._local_server_for_game_index(index))
                if pool is not None:
                    await loop.run_in_executor(pool, functools.partial(self._play_one, *args))
                else:
                    # _setup wasn't called (direct test invocation).
                    await asyncio.to_thread(self._play_one, *args)

        tasks: list[asyncio.Task[None]] = []
        for index, game in enumerate(games):
            game_id = game.game_run.game_id if game.game_run is not None else str(index)
            pass_index = pass_indices_by_game_id.get(game_id, 0)
            pass_indices_by_game_id[game_id] = pass_index + 1
            tasks.append(asyncio.create_task(run_one(index, pass_index, game)))
        try:
            await asyncio.gather(
                *(asyncio.shield(task) for task in tasks), return_exceptions=True
            )
        except asyncio.CancelledError:
            self._stop_event.set()
            await self._drain_game_tasks(tasks)
            self._finish_remaining(games)
            raise

    async def _drain_game_tasks(self, tasks: list[asyncio.Task[None]]) -> None:
        if not tasks:
            return
        timeout = max(0.0, float(self.cancel_drain_timeout_s))
        if timeout == 0.0:
            return
        done, _pending = await asyncio.wait(tasks, timeout=timeout)
        if done:
            await asyncio.gather(*done, return_exceptions=True)

    def _start_local_servers(self) -> None:
        server_count = self._resolved_local_server_count()
        started: list[_LocalServerRuntime] = []
        self._capture_local_server_process_env()
        try:
            for server_index in range(server_count):
                runtime = self._local_server_settings(
                    server_index=server_index, server_count=server_count
                )
                print(
                    "Starting local inference server inside solver setup "
                    f"(server {server_index + 1}/{server_count})"
                )
                subprocess.run(
                    ["make", "server"],
                    cwd=runtime.repo_dir,
                    env=self._local_server_env(runtime.env_overrides),
                    check=True,
                )
                if runtime.api_key_file.is_file():
                    runtime.api_key = runtime.api_key_file.read_text(
                        encoding="utf-8"
                    ).strip()
                started.append(runtime)
        except Exception:
            self._local_servers = started
            self._local_server_started = bool(started)
            with contextlib.suppress(Exception):
                self._stop_local_servers()
            raise

        self._local_servers = started
        self._local_server_started = bool(started)
        if started:
            first = started[0]
            self._local_server_cwd = str(first.repo_dir)
            self._local_server_env_overrides = first.env_overrides
            self._local_server_api_key = first.api_key
            self._local_server_base_url = first.base_url
            if first.api_key:
                os.environ["LOCAL_ANALYZER_API_KEY"] = first.api_key
                os.environ["OPENAI_API_KEY"] = first.api_key
            if first.base_url:
                os.environ["LOCAL_ANALYZER_BASE_URL"] = first.base_url
                os.environ["OPENAI_BASE_URL"] = first.base_url
                os.environ["LOCAL_ANALYZER_PROVIDER"] = "vllm"
                os.environ["OPENAI_PROVIDER"] = "vllm"

    def _stop_local_servers(self) -> None:
        runtimes = list(reversed(self._local_servers))
        if not runtimes and self._local_server_env_overrides:
            repo_dir = (
                Path(self._local_server_cwd)
                if self._local_server_cwd
                else self._local_server_repo_dir()
            )
            runtimes = [
                _LocalServerRuntime(
                    index=0,
                    repo_dir=repo_dir,
                    api_key_file=Path(
                        self._local_server_env_overrides.get("SERVER_API_KEY_FILE", "")
                    ),
                    env_overrides=self._local_server_env_overrides,
                    base_url=self._local_server_base_url,
                    api_key=self._local_server_api_key,
                )
            ]
        try:
            for runtime in runtimes:
                subprocess.run(
                    ["make", "stop-server"],
                    cwd=runtime.repo_dir,
                    env=self._local_server_env(runtime.env_overrides),
                    check=False,
                )
        finally:
            self._local_servers = []
            self._local_server_started = False
            self._restore_local_server_process_env()

    def _capture_local_server_process_env(self) -> None:
        self._local_server_original_env = {
            key: os.environ.get(key) for key in _LOCAL_SERVER_PROCESS_ENV_KEYS
        }

    def _restore_local_server_process_env(self) -> None:
        if not self._local_server_original_env:
            return
        for key, value in self._local_server_original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._local_server_original_env = {}

    def _local_server_settings(
        self, *, server_index: int, server_count: int
    ) -> _LocalServerRuntime:
        config_path = self.local_server_config.strip()
        if not config_path:
            raise ValueError(
                "local_server_config is required when start_local_server is enabled."
            )

        repo_dir = self._local_server_repo_dir()
        run_dir = (self.job_dir or Path.cwd()).resolve()
        api_key_file = self._local_server_api_key_path(
            server_index=server_index,
            server_count=server_count,
            run_dir=run_dir,
        )
        pid_path = run_dir / (
            "server.pid" if server_count <= 1 else f"server-{server_index}.pid"
        )
        log_path = run_dir / (
            "server.log" if server_count <= 1 else f"server-{server_index}.log"
        )
        port = self._local_server_port(server_index=server_index)
        base_url = f"http://127.0.0.1:{port}/v1" if port is not None else ""
        env_overrides = {
            "CONFIG_PATH": config_path,
            "SERVER_API_KEY_FILE": str(api_key_file),
            "SERVER_PID": str(pid_path),
            "SERVER_LOG": str(log_path),
            "SERVER_TAIL_ON_WAIT": "true",
            "UV_PROJECT_ENVIRONMENT": str(repo_dir / ".venv"),
        }
        venv_python = self._local_server_venv_python(repo_dir)
        if venv_python is not None:
            env_overrides["SERVER_VENV_PYTHON"] = str(venv_python)
            env_overrides["PYTHON"] = str(venv_python)
        if port is not None:
            env_overrides.update(
                {
                    "SERVER_PORT": str(port),
                    "LOCAL_ANALYZER_BASE_URL": base_url,
                    "OPENAI_BASE_URL": base_url,
                    "LOCAL_ANALYZER_PROVIDER": "vllm",
                    "OPENAI_PROVIDER": "vllm",
                }
            )
        if self.local_server_tensor_parallel_size is not None:
            env_overrides["SERVER_TENSOR_PARALLEL_SIZE"] = str(
                int(self.local_server_tensor_parallel_size)
            )
        if server_count > 1:
            env_overrides["CUDA_VISIBLE_DEVICES"] = (
                self._cuda_visible_device_for_server(server_index)
            )
        return _LocalServerRuntime(
            index=server_index,
            repo_dir=repo_dir,
            api_key_file=api_key_file,
            env_overrides=env_overrides,
            base_url=base_url,
        )

    def _resolved_local_server_count(self) -> int:
        if not self.start_local_server:
            return 0
        return max(1, int(self.local_server_count or 1))

    def _local_server_port(self, *, server_index: int) -> int | None:
        if self.local_server_port is None:
            return None
        return int(self.local_server_port) + int(server_index)

    def _local_server_api_key_path(
        self, *, server_index: int, server_count: int, run_dir: Path
    ) -> Path:
        default_name = (
            "server-api-key" if server_count <= 1 else f"server-{server_index}-api-key"
        )
        base_path = self._resolve_local_server_path(
            self.local_server_api_key_file, default=run_dir / default_name
        )
        if server_count <= 1 or not str(self.local_server_api_key_file or "").strip():
            return base_path
        suffix = base_path.suffix
        stem = base_path.name[: -len(suffix)] if suffix else base_path.name
        return base_path.with_name(f"{stem}-{server_index}{suffix}")

    def _cuda_visible_device_for_server(self, server_index: int) -> str:
        visible_devices = [
            device.strip()
            for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if device.strip()
        ]
        if server_index < len(visible_devices):
            return visible_devices[server_index]
        return str(server_index)

    def _local_server_repo_dir(self) -> Path:
        repo_dir = (
            Path(self.local_server_repo_dir).expanduser()
            if self.local_server_repo_dir
            else Path(__file__).parents[2]
        )
        repo_dir = repo_dir.resolve()
        if not repo_dir.is_dir():
            raise ValueError(f"local_server_repo_dir does not exist: {repo_dir}")
        return repo_dir

    def _local_server_venv_python(self, repo_dir: Path) -> Path | None:
        repo_venv_python = repo_dir / ".venv" / "bin" / "python"
        if repo_venv_python.is_file():
            return repo_venv_python
        return None

    def _resolve_local_server_path(self, raw_value: str, *, default: Path) -> Path:
        raw = str(raw_value or "").strip()
        if not raw:
            return default.resolve()
        path = Path(raw).expanduser()
        if path.is_absolute():
            return path
        return (self._local_server_repo_dir() / path).resolve()

    def _local_server_env(
        self, overrides: dict[str, str] | None = None
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update(overrides or self._local_server_env_overrides)
        return env

    def soft_time_remaining_seconds(self) -> float | None:
        if self.soft_end_time is None:
            return None
        now = (
            datetime.now(self.soft_end_time.tzinfo)
            if self.soft_end_time.tzinfo
            else datetime.now()
        )
        return max(0.0, (self.soft_end_time - now).total_seconds())

    def _local_server_for_game_index(
        self, game_index: int
    ) -> _LocalServerRuntime | None:
        if not self._local_servers:
            return None
        return self._local_servers[int(game_index) % len(self._local_servers)]

    def _make_analyzer(
        self,
        game: taaf.game.Game,
        index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> Any:
        if self.analyzer_factory is not None:
            return self.analyzer_factory(game, index)
        return ToolAgent(
            model=self.model,
            timeout=self.analyzer_timeout,
            save_request_logs=self.save_request_logs,
            api_key=(
                local_server.api_key
                if local_server is not None
                else self._local_server_api_key
            )
            or None,
            base_url=(
                local_server.base_url
                if local_server is not None
                else self._local_server_base_url
            )
            or None,
            provider="vllm" if local_server is not None else None,
            hard_noop_guard=self.hard_noop_guard,
            animation_awareness=self.animation_awareness,
        )

    def _play_one(
        self,
        game: taaf.game.Game,
        index: int,
        pass_index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> None:
        try:
            assert game.game_run is not None
            run = game.game_run
            run_stem = self._run_stem(run.game_id, pass_index)
            state_path = self._artifacts_dir() / f"{run_stem}_{RUNTIME_STATE_FILENAME}"
            viewer_data_path = self._artifacts_dir() / f"{run_stem}_viewer_data.json"
            transcript_path = self._transcripts_dir() / f"{run_stem}.txt"
            analysis_relpath = f"solver_analysis/{run_stem}.html"
            analyzer = self._make_analyzer(game, index, local_server)
            session = _HarnessGameSession(
                solver=self,
                game=game,
                analyzer=analyzer,
                game_index=index,
                pass_index=pass_index,
                state_path=state_path,
                transcript_path=transcript_path,
                analysis_html_relpath=analysis_relpath,
                stop_event=self._stop_event,
                viewer_data_path=viewer_data_path,
            )
            session.play()
        except Exception as exc:
            self._finish_after_error(game, exc)

    def _artifacts_dir(self) -> Path:
        root = self.job_dir or Path.cwd() / "taaf_harness_artifacts"
        path = root / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _transcripts_dir(self) -> Path:
        root = self.job_dir or Path.cwd() / "taaf_harness_artifacts"
        path = root / "transcripts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _run_stem(self, game_id: str, index: int) -> str:
        return f"{artifact_stem(game_id)}_p{index}"

    def _finish_remaining(self, games: list[taaf.game.Game]) -> None:
        for game in games:
            run = game.game_run
            if run is not None and run.final_score is None:
                try:
                    if self._stop_event.is_set() and run.state == "playing":
                        run.state = "cancelled"
                    game.finish_game()
                except Exception:
                    pass

    def _finish_after_error(self, game: taaf.game.Game, exc: Exception) -> None:
        run = game.game_run
        if run is None or run.final_score is not None:
            return
        run.solver_note = f"error: {type(exc).__name__}: {exc}"
        if run.state == "playing":
            run.state = "crashed"
        with contextlib.suppress(Exception):
            game.finish_game()
        if run.final_score is None:
            with contextlib.suppress(Exception):
                run.final_score = run._compute_final_score()

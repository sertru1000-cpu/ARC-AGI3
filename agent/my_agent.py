"""Your ARC-AGI-3 agent. This is the *only* file you should normally edit.

Two interchangeable brains, selected by env var AGENT_BRAIN:
  - "explorer" (default): HeuristicExplorer, no LLM — fast floor baseline.
  - "llm": Duck-style loop — the model writes python in a sandbox, inspects
    segmentation/diffs, and executes batched actions via action([...]).
    Backend comes from LLM_BASE_URL (OpenAI-compatible) or falls back to a
    MockLLM for GPU-less smoke tests.

Contract (enforced by the ARC-AGI-3-Agents framework):
  - Subclass `agents.agent.Agent`, class named `MyAgent`,
    implement `is_done` and `choose_action`.
The LLM brain additionally overrides `main()` because its sandbox steps the
environment directly (multiple actions per turn), which the one-action-per-
call `choose_action` loop cannot express.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

try:
    from agent.harness.explorer import HeuristicExplorer
    from agent.harness.llm import default_backend
    from agent.harness.llm_policy import LLMPolicy
    from agent.harness.sandbox import Sandbox
    from agent.harness.journal import REFLECTION_PROMPT, CrossGameJournal
    from agent.harness.probe import probe_actions
    from agent.harness.prompts import DEATH_REPLAY_NOTE, LEVEL_UP_NOTE, memo_digest
    from agent.harness.speedrun import ReplayStep, build_replay_plan, execute_replay
    from agent.harness.vision import VisionLLM
except ImportError:  # Kaggle notebook: harness modules are inlined beside us
    from harness.explorer import HeuristicExplorer  # type: ignore
    from harness.journal import REFLECTION_PROMPT, CrossGameJournal  # type: ignore
    from harness.llm import default_backend  # type: ignore
    from harness.llm_policy import LLMPolicy  # type: ignore
    from harness.probe import probe_actions  # type: ignore
    from harness.prompts import DEATH_REPLAY_NOTE, LEVEL_UP_NOTE, memo_digest  # type: ignore
    from harness.sandbox import Sandbox  # type: ignore
    from harness.speedrun import ReplayStep, build_replay_plan, execute_replay  # type: ignore
    from harness.vision import VisionLLM  # type: ignore

import numpy as np

logger = logging.getLogger()

BRAIN = os.getenv("AGENT_BRAIN", "explorer").strip().lower()
MAX_LLM_TURNS = int(os.getenv("MY_AGENT_MAX_TURNS", "60"))
GAME_SECONDS = float(os.getenv("MY_AGENT_GAME_SECONDS", "480"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
PROBE_ENABLED = os.getenv("MY_AGENT_PROBE", "1") != "0"
MEMO_NOTES_ENABLED = os.getenv("MY_AGENT_MEMO_NOTES", "1") != "0"
DEATH_REPLAY_ENABLED = os.getenv("MY_AGENT_DEATH_REPLAY", "1") != "0"
CROSS_MEMORY_ENABLED = os.getenv("MY_AGENT_CROSS_MEMORY", "1") != "0"
VISION_ENABLED = os.getenv("MY_AGENT_VISION", "0") == "1"
# Episode-kill thresholds. Defaults = competition behaviour (don't burn a
# hidden game's wall time on a dead model). Teacher collection raises both
# to "effectively never" (collect_teacher.py): an expert trace that recovers
# from its own mistakes is exactly the data we want, not something to cut.
MAX_TURN_FAILURES = int(os.getenv("MY_AGENT_MAX_TURN_FAILURES", "3"))
MAX_NO_CODE_STRIKES = int(os.getenv("MY_AGENT_MAX_NO_CODE_STRIKES", "5"))
VISION_SCALE = int(os.getenv("MY_AGENT_VISION_SCALE", "8"))
# Hybrid floor (backlog item 2, 20.08): spend this many real actions on the
# heuristic explorer before handing control to the LLM, so a stalled/looping
# model can never do worse than the explorer-only baseline for that budget.
FLOOR_ACTIONS = int(os.getenv("MY_AGENT_FLOOR_ACTIONS", "20"))

# Global wall-clock budget shared by every game in this process. Keeps the
# whole run inside the notebook limit even if the hidden set is large.
import time as _time_mod

_GLOBAL_START = _time_mod.time()
TOTAL_SECONDS = float(os.getenv("MY_AGENT_TOTAL_SECONDS", "0") or 0)  # 0 = off
EXPECTED_GAMES = int(os.getenv("MY_AGENT_EXPECTED_GAMES", "30"))


PARALLEL_GAMES = os.getenv("MY_AGENT_PARALLEL", "0") != "0"


def _game_time_allowance(games_started: int) -> float:
    """Per-game seconds: the fixed cap, shrunk if the global budget runs hot.

    Sequential mode divides the remaining window across games still queued.
    Parallel mode (Swarm runs every game in its own thread CONCURRENTLY) must
    NOT divide: all games share the same wall-clock window, so each one may
    use the whole remainder. Dividing here was the bug that stopped all 30
    hidden games after 720s while 7h of budget sat unused.
    """
    if TOTAL_SECONDS <= 0:
        return GAME_SECONDS
    remaining = TOTAL_SECONDS - (_time_mod.time() - _GLOBAL_START)
    if PARALLEL_GAMES:
        return max(60.0, min(GAME_SECONDS, remaining))
    games_left = max(1, EXPECTED_GAMES - games_started)
    return max(60.0, min(GAME_SECONDS, remaining / games_left))


class MyAgent(Agent):
    """ARC-AGI-3 agent with switchable explorer/LLM brains."""

    MAX_ACTIONS = int(os.getenv("MY_AGENT_MAX_ACTIONS", "400"))
    _games_started: int = 0  # class-level: shared across games in this process

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.explorer = HeuristicExplorer(seed=hash(self.game_id) & 0xFFFF)
        self.policy: Optional[LLMPolicy] = None
        # Every executed action lands here for the post-win speedrun phase.
        self.replay_log: list[dict] = []

    def take_action(self, action: GameAction):  # type: ignore[override]
        frame = super().take_action(action)
        if frame is not None and frame.frame:
            try:
                data = action.action_data.model_dump()
            except Exception:
                data = {}
            self.replay_log.append({
                "id": int(action.value),
                "data": {k: v for k, v in data.items() if k in ("x", "y")},
                "grid": np.asarray(frame.frame[-1], dtype=np.int8),
                "level": int(frame.levels_completed or 0),
                "state": str(frame.state),
            })
        return frame

    # ── shared plumbing ───────────────────────────────────────────────────
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    @staticmethod
    def _normalize_available(latest_frame: FrameData) -> list[GameAction]:
        raw = latest_frame.available_actions or list(GameAction)
        return [a if isinstance(a, GameAction) else GameAction.from_id(int(a)) for a in raw]

    # ── explorer brain: standard one-action-per-call loop ─────────────────
    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.explorer.state.prev_grid = None
            self.explorer.state.last_action_key = None
            return GameAction.RESET

        self.explorer.observe(latest_frame)
        available = self._normalize_available(latest_frame)
        simple = [a.name for a in available if a not in (GameAction.RESET, GameAction.ACTION6)]
        has_click = GameAction.ACTION6 in available

        name, payload = self.explorer.decide(latest_frame, simple, has_click)
        action = GameAction[name]
        if payload is not None:
            action.set_data(payload)
        action.reasoning = f"explorer:{name}"
        return action

    # ── main loop: explorer or LLM brain, then the speedrun phase ─────────
    def main(self) -> None:
        if BRAIN != "llm":
            super().main()
            try:
                self._speedrun_if_won()
            except Exception as exc:
                logger.warning(f"{self.game_id}: speedrun phase failed harmlessly: {exc!r}")
            return

        import time as _time

        self.timer = _time.time()
        games_before_me = MyAgent._games_started
        MyAgent._games_started += 1
        my_seconds = _game_time_allowance(games_before_me)
        backend = default_backend()
        logger.info(
            f"{self.game_id}: LLM brain online, backend={backend.name}, "
            f"time allowance {my_seconds:.0f}s"
        )

        sandbox = Sandbox(env_step=self._env_step, budget_left=self._budget_left)

        # Vision (teacher collection): attach the current board as a PNG to
        # every LLM request. Call-time only -- message history and traces stay
        # text-only, so SFT data for the (text-only) student is unaffected.
        if VISION_ENABLED:
            backend = VisionLLM(
                backend, lambda: sandbox.current.grid if sandbox.current is not None else None,
                scale=VISION_SCALE)
            logger.info(f"{self.game_id}: vision on, backend={backend.name}")

        # Seed: RESET to obtain the first real frame.
        first = self._env_step("RESET", None)
        sandbox.update_frame(first)  # no action_name: seed frame, not history
        win_levels = int(first.win_levels or 0)

        # Deterministic opening probe: a control legend for the model's first turn.
        probe_summary = ""
        if PROBE_ENABLED:
            raw = first.available_actions or []
            avail_ids = {int(getattr(a, "value", a)) for a in raw}

            def _probe_step(aid: int, data: dict | None):
                frame = self._env_step_by_id(aid, data)
                sandbox.update_frame(frame, f"ACTION{aid}" if aid else "RESET")
                return frame

            probe_summary, _ = probe_actions(_probe_step, first, avail_ids)

        # Hybrid floor: burn a small opening budget on the heuristic explorer
        # (same brain as AGENT_BRAIN=explorer) before handing control to the
        # LLM. Real actions, real budget spend — not free — but it means a
        # model that stalls out of the gate still leaves the episode no worse
        # than the explorer-only baseline for those N actions. Stops early on
        # a level-up so the LLM inherits genuine progress instead of noise.
        floor_spent = 0
        if FLOOR_ACTIONS > 0:
            floor_level_start = sandbox.current.level if sandbox.current else 0
            latest = first
            while floor_spent < FLOOR_ACTIONS and self._budget_left() > 0:
                state = str(getattr(latest, "state", "")).split(".")[-1]
                if state == "WIN":
                    break
                if state in ("NOT_PLAYED", "GAME_OVER"):
                    latest = self._env_step("RESET", None)
                    sandbox.update_frame(latest)
                    self.explorer.state.prev_grid = None
                    self.explorer.state.last_action_key = None
                    continue
                self.explorer.observe(latest)
                available = self._normalize_available(latest)
                simple = [a.name for a in available
                          if a not in (GameAction.RESET, GameAction.ACTION6)]
                has_click = GameAction.ACTION6 in available
                name, payload = self.explorer.decide(latest, simple, has_click)
                aid = GameAction[name].value
                latest = self._env_step_by_id(aid, payload)
                sandbox.update_frame(latest, f"ACTION{aid}", payload)
                floor_spent += 1
                if sandbox.current and sandbox.current.level > floor_level_start:
                    break  # real progress -- let the LLM build on it, not repeat it
            if floor_spent:
                logger.info(f"{self.game_id}: hybrid floor spent {floor_spent} "
                            f"actions, level={sandbox.current.level if sandbox.current else 0}")

        journal = CrossGameJournal() if CROSS_MEMORY_ENABLED else None
        cross_note = journal.summary_text() if journal else ""
        if floor_spent:
            cross_note = (cross_note + "\n\n" if cross_note else "") + (
                f"[harness note] A heuristic opening burst already spent "
                f"{floor_spent} actions before you took over (see the probe/"
                "history above for what was learned). Build on it -- don't "
                "repeat blind exploration from scratch.")
        # Optional human hint for this game (teacher data collection only):
        # MY_AGENT_HINT_<gameid>=text. Injected into the initial user message
        # so the reasoning in the trace stays honest — the model still has to
        # discover controls and verify the mechanics itself.
        hint = os.getenv(f"MY_AGENT_HINT_{self.game_id.split('-')[0].upper()}", "")
        if hint:
            cross_note = (cross_note + "\n\n" if cross_note else "") + \
                f"[human hint] {hint}"

        self.policy = LLMPolicy(
            backend=backend, sandbox=sandbox, game_id=self.game_id,
            win_levels=win_levels, max_tokens=LLM_MAX_TOKENS,
        )
        self.policy.start(probe_summary, cross_note)

        turns = 0
        stop_reason: str | None = None  # set by break paths; else classified after the loop
        last_level = sandbox.current.level if sandbox.current else 0
        while (
            self.frames[-1].state is not GameState.WIN
            and self._budget_left() > 0
            and turns < MAX_LLM_TURNS
            and (_time.time() - self.timer) < my_seconds
        ):
            # Auto-recover if the model left the game dead between turns.
            if self.frames[-1].state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                was_dead = self.frames[-1].state is GameState.GAME_OVER
                if was_dead:
                    # Pain memory: the fatal move goes into persistent memo,
                    # not just a context note (notes get trimmed away; memo
                    # survives the whole episode and reaches the SFT data).
                    fatal = sandbox.history[-1][0] if sandbox.history else "?"
                    lvl = sandbox.current.level if sandbox.current else 0
                    dangers = sandbox.memo.setdefault("dangers", [])
                    dangers.append(f"L{lvl}: died after {fatal} "
                                   f"(step {sandbox.step_counter})")
                    del dangers[:-8]  # keep the most recent lessons
                sandbox.update_frame(self._env_step("RESET", None))
                if was_dead and DEATH_REPLAY_ENABLED:
                    replayed = self._replay_last_attempt(sandbox)
                    if replayed:
                        self.policy.note(DEATH_REPLAY_NOTE)
            try:
                info = self.policy.play_turn()
                self._turn_failures = 0
            except Exception as exc:
                # One bad turn (LLM outage, unexpected sandbox blow-up) must
                # not kill the whole game — or, worse, the whole notebook.
                turn_failures = getattr(self, "_turn_failures", 0) + 1
                self._turn_failures = turn_failures
                logger.warning(f"{self.game_id}: turn failed ({exc!r}), "
                               f"strike {turn_failures}/{MAX_TURN_FAILURES}")
                # Log it (the trace is the debugging record) and tell the
                # model its turn was lost -- soft feedback, not a penalty.
                if self.policy.trace:
                    self.policy.trace.write({"turn": self.policy.turns, "turn_failed": repr(exc)[:500],
                                             "strike": turn_failures})
                try:
                    self.policy.note(
                        f"[harness error] Your previous turn was lost to an infrastructure "
                        f"error ({type(exc).__name__}); no code ran and the board is "
                        "unchanged. Simply continue from your last plan.")
                except Exception:
                    pass
                if turn_failures >= MAX_TURN_FAILURES:
                    logger.warning(f"{self.game_id}: giving up after repeated turn failures")
                    break
                continue
            turns += 1
            cur_level = sandbox.current.level if sandbox.current else 0
            if cur_level != last_level:
                last_level = cur_level
                note = LEVEL_UP_NOTE
                if MEMO_NOTES_ENABLED:
                    note += memo_digest(sandbox.memo)
                self.policy.note(note)
            logger.info(
                f"{self.game_id}: turn {turns} actions={info['actions']} "
                f"level={sandbox.current.level if sandbox.current else 0}/{win_levels} "
                f"budget={self._budget_left()} err={bool(info['error'])}"
            )
            if info["win"]:
                break
            if self.policy.no_code_strikes >= MAX_NO_CODE_STRIKES:
                logger.warning(f"{self.game_id}: model stopped emitting code, ending game")
                stop_reason = "no_code"
                break

        # WHY the game ended. Without this the only way to tell "ran out of
        # wall-clock" from "ran out of turns" is to diff log timestamps by
        # hand, and on a Phase B rerun the logs are barely reachable at all.
        # If time_cap shows up across many games, the shared window is the
        # binding budget and PHASE_B_TOTAL_SECONDS is set too low (or the
        # margin under Kaggle's 9 h cap is too thin).
        elapsed = _time.time() - self.timer
        if stop_reason is None:
            if self.frames[-1].state is GameState.WIN:
                stop_reason = "win"
            elif self._budget_left() <= 0:
                stop_reason = "action_cap"
            elif turns >= MAX_LLM_TURNS:
                stop_reason = "turn_cap"
            elif elapsed >= my_seconds:
                stop_reason = "time_cap"
            else:
                stop_reason = "other"
        logger.info(
            f"{self.game_id}: END reason={stop_reason} turns={turns}/{MAX_LLM_TURNS} "
            f"actions_left={self._budget_left()} "
            f"elapsed={elapsed:.0f}s/{my_seconds:.0f}s "
            f"level={sandbox.current.level if sandbox.current else 0}/{win_levels} "
            f"no_code_retries={getattr(self.policy, 'no_code_retries', 0)} "
            f"repeats_blocked={getattr(self.policy, 'repeats_blocked', 0)}"
        )
        if self.policy is not None and self.policy.trace:
            self.policy.trace.write({"event": "game_end", "reason": stop_reason,
                                     "turns": turns, "elapsed_s": round(elapsed),
                                     "allowance_s": round(my_seconds),
                                     "actions_left": self._budget_left(),
                                     "no_code_retries": getattr(self.policy, "no_code_retries", 0),
                                     "repeats_blocked": getattr(self.policy, "repeats_blocked", 0),
                                     "level": sandbox.current.level if sandbox.current else 0})

        try:
            self._speedrun_if_won()
        except Exception as exc:
            logger.warning(f"{self.game_id}: speedrun phase failed harmlessly: {exc!r}")

        # Cross-game journal: stats always; an LLM reflection only on success.
        if journal is not None:
            import json as _json

            lessons: list[str] = []
            lvls = int(self.frames[-1].levels_completed or 0)
            if lvls > 0:
                try:
                    wm = _json.dumps(sandbox.memo.get("world_model", {}), ensure_ascii=False)
                    txt = backend.chat(
                        [{"role": "user", "content": REFLECTION_PROMPT.format(
                            levels=lvls, world_model=wm)}],
                        max_tokens=400, temperature=0.4,
                    )
                    lessons = [l.strip() for l in txt.splitlines() if l.strip()][:2]
                except Exception as exc:
                    logger.warning(f"{self.game_id}: reflection failed harmlessly: {exc!r}")
            try:
                journal.record_game(self.replay_log, lvls, lessons)
                logger.info(f"{self.game_id}: journal updated ({journal.data['games']} games)")
            except Exception as exc:
                logger.warning(f"{self.game_id}: journal update failed: {exc!r}")

        self.cleanup()

    def _replay_last_attempt(self, sandbox: Sandbox) -> int:
        """After a death+reset, fast-forward through the just-lost attempt.

        Replays the non-noop actions of the current level's last attempt,
        minus the fatal final action, verifying frames as we go. Saves
        wall-clock time (no LLM turns), costs the same actions the model
        would spend walking back anyway. Returns steps replayed.
        """
        # Slice replay_log: entries after the last RESET (the one we just did
        # is not logged via take_action wrapper order — find prior attempt).
        # Walk back to the RESET that *started* the lost attempt.
        log = self.replay_log
        end = len(log)
        # Drop trailing reset entries (the recovery reset we just made).
        while end > 0 and log[end - 1]["id"] == 0:
            end -= 1
        if end == 0:
            return 0
        attempt_level = log[end - 1]["level"]
        start = end
        while start > 0 and log[start - 1]["id"] != 0 and log[start - 1]["level"] == attempt_level:
            start -= 1
        # If we stopped on a lower-level entry, log[start] is the action that
        # completed the previous level — it belongs to that level, skip it.
        if 0 < start < end and log[start - 1]["id"] != 0 and log[start - 1]["level"] < attempt_level:
            start += 1
        attempt = log[start:end]
        if len(attempt) < 3:
            return 0  # nothing worth replaying
        attempt = attempt[:-1]  # drop the fatal action

        # Distill no-ops out, keep expected grids for verification.
        plan: list[ReplayStep] = []
        prev_grid = None
        for e in attempt:
            grid = e["grid"]
            if prev_grid is not None and grid is not None and prev_grid.shape == grid.shape \
                    and bool((prev_grid == grid).all()):
                prev_grid = grid
                continue
            plan.append(ReplayStep(e["id"], dict(e["data"] or {}), grid, e["level"]))
            prev_grid = grid
        if not plan or len(plan) > self._budget_left():
            return 0

        replayed = 0
        for step in plan:
            frame = self._env_step_by_id(step.action_id, step.data or None)
            sandbox.update_frame(frame, f"ACTION{step.action_id}")
            replayed += 1
            got = np.asarray(frame.frame[-1], dtype=np.int8) if frame.frame else None
            if got is None or got.shape != step.expected_grid.shape \
                    or not (got == step.expected_grid).all():
                logger.info(f"{self.game_id}: death-replay diverged at {replayed}/{len(plan)}")
                break
            if str(frame.state).endswith(("GAME_OVER", "WIN")):
                break
        logger.info(f"{self.game_id}: death-replay fast-forwarded {replayed} actions")
        return replayed

    def _speedrun_if_won(self) -> None:
        """After a WIN, full-reset and replay the distilled trajectory for a
        fresh scorecard run (game score = max over runs, so риск нулевой)."""
        if self.frames[-1].state is not GameState.WIN:
            return
        plan = build_replay_plan(self.replay_log)
        original = sum(1 for e in self.replay_log if e["id"] != 0)
        logger.info(
            f"{self.game_id}: speedrun plan {len(plan)} actions "
            f"(original run used {original}), budget left {self._budget_left()}"
        )
        result = execute_replay(
            plan,
            env_step=lambda aid, data: self._env_step_by_id(aid, data),
            budget_left=self._budget_left,
        )
        logger.info(f"{self.game_id}: speedrun result {result}")

    def _env_step_by_id(self, action_id: int, payload: dict | None) -> FrameData:
        action = GameAction.from_id(action_id)
        if payload:
            action.set_data(payload)
        action.reasoning = "speedrun"
        frame = self.take_action(action)
        if frame is None:
            raise RuntimeError(f"environment rejected replay action {action_id}")
        self.append_frame(frame)
        self.action_counter += 1
        return frame

    def _budget_left(self) -> int:
        return self.MAX_ACTIONS - self.action_counter

    def _env_step(self, engine_action_name: str, payload: dict | None) -> FrameData:
        """Real environment step used by the sandbox's action()."""
        action = GameAction[engine_action_name] if engine_action_name != "RESET" else GameAction.RESET
        if payload is not None:
            action.set_data(payload)
        action.reasoning = "llm"
        frame = self.take_action(action)
        if frame is None:
            raise RuntimeError(f"environment rejected action {engine_action_name}")
        self.append_frame(frame)
        self.action_counter += 1
        # The sandbox updates its own view after action(); main-loop RESETs
        # call sandbox.update_frame explicitly.
        return frame

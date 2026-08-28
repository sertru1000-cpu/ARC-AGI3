"""Direct OpenAI-compatible tool-calling analyzer for ARC puzzle runs."""
from __future__ import annotations

import copy
import heapq
import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse, urlunparse

import requests

from inference.agent.action_names import to_engine_action, to_model_action
from inference.agent.prompts import (
    ATLAS_EXPLORE_FIRST_CHECKPOINT,
    ATLAS_EXTRACT_CHECKPOINT,
    ATLAS_FORCE_ACT_OVERRIDE,
    ATLAS_FORCE_ROLLBACK_CHECKPOINT,
    ATLAS_GOAL_RECONSIDER_CHECKPOINT,
    ATLAS_MEMO_CHECKPOINT,
    ATLAS_NOTE_ENFORCEMENT_CHECKPOINT,
    ATLAS_PLAN_CHECKPOINT_TEMPLATE,
    ATLAS_PLAN_FORCE_OVERRIDE,
    ATLAS_PLAN_REAL_ESCALATION_DEFAULT,
    ATLAS_PLAN_REAL_ESCALATION_MOUSE,
    ATLAS_PLAN_REAL_FORCE_CHECKPOINT,
    ATLAS_PLAN_REAL_MOUSE_ARGS_HINT,
    ATLAS_THEORY_CHECKPOINT,
    ATLAS_THEORY_FORCE_OVERRIDE,
    COMPACT_TOOL_SESSION_ADDENDUM,
    GAME_OVERVIEW_ADDENDUM,
    PYTHON_ADDENDUM,
    STRUCTURED_RUNTIME_STATE_ADDENDUM,
    MULTIMODAL_CONTEXT_ADDENDUM,
    TOOL_CALL_FORMAT_GUIDANCE,
    VISUAL_GAME_ADDENDUM,
)

# atlas 27.08: found live on r11l -- human baseline solves this MOUSE-only
# game in 2 moves, but the model spent 45 minutes/tens of thousands of
# tokens on verify_theory() without ever trying more than 2 real actions.
# Threshold is deliberately LOWER than _ATLAS_THEORY_NAG_AFTER_CALLS below --
# exploring the control surface is more foundational than any theory about
# it, so this must be checked (and able to fire) before theory does, not
# after. See _atlas_action_kinds_resolved (tracked in _handle_action) and
# the priority chain in _build_user_prompt. Resets every level-up (see
# _atlas_explore_level) -- a new level can introduce new mechanics or make a
# previously-irrelevant control matter, so "resolved" from an earlier level
# doesn't carry over.
_ATLAS_EXPLORE_NUDGE_AFTER_CALLS = 2
# A control kind counts as "resolved" (no longer nagged about) once it
# either (a) visibly changed the board at least once -- the model learned
# something real -- or (b) has been tried this many times with NO visible
# effect. (b) exists so a control that is genuinely inert in this exact
# spot/level (e.g. MOUSE clicks on empty water before finding the fish)
# doesn't nag forever -- the r11l (v12) total-paralysis lesson: an
# unsatisfiable gate ("prove this changed something") can trap a model just
# as badly as "prove verified_accuracy >= 0.6" did.
_ATLAS_EXPLORE_MAX_ATTEMPTS_PER_KIND = 3
# atlas: how many python-tool calls may pass without a plan_with_theory( call
# before the checkpoint nags again. Mirrors our own harness's PLAN_NAG_EVERY.
_ATLAS_PLAN_NAG_EVERY = 3
# atlas 27.08: hard backstop for the soft plan nudge above (see
# ATLAS_PLAN_FORCE_OVERRIDE in prompts.py). 2x the soft nag's own threshold,
# same "let the soft wording get a real chance first" logic already used
# for the theory-force threshold vs. ATLAS_THEORY_NAG_AFTER_CALLS.
_ATLAS_PLAN_FORCE_AFTER_CALLS = _ATLAS_PLAN_NAG_EVERY * 2
# Below this many python-tool calls, transitions probably haven't accumulated
# enough for verify_theory to be worth nagging about yet.
_ATLAS_THEORY_NAG_AFTER_CALLS = 4
# 24.08: was disabled after r11l (v12) showed it can cause TOTAL action
# paralysis (1 real action in 4.4h) -- verify_theory's accuracy is an exact
# whole-board match per transition, so a hard-to-model mechanic can make
# "verified_accuracy >= 0.6" nearly unreachable, and the model read "Do not
# skip this turn -- probing without a theory wastes actions" as a hard gate
# against acting further at all. Disabling it outright turned out to have
# its own cost, found on v15: with NEITHER checkpoint able to fire
# (PLAN_CHECKPOINT only fires after a verified theory exists),
# verify_theory/plan_with_theory/execute_plan dropped to 0 calls across 612
# real python-tool calls, 25 games -- regressed all the way back to C0's
# original "tool ignored" problem this nudge was built to fix (v5: 0->42
# calls). 25.08 v16: re-enabled with SOFTENED wording ("this is a
# suggestion, not a requirement... acting without a verified theory is
# completely fine") + the force-act backstop below. Backstop worked (fired
# 23 times across 13/25 games, no paralysis recurrence), but the soft
# wording gave the model explicit permission to ignore it -- still 0/662
# real calls despite the checkpoint firing 360 times; a live transcript
# check (r11l) showed the model's own reasoning never even mentions it.
# 25.08 v17: wording re-strengthened (imperative "THIS turn, write
# predict()...", no more "just a suggestion" framing) now that the
# force-act backstop is PROVEN to catch paralysis independent of wording --
# safety no longer depends on how forcefully this text reads.
_ATLAS_THEORY_CHECKPOINT_ENABLED = True
# atlas 27.08: hard backstop for a THIRD r11l failure mode (see
# ATLAS_THEORY_FORCE_OVERRIDE in prompts.py for the full history/rationale).
# Unlike _ATLAS_FORCE_ACT_AFTER_CALLS below (tracks calls since the last real
# action(), resets on every action()), this counts total python-tool calls
# this game and never resets -- it only cares whether verify_theory( has
# EVER been called, tracked via the existing _atlas_verify_theory_call_count.
# 2x the soft nag's own threshold: give the soft wording a real chance first.
_ATLAS_THEORY_FORCE_AFTER_CALLS = 8
# atlas 25.08: hard backstop for the r11l failure mode, independent of
# checkpoint wording. Counts real `python` tool calls since the last call
# that actually invoked action() (see _atlas_calls_since_real_action,
# reset/incremented in _run_python_tool). Once this crosses the threshold,
# _build_user_prompt overrides BOTH theory-style checkpoints with
# ATLAS_FORCE_ACT_OVERRIDE for that turn -- unconditional, cannot be
# out-argued by any interpretation of the softer nudge text above.
_ATLAS_FORCE_ACT_AFTER_CALLS = 8
# atlas 25.08: found on dc22 (Gemini teacher data, old harness) -- 221
# verify_theory calls, but the model was cycling through 4 unrelated
# high-level theories of what KIND of mechanic this is (a rotating dial -> a
# camera capture -> a lathe/silhouette -> an assembly arm), each abandoned
# rather than falsified by evidence. THEORY_CHECKPOINT only ever pushes
# "refine predict()" -- it has no way to suggest the DYNAMICS aren't the
# problem, the GOAL model might be. After this many verify_theory( calls
# still below 0.6 accuracy, nudge toward reconsidering the goal instead of
# yet another predict() rewrite. Deliberately 2x _ATLAS_THEORY_NAG_AFTER_CALLS
# -- give one theory a real chance before suggesting the bigger reframe.
_ATLAS_GOAL_RECONSIDER_AFTER_CALLS = 8
# atlas 26.08: found live -- memo was never written to in 81 real actions
# across 3 games (0%, not even the ~0.2% C0 baseline for a passively-worded
# tool). Lowest-priority checkpoint in the chain, so it only fires once
# nothing more urgent (force-act/goal-reconsider/theory/plan) is active.
# Threshold above _ATLAS_THEORY_NAG_AFTER_CALLS so a game gets a real chance
# to need memo before being nagged about it.
_ATLAS_MEMO_NUDGE_AFTER_CALLS = 10
# atlas 26.08: idea #2 from Gemini's strategy review -- extract= is documented
# but, like memo, has near-zero voluntary uptake. Threshold sits between
# theory's (4) and goal-reconsider's (8): give a game a real chance to need
# extract before nagging about it, but nag before the bigger goal-reconsider
# reframe fires. NOTE: in _build_user_prompt's elif chain this is checked
# BEFORE the generic theory checkpoint despite firing "after" it here --
# theory's lower threshold is always already satisfied once this one's is,
# so checking theory first would make this branch unreachable.
_ATLAS_EXTRACT_NUDGE_AFTER_CALLS = 6
# atlas 26.08: idea #1 (rollback), refined per Gemini's risk-mitigation
# design. Trigger A -- "soft stall": this many real actions taken since the
# last level-progress event (tracked via _atlas_actions_since_level_progress,
# reset on level-up) without progress raises the FORCE_ROLLBACK ultimatum.
_ATLAS_ROLLBACK_STALL_AFTER_CALLS = 15
# atlas 27.08 (late): plan_real principle-force -- fires EARLIER than the
# rollback stall trigger above (search constructively before giving up and
# rewinding), and only while no rollback ultimatum is active.
_ATLAS_PLAN_REAL_STALL_AFTER_ACTIONS = 10
# After this many consecutive showings without a plan_real( call, the
# harness runs the search itself (non-MOUSE games only -- for MOUSE games
# it cannot pick candidate clicks, so there the nag just stops after
# _ATLAS_PLAN_REAL_NAG_CAP showings instead of nagging forever).
_ATLAS_PLAN_REAL_AUTO_FORCE_AFTER = 2
_ATLAS_PLAN_REAL_NAG_CAP = 5
# Trigger B -- "hard loop": current board_signature() matches one seen this
# many actions ago (a real repeat, not mere similarity) -- Gemini's own
# example was "вернулся в состояние экрана, в котором уже был 2 хода назад",
# so this is set to exactly that rather than a looser guess.
_ATLAS_ROLLBACK_LOOP_WINDOW = 2
# After this many consecutive turns where the model was given the two-step
# coercion ultimatum and still did not call rollback(), the harness performs
# the rollback itself (to the most recent auto-anchor) as a hard backstop --
# mirrors the project's C0 lesson that voluntary adoption of an unfamiliar
# tool under pressure cannot be assumed even with an explicit ultimatum.
_ATLAS_ROLLBACK_AUTO_FORCE_AFTER = 3
# atlas 26.08: idea #3 from Gemini's strategy review -- context sanitizer.
# Games run up to ~4h; the raw action/observation transcript accumulates
# false theories, analyzer-timeout artifacts, and dead-end attempts that
# token-budget trimming only prunes reactively (oldest-block-first, once the
# context gets too big), never deliberately. After this many real actions
# since the last sanitize with no level-up (see _atlas_context_sanitize_level),
# a HOST-triggered (not model-voluntary -- same C0 lesson as rollback/memo)
# sanitize fires, same as a level-up always does regardless of this count.
_ATLAS_CONTEXT_SANITIZE_EVERY_CALLS = 20
_ATLAS_ACCURACY_RE = re.compile(r"['\"]accuracy['\"]\s*:\s*([0-9]*\.?[0-9]+)")
# atlas 27.08: found on wa30 (Gemini-flagged gate-bypass) -- a verify_theory(
# call made right after rollback() (which wipes the transitions history) is
# technically compliant with the theory-force override but tests 0
# transitions, so it carries zero diagnostic value. This regex lets the
# force-override gate tell a real call from a vacuous one, the same way
# _ATLAS_ACCURACY_RE lets it read the accuracy out of the printed dict.
_ATLAS_TRANSITIONS_TESTED_RE = re.compile(r"['\"]transitions_tested['\"]\s*:\s*([0-9]+)")
# atlas 27.08 (direction change, user: "мы занимаемся костылями, не идём к
# agi"): real-engine speculative execution. The OFFLINE engine snapshot/
# restore built for rollback() doubles as a PERFECT simulator -- game_run
# (whose actions_per_level/levels_completed feed _compute_final_score) is
# part of the snapshot, so actions taken inside a probe and then rewound
# never reach the score, while the engine scorecard is a monotonic max()
# that keeps any level a probe happens to complete. This removes the need
# for a model-authored predict() for PLANNING entirely -- the exact
# authorship friction the whole theory-checkpoint chain has been fighting:
# the model (or the harness BFS below) can just try sequences for real and
# rewind. Budgets: everything here runs host-side while the sandbox blocks
# inside the model's 30s python-call limit, so the search must leave the
# model's own code room to finish.
_ATLAS_PROBE_MAX_SEQUENCES = 16
_ATLAS_PROBE_MAX_TOTAL_STEPS = 240
# 29.08 (Gemini round 6, D2/HailMary): hard caps raised -- the wall budget
# is now per-request ("wall_seconds", clamped to [1, 60]): model-called
# searches keep the 10s default (they run inside the sandbox's 30s reply
# window), while HOST-side searches (proactive level entry, hail-mary) may
# spend 40-60s -- engine-bound CPU time that costs zero LLM turns.
_ATLAS_SEARCH_MAX_NODES = 5000
_ATLAS_SEARCH_MAX_DEPTH = 12
_ATLAS_SEARCH_WALL_SECONDS = 10.0
_ATLAS_SEARCH_WALL_MAX_SECONDS = 60.0
# D2: progressive proactive budget -- level 2+ carries 2x+ score weight and
# a deeper state space; the level-entry search there gets the big budget.
_ATLAS_PROACTIVE_L1_BUDGET = {"max_depth": 6, "max_nodes": 250, "wall_seconds": 10}
_ATLAS_PROACTIVE_DEEP_BUDGET = {"max_depth": 10, "max_nodes": 1000, "wall_seconds": 40}
_ATLAS_HAIL_MARY_BUDGET = {"max_depth": 12, "max_nodes": 5000, "wall_seconds": 60}
# HailMary trigger: this many seconds (or less) left on the game's wall
# budget while on level 2+ -> the harness stops waiting for the model and
# brute-forces the board once.
_ATLAS_HAIL_MARY_REMAINING_S = 600.0
# atlas 27.08 (plan_real v2): Monte-Carlo deep rollouts -- after the
# systematic frontier is exhausted (or its node budget spent), leftover
# wall-clock goes to random playouts of up to this depth from the most
# novel frontier states. Catches solutions deeper than the systematic
# depth cap at the cost of completeness (a miss proves nothing). Classic
# UCT/MCTS value-backprop is deliberately NOT used: rewards here are
# sparse/binary (level completed or not) and the engine is deterministic,
# so a bandit layer degenerates to random search with extra bookkeeping.
_ATLAS_ROLLOUT_DEPTH = 24
_ATLAS_MAX_ROLLOUTS = 300
# atlas 27.08 (probe/checkpoint integration, found live on the first
# OFFLINE pod run): probes displaced BOTH real actions and verify_theory,
# and the checkpoint ladder -- designed before probes existed -- responded
# by spamming contradictory imperatives (explore-first fired 153x,
# theory-force 52x in ~37 min, real actions halved vs baseline). A probe
# that just ran IS empirical theory work: the engine itself answered
# "what happens if". So the theory nag and theory-force stay quiet for
# this many python calls after any successful probe execution.
_ATLAS_PROBE_THEORY_GRACE_CALLS = 4
# atlas 28.08 (Gemini round 3): probe rationing -- probes are score-free
# but consume LLM turns and wall-clock inside the game's fixed time
# budget (measured live: probe-heavy runs halved real actions and scored
# 0.39-0.44 vs 0.96-1.10 for checkpoint-only). After this many
# consecutive probe-executing calls with no real action() in between, a
# one-line "convert knowledge into moves" nudge is appended.
_ATLAS_PROBE_RATION_FREE = 3
# atlas 28.08 (Gemini round 5, L1): proactive plan_real -- at every level
# entry (incl. game start) the harness runs the now-fast search ITSELF,
# before the model spends any turns, and executes a found plan. Zero model
# turns per auto-solved level; a miss costs ~10s wall and stays silent.
_ATLAS_PLAN_REAL_PROACTIVE = os.environ.get("ATLAS_PLAN_REAL_PROACTIVE", "1") != "0"
# atlas 28.08 (Gemini round 5, L2): auto-replay of SOLVED levels. The run
# records the action sequence that completed each level; when the game
# falls back to an earlier level (RESET after game over, or a deliberate
# full restart), the harness batch-replays the known solutions instead of
# letting the model re-derive them turn by turn. Saves model TURNS and
# wall-clock, not recorded actions (the play's action counter is
# cumulative either way). Stops honestly on the first divergence.
_ATLAS_LEVEL_AUTO_REPLAY = os.environ.get("ATLAS_LEVEL_AUTO_REPLAY", "1") != "0"
# 29.08 (Gemini round 6, D4): kill-switch for the level-up mechanic handoff.
_ATLAS_MECHANIC_HANDOFF = os.environ.get("ATLAS_MECHANIC_HANDOFF", "1") != "0"
# atlas 28.08 (Gemini round 5, L3): cap CONCURRENT LLM HTTP requests
# below the game concurrency -- 55 simultaneous vLLM requests thrash the
# KV cache into eviction/recompute spirals and starve every game (observed
# live: 5 of 25 games executed ONE action in 105 min). 0 = no gate.
_ATLAS_LLM_MAX_CONCURRENT = int(os.environ.get("ATLAS_LLM_MAX_CONCURRENT_REQUESTS", "0") or "0")
_ATLAS_LLM_REQUEST_GATE = (
    threading.Semaphore(_ATLAS_LLM_MAX_CONCURRENT) if _ATLAS_LLM_MAX_CONCURRENT > 0 else None
)
# 29.08 (Gemini round 6, D3 "zombie cull"): games stuck on level 1 with no
# progress must not monopolize the request gate and starve the games that
# actually reached level 2+ (measured: 82 zombies crowding 25 slots left
# the median level-2 game FOUR real actions). Zombies (level 1, >=N real
# actions since progress) must additionally hold one of these fewer slots
# -- deep and fresh games keep priority access to the main gate.
_ATLAS_ZOMBIE_AFTER_ACTIONS = int(os.environ.get("ATLAS_ZOMBIE_AFTER_ACTIONS", "25") or "25")
_ATLAS_LLM_ZOMBIE_SLOTS = int(os.environ.get("ATLAS_LLM_ZOMBIE_SLOTS", "0") or "0")
if _ATLAS_LLM_ZOMBIE_SLOTS <= 0 and _ATLAS_LLM_MAX_CONCURRENT > 0:
    _ATLAS_LLM_ZOMBIE_SLOTS = max(1, int(_ATLAS_LLM_MAX_CONCURRENT * 0.4))
_ATLAS_LLM_ZOMBIE_GATE = (
    threading.Semaphore(_ATLAS_LLM_ZOMBIE_SLOTS)
    if (_ATLAS_LLM_REQUEST_GATE is not None and _ATLAS_LLM_ZOMBIE_SLOTS > 0)
    else None
)
# Cap on the per-level probe-findings memory injected into the prompt so
# the model does not re-probe what it already learned.
_ATLAS_PROBE_FINDINGS_MAX = 12
# Auto-derived MOUSE candidate clicks for plan_real (centers of the
# largest non-background segmentation objects) -- lets click games be
# searched with zero model authorship. Small K: every candidate
# multiplies the frontier.
_ATLAS_MOUSE_AUTO_CANDIDATES = 6
# atlas: a non-null 'note' in a printed/returned plan_with_theory() result --
# res['note'] is only set when the found plan has more than one step, so this
# is a best-effort stand-in for "the model got back a multi-step plan" when
# the structured result dict isn't available and we fall back to stdout.
_ATLAS_NOTE_PRESENT_RE = re.compile(r"['\"]note['\"]\s*:\s*(?!None\b)\S")

# atlas A4: action-effect summary, computed from history_entries (already
# available to the host each turn) rather than asked of the model. Backlog
# note (22.08, ported from public non-LLM ARC-AGI-3 notebooks -- stochasticgoose's
# ActionEffectAttention, ForgeNet): the transition data needed to answer "what
# does each action change" already exists in the frame history; nobody was
# aggregating it, so the model re-derives the obvious by trial and error every
# episode instead of getting a head start. Only look at the last N history
# entries -- cheap, and biases toward the CURRENT level's mechanics rather
# than diluting with a level that may play by different rules.
_ATLAS_ACTION_EFFECT_HISTORY_WINDOW = 60
_ATLAS_ACTION_EFFECT_MIN_TRANSITIONS = 4
_ATLAS_ACTION_EFFECT_HUD_SHARE = 0.7
# Suppress the invariant-region hint once the "active" bbox covers this much
# of the board -- past that, the hint no longer narrows anything (found live
# on re86, 24.08: a bbox spanning the whole 64x64 board).
_ATLAS_INVARIANT_MAX_BBOX_SHARE = 0.8
_ATLAS_MOUSE_ACTION_RE = re.compile(r"^MOUSE\(row=(-?\d+), col=(-?\d+)\)$")


def _atlas_action_effect_summary(history_entries: list[HistoryEntry]) -> list[str]:
    window = history_entries[-(_ATLAS_ACTION_EFFECT_HISTORY_WINDOW + 1):]
    per_action: dict[str, list[tuple[tuple[int, int] | None, list[tuple[int, int]]]]] = {}
    changed_counts: dict[tuple[int, int], int] = {}
    total = 0
    board_cells = 0
    for prev, cur in zip(window, window[1:]):
        action = (cur.action or "").strip()
        before, after = prev.frame, cur.frame
        if not action or before is None or after is None:
            continue
        before_grid, after_grid = before.grid, after.grid
        if len(before_grid) != len(after_grid):
            continue
        board_cells = max(board_cells, len(after_grid) * (len(after_grid[0]) if after_grid else 0))
        changed: list[tuple[int, int]] = []
        for r, (before_row, after_row) in enumerate(zip(before_grid, after_grid)):
            if len(before_row) != len(after_row):
                continue
            for c, (b, a) in enumerate(zip(before_row, after_row)):
                if b != a:
                    changed.append((r, c))
                    changed_counts[(r, c)] = changed_counts.get((r, c), 0) + 1
        total += 1
        mouse_match = _ATLAS_MOUSE_ACTION_RE.match(action)
        if mouse_match:
            base, click = "MOUSE", (int(mouse_match.group(1)), int(mouse_match.group(2)))
        else:
            base, click = action.split("(", 1)[0].strip().upper(), None
        per_action.setdefault(base, []).append((click, changed))

    if total < _ATLAS_ACTION_EFFECT_MIN_TRANSITIONS:
        return []

    # HUD cells first, so per-action stats below can exclude them -- without
    # this, a HUD cell that changes on every transition contaminates every
    # action's bbox (and MOUSE's relative-to-click bbox) with its own
    # unrelated position, defeating the whole point of separating mechanic
    # from noise.
    hud_cells = {
        cell for cell, count in changed_counts.items() if count / total >= _ATLAS_ACTION_EFFECT_HUD_SHARE
    }
    if not (hud_cells and 0 < len(hud_cells) <= max(64, board_cells // 100)):
        hud_cells = set()

    def _bbox(cells: list[tuple[int, int]]) -> str:
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        return f"rows {min(rows)}..{max(rows)}, cols {min(cols)}..{max(cols)}"

    lines = ["Action-effect summary (aggregated from recent history, costs nothing to read):"]
    for base in sorted(per_action):
        entries = [(click, [cell for cell in changed if cell not in hud_cells]) for click, changed in per_action[base]]
        n = len(entries)
        avg_changed = sum(len(changed) for _, changed in entries) / n
        if base == "MOUSE":
            relative_cells = [
                (r - click[0], c - click[1])
                for click, changed in entries
                if click is not None
                for r, c in changed
            ]
            if relative_cells:
                lines.append(
                    f"- MOUSE ({n} clicks): avg {avg_changed:.1f} cell(s) change, "
                    f"relative to the click point mostly within {_bbox(relative_cells)}."
                )
            else:
                lines.append(f"- MOUSE ({n} clicks): avg {avg_changed:.1f} cell(s) change; none recorded yet.")
        else:
            all_cells = [cell for _, changed in entries for cell in changed]
            if all_cells:
                lines.append(f"- {base} ({n}x): avg {avg_changed:.1f} cell(s) change, mostly within {_bbox(all_cells)}.")
            else:
                lines.append(f"- {base} ({n}x): no cell ever changed -- likely a no-op here.")

    if hud_cells:
        rows = sorted({r for r, _ in hud_cells})
        lines.append(
            f"- {len(hud_cells)} cell(s) change in >={_ATLAS_ACTION_EFFECT_HUD_SHARE:.0%} of ALL recent "
            f"transitions regardless of action (rows {rows[:6]}{', ...' if len(rows) > 6 else ''}) "
            "-- likely a HUD/timer element, not gameplay; excluded from the stats above."
        )

    # The mirror of the HUD line: cells that NEVER changed in ANY observed
    # transition, regardless of action. Reported as the bounding box of
    # everything that DID ever change (the "active region") rather than
    # listing every static cell -- knowing where the mechanic can't possibly
    # be narrows the search space the same way knowing where it lives does.
    #
    # Degenerate case found live (re86, 24.08): different actions can each
    # move a spatially-tight but DIFFERENT region (UP moves the top, DOWN
    # moves the bottom, etc.) -- the union bbox then sprawls across nearly
    # the whole board even though most INDIVIDUAL cells are static, and the
    # hint stops narrowing anything ("static outside rows 0..63" on a 64-row
    # board is not a hint). Suppress it once the bbox itself covers most of
    # the board, regardless of how high the static-cell percentage reads.
    if changed_counts and board_cells:
        active_rows = [r for r, _ in changed_counts]
        active_cols = [c for _, c in changed_counts]
        static_cells = board_cells - len(changed_counts)
        bbox_cells = (max(active_rows) - min(active_rows) + 1) * (max(active_cols) - min(active_cols) + 1)
        if static_cells > 0 and bbox_cells / board_cells <= _ATLAS_INVARIANT_MAX_BBOX_SHARE:
            lines.append(
                f"- {static_cells}/{board_cells} cell(s) ({static_cells / board_cells:.0%}) never "
                f"changed in any observed transition -- everything outside rows "
                f"{min(active_rows)}..{max(active_rows)}, cols {min(active_cols)}..{max(active_cols)} "
                "has been static; the mechanic can only live inside that region."
            )
    return lines

from inference.agent.vision_context import (
    current_grid_image_enabled,
    current_grid_image_part,
)

from inference.agent.noop_guard import NoopGuard, board_signature, normalize_action_signature
from inference.utils.animation import (
    ANIMATION_HINT_COOLDOWN_TURNS,
    ANIMATION_HINT_FOLLOW_WINDOW_TURNS,
    ANIMATION_HINT_MIN_TRANSIENT_PIXELS,
    animation_hint_text,
    build_animation_view,
    describe_animation,
    pick_animation,
    should_suggest_animation,
)
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.utils.grid_utils import ARC_COLOR_CHARS
from inference.utils.segmentation import segment_layer
from inference.agent.runtime_state import Frame, HistoryEntry, RUNTIME_STATE_FILENAME, load_runtime_state
from inference.utils.openai_compat import build_chat_payload, build_headers

log = logging.getLogger(__name__)

_LOCAL_ANALYZER_MODEL_ID = os.environ.get("LOCAL_ANALYZER_MODEL_ID", "")
_LOCAL_ANALYZER_BASE_URL = os.environ.get("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1")
_DEFAULT_ANALYZER_MODEL = os.environ.get(
    "INFERENCE_ANALYZER_MODEL",
    _LOCAL_ANALYZER_MODEL_ID,
)
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    flags=re.DOTALL | re.IGNORECASE,
)
_TOOL_CALL_PARAMETER_RE = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
    flags=re.DOTALL | re.IGNORECASE,
)
_THINK_TAG_RE = re.compile(r"</?think>", flags=re.IGNORECASE)


def _get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _contains_tool_call_markup(*chunks: str) -> bool:
    for chunk in chunks:
        lowered = chunk.lower()
        if "<tool_call" in lowered or "<function=" in lowered:
            return True
    return False


def _strip_tool_call_markup(text: str) -> str:
    if not text.strip():
        return ""
    stripped = _TOOL_CALL_BLOCK_RE.sub("", text)
    return stripped.strip()


def _recover_tool_calls_from_markup(*chunks: str) -> list[dict[str, Any]]:
    recovered: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        if not chunk.strip():
            continue
        for match in _TOOL_CALL_BLOCK_RE.finditer(chunk):
            tool_name = str(match.group(1) or "").strip()
            if not tool_name:
                continue
            raw_body = str(match.group(2) or "")
            arguments = {
                str(parameter_name).strip(): value
                for parameter_name, value in _TOOL_CALL_PARAMETER_RE.findall(raw_body)
                if str(parameter_name).strip()
            }
            cache_key = (
                tool_name,
                json.dumps(arguments, ensure_ascii=True, sort_keys=True),
            )
            if cache_key in seen:
                continue
            seen.add(cache_key)
            recovered.append(
                {
                    "id": f"markup-call-{len(recovered) + 1}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments, ensure_ascii=True),
                    },
                }
            )
    return recovered


def _get_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


_LOCAL_ANALYZER_MAX_OUTPUT = _get_env_int("LOCAL_ANALYZER_MAX_OUTPUT", 0)
_LOCAL_ANALYZER_CONTEXT_WINDOW = _get_env_int("LOCAL_ANALYZER_CONTEXT_WINDOW", 32768)
_LOCAL_ANALYZER_TIMEOUT = _get_env_float("LOCAL_ANALYZER_TIMEOUT", 0.0)
_LOCAL_ANALYZER_TOOL_STEPS = _get_env_int("LOCAL_ANALYZER_TOOL_STEPS", 12)
_LOCAL_ANALYZER_TOOL_TIMEOUT = _get_env_int("LOCAL_ANALYZER_TOOL_TIMEOUT", 30)
_LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS = _get_env_int("LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS", 1024)
_LOCAL_ANALYZER_YIELD_SECONDS = _get_env_float("LOCAL_ANALYZER_YIELD_SECONDS", 0.0)
_LOCAL_ANALYZER_ENABLE_THINKING = _get_env_bool("LOCAL_ANALYZER_ENABLE_THINKING", True)
_LOCAL_ANALYZER_TEMPERATURE = _get_env_float("LOCAL_ANALYZER_TEMPERATURE", 0.6)
_LOCAL_ANALYZER_TOP_P = _get_env_float("LOCAL_ANALYZER_TOP_P", 0.95)
_LOCAL_ANALYZER_TOP_K = _get_env_int("LOCAL_ANALYZER_TOP_K", 20)
_LOCAL_ANALYZER_SEED = _get_env_int("LOCAL_ANALYZER_SEED", -1)
_REQUEST_SAFETY_MARGIN_TOKENS = 512
_CONTEXT_OVERFLOW_RETRY_TRIM_TOKENS = 512
# Experiment 2 (Known-Noop-Guard, 2026-07-18): actively blocks re-executing a
# single action already proven to have no effect in the exact same board
# state, instead of only mentioning it in context (see Experiment 1 / K1,
# reverted for relying on model cooperation that didn't hold).
_HARD_NOOP_GUARD_ENABLED = _get_env_bool("ARC3_HARD_NOOP_GUARD", True)
# Experiment 3 (Animation awareness, 2026-08-07): one action can return a
# whole list of frames; the harness only ever showed the last one. This flag
# gates all three stages -- compact per-action metadata, the on-demand frame
# retrieval tool, and the proactive hint. The no-op guard's frame-count fix is
# deliberately *not* gated: it is a bug fix, not part of the experiment arm.
_ANIMATION_AWARENESS_ENABLED = _get_env_bool("ARC3_ANIMATION_AWARENESS", True)
_PERSISTENT_HISTORY_ASSISTANT_TURNS = 30
_RESPONSE_META_MAX_CHARS = 4000

_PYTHON_TOOL_DESCRIPTION = (
    "Run one ephemeral Python snippet against preloaded ASCII game state. Available globals: "
    "`current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`, "
    "`valid_actions`, `last_action_result`, "
    "`action(actions)` for executing one or more real environment actions, "
    "and `animation()` for a compact diff timeline of the frames the last animated action produced. "
    "`current_frame` and each `history[*].frame` expose only `.ascii`, `.segmentation`, `.step`, and `.level`; "
    "`history[-1].frame` is the current post-action frame, not the previous frame. "
    "For before/after diffs, compare `previous_frame` to `current_frame` or use `last_transition.before_frame` and `.after_frame`. "
    "For MOUSE, pass `row` and `col` integer fields; legacy x/y fields are rejected. "
    "The raw numeric grid is not available. Use `.segmentation` as the primary view; use `.ascii` only to read a small, specific region. "
    "Use `print(...)` for compact output or assign final data to `result`."
)

def _normalize_valid_actions(valid_actions: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in valid_actions or []:
        engine_name = to_engine_action(value)
        name = to_model_action(engine_name or value)
        if name and name not in names:
            names.append(name)
    return names


def _pending_action_signature(action: dict[str, Any]) -> str:
    """Signature for a not-yet-executed action, matching the format the
    environment reports back as ``action_display``/``executed_actions`` after
    execution (see ``_format_action_display`` in solver.py), so a pending
    action can be checked against the no-op guard before it ever runs.

    Returns "" if a MOUSE action is missing valid row/col (validation of that
    is the environment's job; the guard simply won't match on it).
    """
    engine_name = to_engine_action(action.get("action"))
    display_name = to_model_action(engine_name) if engine_name else str(action.get("action") or "")
    if display_name == "MOUSE":
        try:
            row = max(0, min(63, int(action.get("row"))))
            col = max(0, min(63, int(action.get("col"))))
        except (TypeError, ValueError):
            return ""
        return normalize_action_signature(f"MOUSE(row={row}, col={col})")
    return normalize_action_signature(display_name)


def _payload_frame_count(payload: dict[str, Any]) -> int:
    """Frames the environment returned for one action, defaulting to 1.

    Older payloads (and the synthetic ones built for blocked/terminal
    actions) carry no ``frame_count``; treating those as a single frame keeps
    the no-op guard's behaviour unchanged for them.
    """
    try:
        return max(1, int(payload.get("frame_count") or 1))
    except (TypeError, ValueError):
        return 1


def _action_animated(payload: dict[str, Any]) -> bool:
    """True if this action produced a multi-frame animation.

    An animated action is never a no-op, regardless of ``board_changed`` --
    see ``NoopGuard.observe``.
    """
    return _payload_frame_count(payload) > 1


def _format_valid_action_line(valid_actions: list[str] | None) -> str:
    names = _normalize_valid_actions(valid_actions)
    if not names:
        return "unknown"
    return ", ".join(names)


def _terminal_action_reason(result: dict[str, Any]) -> str | None:
    if result.get("run_complete"):
        return "run_complete"
    if result.get("game_over"):
        return "game_over"
    if result.get("level_completed"):
        return "level_completed"
    if result.get("done"):
        return "done"
    return None


def _terminal_action_stop_detail(reason: str | None) -> str:
    if reason == "run_complete":
        return "No further actions were executed because the run is already complete."
    if reason == "game_over":
        return (
            "No further actions were executed because the previous action reached GAME_OVER; "
            "the runner will auto-reset before the next analyzer turn."
        )
    if reason == "level_completed":
        return (
            "No further actions were executed because the previous action completed a level; "
            "re-ground on the new scene before acting again."
        )
    if reason == "done":
        return "No further actions were executed because the environment reported done."
    return "No further actions were executed because the previous action reached a terminal state."


def _aggregate_action_batch_result(
    *,
    requested_count: int,
    executed_results: list[dict[str, Any]],
    blocked_actions: list[str],
    last_failed: dict[str, Any] | None,
    valid_actions: list[str],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    """Combine per-action results from walking a batch through the no-op
    guard one action at a time (see the ``len(normalized_actions) > 1``
    branch of ``_handle_action``) into the same shape ``_compact_action_result``
    produces for one atomic ``step_env`` batch call.
    """
    last_executed = executed_results[-1] if executed_results else None
    base = last_executed or last_failed or fallback

    executed_actions: list[str] = []
    total_reward = 0.0
    board_changed = False
    frame_count = 0
    for item in executed_results:
        names = item.get("executed_actions")
        if isinstance(names, list) and names:
            executed_actions.extend(str(name) for name in names)
        elif item.get("action_display"):
            executed_actions.append(str(item["action_display"]))
        try:
            total_reward += float(item.get("reward") or 0.0)
        except (TypeError, ValueError):
            pass
        board_changed = board_changed or bool(item.get("board_changed"))
        # Max, not sum -- see the same aggregation in solver.step_env.
        frame_count = max(frame_count, _payload_frame_count(item))

    result: dict[str, Any] = {
        "executed": bool(executed_results),
        "action_num": base.get("action_num"),
        "level": base.get("level"),
        "score": base.get("score"),
        "reward": total_reward,
        "state": base.get("state"),
        "valid_actions": list(valid_actions),
        "board_changed": board_changed,
        "frame_count": frame_count or 1,
        "done": bool(last_executed and last_executed.get("done")),
        "level_completed": bool(last_executed and last_executed.get("level_completed")),
        "game_over": bool(last_executed and last_executed.get("game_over")),
        "run_complete": bool(last_executed and last_executed.get("run_complete")),
        "requested_count": requested_count,
        "executed_count": len(executed_results),
    }
    if executed_actions:
        result["executed_actions"] = executed_actions
    batch_animation = pick_animation([item.get("animation") for item in executed_results])
    if batch_animation is not None:
        result["animation"] = batch_animation
    processed = len(executed_results) + len(blocked_actions) + (1 if last_failed is not None else 0)
    result["stopped_early"] = processed < requested_count or last_failed is not None
    if blocked_actions:
        result["blocked_count"] = len(blocked_actions)
        result["blocked_actions"] = list(blocked_actions)

    terminal_reason = _terminal_action_reason(last_executed) if last_executed else None
    if terminal_reason:
        result["stop_reason"] = terminal_reason
        result["stop_detail"] = _terminal_action_stop_detail(terminal_reason)
    elif last_failed is not None:
        result["stop_reason"] = "action_error"
        if last_failed.get("error"):
            result["error"] = last_failed.get("error")
        result["stop_detail"] = str(last_failed.get("error") or "An action in the batch failed.")
    elif blocked_actions:
        result["stop_reason"] = "known_noop"
        result["stop_detail"] = (
            f"{len(blocked_actions)} action(s) in this batch were blocked before execution -- "
            "already known to have no effect in the exact board state attempted: "
            + ", ".join(blocked_actions)
            + "."
        )
    return result


def _display_action_number(action_num: int) -> int:
    return max(1, int(action_num) + 1)


def _normalize_summary_text(value: Any, *, max_chars: int | None = 280) -> str:
    text = " ".join(str(value or "").split())
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars].rstrip()}... [{omitted} chars omitted]"


def _extract_labeled_blocks(content: str, labels: list[str]) -> dict[str, str]:
    normalized_labels = {label.lower(): label for label in labels}
    targets = tuple(f"{label.lower()}:" for label in labels)
    extracted: dict[str, list[str]] = {label: [] for label in labels}
    current_label: str | None = None

    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        candidate = stripped
        while candidate.startswith(("-", "*")):
            candidate = candidate[1:].lstrip()
        lowered = candidate.lower()

        matched_label: str | None = None
        inline_value = ""
        for target in targets:
            if lowered.startswith(target):
                matched_label = normalized_labels[target[:-1]]
                inline_value = candidate[len(target):].strip()
                break

        if matched_label is not None:
            current_label = matched_label
            if inline_value:
                extracted[current_label].append(inline_value)
            continue

        if current_label is not None and stripped:
            extracted[current_label].append(stripped)

    return {
        label: _normalize_summary_text("\n".join(lines).strip(), max_chars=None)
        for label, lines in extracted.items()
        if "\n".join(lines).strip()
    }


def _extract_scientist_note(content: str) -> dict[str, str]:
    if not content.strip():
        return {}
    extracted = _extract_labeled_blocks(
        content,
        [
            "World model",
            "Goal model",
            "Action model",
            "Recent findings",
            "Open questions",
            "Plan",
            "Cross-level notes",
            "Hypothesis",
            "History check",
            "Next test",
        ],
    )
    result = {
        "world_model": extracted.get("World model", ""),
        "goal_model": extracted.get("Goal model", ""),
        "action_model": extracted.get("Action model", ""),
        "recent_findings": extracted.get("Recent findings", ""),
        "open_questions": extracted.get("Open questions", ""),
        "current_plan": extracted.get("Plan", ""),
        "cross_level_notes": extracted.get("Cross-level notes", ""),
    }
    if not result["world_model"]:
        result["world_model"] = extracted.get("Hypothesis", "")
    if not result["recent_findings"]:
        result["recent_findings"] = extracted.get("History check", "")
    if not result["current_plan"]:
        result["current_plan"] = extracted.get("Next test", "")
    return result


def _empty_world_model() -> dict[str, str]:
    return {
        "world_model": "",
        "goal_model": "",
        "action_model": "",
        "recent_findings": "",
        "open_questions": "",
        "current_plan": "",
        "cross_level_notes": "",
    }


def _request_tool_choice(tools: list[dict[str, Any]] | None) -> str | None:
    return "auto" if tools else None


def _trim_log_text(text: str, *, max_chars: int = _RESPONSE_META_MAX_CHARS) -> str:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    omitted = len(stripped) - max_chars
    return f"{stripped[:max_chars].rstrip()}\n... [truncated {omitted} chars]"


def _format_model_response_meta(
    *,
    finish_reason: str,
    reasoning: str,
    content: str,
    tool_calls: list[dict[str, Any]],
    tool_call_markup_in_text: bool,
    recovered_tool_calls_from_markup: bool,
    malformed_argument_errors: list[str],
) -> str:
    lines = [
        f"finish_reason: {finish_reason or '(empty)'}",
        f"tool_call_count: {len(tool_calls)}",
        f"content_chars: {len(content)}",
        f"reasoning_chars: {len(reasoning)}",
        f"tool_call_markup_in_text: {'yes' if tool_call_markup_in_text else 'no'}",
        f"tool_calls_recovered_from_markup: {'yes' if recovered_tool_calls_from_markup else 'no'}",
    ]
    if malformed_argument_errors:
        lines.append("tool_call_argument_issues:")
        lines.extend(f"- {issue}" for issue in malformed_argument_errors)
    if tool_calls:
        lines.append("raw_tool_calls:")
        lines.append(_trim_log_text(json.dumps(tool_calls, indent=2, ensure_ascii=True)))
    return "\n".join(lines)


def _build_system_prompt(*, tool_output_tokens: int) -> str:
    prompt = "You are a coding agent solving a grid-based puzzle game."
    prompt += GAME_OVERVIEW_ADDENDUM
    prompt += STRUCTURED_RUNTIME_STATE_ADDENDUM
    if current_grid_image_enabled():
        prompt += MULTIMODAL_CONTEXT_ADDENDUM
    prompt += VISUAL_GAME_ADDENDUM
    prompt += PYTHON_ADDENDUM
    prompt += COMPACT_TOOL_SESSION_ADDENDUM.format(tool_output_tokens=tool_output_tokens)
    return prompt


@dataclass(frozen=True)
class AnalyzerModelConfig:
    provider: str
    base_url: str
    model_id: str


@dataclass(frozen=True)
class AnalyzerTurnResult:
    step_executed: bool
    retryable_failure: bool = False
    reasoning: str = ""
    yielded_control: bool = False


@dataclass(frozen=True)
class _ToolDispatchResult:
    content: str
    step_executed: bool = False


@dataclass(frozen=True)
class _AsciiFrameView:
    ascii: str
    step: int
    level: int
    shape: tuple[int, int]

    def __str__(self) -> str:
        rows, cols = self.shape
        return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

    __repr__ = __str__


@dataclass(frozen=True)
class _AsciiHistoryEntryView:
    action: str
    frame: _AsciiFrameView

    def __str__(self) -> str:
        return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"

    __repr__ = __str__


def _to_ascii_frame_view(frame: Frame | None) -> _AsciiFrameView | None:
    if frame is None:
        return None
    return _AsciiFrameView(
        ascii=frame.ascii,
        step=frame.step,
        level=frame.level,
        shape=frame.shape,
    )


def _to_ascii_history_views(history_entries: list[HistoryEntry]) -> list[_AsciiHistoryEntryView]:
    views: list[_AsciiHistoryEntryView] = []
    for entry in history_entries:
        frame_view = _to_ascii_frame_view(entry.frame)
        if frame_view is None:
            continue
        views.append(_AsciiHistoryEntryView(action=entry.action, frame=frame_view))
    return views


def _ascii_frame_view_payload(frame: Frame | None) -> dict[str, Any] | None:
    view = _to_ascii_frame_view(frame)
    if view is None:
        return None
    return {
        "ascii": view.ascii,
        "step": view.step,
        "level": view.level,
        "shape": [int(view.shape[0]), int(view.shape[1])],
        "grid": [list(row) for row in frame.grid],
    }


def _ascii_history_view_payload(history_entries: list[HistoryEntry]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for entry in history_entries:
        frame_payload = _ascii_frame_view_payload(entry.frame)
        if frame_payload is None:
            continue
        payload.append({"action": entry.action, "frame": frame_payload})
    return payload


def _format_action_span(start_action_num: int | None, end_action_num: int | None) -> str | None:
    if start_action_num is None or end_action_num is None:
        return None
    if start_action_num <= 0 or end_action_num <= 0:
        return None
    if start_action_num == end_action_num:
        return f"{start_action_num}"
    return f"{start_action_num}-{end_action_num}"


def _estimate_tokens(value: Any) -> int:
    try:
        rendered = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        rendered = str(value)
    return max(1, (len(rendered) + 2) // 3)


def _host_accessible_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").strip().lower()
    if hostname != "host.docker.internal":
        return base_url
    netloc = "127.0.0.1"
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


def _resolve_analyzer_model(model: str) -> AnalyzerModelConfig:
    requested = (model or "").strip()
    lowered = requested.lower()
    if lowered in {"local", "local-qwen", "qwen-local", "qwen"}:
        configured_base_url = os.environ.get("LOCAL_ANALYZER_BASE_URL", _LOCAL_ANALYZER_BASE_URL).strip()
        if not configured_base_url:
            raise ValueError("LOCAL_ANALYZER_BASE_URL must be set for the local analyzer preset.")

        provider = os.environ.get("LOCAL_ANALYZER_PROVIDER", os.environ.get("OPENAI_PROVIDER", "vllm")).strip().lower()
        if not provider:
            provider = "vllm"
        model_id = os.environ.get("LOCAL_ANALYZER_MODEL_ID", "").strip() or _LOCAL_ANALYZER_MODEL_ID.strip()
        if not model_id:
            raise ValueError("LOCAL_ANALYZER_MODEL_ID must be set for the local analyzer preset.")
        return AnalyzerModelConfig(
            provider=provider,
            base_url=_host_accessible_base_url(configured_base_url),
            model_id=model_id,
        )

    if not requested:
        requested = _LOCAL_ANALYZER_MODEL_ID.strip()
    if not requested:
        raise ValueError(
            "Analyzer model id is required. Set analyzer.model_id in config, pass --model, "
            "or set LOCAL_ANALYZER_MODEL_ID / INFERENCE_ANALYZER_MODEL."
        )

    provider = os.environ.get("OPENAI_PROVIDER", os.environ.get("LOCAL_ANALYZER_PROVIDER", "vllm")).strip().lower()
    if not provider:
        provider = "vllm"
    base_url = _host_accessible_base_url(
        os.environ.get("OPENAI_BASE_URL", os.environ.get("LOCAL_ANALYZER_BASE_URL", _LOCAL_ANALYZER_BASE_URL)).strip()
    )
    if not base_url:
        raise ValueError("OPENAI_BASE_URL or LOCAL_ANALYZER_BASE_URL must be set for direct model ids.")
    return AnalyzerModelConfig(provider=provider, base_url=base_url, model_id=requested)


def _append_transcript_section(log_path: Path, label: str, content: str) -> None:
    rendered_content = content.strip()
    if not rendered_content:
        return
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{label}]\n")
        f.write(rendered_content)
        f.write("\n\n")


def _render_transcript_section(label: str, content: str) -> str:
    rendered_content = content.strip()
    if not rendered_content:
        return ""
    return f"[{label}]\n{rendered_content}\n\n"


def _json_like_payload(value: Any) -> Any | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        return json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _render_scalar_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True)


def _render_human_readable_lines(value: Any, *, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]
        lines: list[str] = []
        for key, item in value.items():
            key_text = str(key)
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key_text}:")
                lines.extend(_render_human_readable_lines(item, indent=indent + 2))
                continue
            if isinstance(item, str) and "\n" in item:
                multiline = item.splitlines() or [""]
                lines.append(f"{prefix}{key_text}: |")
                lines.extend(f"{prefix}  {line}" for line in multiline)
                continue
            lines.append(f"{prefix}{key_text}: {_render_scalar_value(item)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_render_human_readable_lines(item, indent=indent + 2))
                continue
            if isinstance(item, str) and "\n" in item:
                multiline = item.splitlines() or [""]
                lines.append(f"{prefix}- |")
                lines.extend(f"{prefix}  {line}" for line in multiline)
                continue
            lines.append(f"{prefix}- {_render_scalar_value(item)}")
        return lines
    if isinstance(value, str):
        if "\n" in value:
            multiline = value.splitlines() or [""]
            return [f"{prefix}|", *(f"{prefix}  {line}" for line in multiline)]
        return [f"{prefix}{value}"]
    return [f"{prefix}{_render_scalar_value(value)}"]


def _render_human_readable_value(value: Any) -> str:
    return "\n".join(_render_human_readable_lines(value))


def _render_jsonish_text(value: Any) -> str:
    parsed = _json_like_payload(value)
    if parsed is not None:
        return _render_human_readable_value(parsed)
    return _normalize_message_content(value) if not isinstance(value, str) else value.strip()


def _render_tool_parameter_text(value: Any) -> str:
    if isinstance(value, str):
        return value.rstrip("\n")
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=True)
    return str(value)


def _normalize_tool_call_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return json.loads(json.dumps(arguments))
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return {}
        if stripped.startswith("<tool_call>"):
            recovered_tool_calls = _recover_tool_calls_from_markup(stripped)
            if recovered_tool_calls:
                recovered_arguments = recovered_tool_calls[0].get("function", {}).get("arguments", "{}")
                return json.loads(str(recovered_arguments))
            return {}
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("tool call arguments must decode to a JSON object")
    raise ValueError("tool call arguments must be a JSON object or JSON object string")


def _render_tool_call_markup(tool_name: str, arguments: Any) -> str:
    name = str(tool_name or "").strip()
    if not name:
        return ""
    try:
        parsed_arguments = _normalize_tool_call_arguments(arguments)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""

    lines = ["<tool_call>", f"<function={name}>"]
    for parameter_name, parameter_value in parsed_arguments.items():
        lines.append(f"<parameter={parameter_name}>")
        rendered_value = _render_tool_parameter_text(parameter_value)
        if rendered_value:
            lines.extend(rendered_value.splitlines())
        lines.append("</parameter>")
    lines.append("</function>")
    lines.append("</tool_call>")
    return "\n".join(lines)


def _render_tool_result_display(content: Any) -> str:
    parsed = _json_like_payload(content) if isinstance(content, str) else (content if isinstance(content, dict) else None)
    if isinstance(parsed, dict):
        stdout = str(parsed.get("stdout", "") or "").rstrip("\n")
        error = str(parsed.get("error", "") or "").rstrip("\n")
        result = parsed.get("result")
        has_result = result not in (None, "", [], {})
        if stdout and not error and not has_result:
            return stdout

        blocks: list[str] = []
        if stdout:
            blocks.append(stdout)
        if has_result:
            rendered_result = _render_human_readable_value(result)
            if stdout:
                blocks.append(f"result:\n{rendered_result}")
            else:
                blocks.append(rendered_result)
        if error:
            if stdout or has_result:
                blocks.append(f"error:\n{error}")
            else:
                blocks.append(error)
        if blocks:
            return "\n\n".join(block for block in blocks if block.strip())

    return _render_jsonish_text(content)


def _resolve_run_artifact_location(state_path: Path) -> tuple[Path, str | None]:
    parent = state_path.parent
    if parent.name == "artifacts" and parent.parent != parent:
        run_root = parent.parent
        runtime_state_files = list(parent.glob(f"*_{RUNTIME_STATE_FILENAME}"))
        if len(runtime_state_files) <= 1:
            return run_root, None
        runtime_state_stem = Path(RUNTIME_STATE_FILENAME).stem
        suffix = f"_{runtime_state_stem}"
        state_stem = state_path.stem
        game_stem = state_stem[:-len(suffix)] if state_stem.endswith(suffix) else state_stem
        return run_root, game_stem
    return parent, None


def _resolve_named_run_artifact(
    state_path: Path,
    *,
    default_name: str,
    per_game_suffix: str,
    directory_name: str | None = None,
) -> Path:
    run_root, game_stem = _resolve_run_artifact_location(state_path)
    output_root = run_root / directory_name if directory_name else run_root
    if game_stem:
        return output_root / f"{game_stem}{per_game_suffix}"
    return output_root / default_name


def _render_prompt_log_message(message: dict[str, Any]) -> str:
    role = str(message.get("role", "")).strip().upper() or "UNKNOWN"
    header = f"[{role}]"
    tool_call_id = str(message.get("tool_call_id", "")).strip()
    if role == "TOOL" and tool_call_id:
        header = f"[TOOL RESULT: {tool_call_id}]"
    blocks = [header]

    content = _normalize_message_content(message.get("content", ""))
    if content:
        blocks.append(_render_tool_result_display(content) if role == "TOOL" else content)

    reasoning = _extract_reasoning_text(message)
    if reasoning:
        blocks.append("[REASONING]")
        blocks.append(reasoning)

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        for tool_call in tool_calls:
            function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            name = str(function.get("name", "")).strip() or "unknown"
            blocks.append(f"[ASSISTANT TOOL CALL: {name}]")
            tool_call_id = str(tool_call.get("id", "")).strip()
            if tool_call_id:
                blocks.append(f"id: {tool_call_id}")
            rendered_tool_call = _render_tool_call_markup(name, function.get("arguments", "{}"))
            if rendered_tool_call:
                blocks.append(rendered_tool_call)
            else:
                raw_arguments = function.get("arguments", "{}")
                try:
                    parsed_arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                    rendered_arguments = json.dumps(parsed_arguments, indent=2, ensure_ascii=True)
                except (TypeError, ValueError, json.JSONDecodeError):
                    rendered_arguments = str(raw_arguments)
                blocks.append("arguments:")
                blocks.append(rendered_arguments if rendered_arguments.strip() else "{}")

    return "\n".join(blocks)


def _resolve_prompt_log_path(state_path: Path) -> Path:
    return _resolve_named_run_artifact(
        state_path,
        default_name="prompt.log",
        per_game_suffix=".log",
        directory_name="prompts",
    )


def _resolve_request_log_path(state_path: Path) -> Path:
    return _resolve_named_run_artifact(
        state_path,
        default_name="requests.jsonl",
        per_game_suffix="_requests.jsonl",
    )


def _append_request_snapshot(
    log_path: Path,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    event: str | None = None,
    tool_choice: str | None = None,
    finish_reason: str | None = None,
    analysis_step: int | None = None,
    action: int | None = None,
    request_index_within_turn: int | None = None,
) -> None:
    payload = {
        "messages": messages,
        "tools": tools or [],
    }
    if event:
        payload["event"] = event
    if tool_choice:
        payload["tool_choice"] = tool_choice
    if finish_reason is not None:
        payload["finish_reason"] = str(finish_reason)
    if analysis_step is not None:
        payload["analysis_step"] = analysis_step
    if action is not None:
        payload["action"] = action
    if request_index_within_turn is not None:
        payload["request_index_within_turn"] = request_index_within_turn
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                payload,
                ensure_ascii=True,
            )
        )
        f.write("\n")


def _write_prompt_log_snapshot(
    log_path: Path,
    *,
    model_id: str,
    base_url: str,
    display_action_num: int,
    analysis_step: int | None,
    request_index: int,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: str | None,
    transcript: str,
) -> None:
    rendered_messages = "\n\n".join(_render_prompt_log_message(message) for message in messages)
    rendered_tools: list[str] = []
    for tool in tools or []:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = str(function.get("name", "")).strip() or "unknown"
        description = str(function.get("description", "")).strip()
        if description:
            rendered_tools.append(f"- {name}: {description}")
        else:
            rendered_tools.append(f"- {name}")
    analysis_label = str(analysis_step) if analysis_step is not None else "n/a"
    transcript_text = transcript.strip()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("LATEST MODEL CALL SNAPSHOT\n")
        f.write(f"model: {model_id}\n")
        f.write(f"base_url: {base_url}\n")
        f.write(f"analysis_step: {analysis_label}\n")
        f.write(f"action: {display_action_num}\n")
        f.write(f"request_index_within_turn: {request_index}\n")
        f.write(f"message_count: {len(messages)}\n")
        f.write(f"tool_choice: {tool_choice or '(none)'}\n")
        f.write("\n[AVAILABLE TOOLS]\n")
        f.write("\n".join(rendered_tools) if rendered_tools else "(none)")
        f.write("\n\n[MODEL INPUT]\n")
        f.write(rendered_messages.strip())
        f.write("\n\n[TURN TRANSCRIPT SO FAR]\n")
        f.write(transcript_text)
        f.write("\n")


def _normalize_message_content(content: Any) -> str:
    def _strip_think_tags(text: str) -> str:
        cleaned = _THINK_TAG_RE.sub("", text)
        cleaned = "\n".join(line for line in cleaned.splitlines() if line.strip())
        return cleaned.strip()

    if isinstance(content, str):
        return _strip_think_tags(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return _strip_think_tags("\n".join(part for part in parts if part))
    return ""


def _extract_reasoning_text(message: dict[str, Any]) -> str:
    reasoning = message.get("reasoning")
    if reasoning in (None, ""):
        reasoning = message.get("reasoning_content", "")
    return _normalize_message_content(reasoning)


def _is_context_length_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "maximum context length" in message
        or "reduce the length of the input prompt" in message
        or "parameter=input_tokens" in message
        or '"param":"input_tokens"' in message
    )


@dataclass
class _ChatCompletionResult:
    message: dict[str, Any]
    finish_reason: str = ""
    usage: dict[str, Any] | None = None


class ToolAgent:
    """Direct tool-calling analyzer compatible with OpenAI-style endpoints."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_ANALYZER_MODEL,
        timeout: Optional[float] = None,
        save_request_logs: bool = False,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        hard_noop_guard: bool | None = None,
        animation_awareness: bool | None = None,
    ) -> None:
        resolved_model = _resolve_analyzer_model(model)
        if base_url is not None or provider is not None:
            resolved_model = AnalyzerModelConfig(
                provider=str(provider or resolved_model.provider).strip() or resolved_model.provider,
                base_url=(
                    _host_accessible_base_url(str(base_url).strip())
                    if base_url is not None and str(base_url).strip()
                    else resolved_model.base_url
                ),
                model_id=resolved_model.model_id,
            )
        self._model = resolved_model
        configured_timeout = _LOCAL_ANALYZER_TIMEOUT if timeout is None else timeout
        self._timeout = None if configured_timeout is None or configured_timeout <= 0 else float(configured_timeout)
        self._api_key = str(api_key or "").strip()
        self._tool_steps = None if _LOCAL_ANALYZER_TOOL_STEPS <= 0 else max(1, _LOCAL_ANALYZER_TOOL_STEPS)
        self._python_timeout = min(30, max(1, _LOCAL_ANALYZER_TOOL_TIMEOUT))
        self._yield_seconds = None if _LOCAL_ANALYZER_YIELD_SECONDS <= 0 else float(_LOCAL_ANALYZER_YIELD_SECONDS)
        configured_max_output = _LOCAL_ANALYZER_MAX_OUTPUT
        self._max_output_tokens = None if configured_max_output <= 0 else max(1, configured_max_output)
        self._reply_reserve_tokens = self._max_output_tokens or 512
        self._tool_output_tokens = max(64, _LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS)
        self._tool_output_chars = max(256, self._tool_output_tokens * 4)
        self._save_request_logs = bool(save_request_logs)
        self._system_prompt = _build_system_prompt(
            tool_output_tokens=self._tool_output_tokens,
        )
        self._request_safety_margin_tokens = _REQUEST_SAFETY_MARGIN_TOKENS
        self._context_budget_tokens = max(
            1024,
            _LOCAL_ANALYZER_CONTEXT_WINDOW - self._reply_reserve_tokens - self._request_safety_margin_tokens,
        )
        self._history_messages: list[dict[str, Any]] = []
        self._session_runtime_dir: Path | None = None
        self._session_total_tokens = 0
        self._session_generated_tokens = 0
        self._step_env_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        self._current_valid_actions: list[str] = []
        self._last_step_summary: dict[str, Any] | None = None
        self._last_action_result: dict[str, Any] | None = None
        self._summarized_knowledge = _empty_world_model()
        # atlas: harness-triggered nags for verify_theory/plan_with_theory --
        # see _run_python_tool and _build_user_prompt. Counts python-tool
        # calls, not turns (a turn can invoke python zero or more times).
        self._atlas_python_call_index = 0
        self._atlas_last_plan_call_index = -99
        self._atlas_last_verified_accuracy: float | None = None
        # atlas: real python-tool calls since the last one that actually
        # called action() -- the force-act circuit breaker's trigger. See
        # _ATLAS_FORCE_ACT_AFTER_CALLS and _run_python_tool.
        self._atlas_calls_since_real_action = 0
        # atlas: total verify_theory( calls this game -- the goal-reconsider
        # checkpoint's trigger. See _ATLAS_GOAL_RECONSIDER_AFTER_CALLS.
        self._atlas_verify_theory_call_count = 0
        # atlas 27.08: True once a verify_theory( call has tested >= 1 real
        # transition -- the theory-force override's actual gate (see
        # _ATLAS_TRANSITIONS_TESTED_RE). Deliberately NOT the same thing as
        # verify_theory_call_count > 0: a call made right after rollback()
        # (which wipes transitions) is a real call but tests 0 transitions,
        # so it must NOT satisfy this -- found live on wa30 (Gemini-flagged
        # gate-bypass). Once True, stays True for the rest of the game (the
        # force override's job -- prove the model CAN write a real theory
        # call -- is done; it does not need to re-prove this after every
        # later rollback).
        self._atlas_verify_theory_real_ever = False
        # atlas 27.08: the python-call index the theory-force override counts
        # FROM, not game start -- reset to the current call index on every
        # rollback (see _restore_to_checkpoint) so the model gets a full
        # fresh _ATLAS_THEORY_FORCE_AFTER_CALLS-call runway to naturally earn
        # a real transition post-rollback, instead of being immediately
        # re-forced into another guaranteed-vacuous call.
        self._atlas_theory_force_eligible_from_call = 0
        # atlas: set when a turn calls plan_with_theory( and gets back a plan
        # of >1 step (res['note'] non-null) AND fires it via a SINGLE action()
        # call in that same script -- exactly the pattern that failed live
        # (ls20: a 7-step plan verified at 1.0 accuracy, executed whole,
        # didn't complete the level). Injected into the NEXT turn's prompt
        # once, then cleared -- see _run_python_tool/_build_user_prompt.
        self._atlas_note_incident: str | None = None
        # atlas: persistent scratch memory across turns, this episode only --
        # mirrors our own harness's Sandbox.memo. See python_tool_sandbox.py
        # for the round-trip (their sandbox is a fresh subprocess per turn,
        # so this can't live there; it lives here and gets threaded through).
        self._atlas_memo: dict[str, Any] = {}
        # atlas: has the model EVER written to memo this game -- the memo
        # checkpoint's trigger. See _ATLAS_MEMO_NUDGE_AFTER_CALLS.
        self._atlas_memo_ever_written = False
        # atlas 26.08: has extract= EVER been passed to verify_theory this
        # game -- the extract-suggestion checkpoint's trigger. See
        # _ATLAS_EXTRACT_NUDGE_AFTER_CALLS.
        self._atlas_extract_ever_used = False
        # atlas 27.08: which control "kinds" (MOUSE/UP/etc, ignoring MOUSE
        # coordinates) are RESOLVED this level -- either they visibly changed
        # the board at least once, or they've been tried
        # _ATLAS_EXPLORE_MAX_ATTEMPTS_PER_KIND times with no visible effect
        # (accepted as "seems inert here" rather than nagged forever). The
        # explore-first checkpoint's trigger. Independent of the checkpoint/
        # rollback feature (works in ONLINE mode too), so tracked with its
        # own level pointer rather than reusing _atlas_current_level.
        self._atlas_action_kinds_resolved: set[str] = set()
        self._atlas_action_kind_attempts: dict[str, int] = {}
        self._atlas_explore_level = 1
        # atlas 26.08: save_checkpoint/rollback state -- see
        # _HarnessGameSession.atlas_snapshot_env/atlas_restore_env for the
        # actual engine-state deepcopy (OFFLINE mode only; None in ONLINE
        # mode disables the whole feature for that session).
        self._checkpoint_env_callback: Callable[[], dict[str, Any] | None] | None = None
        self._restore_env_callback: Callable[[dict[str, Any] | None], bool] | None = None
        self._atlas_checkpoints: dict[str, dict[str, Any]] = {}
        self._atlas_checkpoint_counter = 0
        self._atlas_checkpoint_available = False  # set True on first successful save this session
        # Trigger A (soft stall): real actions since the last level-up.
        self._atlas_actions_since_level_progress = 0
        self._atlas_current_level = 1
        # Trigger B (hard loop): rolling window of board signatures after
        # each real action, to catch "back to a state seen a couple of
        # actions ago" ping-pong loops.
        self._atlas_recent_board_sigs: list[Any] = []
        # One-shot injection for the turn right after a rollback lands --
        # same "queue it, show it once, clear it" pattern as
        # _atlas_note_incident.
        self._atlas_rollback_lesson: str | None = None
        self._atlas_rollback_target_checkpoint: str | None = None
        # Freeform reason text for whichever trigger set the target above --
        # threaded into ATLAS_FORCE_ROLLBACK_CHECKPOINT's {reason} slot.
        self._atlas_rollback_trigger_reason: str | None = None
        # How many turns in a row the force-rollback ultimatum has fired
        # without the model complying -- once this crosses
        # _ATLAS_ROLLBACK_AUTO_FORCE_AFTER, the harness performs the
        # rollback itself with a generic lesson instead of nagging forever.
        self._atlas_rollback_ultimatum_streak = 0
        # Most recently created checkpoint id (auto or manual) -- what a
        # newly-fired trigger points the ultimatum at.
        self._atlas_last_checkpoint_id: str | None = None
        # Set by _build_user_prompt once the ultimatum streak caps out;
        # consumed at the top of the NEXT _run_python_tool call, which has
        # state_path and performs the actual restore there (same code path
        # as a model-initiated rollback()).
        self._atlas_pending_auto_rollback: str | None = None
        # atlas 27.08 (late): plan_real principle-force state -- same
        # two-layer design as the rollback ultimatum above. Per-level usage
        # flag + showing streak + auto-run bookkeeping; all reset on
        # level-up in _atlas_note_action_progress.
        self._atlas_plan_real_used_this_level = False
        self._atlas_plan_real_force_streak = 0
        self._atlas_plan_real_auto_done_this_level = False
        self._atlas_pending_auto_plan_real = False
        self._atlas_pending_auto_plan_real_proactive = False
        self._atlas_plan_real_auto_note: str | None = None
        self._atlas_level_solutions: dict[int, list[dict[str, Any]]] = {}
        self._atlas_current_level_actions: list[dict[str, Any]] = []
        self._atlas_auto_replay_note: str | None = None
        self._atlas_in_auto_replay = False
        self._atlas_pending_mechanic_handoff: int | None = None
        self._atlas_mechanic_handoff_note: str | None = None
        self._atlas_hail_mary_done = False
        self._atlas_time_remaining_callback: Callable[[], float] | None = None
        # atlas 27.08 (probe/checkpoint integration): python-call index of
        # the last SUCCESSFULLY EXECUTED probe (try_actions/plan_real that
        # actually ran on the engine, set host-side in _handle_checkpoint,
        # not the in-code substring tracker) -- the theory nag/force stay
        # quiet for _ATLAS_PROBE_THEORY_GRACE_CALLS calls after it.
        self._atlas_last_probe_call_index = -99
        # atlas 28.08: consecutive probe-executing python calls without a
        # real action() in between (rationing nudge trigger), and the
        # per-level compact memory of probe findings injected into the
        # prompt so learned effects are not re-probed.
        self._atlas_probes_since_real_action = 0
        self._atlas_probe_findings: list[str] = []
        # atlas 26.08: context sanitizer (idea #3) -- independent of the
        # rollback/checkpoint feature above (works in ONLINE mode too, no
        # env-snapshot needed), so tracked with its own counters rather than
        # piggybacking on _atlas_current_level/_atlas_actions_since_level_progress.
        self._atlas_calls_since_sanitize = 0
        self._atlas_context_sanitize_level = 1
        self._atlas_context_sanitize_pending = False
        self._atlas_context_sanitize_reason: str | None = None
        # Frozen at TRIGGER time (inside _handle_action), not at execution
        # time (start of the NEXT analyze() call) -- a level-up trigger fires
        # right before _run_python_tool's end-of-call bookkeeping WIPES
        # _summarized_knowledge for the level that just ended (see
        # _update_summarized_knowledge_from_step_summary), so capturing late
        # would lose exactly the content this feature exists to preserve.
        self._atlas_context_sanitize_input: dict[str, Any] | None = None
        self._atlas_context_snapshot: str | None = None
        self._atlas_context_sanitize_count = 0
        # Explicit ctor arg (e.g. from a pickled HarnessSolver deployed to
        # Kaggle) takes precedence over the local process environment, so the
        # flag state chosen at deploy time survives the trip into the kernel.
        self._hard_noop_guard_enabled = (
            _HARD_NOOP_GUARD_ENABLED if hard_noop_guard is None else bool(hard_noop_guard)
        )
        self._noop_guard: NoopGuard | None = NoopGuard() if self._hard_noop_guard_enabled else None
        self._animation_awareness_enabled = (
            _ANIMATION_AWARENESS_ENABLED if animation_awareness is None else bool(animation_awareness)
        )
        # Merged into the per-attempt experiment event by the solver session,
        # so stages 2 and 3 stay separable from stage 1 in log analysis.
        self.animation_counters: dict[str, int] = {}
        self._reset_animation_hint_state()

    def _headers(self) -> dict[str, str]:
        api_key = (
            self._api_key
            or os.environ.get("LOCAL_ANALYZER_API_KEY", "").strip()
            or os.environ.get("OPENROUTER_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
        )
        site_url = os.environ.get("LOCAL_ANALYZER_SITE_URL", "").strip()
        app_name = os.environ.get("LOCAL_ANALYZER_APP_NAME", "ARC3 Agent Harness").strip()
        return build_headers(
            provider=self._model.provider,
            api_key=api_key,
            referer=site_url,
            title=app_name,
        )

    def _ensure_session(self, state_path: Path) -> None:
        runtime_dir = state_path.parent
        if self._session_runtime_dir != runtime_dir:
            self._session_runtime_dir = runtime_dir
            self._history_messages = []
            self._session_total_tokens = 0
            self._session_generated_tokens = 0
            self._last_step_summary = None
            self._last_action_result = None
            self._summarized_knowledge = _empty_world_model()
            self._atlas_python_call_index = 0
            self._atlas_last_plan_call_index = -99
            self._atlas_last_verified_accuracy = None
            self._atlas_calls_since_real_action = 0
            self._atlas_verify_theory_call_count = 0
            self._atlas_verify_theory_real_ever = False
            self._atlas_theory_force_eligible_from_call = 0
            self._atlas_note_incident = None
            self._atlas_memo = {}
            self._atlas_memo_ever_written = False
            self._atlas_extract_ever_used = False
            self._atlas_action_kinds_resolved = set()
            self._atlas_action_kind_attempts = {}
            self._atlas_explore_level = 1
            self._atlas_checkpoints = {}
            self._atlas_checkpoint_counter = 0
            self._atlas_checkpoint_available = False
            self._atlas_actions_since_level_progress = 0
            self._atlas_current_level = 1
            self._atlas_recent_board_sigs = []
            self._atlas_rollback_lesson = None
            self._atlas_rollback_target_checkpoint = None
            self._atlas_rollback_trigger_reason = None
            self._atlas_rollback_ultimatum_streak = 0
            self._atlas_last_checkpoint_id = None
            self._atlas_pending_auto_rollback = None
            self._atlas_plan_real_used_this_level = False
            self._atlas_plan_real_force_streak = 0
            self._atlas_plan_real_auto_done_this_level = False
            self._atlas_pending_auto_plan_real = False
            self._atlas_pending_auto_plan_real_proactive = False
            self._atlas_plan_real_auto_note = None
            self._atlas_level_solutions = {}
            self._atlas_current_level_actions = []
            self._atlas_auto_replay_note = None
            self._atlas_in_auto_replay = False
            self._atlas_pending_mechanic_handoff = None
            self._atlas_mechanic_handoff_note = None
            self._atlas_hail_mary_done = False
            self._atlas_last_probe_call_index = -99
            self._atlas_probes_since_real_action = 0
            self._atlas_probe_findings = []
            self._atlas_calls_since_sanitize = 0
            self._atlas_context_sanitize_level = 1
            self._atlas_context_sanitize_pending = False
            self._atlas_context_sanitize_reason = None
            self._atlas_context_sanitize_input = None
            self._atlas_context_snapshot = None
            self._atlas_context_sanitize_count = 0
            self._noop_guard = NoopGuard() if self._hard_noop_guard_enabled else None
            self.animation_counters = {}
            self._reset_animation_hint_state()
            if self._checkpoint_env_callback is not None:
                snapshot = self._checkpoint_env_callback()
                if snapshot is not None:
                    self._atlas_checkpoint_counter += 1
                    checkpoint_id = "sys_start"
                    self._atlas_checkpoints[checkpoint_id] = {
                        "label": "game start",
                        "env_snapshot": snapshot,
                        "memo": {},
                        "level": 1,
                        "auto": True,
                    }
                    self._atlas_checkpoint_available = True
                    self._atlas_last_checkpoint_id = checkpoint_id
                    print("atlas: auto-anchor created (sys_start)", flush=True)
                    # atlas 28.08 (Gemini round 5, L1): proactive search at
                    # game start -- fires on the first python call, before
                    # the model has spent any turns on level 1.
                    if _ATLAS_PLAN_REAL_PROACTIVE and self._step_env_callback is not None:
                        self._atlas_pending_auto_plan_real = True
                        self._atlas_pending_auto_plan_real_proactive = True

    @property
    def total_tokens(self) -> int:
        return max(0, int(self._session_total_tokens))

    @property
    def generated_tokens(self) -> int:
        return max(0, int(self._session_generated_tokens))

    def _accumulate_usage_tokens(self, usage: dict[str, Any] | None) -> None:
        if not isinstance(usage, dict):
            return
        generated_token_count = 0
        for key in ("completion_tokens", "output_tokens", "generated_tokens"):
            raw_value = usage.get(key)
            try:
                generated_token_count = max(0, int(raw_value))
                break
            except (TypeError, ValueError):
                continue
        self._session_generated_tokens += generated_token_count

        total_tokens = usage.get("total_tokens")
        try:
            if total_tokens is not None:
                self._session_total_tokens += max(0, int(total_tokens))
                return
        except (TypeError, ValueError):
            pass

        token_count = 0
        for key in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens"):
            raw_value = usage.get(key)
            try:
                token_count += max(0, int(raw_value))
            except (TypeError, ValueError):
                continue
        self._session_total_tokens += token_count

    def _bump_animation_counter(self, key: str, amount: int = 1) -> None:
        self.animation_counters[key] = self.animation_counters.get(key, 0) + amount

    def _reset_animation_hint_state(self) -> None:
        self._animation_hint_level: Any = None
        self._animation_turns_without_progress = 0
        self._animation_transient_animations = 0
        # Start at the cooldown so the very first qualifying turn may hint.
        self._animation_turns_since_hint = ANIMATION_HINT_COOLDOWN_TURNS
        self._animation_hint_follow_window = 0
        self._animation_counted_action: Any = None

    def _animation_hint_line(
        self, previous_step_summary: dict[str, Any] | None, current_level: Any
    ) -> str:
        """Stage 3: suggest ``animation()`` when the agent is stuck on a level
        whose animations demonstrably hide something."""
        if not self._animation_awareness_enabled:
            return ""
        if self._animation_hint_level != current_level:
            # A new level is progress by definition; nothing carries over.
            self._animation_hint_level = current_level
            self._animation_turns_without_progress = 0
            self._animation_transient_animations = 0
            self._animation_hint_follow_window = 0
        if self._animation_hint_follow_window > 0:
            self._animation_hint_follow_window -= 1
        self._animation_turns_since_hint += 1

        summary = previous_step_summary or {}
        if summary.get("level_transition") or summary.get("run_complete"):
            self._animation_turns_without_progress = 0
            self._animation_transient_animations = 0
            return ""
        self._animation_turns_without_progress += 1

        animation = summary.get("animation") or {}
        marker = summary.get("end_action_num")
        # The same summary is re-shown on inspection-only turns, so count each
        # animation once rather than once per turn it stays visible.
        if animation and marker != self._animation_counted_action:
            self._animation_counted_action = marker
            if int(animation.get("transient_pixels") or 0) >= ANIMATION_HINT_MIN_TRANSIENT_PIXELS:
                self._animation_transient_animations += 1

        if not should_suggest_animation(
            turns_without_progress=self._animation_turns_without_progress,
            transient_animations=self._animation_transient_animations,
            turns_since_last_hint=self._animation_turns_since_hint,
        ):
            return ""
        self._animation_turns_since_hint = 0
        self._animation_hint_follow_window = ANIMATION_HINT_FOLLOW_WINDOW_TURNS
        self._bump_animation_counter("stage3_hint_emitted")
        return animation_hint_text(
            self._animation_turns_without_progress, self._animation_transient_animations
        )

    def _summarize_step_sequence(self, action_results: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not action_results:
            return None
        executed_results = [item for item in action_results if item.get("executed")]
        if not executed_results:
            return None

        total_executed = 0
        executed_actions: list[str] = []
        for item in executed_results:
            count = item.get("executed_count")
            try:
                parsed = int(count) if count is not None else 1
            except (TypeError, ValueError):
                parsed = 1
            total_executed += max(1, parsed)
            action_names = item.get("executed_actions")
            if isinstance(action_names, list):
                executed_actions.extend(str(name).strip() for name in action_names if str(name).strip())
            else:
                fallback_action = str(item.get("action_display") or "").strip()
                if fallback_action:
                    executed_actions.append(fallback_action)

        last = executed_results[-1]
        try:
            end_action_num = int(last.get("action_num"))
        except (TypeError, ValueError):
            end_action_num = None
        start_action_num = None
        if end_action_num is not None and total_executed > 0:
            start_action_num = max(1, end_action_num - total_executed + 1)

        summary = {
            "start_action_num": start_action_num,
            "end_action_num": end_action_num,
            "executed_count": total_executed,
            "executed_actions": executed_actions,
            "level": last.get("level"),
            "level_transition": any(bool(item.get("level_completed")) for item in executed_results),
            "run_complete": any(bool(item.get("run_complete")) for item in executed_results),
            "game_over": any(bool(item.get("game_over")) for item in executed_results),
            "board_changed": any(bool(item.get("board_changed")) for item in executed_results),
            "stop_reason": last.get("stop_reason"),
        }
        animation = pick_animation([item.get("animation") for item in executed_results])
        if animation is not None:
            summary["animation"] = animation
        return summary

    def _describe_last_outcome(self, summary: dict[str, Any] | None) -> str:
        if not summary:
            return ""
        span = _format_action_span(
            summary.get("start_action_num"),
            summary.get("end_action_num"),
        )
        count = summary.get("executed_count")
        prefix = "Last executed sequence"
        if span and count:
            prefix = f"Actions {span} ({count} total)"
        elif span:
            prefix = f"Action span {span}"
        elif count:
            prefix = f"Last executed sequence ({count} total)"

        level = summary.get("level")
        if summary.get("level_transition"):
            level_text = f" to level {level}" if level is not None else ""
            return f"{prefix} triggered a level transition{level_text}; re-ground on the new scene."
        if summary.get("run_complete"):
            return f"{prefix} completed the run."
        if summary.get("game_over"):
            return f"{prefix} reached GAME_OVER."

        pieces = [prefix]
        if summary.get("board_changed"):
            pieces.append("produced a board change; verify that it affected gameplay objects rather than only HUD elements.")
        else:
            pieces.append("did not show a confirmed board change; treat this as weak evidence until verified.")
        stop_reason = _normalize_summary_text(summary.get("stop_reason"))
        if stop_reason:
            pieces.append(f"stop_reason={stop_reason}.")
        return " ".join(pieces)

    def _update_summarized_knowledge_from_assistant(self, content: str) -> None:
        note = _extract_scientist_note(content)
        if not note:
            return
        for key, value in note.items():
            if value:
                self._summarized_knowledge[key] = value

    def _update_summarized_knowledge_from_step_summary(self) -> None:
        summary = self._last_step_summary
        if not summary:
            return
        if summary.get("level_transition") or summary.get("run_complete") or summary.get("game_over"):
            for key in (
                "world_model",
                "goal_model",
                "action_model",
                "recent_findings",
                "open_questions",
                "current_plan",
            ):
                self._summarized_knowledge[key] = ""

    def _summarized_knowledge_lines(self) -> list[str]:
        entries = [
            ("World model", self._summarized_knowledge.get("world_model", "")),
            ("Goal model", self._summarized_knowledge.get("goal_model", "")),
            ("Action model", self._summarized_knowledge.get("action_model", "")),
            ("Recent findings", self._summarized_knowledge.get("recent_findings", "")),
            ("Open questions", self._summarized_knowledge.get("open_questions", "")),
            ("Plan", self._summarized_knowledge.get("current_plan", "")),
            ("Cross-level notes", self._summarized_knowledge.get("cross_level_notes", "")),
        ]
        lines = [f"- {label}: {value}" for label, value in entries if value]
        if not lines:
            return []
        return [
            "Working world model carried from earlier turns:",
            *lines,
            "- Revise any item above immediately if `current_frame` or `history` contradicts it.",
        ]

    def _build_user_message(self, user_prompt: str, current_frame: Frame | None) -> dict[str, Any]:
        image_part = current_grid_image_part(current_frame)
        if image_part is None:
            return {"role": "user", "content": user_prompt}

        return {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{user_prompt}\n\nCurrent grid image:"},
                image_part,
            ],
        }


    def _build_user_prompt(
        self,
        action_num: int,
        *,
        valid_actions: list[str] | None,
        current_frame: Frame | None = None,
        history_entries: list[HistoryEntry] | None = None,
        previous_step_summary: dict[str, Any] | None = None,
    ) -> str:
        history_entries = history_entries or []
        current_step = max(current_frame.step if current_frame is not None else 0, max(0, action_num)) + 1
        current_level = current_frame.level if current_frame is not None else 1
        summary_level = None
        if previous_step_summary is not None:
            try:
                summary_level = int(previous_step_summary.get("level"))
            except (TypeError, ValueError):
                summary_level = None
        if summary_level is not None:
            current_level = max(current_level, summary_level)
        observed_max_level = max(
            [current_level, *[entry.frame.level for entry in history_entries if entry.frame is not None]],
            default=current_level,
        )
        lines: list[str] = []
        if previous_step_summary:
            count = previous_step_summary.get("executed_count")
            try:
                normalized_count = int(count) if count is not None else None
            except (TypeError, ValueError):
                normalized_count = None
            action_label = "action" if normalized_count == 1 else "actions"
            lines.append(f"The code executed {normalized_count or 0} {action_label} in the previous sequence.")
            executed_actions = previous_step_summary.get("executed_actions")
            rendered_actions: list[str] = []
            if isinstance(executed_actions, list):
                rendered_actions = [str(name).strip() for name in executed_actions if str(name).strip()]
            if rendered_actions:
                action_prefix = "Executed actions (first 10):" if len(rendered_actions) > 10 else "Executed actions:"
                lines.append(f"{action_prefix} {', '.join(rendered_actions[:10])}.")
            else:
                lines.append("Executed actions: none.")
            if previous_step_summary.get("run_complete"):
                lines.append("You have completed the run!")
            elif previous_step_summary.get("level_transition"):
                lines.append("You have progressed to a new level!")
            else:
                lines.append("You are still on the same level.")
            if previous_step_summary.get("game_over"):
                lines.append("The game is over.")
            animation_line = describe_animation(previous_step_summary.get("animation"))
            if animation_line:
                lines.append(animation_line)
        elif (current_frame is not None and current_frame.step > 0) or action_num > 0:
            lines.append("No previous action sequence was captured.")
        else:
            lines.append("No previous sequence has been executed yet.")
        hint_line = self._animation_hint_line(previous_step_summary, current_level)
        if hint_line:
            lines.append(hint_line)
        state_line = f"Current state: step {current_step}, level {current_level}"
        if observed_max_level > current_level:
            state_line += f" out of observed max level {observed_max_level} so far"
        state_line += "."
        lines.extend(
            [
                state_line,
                f"Valid actions right now: {_format_valid_action_line(valid_actions)}.",
                "Only tool: `python`. It receives `current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`, `valid_actions`, `last_action_result`, and `action(actions)`.",
                "Only letter-coded board views and lightweight metadata are exposed; raw numeric color IDs are not available.",
                "Keep tool output compact: use `current_frame.segmentation` as the primary view, and `current_frame.ascii` only for a small specific region; never print full boards.",
                "For the most recent change, compare `previous_frame` to `current_frame`, or `last_transition.before_frame` to `last_transition.after_frame`; `history[-1].frame` is the current frame, not the previous one.",
                "Use Python to inspect the evidence, refine that world model from the newest history, and search or score candidate actions or short sequences against the current goal as you currently understand it.",
                "Maintain a compact working world model of what the current level seems to contain, what actions appear to do, what the goal seems to be, what is still uncertain, and what plan currently looks best.",
                "Below you are provided with the current world model from the previous turn. The default behavior is to copy it and add or remove things based on the evidence that you gathered. BEFORE EXECUTING NEW ACTIONS YOU MUST ALWAYS GIVE THE REVISED VERSION OF THE WORLD MODEL.",
            ]
        )
        lines.append(
            "You may call `action(actions)` more than once in one Python snippet if your search or control loop needs it, "
            "but stop immediately if a result reports `game_over`, `run_complete`, `level_completed`, or `done`."
        )
        lines.extend(self._summarized_knowledge_lines())
        lines.append("end of world model. ")
        # atlas: harness-triggered nags, not static prompt furniture -- a
        # tool mentioned once in instructions and never enforced gets used in
        # ~0.2% of turns (measured on our own harness's C0 mechanism before
        # this same fix). Only one checkpoint fires per turn: verifying comes
        # before planning, so a still-unverified theory takes priority.
        if self._atlas_calls_since_real_action >= _ATLAS_FORCE_ACT_AFTER_CALLS:
            lines.append(ATLAS_FORCE_ACT_OVERRIDE.format(calls=self._atlas_calls_since_real_action))
            print(
                f"atlas: force-act override injected (action_num={action_num}, "
                f"calls_since_real_action={self._atlas_calls_since_real_action})",
                flush=True,
            )
        elif self._atlas_rollback_target_checkpoint is not None:
            self._atlas_rollback_ultimatum_streak += 1
            if self._atlas_rollback_ultimatum_streak > _ATLAS_ROLLBACK_AUTO_FORCE_AFTER:
                self._atlas_pending_auto_rollback = self._atlas_rollback_target_checkpoint
                lines.append(
                    "[atlas checkpoint] The rollback ultimatum below was shown "
                    f"{self._atlas_rollback_ultimatum_streak - 1} time(s) in a row without compliance. "
                    f"The harness will perform rollback('{self._atlas_rollback_target_checkpoint}') itself "
                    "before your next `python` call runs, with a generic note instead of your own diagnosis."
                )
                print(
                    f"atlas: rollback ultimatum auto-force scheduled (action_num={action_num}, "
                    f"target={self._atlas_rollback_target_checkpoint})",
                    flush=True,
                )
            else:
                lines.append(
                    ATLAS_FORCE_ROLLBACK_CHECKPOINT.format(
                        reason=self._atlas_rollback_trigger_reason or "Progress has stalled.",
                        checkpoint_id=self._atlas_rollback_target_checkpoint,
                        streak=self._atlas_rollback_ultimatum_streak,
                    )
                )
                print(
                    f"atlas: force-rollback checkpoint injected (action_num={action_num}, "
                    f"target={self._atlas_rollback_target_checkpoint}, "
                    f"streak={self._atlas_rollback_ultimatum_streak})",
                    flush=True,
                )
        elif (
            (self._atlas_last_verified_accuracy is None or self._atlas_last_verified_accuracy < 0.6)
            and self._atlas_python_call_index >= _ATLAS_EXPLORE_NUDGE_AFTER_CALLS
            and (set(_normalize_valid_actions(valid_actions)) - self._atlas_action_kinds_resolved)
            # atlas 28.08 (Gemini round 3): probing IS empirical
            # exploration -- while the model is actively running
            # simulations, nagging it to "explore first" is contradictory
            # prompt noise (measured: 280-296 injections per probe run).
            and (self._atlas_python_call_index - self._atlas_last_probe_call_index)
            >= _ATLAS_PROBE_THEORY_GRACE_CALLS
        ):
            # atlas 27.08: checked BEFORE goal-reconsider/theory/extract --
            # exploring what the available controls DO is more foundational
            # than any theory, goal reframe, or extract= refinement built on
            # top of that exploration. See _ATLAS_EXPLORE_NUDGE_AFTER_CALLS.
            # "Resolved" means visibly changed the board at least once, or
            # tried enough times with no effect to accept it's inert here --
            # NOT just "called once", which a single unlucky MOUSE click
            # into empty water would have wrongly satisfied.
            untried = sorted(set(_normalize_valid_actions(valid_actions)) - self._atlas_action_kinds_resolved)
            lines.append(
                ATLAS_EXPLORE_FIRST_CHECKPOINT.format(
                    valid_actions=_format_valid_action_line(valid_actions),
                    tried=len(self._atlas_action_kinds_resolved),
                    untried=", ".join(untried),
                )
            )
            print(
                f"atlas: explore-first checkpoint injected (action_num={action_num}, "
                f"untried={untried})",
                flush=True,
            )
        elif (
            self._atlas_actions_since_level_progress >= _ATLAS_PLAN_REAL_STALL_AFTER_ACTIONS
            and not self._atlas_plan_real_used_this_level
            and not self._atlas_plan_real_auto_done_this_level
            and self._atlas_rollback_target_checkpoint is None
            and self._atlas_checkpoint_available
            and self._step_env_callback is not None
        ):
            # atlas 27.08 (late): plan_real principle-force. Fires at a
            # LOWER stall threshold than the rollback trigger (search
            # constructively before giving up), after explore-first (know
            # your controls before searching over them), and never while a
            # rollback ultimatum is active. MOUSE-only games get the
            # clicks-variant text and never auto-run (the harness cannot
            # pick candidate clicks); their nag stops after
            # _ATLAS_PLAN_REAL_NAG_CAP showings instead of running forever.
            self._atlas_plan_real_force_streak += 1
            mouse_only = not (
                set(_normalize_valid_actions(valid_actions)) - {"MOUSE", "RESET"}
            )
            # atlas 28.08 (Gemini round 3): with segmentation-derived MOUSE
            # candidates the harness CAN now auto-run click games too --
            # the MOUSE-only branch keeps its clicks-hint wording (the
            # model's own candidates are better than object centers) but
            # escalates to auto-run like everyone else.
            if self._atlas_plan_real_force_streak > _ATLAS_PLAN_REAL_AUTO_FORCE_AFTER:
                self._atlas_pending_auto_plan_real = True
                self._atlas_plan_real_auto_done_this_level = True
                lines.append(
                    "[atlas checkpoint] The plan_real() directive was shown "
                    f"{self._atlas_plan_real_force_streak - 1} time(s) in a row without compliance. "
                    "The harness will run the search itself before your next `python` call executes, "
                    "with default settings, and report the result."
                )
                print(
                    f"atlas: plan_real auto-run scheduled (action_num={action_num}, "
                    f"streak={self._atlas_plan_real_force_streak - 1})",
                    flush=True,
                )
            else:
                lines.append(
                    ATLAS_PLAN_REAL_FORCE_CHECKPOINT.format(
                        stalled=self._atlas_actions_since_level_progress,
                        args_hint=ATLAS_PLAN_REAL_MOUSE_ARGS_HINT if mouse_only else "",
                        escalation=(
                            ATLAS_PLAN_REAL_ESCALATION_MOUSE if mouse_only
                            else ATLAS_PLAN_REAL_ESCALATION_DEFAULT.format(
                                streak=self._atlas_plan_real_force_streak
                            )
                        ),
                    )
                )
            if not self._atlas_pending_auto_plan_real:
                print(
                    f"atlas: plan_real force checkpoint injected (action_num={action_num}, "
                    f"streak={self._atlas_plan_real_force_streak}, mouse_only={mouse_only})",
                    flush=True,
                )
        elif (
            self._atlas_last_verified_accuracy is None
            or self._atlas_last_verified_accuracy < 0.6
        ) and self._atlas_verify_theory_call_count >= _ATLAS_GOAL_RECONSIDER_AFTER_CALLS:
            lines.append(ATLAS_GOAL_RECONSIDER_CHECKPOINT.format(calls=self._atlas_verify_theory_call_count))
            print(
                f"atlas: goal-reconsider checkpoint injected (action_num={action_num}, "
                f"verify_theory_calls={self._atlas_verify_theory_call_count})",
                flush=True,
            )
        elif (
            (self._atlas_last_verified_accuracy is None or self._atlas_last_verified_accuracy < 0.6)
            and not self._atlas_extract_ever_used
            and self._atlas_verify_theory_call_count >= _ATLAS_EXTRACT_NUDGE_AFTER_CALLS
        ):
            # atlas 26.08: checked BEFORE the generic theory checkpoint below,
            # not after -- its threshold (verify_theory_call_count) is always
            # >= python_call_index's floor for the same call count, so if it
            # were placed after theory in this chain, theory's lower
            # threshold (4) would ALWAYS already be true by the time this
            # one's condition is, making this branch unreachable dead code.
            # Same reasoning as goal-reconsider (also a bigger, more specific
            # reframe) already being checked ahead of the generic nag.
            lines.append(ATLAS_EXTRACT_CHECKPOINT.format(calls=self._atlas_verify_theory_call_count))
            print(
                f"atlas: extract-suggestion checkpoint injected (action_num={action_num}, "
                f"verify_theory_calls={self._atlas_verify_theory_call_count})",
                flush=True,
            )
        elif (
            _ATLAS_THEORY_CHECKPOINT_ENABLED
            and not self._atlas_verify_theory_real_ever
            and (self._atlas_python_call_index - self._atlas_theory_force_eligible_from_call)
            >= _ATLAS_THEORY_FORCE_AFTER_CALLS
            and (self._atlas_python_call_index - self._atlas_last_probe_call_index)
            >= _ATLAS_PROBE_THEORY_GRACE_CALLS
        ):
            # atlas 27.08: checked BEFORE the soft theory nag below, not
            # after -- its threshold (8) is always >= the soft nag's (4), so
            # placed after it the soft nag would always already be true by
            # the time this one's condition is, making this branch
            # unreachable. Mutually exclusive with goal-reconsider/extract
            # above (both require verify_theory_call_count >= 6/8; this
            # gates on verify_theory_real_ever instead of call_count, so a
            # vacuous post-rollback call -- found live on wa30 -- does not
            # satisfy it), so there's no ordering conflict with those.
            calls_since_eligible = self._atlas_python_call_index - self._atlas_theory_force_eligible_from_call
            lines.append(ATLAS_THEORY_FORCE_OVERRIDE.format(calls=calls_since_eligible))
            print(
                f"atlas: theory-force override injected (action_num={action_num}, "
                f"calls_since_eligible={calls_since_eligible})",
                flush=True,
            )
        elif (
            _ATLAS_THEORY_CHECKPOINT_ENABLED
            and (
                self._atlas_last_verified_accuracy is None
                or self._atlas_last_verified_accuracy < 0.6
            )
            # atlas 27.08 (probe/checkpoint integration): a recent probe IS
            # empirical dynamics work -- do not nag for a coded theory
            # while the model is actively asking the engine directly.
            and (self._atlas_python_call_index - self._atlas_last_probe_call_index)
            >= _ATLAS_PROBE_THEORY_GRACE_CALLS
        ) and self._atlas_python_call_index >= _ATLAS_THEORY_NAG_AFTER_CALLS:
            lines.append(ATLAS_THEORY_CHECKPOINT)
            print(f"atlas: theory checkpoint injected (action_num={action_num})", flush=True)
        elif (
            self._atlas_last_verified_accuracy is not None
            and self._atlas_last_verified_accuracy >= 0.6
            and self._atlas_python_call_index - self._atlas_last_plan_call_index
            >= _ATLAS_PLAN_FORCE_AFTER_CALLS
        ):
            # atlas 27.08: checked BEFORE the soft plan nag below -- its
            # threshold is 2x the soft nag's, same "let the soft version get
            # a real chance first" ordering as theory-force vs. the soft
            # theory nag. Gated on the ATTEMPT only (a real plan_with_theory(
            # call resets _atlas_last_plan_call_index regardless of whether a
            # plan was found), so this can never become an unreachable gate.
            calls_since_plan = self._atlas_python_call_index - self._atlas_last_plan_call_index
            lines.append(
                ATLAS_PLAN_FORCE_OVERRIDE.format(
                    acc=self._atlas_last_verified_accuracy, calls=calls_since_plan
                )
            )
            print(
                f"atlas: plan-force override injected (action_num={action_num}, "
                f"acc={self._atlas_last_verified_accuracy:.2f}, calls_since_plan={calls_since_plan})",
                flush=True,
            )
        elif (
            self._atlas_last_verified_accuracy is not None
            and self._atlas_last_verified_accuracy >= 0.6
            and self._atlas_python_call_index - self._atlas_last_plan_call_index
            >= _ATLAS_PLAN_NAG_EVERY
        ):
            lines.append(
                ATLAS_PLAN_CHECKPOINT_TEMPLATE.format(acc=self._atlas_last_verified_accuracy)
            )
            print(
                f"atlas: plan checkpoint injected (action_num={action_num}, "
                f"acc={self._atlas_last_verified_accuracy:.2f})",
                flush=True,
            )
        elif (
            not self._atlas_memo_ever_written
            and self._atlas_python_call_index >= _ATLAS_MEMO_NUDGE_AFTER_CALLS
        ):
            lines.append(ATLAS_MEMO_CHECKPOINT.format(calls=self._atlas_python_call_index))
            print(
                f"atlas: memo checkpoint injected (action_num={action_num}, "
                f"python_calls={self._atlas_python_call_index})",
                flush=True,
            )
        # atlas: one-shot reflection after a specific past incident, not an
        # ongoing-readiness nudge like the two above -- fires exactly once
        # right after the model fired a multi-step plan in a single
        # action() call, then clears itself. Independent of the if/elif
        # chain above so it can co-occur with either checkpoint.
        if self._atlas_note_incident:
            lines.append(ATLAS_NOTE_ENFORCEMENT_CHECKPOINT.format(detail=self._atlas_note_incident))
            print(f"atlas: note enforcement checkpoint injected (action_num={action_num})", flush=True)
            self._atlas_note_incident = None
        # atlas 26.08: one-shot, same pattern as the note-incident block above
        # -- the ONE thing that survives a rollback's context wipe, injected
        # as a message from the model's own past self the turn right after
        # the rollback (model-initiated or harness auto-forced) lands.
        if self._atlas_rollback_lesson:
            lines.append(
                "[rollback landed] A note from your past self before the rollback: "
                f"{self._atlas_rollback_lesson}"
            )
            print(f"atlas: rollback lesson injected (action_num={action_num})", flush=True)
            self._atlas_rollback_lesson = None
        # atlas 28.08 (Gemini round 3): persistent per-level probe memory --
        # a probe's result must outlive the turn it ran in, or the model
        # re-probes what it already knows (turn displacement tax).
        if self._atlas_probe_findings:
            lines.append(
                "Probe memory (free simulations already run this level): "
                + "; ".join(self._atlas_probe_findings[-_ATLAS_PROBE_FINDINGS_MAX:])
            )
        # atlas 28.08 (Gemini round 3): probe rationing -- knowledge only
        # scores once converted into real moves.
        if self._atlas_probes_since_real_action >= _ATLAS_PROBE_RATION_FREE:
            lines.append(
                f"[atlas] You have run {self._atlas_probes_since_real_action} probe calls in a row "
                "without advancing the real game -- try_actions is now LOCKED and will return an "
                "error until you execute a real action(...). Convert your best probed option into "
                "real moves now; probe knowledge only scores once converted."
            )
            print(
                f"atlas: probe ration nudge injected (probes_in_row={self._atlas_probes_since_real_action})",
                flush=True,
            )
        # atlas 27.08 (late): one-shot result of a harness-auto plan_real run.
        if self._atlas_plan_real_auto_note:
            lines.append(self._atlas_plan_real_auto_note)
            print(f"atlas: auto-plan_real note injected (action_num={action_num})", flush=True)
            self._atlas_plan_real_auto_note = None
        # atlas 28.08 (Gemini round 5, L2): one-shot result of an auto-replay
        # of solved levels after a game restart.
        if self._atlas_auto_replay_note:
            lines.append(self._atlas_auto_replay_note)
            print(f"atlas: auto-replay note injected (action_num={action_num})", flush=True)
            self._atlas_auto_replay_note = None
        # 29.08 (Gemini round 6, D4): one-shot mechanic-handoff diagnostic.
        if self._atlas_mechanic_handoff_note:
            lines.append(self._atlas_mechanic_handoff_note)
            print(f"atlas: mechanic-handoff note injected (action_num={action_num})", flush=True)
            self._atlas_mechanic_handoff_note = None
        action_effect_lines = _atlas_action_effect_summary(history_entries)
        if action_effect_lines:
            print(
                f"atlas: action-effect summary injected (action_num={action_num}, "
                f"{len(action_effect_lines)} line(s))",
                flush=True,
            )
        lines.extend(action_effect_lines)
        if action_num == 0:
            lines.append(
                "Ground yourself in `current_frame` before acting, but start with a compact structural summary rather than restating the full frame."
            )
        else:
            lines.append(
                "Focus on what changed most recently in `history`, update the target environment change if needed, and separate gameplay-object changes from HUD-only changes."
            )
        lines.extend(
            [
                "When ready, call `action(actions)` from inside the `python` tool with the best valid action or ordered batch selected by your code. If your code has found a reliable short sequence, prefer batching it in one call.",
                "You may call `action(actions)` more than once in one Python snippet if your search or control loop needs it.",
                "If you include assistant text before a tool call, keep it short and use it to update the world model. Helpful optional prefixes are `World model:`, `Goal model:`, `Action model:`, `Recent findings:`, `Open questions:`, `Plan:`, and `Cross-level notes:`.",
                TOOL_CALL_FORMAT_GUIDANCE,
            ]
        )
        if "MOUSE" in _normalize_valid_actions(valid_actions):
            lines.append("If you use MOUSE, include integer row and col arguments.")
        return "\n".join(lines)

    def _tools(self, state_path: Path) -> list[dict[str, Any]]:
        self._ensure_session(state_path)
        return [
            {
                "type": "function",
                "function": {
                    "name": "python",
                    "description": _PYTHON_TOOL_DESCRIPTION,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": (
                                    "Python code to run. The snippet is ephemeral and is not saved across tool calls."
                                ),
                            },
                        },
                        "required": ["code"],
                    },
                },
            }
        ]

    def _chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        request_timeout_seconds: float | None = None,
    ) -> _ChatCompletionResult:
        payload = build_chat_payload(
            provider=self._model.provider,
            model=self._model.model_id,
            messages=messages,
            max_tokens=self._max_output_tokens,
            temperature=_LOCAL_ANALYZER_TEMPERATURE,
            top_p=_LOCAL_ANALYZER_TOP_P,
            top_k=_LOCAL_ANALYZER_TOP_K,
            thinking=bool(_LOCAL_ANALYZER_ENABLE_THINKING),
            tools=tools,
            tool_choice=_request_tool_choice(tools),
            seed=_LOCAL_ANALYZER_SEED,
        )
        def post_chat(request_payload: dict[str, Any]) -> requests.Response:
            return requests.post(
                f"{self._model.base_url.rstrip('/')}/chat/completions",
                headers=self._headers(),
                json=request_payload,
                timeout=request_timeout_seconds if request_timeout_seconds is not None else self._timeout,
            )

        is_zombie = (
            _ATLAS_LLM_ZOMBIE_GATE is not None
            and self._atlas_current_level == 1
            and self._atlas_actions_since_level_progress >= _ATLAS_ZOMBIE_AFTER_ACTIONS
        )
        if _ATLAS_LLM_REQUEST_GATE is not None and is_zombie:
            with _ATLAS_LLM_ZOMBIE_GATE:
                with _ATLAS_LLM_REQUEST_GATE:
                    response = post_chat(payload)
        elif _ATLAS_LLM_REQUEST_GATE is not None:
            with _ATLAS_LLM_REQUEST_GATE:
                response = post_chat(payload)
        else:
            response = post_chat(payload)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            message = f"{exc}"
            if detail:
                message += f" | response: {detail}"
            raise requests.RequestException(message) from exc
        if getattr(response, "status_code", 200) >= 400:
            detail = response.text.strip()
            message = f"{response.status_code} Error"
            if detail:
                message += f" | response: {detail}"
            raise requests.RequestException(message)
        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            raise requests.RequestException("server returned no choices")
        choice = choices[0]
        return _ChatCompletionResult(
            message=choice.get("message", {}),
            finish_reason=str(choice.get("finish_reason", "") or ""),
            usage=payload.get("usage"),
        )

    def _trim_tool_text(self, text: str) -> tuple[str, bool]:
        if len(text) <= self._tool_output_chars:
            return text, False
        omitted = len(text) - self._tool_output_chars
        return f"{text[:self._tool_output_chars]}\n... [truncated {omitted} chars]", True

    def _summarize_planned_actions(self, value: Any) -> Any:
        if isinstance(value, dict):
            compacted = {
                key: self._summarize_planned_actions(item)
                for key, item in value.items()
            }
            planned_actions = compacted.pop("planned_actions", None)
            if isinstance(planned_actions, list):
                compacted["planned_action_count"] = len(planned_actions)
                action_result = compacted.get("action_result")
                if isinstance(action_result, dict):
                    executed_count = action_result.get("executed_count")
                    try:
                        compacted["executed_action_count"] = int(executed_count)
                    except (TypeError, ValueError):
                        compacted["executed_action_count"] = 1 if action_result.get("executed") else 0
            return compacted
        if isinstance(value, list):
            return [self._summarize_planned_actions(item) for item in value]
        return value

    def _render_tool_payload(self, payload: dict[str, Any], *, truncate_fields: tuple[str, ...] = ()) -> str:
        result = self._summarize_planned_actions(dict(payload))
        truncated = False
        for field in truncate_fields:
            value = result.get(field)
            if isinstance(value, str):
                result[field], field_truncated = self._trim_tool_text(value)
                truncated = truncated or field_truncated
        if truncated:
            result["truncated"] = True
            result["truncation_note"] = (
                f"Tool output was cut off to stay within the ~{self._tool_output_tokens}-token response budget."
            )
        return json.dumps(result, indent=2)

    def _normalize_python_actions(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, dict):
            items = [value]
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            raise TypeError(
                "action(actions) expects a string, an action object, or a list of action strings/objects."
            )
        if not items:
            raise ValueError("action(actions) requires at least one action.")

        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                action_name = item.strip()
                if not action_name:
                    raise ValueError(f"Action {index} is empty.")
                normalized.append({"action": action_name})
                continue
            if isinstance(item, dict):
                action_name = str(item.get("action", "")).strip()
                if not action_name:
                    raise ValueError(f"Action {index} is missing an `action` field.")
                entry = {"action": action_name}
                if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                    raise ValueError(f"Action {index} uses legacy MOUSE x/y fields; use row and col.")
                if "row" in item:
                    entry["row"] = item.get("row")
                if "col" in item:
                    entry["col"] = item.get("col")
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        return normalized

    def _compact_action_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        compact = {
            "executed": bool(payload.get("executed")),
            "action_num": payload.get("action_num"),
            "level": payload.get("level"),
            "score": payload.get("score"),
            "reward": payload.get("reward"),
            "state": payload.get("state"),
            "valid_actions": payload.get("valid_actions", []),
            "board_changed": bool(payload.get("board_changed")),
            "frame_count": _payload_frame_count(payload),
            "done": bool(payload.get("done")),
            "level_completed": bool(payload.get("level_completed")),
            "game_over": bool(payload.get("game_over")),
            "run_complete": bool(payload.get("run_complete")),
            "action_display": payload.get("action_display") or payload.get("action_name"),
        }
        executed_actions = payload.get("executed_actions")
        if isinstance(executed_actions, list) and executed_actions:
            compact["executed_actions"] = [str(action).strip() for action in executed_actions if str(action).strip()]
        elif compact.get("action_display"):
            compact["executed_actions"] = [str(compact["action_display"]).strip()]
        batch_size = int(payload.get("requested_count") or payload.get("executed_count") or 1)
        if batch_size > 1 or bool(payload.get("stopped_early")):
            compact["requested_count"] = payload.get("requested_count", batch_size)
            compact["executed_count"] = payload.get("executed_count", batch_size)
            compact["stopped_early"] = bool(payload.get("stopped_early"))
        if payload.get("stop_reason"):
            compact["stop_reason"] = payload.get("stop_reason")
        if payload.get("stop_detail"):
            compact["stop_detail"] = payload.get("stop_detail")
        for timing_key in ("run_elapsed_seconds", "time_remaining_seconds"):
            if timing_key in payload:
                compact[timing_key] = payload.get(timing_key)
        if payload.get("error"):
            compact["error"] = payload.get("error")
        animation = payload.get("animation")
        if self._animation_awareness_enabled and isinstance(animation, dict) and animation:
            compact["animation"] = dict(animation)
        return compact

    def _restore_to_checkpoint(self, checkpoint_id: str, lesson_learned: str) -> bool:
        """Shared restore path for both a model-initiated rollback() call and
        the harness's own auto-force backstop. Only mutates env/memo/rollback
        bookkeeping; callers refresh their own local frame/board-sig state
        (via load_runtime_state) since write_runtime_state() already ran
        inside atlas_restore_env by the time this returns True.
        """
        entry = self._atlas_checkpoints.get(checkpoint_id)
        if entry is None or self._restore_env_callback is None:
            return False
        restored = self._restore_env_callback(entry.get("env_snapshot"))
        if not restored:
            return False
        self._atlas_memo = copy.deepcopy(entry.get("memo") or {})
        self._atlas_current_level = int(entry.get("level") or self._atlas_current_level)
        # L2 (28.08): the rolled-back attempt's actions are not part of any
        # solution -- and updating _atlas_current_level here is also what
        # keeps a VOLUNTARY rollback from tripping the auto-replay's
        # fell-back-a-level detection (that path is for engine RESETs).
        self._atlas_current_level_actions = []
        self._atlas_actions_since_level_progress = 0
        self._atlas_recent_board_sigs = []
        self._atlas_rollback_lesson = lesson_learned
        self._atlas_rollback_target_checkpoint = None
        self._atlas_rollback_trigger_reason = None
        self._atlas_rollback_ultimatum_streak = 0
        # atlas 27.08: give the theory-force override a fresh runway from
        # right now -- rollback just wiped the transitions history, so an
        # immediately-forced verify_theory( call would be guaranteed vacuous
        # (found live on wa30). Only relevant while
        # _atlas_verify_theory_real_ever is still False; once the model has
        # proven it can write a real theory call once, the override never
        # fires again regardless of this counter.
        self._atlas_theory_force_eligible_from_call = self._atlas_python_call_index
        return True

    @staticmethod
    def _atlas_grid_diff_count(before: Any, after: Any) -> int:
        """Cells differing between two grids; shape mismatch counts every
        cell of the larger grid (a resize IS a change, not an error)."""
        before_rows = list(before or ())
        after_rows = list(after or ())
        count = 0
        for r in range(max(len(before_rows), len(after_rows))):
            row_b = list(before_rows[r]) if r < len(before_rows) else []
            row_a = list(after_rows[r]) if r < len(after_rows) else []
            for c in range(max(len(row_b), len(row_a))):
                cell_b = row_b[c] if c < len(row_b) else None
                cell_a = row_a[c] if c < len(row_a) else None
                if cell_b != cell_a:
                    count += 1
        return count

    def _atlas_restore_probe(self, snapshot: Any) -> None:
        """Restore inside a probe/search inner loop: ask the callback to
        skip its runtime-state disk write (solver-side probe flag, 28.08
        speed fix). Falls back to a plain restore for callbacks that don't
        take the flag (tests, older wiring). The caller's final non-probe
        restore always rewrites the file once at the end."""
        try:
            self._restore_env_callback(snapshot, probe=True)
        except TypeError:
            self._restore_env_callback(snapshot)

    def _atlas_probe_sequences(
        self, request: dict[str, Any], state_path: Path, snapshot: dict[str, Any]
    ) -> dict[str, Any]:
        """try_actions(): execute candidate sequences on the REAL engine,
        report what each one did, and let the caller's finally-restore
        rewind everything. Nothing here reaches the recorded run: game_run
        (score bookkeeping) is inside the snapshot, and the engine
        scorecard's monotonic max() means a probe that happens to complete
        a level is kept, never lost.
        """
        sequences = request.get("sequences") or []
        if not isinstance(sequences, list) or not sequences:
            return {"error": "try_actions requires at least one non-empty sequence."}
        dropped = max(0, len(sequences) - _ATLAS_PROBE_MAX_SEQUENCES)
        sequences = sequences[:_ATLAS_PROBE_MAX_SEQUENCES]
        baseline_frame, _ = load_runtime_state(state_path)
        baseline_grid = baseline_frame.grid if baseline_frame is not None else ()
        baseline_level = baseline_frame.level if baseline_frame is not None else 1
        results: list[dict[str, Any]] = []
        total_steps = 0
        for index, seq in enumerate(sequences):
            if total_steps >= _ATLAS_PROBE_MAX_TOTAL_STEPS:
                results.append({"sequence_index": index, "skipped": "probe step budget exhausted"})
                continue
            if index > 0:
                self._atlas_restore_probe(snapshot)
            payload = self._step_env_callback({"actions": list(seq), "probe": True})
            executed = int(payload.get("executed_count") or 0)
            total_steps += max(1, executed)
            # Speed fix A (28.08): the probe payload carries the resulting
            # grid; the state-file read is only the fallback for callbacks
            # that don't implement the probe flag.
            after_grid = payload.get("grid")
            after_level = payload.get("level")
            if after_grid is None:
                frame_after, _ = load_runtime_state(state_path)
                after_grid = frame_after.grid if frame_after is not None else ()
                if after_level is None and frame_after is not None:
                    after_level = frame_after.level
            # atlas 27.08 (probe/checkpoint integration): a probe teaches
            # the model what a control does just as well as a real action
            # -- credit explore-first accordingly, but only for UNIFORM
            # sequences (all actions the same kind), where the observed
            # effect is unambiguously attributable to that kind. Mixed
            # sequences credit nothing. Inert uniform probes count toward
            # the same 3-attempt inert-resolution as real actions.
            if executed:
                kinds = {
                    str(s.get("action") if isinstance(s, dict) else s).strip().upper()
                    for s in list(seq)[:executed]
                }
                if len(kinds) == 1:
                    self._atlas_note_action_kind_tried(
                        next(iter(kinds)), baseline_level, bool(payload.get("board_changed"))
                    )
            cells_changed = self._atlas_grid_diff_count(baseline_grid, after_grid or ())
            # Compact result (Gemini round 3): drop what the model already
            # knows (its own requested sequence) and every falsy flag --
            # probe output is prompt-context it pays for on every later turn.
            entry: dict[str, Any] = {
                "sequence_index": index,
                "executed_count": executed,
                "board_changed": bool(payload.get("board_changed")),
                "cells_changed_vs_start": cells_changed,
                "level_after": int(after_level) if after_level is not None else baseline_level,
                "level_completed": bool(payload.get("level_completed")),
            }
            if payload.get("error"):
                entry["error"] = payload.get("error")
            if payload.get("game_over"):
                entry["game_over"] = True
            if payload.get("run_complete"):
                entry["run_complete"] = True
            if payload.get("stop_reason"):
                entry["stop_reason"] = payload.get("stop_reason")
            results.append(entry)
            # Persist a one-line finding for the prompt-side probe memory.
            seq_label = ",".join(
                str(sp.get("action") if isinstance(sp, dict) else sp)
                + (f"({sp.get('row')},{sp.get('col')})" if isinstance(sp, dict) and "row" in sp else "")
                for sp in list(seq)[:4]
            )
            if len(seq) > 4:
                seq_label += f"..x{len(seq)}"
            outcome_label = (
                "WIN" if payload.get("run_complete")
                else "LEVEL DONE" if payload.get("level_completed")
                else "game_over" if payload.get("game_over")
                else f"{cells_changed}c changed" if cells_changed else "no effect"
            )
            self._atlas_probe_findings.append(f"{seq_label}->{outcome_label}")
            if len(self._atlas_probe_findings) > _ATLAS_PROBE_FINDINGS_MAX:
                self._atlas_probe_findings = self._atlas_probe_findings[-_ATLAS_PROBE_FINDINGS_MAX:]
        note = (
            "speculative: the engine was rewound after each sequence -- none of the above is "
            "recorded in the real run. Replay the winner with action(...) to make it count."
        )
        if dropped:
            note += f" ({dropped} sequence(s) over the {_ATLAS_PROBE_MAX_SEQUENCES}-sequence cap were dropped.)"
        print(
            f"atlas: model called try_actions( ({len(results)} sequence(s), {total_steps} engine step(s))",
            flush=True,
        )
        return {"results": results, "note": note}

    @staticmethod
    def _atlas_default_mouse_candidates(grid: Any, k: int = _ATLAS_MOUSE_AUTO_CANDIDATES) -> list[dict[str, Any]]:
        """Auto-derived candidate clicks for plan_real (Gemini round 3):
        centers of the K largest non-background segmentation objects.
        Background = any component covering >30% of the board, plus the
        single largest component. Centroid = mean of the boundary corner
        points (an approximation -- exact for rectangles, good enough for
        a candidate click). Near-duplicate centers (within 2 cells) are
        dropped. Zero model authorship needed; zero-effect candidates are
        pruned naturally by the frontier's dedup."""
        try:
            rows = [list(r) for r in (grid or ())]
            if not rows or not rows[0]:
                return []
            total = len(rows) * len(rows[0])
            nodes = (segment_layer(rows, ARC_COLOR_CHARS) or {}).get("nodes") or []
        except Exception:
            return []
        sized = [n for n in nodes if n.get("pixels")]
        if not sized:
            return []
        largest = max(n["pixels"] for n in sized)
        objects = [
            n for n in sized
            if n["pixels"] < 0.3 * total and n["pixels"] != largest and n.get("boundary")
        ]
        objects.sort(key=lambda n: -n["pixels"])
        out: list[dict[str, Any]] = []
        for n in objects:
            pts = n["boundary"]
            r = int(round(sum(p[0] for p in pts) / len(pts)))
            c = int(round(sum(p[1] for p in pts) / len(pts)))
            if any(abs(r - o["row"]) <= 2 and abs(c - o["col"]) <= 2 for o in out):
                continue
            out.append({"action": "MOUSE", "row": r, "col": c})
            if len(out) >= k:
                break
        return out

    def _atlas_search_real_plan(
        self, request: dict[str, Any], state_path: Path, snapshot: dict[str, Any]
    ) -> dict[str, Any]:
        """plan_real(): breadth-first search over the REAL engine for a
        sequence that completes the current level (or wins the game). No
        predict() involved anywhere -- the engine itself is the world
        model, and the caller's finally-restore rewinds everything the
        search touched. Dedupes on board_signature so no-op moves and
        transpositions don't blow up the frontier; hard-capped on nodes,
        depth, and wall-clock (this all runs inside the model's own 30s
        python-call window).
        """
        try:
            max_depth = max(1, min(_ATLAS_SEARCH_MAX_DEPTH, int(request.get("max_depth") or 6)))
        except (TypeError, ValueError):
            max_depth = 6
        try:
            max_nodes = max(1, min(_ATLAS_SEARCH_MAX_NODES, int(request.get("max_nodes") or 250)))
        except (TypeError, ValueError):
            max_nodes = 250
        try:
            wall_seconds = max(1.0, min(_ATLAS_SEARCH_WALL_MAX_SECONDS,
                                        float(request.get("wall_seconds") or _ATLAS_SEARCH_WALL_SECONDS)))
        except (TypeError, ValueError):
            wall_seconds = _ATLAS_SEARCH_WALL_SECONDS
        baseline_frame, _ = load_runtime_state(state_path)
        candidates = request.get("candidates")
        auto_mouse = 0
        if not candidates:
            candidates = [
                {"action": kind}
                for kind in self._current_valid_actions
                if kind not in ("MOUSE", "RESET")
            ]
            if "MOUSE" in self._current_valid_actions:
                mouse_candidates = self._atlas_default_mouse_candidates(
                    baseline_frame.grid if baseline_frame is not None else ()
                )
                auto_mouse = len(mouse_candidates)
                candidates.extend(mouse_candidates)
        if not candidates:
            return {
                "plan": None,
                "reason": (
                    "no candidate actions -- segmentation found no clickable objects either; pass "
                    "explicit click specs via plan_real(actions=[{'action': 'MOUSE', 'row': r, 'col': c}, ...])."
                ),
            }
        if auto_mouse:
            print(f"atlas: plan_real auto-derived {auto_mouse} MOUSE candidate(s) from segmentation", flush=True)
        baseline_grid = baseline_frame.grid if baseline_frame is not None else ()
        visited = {board_signature(baseline_grid)}
        search_start = time.monotonic()
        deadline = search_start + wall_seconds
        rollouts_enabled = bool(request.get("rollouts", True))
        # Speed fix D (28.08, Gemini round 4): reserve a fixed slice of the
        # wall budget for the rollout phase BEFORE the frontier can exhaust
        # it -- 15 searches lifetime had run 0 rollouts because the frontier
        # always ate the whole budget. Frontier gets 65%, rollouts get the
        # guaranteed rest (the frontier still ends early on max_nodes or
        # true state-space exhaustion, handing rollouts even more time).
        frontier_deadline = (
            search_start + wall_seconds * 0.65
            if rollouts_enabled
            else deadline
        )

        def _plan_found(attempt, payload, nodes, rollouts, found_by):
            executed = int(payload.get("executed_count") or len(attempt))
            plan = attempt[: max(1, executed)]
            self._restore_env_callback(snapshot)
            print(
                f"atlas: plan_real found a plan ({len(plan)} step(s), {nodes} node(s), "
                f"{rollouts} rollout(s), via {found_by})",
                flush=True,
            )
            return {
                "plan": plan,
                "reason": "run_complete" if payload.get("run_complete") else "level_completed",
                "found_by": found_by,
                "nodes_explored": nodes,
                "rollouts": rollouts,
                "note": (
                    "found on the REAL engine, then rewound -- the harness executes it "
                    "immediately (see the final note for the outcome)."
                ),
            }

        # Phase 1 -- novelty-guided frontier (v2): greedy best-first
        # instead of uniform BFS. Priority = how many cells the node's
        # last action changed vs its parent board -- "big movement first",
        # the cheap stand-in for a value function in a sparse-reward,
        # deterministic setting where classic MCTS backprop has nothing to
        # propagate. Dedup on board_signature keeps it complete within
        # max_depth given enough budget, so 'state_space_exhausted' still
        # means what it says.
        frontier: list = [(0, 0, [], baseline_grid)]
        tiebreak = 0
        explored_states: list = [([], baseline_grid)]
        nodes = 0
        budget_hit = False
        while frontier:
            _, _, path, parent_grid = heapq.heappop(frontier)
            for candidate in candidates:
                if nodes >= max_nodes or time.monotonic() >= frontier_deadline:
                    budget_hit = True
                    break
                nodes += 1
                self._atlas_restore_probe(snapshot)
                attempt = [*path, dict(candidate)]
                payload = self._step_env_callback({"actions": attempt, "probe": True})
                if payload.get("error"):
                    continue
                if payload.get("level_completed") or payload.get("run_complete"):
                    return _plan_found(attempt, payload, nodes, 0, "frontier")
                if int(payload.get("executed_count") or 0) < len(attempt):
                    continue
                if payload.get("game_over"):
                    continue
                # Speed fix A (28.08): grid comes back inside the probe
                # payload; the state-file read is only the fallback.
                child_grid = payload.get("grid")
                if child_grid is None:
                    frame_after, _ = load_runtime_state(state_path)
                    child_grid = frame_after.grid if frame_after is not None else ()
                sig = board_signature(child_grid)
                if sig in visited:
                    continue
                visited.add(sig)
                explored_states.append((attempt, child_grid))
                if len(attempt) < max_depth:
                    change = self._atlas_grid_diff_count(parent_grid, child_grid)
                    tiebreak += 1
                    heapq.heappush(frontier, (-change, tiebreak, attempt, child_grid))
            if budget_hit:
                break

        # Phase 2 -- Monte-Carlo deep rollouts (v2): leftover wall-clock
        # goes to random playouts up to _ATLAS_ROLLOUT_DEPTH from randomly
        # chosen already-reached states. This is the genuinely
        # Monte-Carlo part: it can stumble into solutions far deeper than
        # the systematic depth cap, and a batched step_env call stops
        # itself at level completion, so each rollout costs ONE call. A
        # rollout miss proves nothing (unlike frontier exhaustion).
        rollouts = 0
        if rollouts_enabled and explored_states and candidates:
            rng = random.Random(nodes * 1009 + len(visited))
            while rollouts < _ATLAS_MAX_ROLLOUTS and time.monotonic() < deadline:
                rollouts += 1
                start_path, _ = explored_states[rng.randrange(len(explored_states))]
                seq = [dict(rng.choice(candidates)) for _ in range(_ATLAS_ROLLOUT_DEPTH)]
                self._atlas_restore_probe(snapshot)
                attempt = [*start_path, *seq]
                payload = self._step_env_callback({"actions": attempt, "probe": True})
                if payload.get("error"):
                    continue
                if payload.get("level_completed") or payload.get("run_complete"):
                    return _plan_found(attempt, payload, nodes, rollouts, "rollout")

        reason = "budget_exhausted" if budget_hit else "state_space_exhausted"
        print(
            f"atlas: plan_real found no plan ({nodes} node(s), {len(visited) - 1} unique state(s), "
            f"{rollouts} rollout(s), {reason})",
            flush=True,
        )
        return {
            "plan": None,
            "reason": reason,
            "nodes_explored": nodes,
            "unique_states_reached": len(visited) - 1,
            "rollouts": rollouts,
            "note": (
                "no level-completing sequence found: the systematic frontier within max_depth "
                + ("hit its budget" if budget_hit else "was FULLY tried")
                + ((", and " + str(rollouts) + " random deep rollout(s) (up to depth "
                    + str(_ATLAS_ROLLOUT_DEPTH) + ") also found nothing -- though a rollout miss, "
                    "unlike frontier exhaustion, proves nothing") if rollouts else "")
                + ". A deeper/differently-parameterized retry or normal play are both fine."
            ),
        }

    def _atlas_run_mechanic_handoff(self, state_path: Path, prev_level: int) -> str:
        """D4 (Gemini round 6): on entering level N+1, speculatively run the
        exact sequence that solved level N -- one free empirical probe of
        how the rules escalated, handed to the model as ground truth. If
        the old solution happens to solve the NEW level outright, execute
        it for real (zero model turns). Never raises."""
        solution = self._atlas_level_solutions.get(prev_level)
        if (
            not solution
            or self._step_env_callback is None
            or self._checkpoint_env_callback is None
            or self._restore_env_callback is None
        ):
            return ""
        snapshot = self._checkpoint_env_callback()
        if snapshot is None:
            return ""
        baseline_frame, _ = load_runtime_state(state_path)
        baseline_grid = baseline_frame.grid if baseline_frame is not None else ()
        seq_display = ", ".join(
            str(s.get("action")) + (f"({s.get('row')},{s.get('col')})" if "row" in s else "")
            for s in solution
        )
        try:
            payload = self._step_env_callback(
                {"actions": [dict(s) for s in solution], "probe": True}
            ) or {}
        except Exception:
            payload = {}
        finally:
            self._restore_env_callback(snapshot)
        solved = bool(payload.get("level_completed") or payload.get("run_complete"))
        if solved:
            exec_payload = self._step_env_callback({"actions": [dict(s) for s in solution]}) or {}
            try:
                for step in solution[: int(exec_payload.get("executed_count") or 0) or len(solution)]:
                    self._atlas_record_level_action(step, {"executed": True})
                self._atlas_note_action_progress(exec_payload, None)
            except Exception:
                pass
            print(
                f"atlas: mechanic handoff -- level {prev_level}'s solution ALSO solves "
                f"level {prev_level + 1}, harness executed it for real",
                flush=True,
            )
            return (
                f"[HARNESS DIAGNOSTIC] The exact sequence that solved level {prev_level} "
                f"([{seq_display}]) ALSO SOLVES level {prev_level + 1} -- the harness has already "
                "executed it for real. Re-ground on the newest frame; you are on the next level."
            )
        after_grid = payload.get("grid")
        if after_grid is None:
            after_grid = ()
        cells = self._atlas_grid_diff_count(baseline_grid, after_grid) if after_grid else None
        outcome = (
            f"level NOT completed; {cells} cell(s) changed vs level entry"
            if cells is not None
            else f"level NOT completed (executed {int(payload.get('executed_count') or 0)}/{len(solution)} step(s))"
        )
        if payload.get("game_over"):
            outcome += "; ended in game_over (the old pattern is now DANGEROUS)"
        if payload.get("stop_reason"):
            outcome += f"; stop_reason={payload.get('stop_reason')}"
        print(
            f"atlas: mechanic handoff probe ran level {prev_level}'s solution on level "
            f"{prev_level + 1} ({outcome})",
            flush=True,
        )
        return (
            f"[HARNESS DIAGNOSTIC] Level {prev_level} was solved with [{seq_display}]. The harness "
            f"automatically tested that exact sequence on level {prev_level + 1} (then rewound): "
            f"{outcome}. Use this to understand how the rules escalated -- do not re-probe the "
            "same sequence."
        )

    def _atlas_maybe_hail_mary(self, state_path: Path) -> None:
        """(d) 'The Last Gasp' (Gemini round 6): if this game's wall budget
        is nearly dead while on level 2+, stop spending 10-minute model
        turns and throw one 60s/depth-12/5000-node engine search at the
        board. If the game is going to die anyway, let the CPU brute-force
        the points in the final minutes. Fires at most once per game."""
        if (
            self._atlas_hail_mary_done
            or self._atlas_time_remaining_callback is None
            or self._atlas_current_level < 2
            or self._step_env_callback is None
            or self._checkpoint_env_callback is None
        ):
            return
        try:
            remaining = float(self._atlas_time_remaining_callback())
        except Exception:
            return
        if remaining > _ATLAS_HAIL_MARY_REMAINING_S or remaining <= 0:
            return
        self._atlas_hail_mary_done = True
        snapshot = self._checkpoint_env_callback()
        if snapshot is None:
            return
        print(
            f"atlas: HAIL MARY -- {remaining:.0f}s left on level {self._atlas_current_level}, "
            "running the last-gasp search",
            flush=True,
        )
        try:
            result = self._atlas_search_real_plan(dict(_ATLAS_HAIL_MARY_BUDGET), state_path, snapshot)
        except Exception as exc:
            result = {"plan": None, "reason": f"search failed: {exc!r}"}
        finally:
            self._restore_env_callback(snapshot)
        plan = result.get("plan")
        if not plan:
            print(f"atlas: hail-mary search found nothing (reason={result.get('reason')})", flush=True)
            return
        exec_payload = self._step_env_callback({"actions": list(plan)}) or {}
        try:
            for step in list(plan)[: int(exec_payload.get("executed_count") or 0) or len(plan)]:
                self._atlas_record_level_action(step, {"executed": True})
            self._atlas_note_action_progress(exec_payload, None)
        except Exception:
            pass
        outcome = (
            "level COMPLETE" if exec_payload.get("level_completed")
            else "game WON" if exec_payload.get("run_complete")
            else f"stopped early ({exec_payload.get('stop_reason')})"
        )
        print(f"atlas: hail-mary plan executed ({len(plan)} step(s), {outcome})", flush=True)
        self._atlas_plan_real_auto_note = (
            f"[atlas autopilot] Time almost up -- the harness ran a last-gasp deep search and "
            f"executed a {len(plan)}-step engine-verified plan: {outcome}."
        )

    def _atlas_auto_plan_real(self, state_path: Path, proactive: bool = False) -> str:
        """Harness-run plan_real. Two callers: the principle-force second
        layer (after ignored directives) and -- since 28.08, Gemini round 5
        L1 -- the PROACTIVE level-entry search that fires before the model
        spends any turns on a new level. A found plan is EXECUTED
        immediately (it is engine-verified and completes a level -- there is
        no scenario where the model is better served by being handed the
        plan and asked to copy it). Returns the one-shot note for the next
        prompt; a proactive MISS returns "" (silent -- the model just plays
        normally, no noise). Never raises: any failure degrades honestly."""
        if self._step_env_callback is None or self._checkpoint_env_callback is None:
            if proactive:
                return ""
            return "[atlas] the harness tried to run plan_real() itself but the session no longer supports it."
        snapshot = self._checkpoint_env_callback()
        if snapshot is None:
            if proactive:
                return ""
            return "[atlas] the harness tried to run plan_real() itself but no engine snapshot is available."
        # D2 (Gemini round 6): progressive budget -- level 2+ carries the
        # score weight and the deeper state space, and this search is
        # engine-bound CPU that costs zero LLM turns. 40s of search beats
        # a 10-minute queued model turn.
        budget = (
            dict(_ATLAS_PROACTIVE_DEEP_BUDGET)
            if self._atlas_current_level >= 2
            else dict(_ATLAS_PROACTIVE_L1_BUDGET)
        )
        try:
            result = self._atlas_search_real_plan(budget, state_path, snapshot)
        except Exception as exc:
            result = {"plan": None, "reason": f"search failed: {exc!r}"}
        finally:
            self._restore_env_callback(snapshot)
        plan = result.get("plan")
        if not plan:
            if proactive:
                print(
                    f"atlas: proactive plan_real found no plan on level entry "
                    f"(reason={result.get('reason')}, nodes={result.get('nodes_explored')})",
                    flush=True,
                )
                return ""
            return (
                "[atlas] You ignored the plan_real() directive, so the harness ran the search itself: "
                f"NO level-completing sequence was found within depth 6 from here over the default "
                f"move set (non-MOUSE actions plus auto-derived object-center clicks) "
                f"(reason={result.get('reason')}, nodes={result.get('nodes_explored')}, "
                f"unique_states={result.get('unique_states_reached')}). That is real information: this "
                "level needs MOUSE targets, a longer sequence, or a different read of the goal -- "
                "stop repeating single-action probes that cannot work."
            )
        payload = self._step_env_callback({"actions": list(plan)}) or {}
        executed = int(payload.get("executed_count") or 0)
        # Record the executed steps as this level's solution (L2 replay),
        # then run the same level-up bookkeeping a real action would
        # (anchor for the new level, per-level resets, proactive
        # re-seeding for the NEXT level -- auto-solved levels chain).
        try:
            for step in list(plan)[: executed or len(plan)]:
                self._atlas_record_level_action(step, {"executed": True})
            self._atlas_note_action_progress(payload, None)
        except Exception:
            pass
        outcome = (
            "the level is COMPLETE" if payload.get("level_completed")
            else "the game is WON" if payload.get("run_complete")
            else f"execution stopped early (stop_reason={payload.get('stop_reason')})"
        )
        plan_display = ", ".join(
            str(s.get("action")) + (f"({s.get('row')},{s.get('col')})" if "row" in s else "")
            for s in plan
        )
        print(
            f"atlas: harness {'proactively ' if proactive else ''}auto-ran plan_real and executed the plan "
            f"({executed}/{len(plan)} step(s), outcome={outcome})",
            flush=True,
        )
        prefix = (
            "[atlas autopilot] On entering this level the harness ran plan_real() itself, "
            if proactive
            else "[atlas] You ignored the plan_real() directive, so the harness ran the search itself, "
        )
        return (
            prefix
            + f"found a verified sequence [{plan_display}] and EXECUTED it: {outcome}. Analyze that "
            "successful sequence to deduce the game's mechanics -- you will need them on the next "
            "level. Re-ground on the newest frame before acting further."
        )

    def _atlas_note_action_kind_tried(self, action_sig: str, level: int, board_changed: bool) -> None:
        """Tracks which control kinds (MOUSE/UP/etc -- MOUSE coordinates are
        stripped, only the kind matters) are RESOLVED this LEVEL, for the
        explore-first checkpoint. A kind resolves the moment it visibly
        changes the board (the model learned its real effect) -- calling it
        once with no visible effect does NOT resolve it, so a single
        unlucky MOUSE click into empty water doesn't wrongly satisfy "tried
        MOUSE". Capped at _ATLAS_EXPLORE_MAX_ATTEMPTS_PER_KIND attempts,
        after which a still-inert kind resolves anyway -- a control that
        truly does nothing here must not become an unsatisfiable gate (the
        same class of trap as the old "verified_accuracy >= 0.6" one).
        Resets on every level-up (independent level pointer, not shared
        with the rollback feature's _atlas_current_level) since a new level
        can introduce new mechanics or make a previously-irrelevant control
        matter again.
        """
        if level > self._atlas_explore_level:
            self._atlas_explore_level = level
            self._atlas_action_kinds_resolved = set()
            self._atlas_action_kind_attempts = {}
        if not action_sig:
            return
        kind = action_sig.split("(", 1)[0]
        if kind in self._atlas_action_kinds_resolved:
            return
        if board_changed:
            self._atlas_action_kinds_resolved.add(kind)
            return
        attempts = self._atlas_action_kind_attempts.get(kind, 0) + 1
        self._atlas_action_kind_attempts[kind] = attempts
        if attempts >= _ATLAS_EXPLORE_MAX_ATTEMPTS_PER_KIND:
            self._atlas_action_kinds_resolved.add(kind)

    def _atlas_record_level_action(self, spec: Any, compact_payload: dict[str, Any]) -> None:
        """L2 solution recorder (28.08): remember every REAL executed action
        of the current level attempt so a completed level's sequence can be
        replayed after a future full reset. A RESET clears the attempt (its
        actions are not part of any solution)."""
        if not compact_payload.get("executed"):
            return
        kind = str(spec.get("action") if isinstance(spec, dict) else spec).strip().upper()
        if not kind:
            return
        if kind == "RESET":
            self._atlas_current_level_actions = []
            return
        entry: dict[str, Any] = {"action": kind}
        if isinstance(spec, dict) and "row" in spec and "col" in spec:
            try:
                entry["row"] = int(spec["row"])
                entry["col"] = int(spec["col"])
            except (TypeError, ValueError):
                pass
        self._atlas_current_level_actions.append(entry)

    def _atlas_auto_replay_solved_levels(self, from_level: int, upto_level: int) -> None:
        """L2 replay (28.08): after a fall back to `from_level`, batch-replay
        the recorded solutions for from_level..upto_level-1. Each level's
        sequence is engine-history ground truth from THIS play, but the
        replay still verifies each level actually completes and stops
        honestly on the first divergence (mechanics can be stateful)."""
        self._atlas_in_auto_replay = True
        replayed: list[int] = []
        diverged: str | None = None
        try:
            for lv in range(max(1, from_level), upto_level):
                solution = self._atlas_level_solutions.get(lv)
                if not solution:
                    diverged = f"no recorded solution for level {lv}"
                    break
                payload = self._step_env_callback({"actions": [dict(s) for s in solution]}) or {}
                advanced = bool(payload.get("level_completed") or payload.get("run_complete"))
                try:
                    after_level = int(payload.get("level") or 0)
                except (TypeError, ValueError):
                    after_level = 0
                self._atlas_note_action_progress(payload, None)
                if not advanced and after_level <= lv:
                    diverged = (
                        f"level {lv} did not complete on replay "
                        f"(stop_reason={payload.get('stop_reason')}) -- mechanics may be "
                        "stateful/randomized; play it manually"
                    )
                    break
                replayed.append(lv)
                if payload.get("run_complete") or payload.get("game_over"):
                    break
        finally:
            self._atlas_in_auto_replay = False
        if replayed or diverged:
            summary = (
                f"[atlas autopilot] The game restarted from level {max(1, from_level)}. The harness "
                f"auto-replayed your OWN previously successful solutions for level(s) "
                f"{', '.join(str(x) for x in replayed) if replayed else 'none'}"
                + (f"; stopped: {diverged}" if diverged else "")
                + ". Re-ground on the newest frame -- you are past the replayed levels, do not solve them again."
            )
            self._atlas_auto_replay_note = summary
            print(
                f"atlas: auto-replayed solved level(s) {replayed or []} after a game restart"
                + (f" (stopped: {diverged})" if diverged else ""),
                flush=True,
            )

    def _atlas_note_action_progress(self, compact_payload: dict[str, Any], board_sig_after: Any) -> None:
        """Trigger A/B bookkeeping for the force-rollback checkpoint, and
        auto-anchor creation on level-up. Called after every REAL executed
        action from _handle_action (both the single- and batch-action
        paths). No-op when the checkpoint feature is unavailable (ONLINE
        mode) so this stays free of behavior change there.
        """
        if self._checkpoint_env_callback is None or not compact_payload.get("executed"):
            return
        try:
            level = int(compact_payload.get("level"))
        except (TypeError, ValueError):
            level = self._atlas_current_level
        # atlas 28.08 (Gemini round 5, L2): the game fell back to an
        # earlier level -- a full reset (RESET after game over, or a
        # deliberate restart). If the run has recorded solutions for the
        # levels between here and where we were, batch-replay them instead
        # of letting the model re-derive solved levels turn by turn.
        if (
            level < self._atlas_current_level
            and not self._atlas_in_auto_replay
            and not compact_payload.get("game_over")
        ):
            fell_from = self._atlas_current_level
            self._atlas_current_level = max(1, level)
            self._atlas_current_level_actions = []
            if _ATLAS_LEVEL_AUTO_REPLAY and self._step_env_callback is not None:
                self._atlas_auto_replay_solved_levels(self._atlas_current_level, fell_from)
            return
        if compact_payload.get("level_completed") or level > self._atlas_current_level:
            # Record the sequence that just completed the level BEFORE the
            # level pointer moves -- it is the replayable solution for the
            # level we were on.
            if self._atlas_current_level_actions:
                self._atlas_level_solutions[self._atlas_current_level] = list(
                    self._atlas_current_level_actions
                )
                # D4 (Gemini round 6): mechanics usually escalate, not
                # change -- schedule a free harness probe of THIS solution
                # on the next level's board (consumed on the next python
                # call, before the model spends any turns there).
                if _ATLAS_MECHANIC_HANDOFF:
                    self._atlas_pending_mechanic_handoff = self._atlas_current_level
            self._atlas_current_level_actions = []
            self._atlas_current_level = max(self._atlas_current_level, level)
            snapshot = self._checkpoint_env_callback()
            if snapshot is not None:
                self._atlas_checkpoint_counter += 1
                checkpoint_id = f"sys_level_{self._atlas_current_level}"
                self._atlas_checkpoints[checkpoint_id] = {
                    "label": f"start of level {self._atlas_current_level}",
                    "env_snapshot": snapshot,
                    "memo": copy.deepcopy(self._atlas_memo),
                    "level": self._atlas_current_level,
                    "auto": True,
                }
                self._atlas_checkpoint_available = True
                self._atlas_last_checkpoint_id = checkpoint_id
                print(f"atlas: auto-anchor created ({checkpoint_id})", flush=True)
            self._atlas_actions_since_level_progress = 0
            self._atlas_recent_board_sigs = []
            self._atlas_rollback_target_checkpoint = None
            self._atlas_rollback_trigger_reason = None
            self._atlas_rollback_ultimatum_streak = 0
            # atlas 27.08 (late): a new level gets a fresh plan_real
            # runway -- usage flag, showing streak, and the once-per-level
            # auto-run allowance all reset here.
            self._atlas_plan_real_used_this_level = False
            self._atlas_plan_real_force_streak = 0
            self._atlas_plan_real_auto_done_this_level = False
            self._atlas_pending_auto_plan_real = False
            self._atlas_pending_auto_plan_real_proactive = False
            self._atlas_probe_findings = []
            # atlas 28.08 (Gemini round 5, L1): proactive search on level
            # entry -- the harness searches the NEW level before the model
            # spends turns on it. Chains naturally: an auto-solved level
            # lands here again and seeds the next level's search.
            if (
                _ATLAS_PLAN_REAL_PROACTIVE
                and self._atlas_checkpoint_available
                and self._step_env_callback is not None
            ):
                self._atlas_pending_auto_plan_real = True
                self._atlas_pending_auto_plan_real_proactive = True
            return
        self._atlas_actions_since_level_progress += 1
        if board_sig_after is not None:
            self._atlas_recent_board_sigs.append(board_sig_after)
            max_len = _ATLAS_ROLLBACK_LOOP_WINDOW + 1
            if len(self._atlas_recent_board_sigs) > max_len:
                self._atlas_recent_board_sigs = self._atlas_recent_board_sigs[-max_len:]
        if self._atlas_rollback_target_checkpoint is not None or self._atlas_last_checkpoint_id is None:
            return
        trigger_reason: str | None = None
        if self._atlas_actions_since_level_progress >= _ATLAS_ROLLBACK_STALL_AFTER_CALLS:
            trigger_reason = (
                f"You have taken {self._atlas_actions_since_level_progress} real actions since the last "
                "level progress with no advancement."
            )
        elif (
            board_sig_after is not None
            and len(self._atlas_recent_board_sigs) > _ATLAS_ROLLBACK_LOOP_WINDOW
            and board_sig_after == self._atlas_recent_board_sigs[-(_ATLAS_ROLLBACK_LOOP_WINDOW + 1)]
        ):
            trigger_reason = (
                f"The board has returned to a state it was already in {_ATLAS_ROLLBACK_LOOP_WINDOW} "
                "actions ago -- a ping-pong loop."
            )
        if trigger_reason is not None:
            self._atlas_rollback_target_checkpoint = self._atlas_last_checkpoint_id
            self._atlas_rollback_trigger_reason = trigger_reason
            self._atlas_rollback_ultimatum_streak = 0
            print(
                f"atlas: rollback trigger fired ({trigger_reason!r}), "
                f"target={self._atlas_last_checkpoint_id}",
                flush=True,
            )

    def _atlas_note_context_sanitize_progress(self, compact_payload: dict[str, Any], frame: Frame | None) -> None:
        """Trigger detection for the context sanitizer (idea #3): a level-up
        OR _ATLAS_CONTEXT_SANITIZE_EVERY_CALLS real actions since the last
        sanitize. Independent of the checkpoint/rollback feature above --
        works in ONLINE mode too, since it never touches engine state, only
        the chat message history. Called after every REAL executed action
        from _handle_action, same call sites as _atlas_note_action_progress.
        """
        if not compact_payload.get("executed") or self._atlas_context_sanitize_pending:
            return
        try:
            level = int(compact_payload.get("level"))
        except (TypeError, ValueError):
            level = None
        level_up = bool(compact_payload.get("level_completed")) or (
            level is not None and level > self._atlas_context_sanitize_level
        )
        self._atlas_calls_since_sanitize += 1
        if level is not None:
            self._atlas_context_sanitize_level = max(self._atlas_context_sanitize_level, level)
        if not (level_up or self._atlas_calls_since_sanitize >= _ATLAS_CONTEXT_SANITIZE_EVERY_CALLS):
            return
        self._atlas_context_sanitize_pending = True
        self._atlas_context_sanitize_reason = "level_up" if level_up else "step_count"
        # Captured NOW, at trigger time, not when the sanitizer actually runs
        # (start of the NEXT analyze() call) -- see the field's docstring in
        # __init__ for why a level-up trigger specifically would otherwise
        # capture already-wiped knowledge.
        self._atlas_context_sanitize_input = {
            "memo": copy.deepcopy(self._atlas_memo),
            "knowledge_lines": list(self._summarized_knowledge_lines()),
            "level": frame.level if frame is not None else level,
            "step": frame.step if frame is not None else None,
            "ascii": frame.ascii if frame is not None else "",
        }
        print(
            f"atlas: context-sanitize trigger fired (reason={self._atlas_context_sanitize_reason}, "
            f"calls_since_sanitize={self._atlas_calls_since_sanitize})",
            flush=True,
        )

    def _atlas_run_context_sanitizer(self, *, analyzer_log: Path) -> None:
        """Executes the pending context sanitize (idea #3): a separate,
        tool-free LLM call synthesizes the frozen memo/world-model/board
        snapshot captured at trigger time into a compact "state of the
        world", then _history_messages -- the raw action/observation
        transcript accumulated over potentially hours of play -- is REPLACED
        with a single synthetic exchange carrying that synthesis. Called
        from analyze(), which has _chat_completion and transcript logging
        available; a failure here just skips the sanitize for this turn
        (existing history is left untouched), same fail-open pattern as the
        time-bank/retry-storm backstops elsewhere in this file.
        """
        self._atlas_context_sanitize_pending = False
        self._atlas_calls_since_sanitize = 0
        reason = self._atlas_context_sanitize_reason or "step_count"
        self._atlas_context_sanitize_reason = None
        snapshot_input = self._atlas_context_sanitize_input or {}
        self._atlas_context_sanitize_input = None
        if not self._history_messages:
            return
        sanitizer_system = (
            "You are a compact state-of-the-world synthesizer for a long-running ARC-AGI "
            "game-playing agent. You will be given the agent's persistent memo, its running "
            "world/goal/action model notes, and the current board state. Produce a SHORT, dense "
            "synthesis (a few sentences or a compact bulleted list, not a story) that captures "
            "everything a fresh agent would need to keep playing well from here: confirmed facts "
            "about the mechanic, the current goal, what has already been tried and failed (so it is "
            "not repeated), and any invariants worth remembering. Output ONLY the synthesis -- no "
            "preamble, no meta-commentary about this request."
        )
        parts: list[str] = []
        memo = snapshot_input.get("memo") or {}
        if memo:
            parts.append(f"Persistent memo: {json.dumps(memo, ensure_ascii=False)}")
        knowledge_lines = snapshot_input.get("knowledge_lines") or []
        if knowledge_lines:
            parts.append("Running world/goal/action model:\n" + "\n".join(knowledge_lines))
        level = snapshot_input.get("level")
        if level is not None:
            parts.append(f"Level: {level}, step: {snapshot_input.get('step')}")
        ascii_view = snapshot_input.get("ascii") or ""
        if ascii_view:
            parts.append(f"Current board:\n{ascii_view}")
        sanitizer_user = "\n\n".join(parts) or "(no accumulated knowledge yet -- write a minimal placeholder synthesis)"
        try:
            result = self._chat_completion(
                [
                    {"role": "system", "content": sanitizer_system},
                    {"role": "user", "content": sanitizer_user},
                ],
                tools=None,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"atlas: context sanitizer request failed ({exc}) -- keeping existing history untouched", flush=True)
            _append_transcript_section(analyzer_log, "CONTEXT SANITIZER", f"failed ({reason}): {exc}")
            return
        snapshot = str((result.message or {}).get("content", "") or "").strip()
        if not snapshot:
            print("atlas: context sanitizer returned empty content -- keeping existing history untouched", flush=True)
            return
        self._atlas_context_snapshot = snapshot
        self._atlas_context_sanitize_count += 1
        dropped = len(self._history_messages)
        self._history_messages = [
            {
                "role": "user",
                "content": (
                    f"[context sanitizer, {reason}] The raw action/observation history up to this "
                    f"point has been cleared and replaced with this synthesized state of the world:"
                    f"\n\n{snapshot}"
                ),
            },
            {"role": "assistant", "content": "Understood -- continuing from this synthesized state."},
        ]
        _append_transcript_section(
            analyzer_log,
            "CONTEXT SANITIZER",
            f"reason={reason}, dropped {dropped} history message(s)\n\n{snapshot}",
        )
        print(
            f"atlas: context sanitizer ran (reason={reason}, dropped {dropped} history message(s), "
            f"snapshot_chars={len(snapshot)})",
            flush=True,
        )

    def _run_python_tool(self, state_path: Path, arguments: dict[str, Any]) -> _ToolDispatchResult:
        self._ensure_session(state_path)
        if self._atlas_pending_auto_rollback is not None:
            target = self._atlas_pending_auto_rollback
            self._atlas_pending_auto_rollback = None
            restored = self._restore_to_checkpoint(
                target,
                "[harness auto-rollback: the rollback ultimatum was shown "
                f"{_ATLAS_ROLLBACK_AUTO_FORCE_AFTER} times in a row without compliance, so the harness "
                "performed the rollback itself with this generic note instead of your own diagnosis.]",
            )
            if restored:
                print(f"atlas: harness auto-performed rollback (target={target})", flush=True)
        # atlas 27.08 (late): harness-auto plan_real -- the second layer of
        # the plan_real principle-force, mirroring the auto-rollback block
        # above. Runs the search itself with default settings; if a plan is
        # found it is EXECUTED for real (engine-verified, strictly
        # beneficial -- it completes a level), and either way a one-shot
        # note lands in the next prompt.
        # D4 (Gemini round 6): consume a pending mechanic handoff BEFORE the
        # proactive search -- if the previous level's solution solves this
        # level outright, the search below then runs for the NEXT level.
        if self._atlas_pending_mechanic_handoff is not None:
            handoff_level = self._atlas_pending_mechanic_handoff
            self._atlas_pending_mechanic_handoff = None
            try:
                handoff_note = self._atlas_run_mechanic_handoff(state_path, handoff_level)
            except Exception:
                handoff_note = ""
            if handoff_note:
                self._atlas_mechanic_handoff_note = handoff_note
        if self._atlas_pending_auto_plan_real:
            self._atlas_pending_auto_plan_real = False
            was_proactive = self._atlas_pending_auto_plan_real_proactive
            self._atlas_pending_auto_plan_real_proactive = False
            auto_note = self._atlas_auto_plan_real(state_path, proactive=was_proactive)
            if auto_note:
                self._atlas_plan_real_auto_note = auto_note
        # (d) Hail Mary: near-death on level 2+ -> one last-gasp deep search.
        try:
            self._atlas_maybe_hail_mary(state_path)
        except Exception:
            pass
        code = str(arguments.get("code", "")).rstrip()
        if not code:
            return _ToolDispatchResult(json.dumps({"error": "python requires a non-empty `code` string."}, indent=2))
        try:
            compile(code, "<python_tool>", "exec")
        except SyntaxError as exc:
            return _ToolDispatchResult(json.dumps({"error": f"Python syntax error: {exc}"}, indent=2))

        current_frame, history_entries = load_runtime_state(state_path)
        valid_actions = list(_normalize_valid_actions(self._current_valid_actions))

        def _serialized_runtime_state(
            *,
            next_valid_actions: list[str] | None = None,
            last_action_result: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            refreshed_frame, refreshed_history = load_runtime_state(state_path)
            current_frame_payload = _ascii_frame_view_payload(refreshed_frame)
            if isinstance(next_valid_actions, list):
                sanitized_actions = [str(item).strip() for item in next_valid_actions if str(item).strip()]
            else:
                sanitized_actions = list(valid_actions)
            persisted_action_result = (
                last_action_result
                if isinstance(last_action_result, dict)
                else self._last_action_result
            )
            return {
                "current_frame": current_frame_payload,
                "history": _ascii_history_view_payload(refreshed_history),
                "valid_actions": sanitized_actions,
                "last_action_result": (
                    dict(persisted_action_result)
                    if isinstance(persisted_action_result, dict)
                    else {}
                ),
                # atlas: only read by the sandbox bootstrap's initial setup,
                # never by its per-action _refresh_state -- harmless to also
                # include it in the post-action state payload below.
                "memo": self._atlas_memo,
            }

        terminal_action_result: dict[str, Any] | None = None
        # Board signature/level *before* the next real action, so a known
        # no-op can be keyed by the exact state it was tried in. Starts at the
        # pre-call frame and advances after each executed action.
        noop_guard_board_sig = board_signature(current_frame.grid) if current_frame is not None else board_signature(())
        noop_guard_level = current_frame.level if current_frame is not None else 1

        def _handle_action(actions: list[dict[str, Any]]) -> dict[str, Any]:
            nonlocal terminal_action_result, noop_guard_board_sig, noop_guard_level
            if self._step_env_callback is None:
                raise RuntimeError("action(actions) is not available in this session.")
            normalized_actions = self._normalize_python_actions(actions)
            if terminal_action_result is not None:
                reason = _terminal_action_reason(terminal_action_result) or "terminal_state"
                compact_payload = {
                    "executed": False,
                    "action_num": terminal_action_result.get("action_num"),
                    "level": terminal_action_result.get("level"),
                    "score": terminal_action_result.get("score"),
                    "reward": 0.0,
                    "state": terminal_action_result.get("state"),
                    "valid_actions": [],
                    "board_changed": False,
                    "done": bool(terminal_action_result.get("done")),
                    "level_completed": bool(terminal_action_result.get("level_completed")),
                    "game_over": bool(terminal_action_result.get("game_over")),
                    "run_complete": bool(terminal_action_result.get("run_complete")),
                    "requested_count": len(normalized_actions),
                    "executed_count": 0,
                    "stopped_early": True,
                    "stop_reason": f"previous_{reason}",
                    "stop_detail": _terminal_action_stop_detail(reason),
                }
                self._last_action_result = dict(compact_payload)
                return {
                    "action_result": compact_payload,
                    "state": _serialized_runtime_state(
                        next_valid_actions=[],
                        last_action_result=compact_payload,
                    ),
                }
            if len(normalized_actions) == 1:
                pending_action_sig = _pending_action_signature(normalized_actions[0])
                if (
                    self._noop_guard is not None
                    and pending_action_sig
                    and self._noop_guard.is_known_noop(noop_guard_level, noop_guard_board_sig, pending_action_sig)
                ):
                    last_result = self._last_action_result or {}
                    compact_payload = {
                        "executed": False,
                        "action_num": last_result.get("action_num"),
                        "level": noop_guard_level,
                        "score": last_result.get("score"),
                        "reward": 0.0,
                        "state": last_result.get("state"),
                        "valid_actions": list(self._current_valid_actions),
                        "board_changed": False,
                        "done": False,
                        "level_completed": False,
                        "game_over": False,
                        "run_complete": False,
                        "requested_count": 1,
                        "executed_count": 0,
                        "stopped_early": True,
                        "stop_reason": "known_noop",
                        "stop_detail": (
                            f"{pending_action_sig} already had no effect in this exact board state; "
                            "blocked before execution, no action budget spent."
                        ),
                    }
                    self._last_action_result = dict(compact_payload)
                    return {
                        "action_result": compact_payload,
                        "state": _serialized_runtime_state(
                            next_valid_actions=list(self._current_valid_actions),
                            last_action_result=compact_payload,
                        ),
                    }
                raw_payload = self._step_env_callback({"actions": normalized_actions})
                if not isinstance(raw_payload, dict):
                    raise RuntimeError("action(actions) did not return a JSON-like payload.")
                compact_payload = self._compact_action_result(raw_payload)
                next_valid_actions = raw_payload.get("valid_actions")
                if isinstance(next_valid_actions, list):
                    self._current_valid_actions = _normalize_valid_actions(next_valid_actions)
                if compact_payload.get("executed") and _terminal_action_reason(compact_payload):
                    terminal_action_result = compact_payload
                self._last_action_result = dict(compact_payload)
                if self._noop_guard is not None and pending_action_sig and compact_payload.get("executed"):
                    self._noop_guard.observe(
                        level=noop_guard_level,
                        board_before_sig=noop_guard_board_sig,
                        action_sig=pending_action_sig,
                        board_changed=bool(compact_payload.get("board_changed")),
                        animated=_action_animated(compact_payload),
                    )
                refreshed_frame, _ = load_runtime_state(state_path)
                if refreshed_frame is not None:
                    noop_guard_board_sig = board_signature(refreshed_frame.grid)
                    noop_guard_level = refreshed_frame.level
                self._atlas_record_level_action(normalized_actions[0], compact_payload)
                self._atlas_note_action_progress(compact_payload, noop_guard_board_sig)
                self._atlas_note_context_sanitize_progress(compact_payload, refreshed_frame)
                if compact_payload.get("executed"):
                    self._atlas_note_action_kind_tried(
                        pending_action_sig, noop_guard_level, bool(compact_payload.get("board_changed"))
                    )
                return {
                    "action_result": compact_payload,
                    "state": _serialized_runtime_state(
                        next_valid_actions=next_valid_actions if isinstance(next_valid_actions, list) else None,
                        last_action_result=compact_payload,
                    ),
                }

            # Batch of >1 actions: walk them one at a time (each as its own
            # single-action step_env call) instead of forwarding the whole
            # batch atomically, so the guard can block individually known
            # no-ops inside a batch rather than only ever seeing it as a
            # whole. This costs no extra IPC round-trip to the sandbox (the
            # model still made one action(...) call) and no extra env work --
            # step_env already executes and persists state per action
            # internally even for an atomic batch call.
            executed_results: list[dict[str, Any]] = []
            blocked_actions: list[str] = []
            last_failed: dict[str, Any] | None = None
            for action in normalized_actions:
                action_sig = _pending_action_signature(action)
                if (
                    self._noop_guard is not None
                    and action_sig
                    and self._noop_guard.is_known_noop(noop_guard_level, noop_guard_board_sig, action_sig)
                ):
                    blocked_actions.append(action_sig)
                    continue
                raw_payload = self._step_env_callback({"actions": [action]})
                if not isinstance(raw_payload, dict):
                    raise RuntimeError("action(actions) did not return a JSON-like payload.")
                sub_compact = self._compact_action_result(raw_payload)
                next_valid_actions = raw_payload.get("valid_actions")
                if isinstance(next_valid_actions, list):
                    self._current_valid_actions = _normalize_valid_actions(next_valid_actions)
                if not sub_compact.get("executed"):
                    last_failed = sub_compact
                    break
                executed_results.append(sub_compact)
                if self._noop_guard is not None and action_sig:
                    self._noop_guard.observe(
                        level=noop_guard_level,
                        board_before_sig=noop_guard_board_sig,
                        action_sig=action_sig,
                        board_changed=bool(sub_compact.get("board_changed")),
                        animated=_action_animated(sub_compact),
                    )
                refreshed_frame, _ = load_runtime_state(state_path)
                if refreshed_frame is not None:
                    noop_guard_board_sig = board_signature(refreshed_frame.grid)
                    noop_guard_level = refreshed_frame.level
                self._atlas_record_level_action(action, sub_compact)
                self._atlas_note_action_progress(sub_compact, noop_guard_board_sig)
                self._atlas_note_context_sanitize_progress(sub_compact, refreshed_frame)
                self._atlas_note_action_kind_tried(
                    action_sig, noop_guard_level, bool(sub_compact.get("board_changed"))
                )
                if _terminal_action_reason(sub_compact):
                    terminal_action_result = sub_compact
                    break

            compact_payload = _aggregate_action_batch_result(
                requested_count=len(normalized_actions),
                executed_results=executed_results,
                blocked_actions=blocked_actions,
                last_failed=last_failed,
                valid_actions=list(self._current_valid_actions),
                fallback=self._last_action_result or {},
            )
            self._last_action_result = dict(compact_payload)
            return {
                "action_result": compact_payload,
                "state": _serialized_runtime_state(
                    next_valid_actions=list(self._current_valid_actions),
                    last_action_result=compact_payload,
                ),
            }

        def _handle_animation(request: dict[str, Any]) -> dict[str, Any]:
            if self._step_env_callback is None:
                return {"error": "animation() is not available in this session."}
            self._bump_animation_counter("stage2_animation_requests")
            if self._animation_hint_follow_window > 0:
                # Retrieval within the window after a stage-3 hint: count it as
                # the hint being followed, not as the model's own initiative.
                self._bump_animation_counter("stage3_hint_followed")
                self._animation_hint_follow_window = 0
            else:
                self._bump_animation_counter("stage2_animation_requests_unprompted")
            raw = self._step_env_callback(
                {"query": "animation", "action_num": request.get("action_num")}
            )
            record = raw.get("record") if isinstance(raw, dict) else None
            view = build_animation_view(
                record if isinstance(record, dict) else None,
                frame=request.get("frame"),
                region=request.get("region"),
            )
            if not view.get("error"):
                self._bump_animation_counter("stage2_animation_requests_served")
            return view

        def _handle_checkpoint(request: dict[str, Any]) -> dict[str, Any]:
            nonlocal noop_guard_board_sig, noop_guard_level, terminal_action_result
            if self._checkpoint_env_callback is None or self._restore_env_callback is None:
                return {"error": "save_checkpoint/rollback are not available in this session (no undo in ONLINE mode)."}
            action = str(request.get("action") or "").strip()
            if action == "save":
                label = str(request.get("label") or "").strip() or "checkpoint"
                snapshot = self._checkpoint_env_callback()
                if snapshot is None:
                    return {"error": "save_checkpoint failed: environment snapshot unavailable."}
                self._atlas_checkpoint_counter += 1
                checkpoint_id = f"cp{self._atlas_checkpoint_counter}"
                request_memo = request.get("memo")
                self._atlas_checkpoints[checkpoint_id] = {
                    "label": label,
                    "env_snapshot": snapshot,
                    "memo": copy.deepcopy(request_memo) if isinstance(request_memo, dict) else copy.deepcopy(self._atlas_memo),
                    "level": noop_guard_level,
                    "auto": False,
                }
                self._atlas_checkpoint_available = True
                self._atlas_last_checkpoint_id = checkpoint_id
                print(f"atlas: model called save_checkpoint( (id={checkpoint_id}, label={label!r})", flush=True)
                return {"checkpoint_id": checkpoint_id}
            if action in ("probe_sequences", "plan_real"):
                # atlas 27.08: real-engine speculative execution -- reuses
                # the checkpoint request channel (no wire-protocol change)
                # and the exact snapshot/restore pair rollback runs on. The
                # finally-restore is unconditional: whatever the probe or
                # search did to the live engine, the game continues from
                # exactly where it was, and game_run's score bookkeeping
                # (inside the snapshot) never saw any of it.
                if self._step_env_callback is None:
                    return {"error": "try_actions/plan_real are not available: no action executor in this session."}
                # atlas 28.08 (Gemini round 4, probe HARD GATE): the soft
                # ration nudge was ignored in the tail (live streaks of 8
                # consecutive probes). A nudge is a request; a blocked tool
                # is physics. 3 probe calls in a row without a real action
                # -> try_actions stops returning results until a real
                # action() runs (the reset already lives in the real-action
                # bookkeeping). plan_real stays un-gated: it is the
                # harness's own search and fires rarely.
                if (
                    action == "probe_sequences"
                    and self._atlas_probes_since_real_action >= _ATLAS_PROBE_RATION_FREE
                ):
                    print(
                        f"atlas: probe hard gate blocked try_actions "
                        f"(probes_in_row={self._atlas_probes_since_real_action})",
                        flush=True,
                    )
                    return {
                        "error": (
                            f"Probe budget exhausted: you have run "
                            f"{self._atlas_probes_since_real_action} probe calls in a row. "
                            "try_actions is LOCKED until you execute a real action(...) -- "
                            "convert your best probed option into real moves now; the lock "
                            "lifts on the next real action."
                        )
                    }
                snapshot = self._checkpoint_env_callback()
                if snapshot is None:
                    return {
                        "error": (
                            "try_actions/plan_real are not available in this session "
                            "(no engine snapshot in ONLINE mode)."
                        )
                    }
                try:
                    if action == "probe_sequences":
                        probe_payload = self._atlas_probe_sequences(request, state_path, snapshot)
                    else:
                        probe_payload = self._atlas_search_real_plan(request, state_path, snapshot)
                finally:
                    self._restore_env_callback(snapshot)
                    refreshed_frame, _ = load_runtime_state(state_path)
                    if refreshed_frame is not None:
                        noop_guard_board_sig = board_signature(refreshed_frame.grid)
                        noop_guard_level = refreshed_frame.level
                if not probe_payload.get("error"):
                    # atlas 27.08 (probe/checkpoint integration): a probe
                    # that actually EXECUTED is empirical theory work --
                    # opens the theory-nag grace window. Set host-side (not
                    # in the in-code substring tracker) so a failed/
                    # unavailable call never earns the grace.
                    self._atlas_last_probe_call_index = self._atlas_python_call_index
                    self._atlas_probes_since_real_action += 1
                if (
                    action == "plan_real"
                    and not probe_payload.get("error")
                    and isinstance(probe_payload.get("plan"), list)
                    and probe_payload.get("plan")
                ):
                    # atlas 28.08 (Gemini round 5, the missing lever): a
                    # found plan is engine-VERIFIED to complete the level.
                    # Handing it back and hoping the model retypes it as
                    # action([...]) next turn costs a turn and risks a typo
                    # -- the harness executes it on the spot instead. Zero
                    # extra model turns between reasoning and score.
                    plan_steps = list(probe_payload["plan"])
                    exec_payload = self._step_env_callback({"actions": plan_steps}) or {}
                    exec_count = int(exec_payload.get("executed_count") or 0)
                    try:
                        for step in plan_steps[: exec_count or len(plan_steps)]:
                            self._atlas_record_level_action(step, {"executed": True})
                        self._atlas_note_action_progress(exec_payload, None)
                    except Exception:
                        pass
                    exec_outcome = (
                        "the level is COMPLETE" if exec_payload.get("level_completed")
                        else "the game is WON" if exec_payload.get("run_complete")
                        else f"execution stopped early (stop_reason={exec_payload.get('stop_reason')})"
                    )
                    probe_payload["executed_by_harness"] = True
                    probe_payload["execution_outcome"] = exec_outcome
                    probe_payload["note"] = (
                        f"[Harness Auto-Action] plan_real found a {len(plan_steps)}-step engine-verified "
                        f"solution and the harness EXECUTED it for real ({exec_count}/{len(plan_steps)} "
                        f"step(s); {exec_outcome}). Do NOT replay it. Analyze the executed sequence in "
                        "res['plan'] to deduce the game's mechanics -- you will need them on the next level."
                    )
                    self._atlas_probes_since_real_action = 0
                    self._atlas_calls_since_real_action = 0
                    refreshed_frame, _ = load_runtime_state(state_path)
                    if refreshed_frame is not None:
                        noop_guard_board_sig = board_signature(refreshed_frame.grid)
                        noop_guard_level = refreshed_frame.level
                    print(
                        f"atlas: harness auto-executed the model-found plan_real plan "
                        f"({exec_count}/{len(plan_steps)} step(s), outcome={exec_outcome})",
                        flush=True,
                    )
                probe_payload["state"] = _serialized_runtime_state(
                    next_valid_actions=list(self._current_valid_actions)
                )
                return probe_payload
            if action == "rollback":
                checkpoint_id = str(request.get("checkpoint_id") or "").strip()
                lesson_learned = str(request.get("lesson_learned") or "").strip()
                if checkpoint_id not in self._atlas_checkpoints:
                    return {
                        "error": (
                            f"Unknown checkpoint_id {checkpoint_id!r}. Use an id returned by save_checkpoint(...) "
                            "or one of the announced auto-anchors (sys_start, sys_level_N)."
                        )
                    }
                if not lesson_learned:
                    return {
                        "error": (
                            "rollback(checkpoint_id, lesson_learned) requires a non-empty lesson_learned -- "
                            "state what specifically will be done differently this time."
                        )
                    }
                if not self._restore_to_checkpoint(checkpoint_id, lesson_learned):
                    return {"error": "rollback failed: environment restore was rejected."}
                refreshed_frame, _ = load_runtime_state(state_path)
                if refreshed_frame is not None:
                    noop_guard_board_sig = board_signature(refreshed_frame.grid)
                    noop_guard_level = refreshed_frame.level
                terminal_action_result = None
                print(f"atlas: model called rollback( to {checkpoint_id!r}, lesson={lesson_learned!r})", flush=True)
                return {
                    "state": _serialized_runtime_state(next_valid_actions=list(self._current_valid_actions)),
                    # atlas: the sandbox's rollback() applies this to its OWN
                    # local `memo` global (unlike a normal action-result reply,
                    # which deliberately leaves memo alone) -- without this,
                    # the subprocess keeps its stale pre-rollback memo for the
                    # rest of the script, which then overwrites this correct
                    # restore right back to the stale value when the script
                    # ends and self._atlas_memo is synced from sandbox_result.
                    "memo": copy.deepcopy(self._atlas_memo),
                }
            return {"error": f"Unknown checkpoint action {action!r}; expected 'save' or 'rollback'."}

        sandbox_result = run_sandboxed_python(
            code=code,
            timeout_seconds=self._python_timeout,
            initial_state=_serialized_runtime_state(),
            action_handler=_handle_action,
            animation_handler=_handle_animation if self._animation_awareness_enabled else None,
            checkpoint_handler=_handle_checkpoint if self._checkpoint_env_callback is not None else None,
        )

        action_results = [
            item
            for item in sandbox_result.get("action_results") or []
            if isinstance(item, dict)
        ]
        payload: dict[str, Any] = {"tool": "python"}
        rendered_stdout = str(sandbox_result.get("stdout", "") or "")
        rendered_error = str(sandbox_result.get("error", "") or "")

        # atlas: round-trip memo. Only overwrite on a dict reply -- a
        # timed-out or crashed-before-reply subprocess sends no memo at all,
        # and the previous turns' accumulated memo must survive that, not
        # get silently wiped.
        returned_memo = sandbox_result.get("memo")
        if isinstance(returned_memo, dict):
            if returned_memo != self._atlas_memo:
                self._atlas_memo_ever_written = True
                print(
                    f"atlas: model wrote to memo (call #{self._atlas_python_call_index + 1}, "
                    f"keys={sorted(returned_memo.keys())})",
                    flush=True,
                )
            self._atlas_memo = returned_memo

        # atlas: track verify_theory/plan_with_theory usage for the
        # checkpoint nags in _build_user_prompt. Best-effort accuracy
        # extraction: prefer a structured `result` dict (the model did
        # `result = verify_theory(...)`), else regex the stdout repr (the
        # model printed the dict, or a plan_with_theory result that also
        # carries 'verified_accuracy'... 'accuracy' covers both keys' tails).
        self._atlas_python_call_index += 1
        if action_results:
            self._atlas_calls_since_real_action = 0
            self._atlas_probes_since_real_action = 0
        else:
            self._atlas_calls_since_real_action += 1
        if "plan_with_theory(" in code:
            self._atlas_last_plan_call_index = self._atlas_python_call_index
            print(f"atlas: model called plan_with_theory( (call #{self._atlas_python_call_index})", flush=True)
            # atlas: did this SAME script also fire a >1-step plan in one
            # action() call? A multi-step plan is only as reliable as
            # verify_theory's single-step checks (see res['note']) -- firing
            # it whole, with no board_changed check in between, is exactly
            # the pattern that failed live (ls20). action_results already
            # reflects every action() call made in this script; each item's
            # requested_count is the batch size of ONE such call.
            note_result = sandbox_result.get("result")
            note_present = isinstance(note_result, dict) and bool(note_result.get("note"))
            if not note_present:
                note_present = bool(_ATLAS_NOTE_PRESENT_RE.search(rendered_stdout))
            batch_size = max(
                (int(item.get("requested_count") or 0) for item in action_results),
                default=0,
            )
            if note_present and batch_size > 1:
                self._atlas_note_incident = (
                    f"plan_with_theory( returned a plan with more than one step (res['note'] was set) "
                    f"and it was executed via a single action() call requesting {batch_size} steps at once, "
                    "with no board_changed check in between."
                )
                print(
                    f"atlas: multi-step plan fired in one action() call (call #{self._atlas_python_call_index}, "
                    f"{batch_size} steps) -- note enforcement checkpoint queued for next turn",
                    flush=True,
                )
        if "execute_plan(" in code:
            print(f"atlas: model called execute_plan( (call #{self._atlas_python_call_index})", flush=True)
        if "plan_real(" in code or "try_actions(" in code:
            # atlas 27.08 (late): real-engine probe usage -- satisfies the
            # plan_real principle-force for this level and resets its streak.
            self._atlas_plan_real_used_this_level = True
            self._atlas_plan_real_force_streak = 0
            print(
                f"atlas: model called plan_real/try_actions (call #{self._atlas_python_call_index})",
                flush=True,
            )
        if "verify_theory(" in code:
            self._atlas_verify_theory_call_count += 1
            sandbox_payload_result = sandbox_result.get("result")
            accuracy = None
            transitions_tested = None
            if isinstance(sandbox_payload_result, dict):
                if isinstance(sandbox_payload_result.get("accuracy"), (int, float)):
                    accuracy = float(sandbox_payload_result["accuracy"])
                if isinstance(sandbox_payload_result.get("transitions_tested"), (int, float)):
                    transitions_tested = int(sandbox_payload_result["transitions_tested"])
            if accuracy is None:
                match = _ATLAS_ACCURACY_RE.search(rendered_stdout)
                if match:
                    try:
                        accuracy = float(match.group(1))
                    except ValueError:
                        accuracy = None
            if transitions_tested is None:
                match = _ATLAS_TRANSITIONS_TESTED_RE.search(rendered_stdout)
                if match:
                    try:
                        transitions_tested = int(match.group(1))
                    except ValueError:
                        transitions_tested = None
            if transitions_tested is not None and transitions_tested >= 1:
                self._atlas_verify_theory_real_ever = True
            print(
                f"atlas: model called verify_theory( (call #{self._atlas_python_call_index}, "
                f"parsed accuracy={accuracy}, transitions_tested={transitions_tested})",
                flush=True,
            )
            if accuracy is not None:
                self._atlas_last_verified_accuracy = accuracy
            if "extract=" in code or "extract =" in code:
                self._atlas_extract_ever_used = True
        if rendered_error:
            payload["error"] = rendered_error
            if rendered_stdout:
                payload["stdout"] = rendered_stdout
        else:
            payload["returncode"] = 0
            if rendered_stdout:
                payload["stdout"] = rendered_stdout
            elif sandbox_result.get("result") is not None:
                payload["result"] = sandbox_result.get("result")
            elif action_results:
                if len(action_results) == 1:
                    payload["result"] = action_results[-1]
                else:
                    payload["result"] = {
                        "action_calls": len(action_results),
                        "last_action_result": action_results[-1],
                    }

        step_executed = any(bool(item.get("executed")) for item in action_results)
        if step_executed:
            self._last_step_summary = self._summarize_step_sequence(action_results)
            self._update_summarized_knowledge_from_step_summary()
        return _ToolDispatchResult(
            self._render_tool_payload(payload, truncate_fields=("stdout", "error", "result")),
            step_executed=step_executed,
        )

    def _dispatch_tool(self, state_path: Path, name: str, arguments: dict[str, Any]) -> _ToolDispatchResult:
        self._ensure_session(state_path)
        if name == "python":
            return self._run_python_tool(state_path, arguments)
        return _ToolDispatchResult(json.dumps({"error": f"Unknown tool: {name}"}, indent=2))

    def _estimate_request_input_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        payload: dict[str, Any] = {"messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = _request_tool_choice(tools)
        return _estimate_tokens(payload)

    def _drop_oldest_history_block(self, history: list[dict[str, Any]], *, preserve_recent: int) -> bool:
        removable = len(history) - preserve_recent
        if removable <= 0:
            return False
        first = history.pop(0)
        first_role = str(first.get("role", "")).strip()
        if first_role in {"assistant", "tool"}:
            while history and history[0].get("role") == "tool" and len(history) > preserve_recent:
                history.pop(0)
            return True
        while history and history[0].get("role") == "tool" and len(history) > preserve_recent:
            history.pop(0)
        while history and history[0].get("role") != "user" and len(history) > preserve_recent:
            history.pop(0)
        return True

    def _keep_recent_history_turns(
        self,
        messages: list[dict[str, Any]],
        *,
        max_turns: int,
    ) -> list[dict[str, Any]]:
        if max_turns <= 0 or not messages:
            return []

        kept_reversed: list[dict[str, Any]] = []
        assistant_turns = 0
        for message in reversed(messages):
            kept_reversed.append(message)
            if str(message.get("role", "")).strip() == "assistant":
                assistant_turns += 1
                if assistant_turns >= max_turns:
                    break

        kept = list(reversed(kept_reversed))
        while kept and str(kept[0].get("role", "")).strip() == "tool":
            kept.pop(0)
        return kept

    def _drop_until_first_user_message(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        trimmed = list(history)
        while trimmed and str(trimmed[0].get("role", "")).strip() != "user":
            trimmed.pop(0)
        return trimmed

    def _persistent_history_messages(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        trimmed = self._trim_messages_for_context(messages, tools=tools)
        if not trimmed:
            return []
        trimmed_history = trimmed[1:]
        history = self._keep_recent_history_turns(
            trimmed_history,
            max_turns=_PERSISTENT_HISTORY_ASSISTANT_TURNS,
        )
        if (
            history
            and str(history[0].get("role", "")).strip() != "user"
            and len(trimmed_history) > len(history)
        ):
            previous_message = trimmed_history[len(trimmed_history) - len(history) - 1]
            if str(previous_message.get("role", "")).strip() == "user":
                history = [previous_message, *history]
        return self._drop_until_first_user_message(history)

    def _trim_messages_for_context(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        preserve_recent: int = 1,
        extra_safety_tokens: int = 0,
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        system_message = messages[0]
        history = list(messages[1:])
        preserve_recent = max(0, preserve_recent)
        budget_tokens = max(1, self._context_budget_tokens - max(0, extra_safety_tokens))
        while history and self._estimate_request_input_tokens([system_message, *history], tools=tools) > budget_tokens:
            if not self._drop_oldest_history_block(history, preserve_recent=preserve_recent):
                break
        history = self._drop_until_first_user_message(history)
        return [system_message, *history]

    def _force_reduce_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        preserve_recent: int = 1,
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        system_message = messages[0]
        history = list(messages[1:])
        if not self._drop_oldest_history_block(history, preserve_recent=max(0, preserve_recent)):
            return list(messages)
        return [system_message, *history]

    def analyze(
        self,
        state_path: Path,
        action_num: int,
        valid_actions: list[str] | None = None,
        step_env: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        checkpoint_env: Callable[[], dict[str, Any] | None] | None = None,
        restore_env: Callable[[dict[str, Any] | None], bool] | None = None,
        transcript_path: Path | None = None,
        analysis_step: int | None = None,
        transcript_updated: Callable[[str], None] | None = None,
        request_timeout_seconds: float | None = None,
        should_stop: Callable[[], bool] | None = None,
        time_remaining: Callable[[], float] | None = None,
    ) -> AnalyzerTurnResult | None:
        if not state_path.exists():
            return None
        # atlas 26.08: set BEFORE _ensure_session -- a new-session detection
        # there auto-creates the sys_start checkpoint anchor using this exact
        # callback, and this ToolAgent instance is reused across games, so
        # setting it after would let a brand-new game's sys_start snapshot
        # the PREVIOUS game's still-assigned callback for one call.
        self._step_env_callback = step_env
        # atlas 26.08: save_checkpoint/rollback only offered to the model
        # when checkpoint_env is actually wired AND returns a real snapshot
        # -- ONLINE-mode sessions pass None here (see
        # _HarnessGameSession.atlas_snapshot_env), and the sandbox tool is
        # withheld entirely rather than exposing a rollback that can't
        # revert anything.
        self._checkpoint_env_callback = checkpoint_env
        self._restore_env_callback = restore_env
        # 29.08 (Gemini round 6, hail mary): how much of this game's wall
        # budget is left -- lets the harness brute-force the board instead
        # of spending a queued model turn when the game is nearly dead.
        self._atlas_time_remaining_callback = time_remaining
        self._ensure_session(state_path)
        self._current_valid_actions = _normalize_valid_actions(valid_actions)

        analyzer_log = transcript_path or (state_path.parent / f"{state_path.stem}_analyzer.txt")
        prompt_log = _resolve_prompt_log_path(state_path)
        current_frame, history_entries = load_runtime_state(state_path)
        # atlas 26.08: context sanitizer (idea #3) -- runs at the START of
        # the turn AFTER the trigger fired (inside the previous turn's
        # _run_python_tool), before this turn's messages/history snapshot
        # are built below, so a fresh sanitize is what this turn's request
        # actually sends.
        if self._atlas_context_sanitize_pending:
            self._atlas_run_context_sanitizer(analyzer_log=analyzer_log)
        user_prompt = self._build_user_prompt(
            action_num,
            valid_actions=valid_actions,
            current_frame=current_frame,
            history_entries=history_entries,
            previous_step_summary=self._last_step_summary,
        )
        display_action_num = _display_action_number(action_num)

        with open(analyzer_log, "a", encoding="utf-8") as f:
            step_label = f"analysis_step={analysis_step} | " if analysis_step is not None else ""
            transcript_header = (
                f"\n--- {step_label}action={display_action_num} | "
                f"{time.strftime('%H:%M:%S')} | tool-agent ---\n"
            )
            f.write(transcript_header)
        transcript_parts = [transcript_header]

        def append_transcript(label: str, content: str) -> None:
            _append_transcript_section(analyzer_log, label, content)
            transcript_parts.append(_render_transcript_section(label, content))
            if transcript_updated is not None:
                transcript_updated("".join(transcript_parts))

        append_transcript("SYSTEM PROMPT", self._system_prompt)
        append_transcript("USER PROMPT", user_prompt)

        previous_history_messages = list(self._history_messages)
        preserve_history = True
        messages: list[dict[str, Any]] = self._trim_messages_for_context(
            [{"role": "system", "content": self._system_prompt}, *self._history_messages, self._build_user_message(user_prompt, current_frame)],
            tools=self._tools(state_path),
            preserve_recent=1,
        )
        step_executed = False
        captured_reasoning = ""
        latest_request_messages: list[dict[str, Any]] | None = None
        latest_request_tools: list[dict[str, Any]] | None = None
        latest_request_tool_choice: str | None = None
        latest_request_index = 0
        turn_started_at = time.monotonic()
        yielded_control_reason: str | None = None

        def control_yield_reason() -> str | None:
            if should_stop is not None:
                try:
                    if should_stop():
                        return "stop_requested"
                except Exception as exc:
                    log.warning("analyzer stop check failed at action %d: %s", display_action_num, exc)
            if self._yield_seconds is not None and (time.monotonic() - turn_started_at) >= self._yield_seconds:
                return "turn_time_budget"
            return None

        try:
            turn_count = 0
            while self._tool_steps is None or turn_count < self._tool_steps:
                yielded_control_reason = control_yield_reason()
                if yielded_control_reason is not None:
                    break
                turn_count += 1
                tools = self._tools(state_path)
                tool_choice = _request_tool_choice(tools)
                messages = self._trim_messages_for_context(messages, tools=tools)
                latest_request_messages = json.loads(json.dumps(messages))
                latest_request_tools = json.loads(json.dumps(tools))
                latest_request_tool_choice = tool_choice
                latest_request_index = turn_count
                _write_prompt_log_snapshot(
                    prompt_log,
                    model_id=self._model.model_id,
                    base_url=self._model.base_url,
                    display_action_num=display_action_num,
                    analysis_step=analysis_step,
                    request_index=turn_count,
                    messages=latest_request_messages,
                    tools=latest_request_tools,
                    tool_choice=tool_choice,
                    transcript="".join(transcript_parts),
                )
                try:
                    request_kwargs: dict[str, Any] = {"tools": tools}
                    if request_timeout_seconds is not None:
                        request_kwargs["request_timeout_seconds"] = request_timeout_seconds
                    if self._save_request_logs:
                        _append_request_snapshot(
                            _resolve_request_log_path(state_path),
                            messages=latest_request_messages,
                            tools=latest_request_tools,
                            event="request",
                            tool_choice=latest_request_tool_choice,
                            analysis_step=analysis_step,
                            action=display_action_num,
                            request_index_within_turn=latest_request_index,
                        )
                    result = self._chat_completion(messages, **request_kwargs)
                    self._accumulate_usage_tokens(result.usage)
                    if self._save_request_logs:
                        _append_request_snapshot(
                            _resolve_request_log_path(state_path),
                            messages=latest_request_messages,
                            tools=latest_request_tools,
                            event="response",
                            tool_choice=latest_request_tool_choice,
                            analysis_step=analysis_step,
                            action=display_action_num,
                            request_index_within_turn=latest_request_index,
                            finish_reason=result.finish_reason,
                        )
                except requests.RequestException as exc:
                    if not _is_context_length_error(exc):
                        raise
                    trimmed_messages = self._trim_messages_for_context(
                        messages,
                        tools=tools,
                        extra_safety_tokens=_CONTEXT_OVERFLOW_RETRY_TRIM_TOKENS,
                    )
                    if trimmed_messages == messages:
                        trimmed_messages = self._force_reduce_messages(messages)
                    if trimmed_messages == messages:
                        raise
                    append_transcript(
                        "ANALYZER STATUS",
                        "context_overflow_recovered: dropped older history after server rejected the request as too long.",
                    )
                    messages = trimmed_messages
                    continue
                raw_reasoning = _extract_reasoning_text(result.message)
                raw_content = _normalize_message_content(result.message.get("content", ""))
                tool_calls = json.loads(json.dumps(result.message.get("tool_calls") or []))
                tool_call_markup_in_text = _contains_tool_call_markup(raw_reasoning, raw_content)
                recovered_tool_calls_from_markup = False
                if not tool_calls and tool_call_markup_in_text:
                    tool_calls = _recover_tool_calls_from_markup(raw_reasoning, raw_content)
                    recovered_tool_calls_from_markup = bool(tool_calls)
                reasoning = _strip_tool_call_markup(raw_reasoning) if tool_call_markup_in_text else raw_reasoning
                content = _strip_tool_call_markup(raw_content) if tool_call_markup_in_text else raw_content
                malformed_argument_errors: list[str] = []
                for tool_call in tool_calls:
                    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                    tool_name = str(function.get("name", "")).strip() or "unknown"
                    raw_arguments = function.get("arguments", "{}")
                    if isinstance(raw_arguments, str):
                        try:
                            json.loads(raw_arguments)
                        except json.JSONDecodeError as exc:
                            malformed_argument_errors.append(f"{tool_name}: invalid JSON arguments ({exc})")
                response_meta = _format_model_response_meta(
                    finish_reason=result.finish_reason,
                    reasoning=reasoning,
                    content=content,
                    tool_calls=tool_calls,
                    tool_call_markup_in_text=tool_call_markup_in_text,
                    recovered_tool_calls_from_markup=recovered_tool_calls_from_markup,
                    malformed_argument_errors=malformed_argument_errors,
                )
                append_transcript(
                    "MODEL RESPONSE META",
                    response_meta,
                )
                assistant_message: dict[str, Any] = {"role": "assistant"}

                if reasoning:
                    captured_reasoning = reasoning
                    append_transcript("THINKING", reasoning)
                    assistant_message["reasoning"] = reasoning

                if not tool_calls:
                    if content:
                        self._update_summarized_knowledge_from_assistant(content)
                        append_transcript("ASSISTANT", content)
                        assistant_message["content"] = content
                    elif reasoning:
                        assistant_message["content"] = None

                    if content or reasoning:
                        messages.append(assistant_message)
                    yielded_control_reason = control_yield_reason()
                    if yielded_control_reason is not None:
                        break
                    followup_prefix = "You have not acted yet. Investigate first. "
                    if tool_call_markup_in_text:
                        followup_prefix = (
                            "You did not call a tool. We detected `<tool_call>` markup inside your reasoning or assistant text, "
                            "so no parsed tool call was executed. On this retry, do not add a note or explanation first. "
                            "Emit exactly one `python` tool call directly as your next response. "
                            "Do not place `<tool_call>` markup inside reasoning, explanation, or notes. "
                        )
                    followup_prompt = (
                        f"{followup_prefix}"
                        "Then investigate and revise your working world model of what the level contains, what actions appear to do, what the current goal seems to be, and what plan looks best. "
                        "If helpful, include short world-model update lines such as `World model:`, `Goal model:`, `Action model:`, `Recent findings:`, `Open questions:`, `Plan:`, or `Cross-level notes:`. "
                        "Call the `python` tool with code that inspects `current_frame`, `previous_frame`, `last_transition`, `history`, or `valid_actions` -- use `current_frame.segmentation` as the primary view, and `.ascii` only for a small specific region -- "
                        "compare `previous_frame` to `current_frame` for the most recent change, "
                        "derives a compact board summary, programs a small search or scorer over candidate actions or short sequences, "
                        "then call `action(actions)` inside Python with the best valid action or ordered batch that your code selected. "
                        f"{TOOL_CALL_FORMAT_GUIDANCE}"
                    )
                    append_transcript("USER PROMPT", followup_prompt)
                    messages.append({"role": "user", "content": followup_prompt})
                    continue

                if content:
                    self._update_summarized_knowledge_from_assistant(content)
                    append_transcript("ASSISTANT", content)
                    assistant_message["content"] = content
                assistant_message["tool_calls"] = tool_calls
                messages.append(assistant_message)

                for tool_index, tool_call in enumerate(tool_calls):
                    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                    tool_name = str(function.get("name", "")).strip()
                    raw_args = function.get("arguments", "{}")
                    try:
                        if isinstance(raw_args, str):
                            arguments = json.loads(raw_args)
                        elif isinstance(raw_args, dict):
                            arguments = json.loads(json.dumps(raw_args))
                        else:
                            arguments = {}
                    except json.JSONDecodeError:
                        arguments = {}
                    rendered_tool_call = _render_tool_call_markup(tool_name, raw_args)
                    append_transcript(
                        f"TOOL CALL: {tool_name}",
                        rendered_tool_call or (json.dumps(arguments, indent=2) if arguments else "{}"),
                    )
                    dispatch = self._dispatch_tool(state_path, tool_name, arguments)
                    if dispatch.step_executed:
                        step_executed = True
                    append_transcript(f"TOOL RESULT: {tool_name}", _render_tool_result_display(dispatch.content))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", ""),
                            "content": dispatch.content,
                        }
                    )
                    if dispatch.step_executed:
                        if tool_index < len(tool_calls) - 1:
                            preserve_history = False
                        break
                    yielded_control_reason = control_yield_reason()
                    if yielded_control_reason is not None:
                        if tool_index < len(tool_calls) - 1:
                            preserve_history = False
                        break
                if yielded_control_reason is not None:
                    break
                if step_executed:
                    break

        except requests.RequestException as exc:
            append_transcript("ANALYZER STATUS", f"request_error: {exc}")
            preserve_history = False
            if latest_request_messages is not None:
                _write_prompt_log_snapshot(
                    prompt_log,
                    model_id=self._model.model_id,
                    base_url=self._model.base_url,
                    display_action_num=display_action_num,
                    analysis_step=analysis_step,
                    request_index=latest_request_index,
                    messages=latest_request_messages,
                    tools=latest_request_tools,
                    tool_choice=latest_request_tool_choice,
                    transcript="".join(transcript_parts),
                )
            log.warning("analyzer request failed at action %d: %s", display_action_num, exc)
            return AnalyzerTurnResult(step_executed=False, retryable_failure=True, reasoning=captured_reasoning)
        except Exception as exc:
            append_transcript("ANALYZER STATUS", f"error: {exc}")
            preserve_history = False
            if latest_request_messages is not None:
                _write_prompt_log_snapshot(
                    prompt_log,
                    model_id=self._model.model_id,
                    base_url=self._model.base_url,
                    display_action_num=display_action_num,
                    analysis_step=analysis_step,
                    request_index=latest_request_index,
                    messages=latest_request_messages,
                    tools=latest_request_tools,
                    tool_choice=latest_request_tool_choice,
                    transcript="".join(transcript_parts),
                )
            log.warning("analyzer failed at action %d: %s", display_action_num, exc)
            return None
        finally:
            if preserve_history:
                self._history_messages = self._persistent_history_messages(messages, tools=self._tools(state_path))
            else:
                self._history_messages = previous_history_messages
            self._step_env_callback = None
            self._current_valid_actions = []

        if step_executed:
            status_message = "Step executed."
        elif yielded_control_reason is not None:
            status_message = f"Yielded control to solver: {yielded_control_reason}."
        else:
            status_message = "No action(...) call was captured."

        status = (
            f"model: {self._model.model_id}\n"
            f"base_url: {self._model.base_url}\n"
            f"max_output_tokens: {self._max_output_tokens if self._max_output_tokens is not None else 'server default'}\n"
            f"reply_reserve_tokens: {self._reply_reserve_tokens}\n"
            f"context_budget_tokens: {self._context_budget_tokens}\n"
            f"hard_noop_guard: {self._hard_noop_guard_enabled}\n"
            f"request_safety_margin_tokens: {self._request_safety_margin_tokens}\n"
            f"tool_output_tokens: {self._tool_output_tokens}\n"
            f"yield_seconds: {self._yield_seconds if self._yield_seconds is not None else 'disabled'}\n"
            f"available_tools: python\n"
            f"python_timeout_seconds: {self._python_timeout}\n"
            f"history_messages: {len(self._history_messages)}\n"
            f"step_executed: {step_executed}\n"
            f"message: {status_message}"
        )
        append_transcript("ANALYZER STATUS", status)
        if latest_request_messages is not None:
            _write_prompt_log_snapshot(
                prompt_log,
                model_id=self._model.model_id,
                base_url=self._model.base_url,
                display_action_num=display_action_num,
                analysis_step=analysis_step,
                request_index=latest_request_index,
                messages=latest_request_messages,
                tools=latest_request_tools,
                tool_choice=latest_request_tool_choice,
                transcript="".join(transcript_parts),
            )
        return AnalyzerTurnResult(
            step_executed=step_executed,
            reasoning=captured_reasoning,
            yielded_control=yielded_control_reason is not None,
        )

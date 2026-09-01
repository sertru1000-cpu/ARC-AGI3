"""Build the atlas notebook: the upstream submission notebook plus our own layer.

The bundled source dataset stays untouched. Everything we change is set on
``bm`` / ``bm.solver`` from the notebook's own inline customization hook.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

def _as_bool(raw: str) -> bool:
    return (raw or "0").strip().lower() not in {"0", "false", "no"}


ROOT = Path(__file__).resolve().parent.parent
SRC_NB = ROOT / "notebooks_duck" / "submission.ipynb"
OUT_DIR = ROOT / "notebooks_atlas"
OUT_NB = OUT_DIR / "submission.ipynb"
KERNEL_SLUG = "sergueimakarov/arc3-atlas"
KERNEL_TITLE = "arc3 atlas"

# atlas: mount OUR fork of the Duck source (atlas_src/, published 22.08 as a
# private Kaggle dataset) instead of the upstream jakobbrggen bundle.
OLD_SOURCE_DATASET = "jakobbrggen/taaf-kaggle-source-anim-20260807-anim"
NEW_SOURCE_DATASET = "sergueimakarov/arc3-atlas-src"

# --- the snippet appended to the cell that unpickles the benchmark ----------
PRISTINE_CAPTURE = '''

# atlas: remember what the bundle itself carries, before any override below.
ATLAS_PRISTINE = {
    "analyzer_timeout": getattr(bm.solver, "analyzer_timeout", None),
    "concurrency": getattr(bm.solver, "concurrency", None),
    "max_actions_per_game": getattr(bm.solver, "max_actions_per_game", None),
    "max_runtime_s_per_game": getattr(bm.solver, "max_runtime_s_per_game", None),
    "n_passes": getattr(bm, "n_passes", None),
}
'''

# --- the new cell, inserted after the inline customization hook -------------
ATLAS_CELL = '''# ==========================================================================
# atlas v2 -- our layer on top of the upstream harness.
#
# v1 (22.08, analyzer_timeout 900 -> 180s, concurrency 28 -> 14) measured on
# arc3-atlas itself (22.08 night, 25 public games, 4h24m wall clock):
#
#   * mean score 4.99 (median 2.78), up from the 3.02 v0 baseline -- but:
#   * 512 analyzer requests failed, and 488 of those hit the FULL 180s cap
#     (vs. only 12/33 full-timeout stalls in the 900s baseline). That is
#     488 * 180s = 87840 thread-seconds of pure retry-thrash out of 221774
#     available (14 concurrency * 15841s wall clock) = 39.6% wasted.
#   * root cause, read from vllm-openai-server.log: aggregate generation
#     throughput holds at ~250-270 tok/s regardless of concurrency (GPU
#     compute-bound, KV cache usage only ~35-37%), so at 14 concurrent
#     requests each gets ~18-20 tok/s. LOCAL_ANALYZER_MAX_OUTPUT was 0
#     (uncompleted -- literally unbounded) with thinking enabled, so a
#     turn generating ~4800+ tokens already exceeds a 180s cap. 180s was a
#     guess, not measured -- it undershot badly.
#
# v2 fixes the actual cause instead of re-guessing the timeout in isolation:
#   1. Cap LOCAL_ANALYZER_MAX_OUTPUT at 8000 tokens. This is a MODULE-LEVEL
#      constant read once at import in inference.agent.tool_agent (line
#      ~148) and frozen again per-ToolAgent in __init__ -- setting the env
#      var here would be too late (the module is already imported by the
#      time this cell runs) and silently do nothing. Must patch the
#      module attribute directly so future ToolAgent() constructions (one
#      per game/pass, built fresh at run time -- NOT part of the pickle)
#      pick it up.
#   2. Raise analyzer_timeout to 480s: at the measured worst-case ~18 tok/s
#      (full 14-way contention), 8000 tokens takes ~444s: 480s gives ~35s
#      of margin for prefill/network on top of the now-bounded worst case.
#
# Score is driven by DEPTH, so time lost to queueing/retry-thrash is levels
# not reached.
# ==========================================================================

ATLAS_ANALYZER_TIMEOUT_S = 480.0       # v1 tried 180 (measured: far too short)
ATLAS_ANALYZER_MAX_OUTPUT_TOKENS = 8000  # v1 had this unbounded (0) -- the real bug
ATLAS_CONCURRENCY = __ATLAS_CONCURRENCY__  # substituted by the BUILDER (env ATLAS_CONCURRENCY_BUILD, default 20)
                                        # 29.08 evening (user, option B): BACK to
                                        # v23's proven waves-of-20. The one-wave
                                        # 110 config just scored 0.65 on the
                                        # hidden set with v23's own code (v24,
                                        # 55860197) vs 0.92 for waves-of-20 --
                                        # and the same-day testbed showed WHY:
                                        # 55 concurrent games starve each other
                                        # of LLM turns (wall time is cheap, LLM
                                        # turns are the scarce resource). This
                                        # build = round-6+8 depth code on the
                                        # config that actually scored 0.92.
                                        # Final-rescore (110 games -> 80 min/game)
                                        # config question deliberately deferred.
                                        # Previous one-wave rationale kept below
                                        # for history:
                                        # 28.08 (user): ONE WAVE of the ~110-game
                                        # hidden set (55 public-LB + 55 private,
                                        # structure confirmed 28.08). Rehearsals:
                                        # 55x4h and 110x6h both clean on the pod.
                                        # Was 20 through v23 (=6 waves, 80 min/game).
                                        # 27.08: kernel v21 Phase A calibration
                                        # (25 games, real Kaggle RTX Pro 6000
                                        # backend, not just RunPod A100) came
                                        # back clean at concurrency=20 -- no
                                        # retry-storm recurrence -- and the
                                        # user made the deliberate call to KEEP
                                        # 20 for the real (Phase B) submission
                                        # too. See [[arc-agi-3-top10-plan]]
                                        # memory for the full history.
                                        # 25.08: lowered from 14 after the
                                        # retry-storm bug (cn04/lp85/re86 in
                                        # v17 lost 15-30min each retrying one
                                        # analysis_step on request timeouts,
                                        # plausibly the shared local LLM
                                        # backend overloaded under 14x
                                        # concurrent load). Checked the real
                                        # atlas_fit_game_cap() formula first:
                                        # at 25 games, concurrency 9-13 all
                                        # still land on 3 (or fewer) waves,
                                        # so the real per-game cap stays
                                        # pinned at the 8500s ceiling either
                                        # way -- this costs ZERO per-game
                                        # budget, only raises the worst-case
                                        # total wall time (4.72h->7.08h,
                                        # still comfortably inside the 8h
                                        # budget/9h hard cap). Below 9,
                                        # waves=4 and the real cap drops to
                                        # 7200s -- a genuine cost, so 10 was
                                        # chosen for margin above that cliff
                                        # while cutting concurrent load by
                                        # ~29%. Complements, doesn't replace,
                                        # the retry-storm backstop in
                                        # solver.py -- less contention makes
                                        # a storm less likely, the backstop
                                        # bounds the damage if one still
                                        # happens. NOT applied to kernel
                                        # version 19 (already pushed/running
                                        # when this was decided) -- takes
                                        # effect on the next build after it.
                                        # See scripts/test_atlas_fit_game_cap.py
                                        # for the concurrency/waves/cap math.
ATLAS_FALLBACK_GAME_CAP_S = 7920.0     # applied only if the bundle carries none

# Wall-clock guard for the submission rerun. Unlike the offline run, a rerun
# gets soft_end_time=None, so the per-game cap is the ONLY thing standing
# between us and Kaggle killing the notebook at 9 h with no result.
ATLAS_SUBMISSION_BUDGET_S = 28800.0   # 29.08 (option B): back to v23's 8h +
                                       # 1h margin -- the exact budget behind
                                       # 0.92. Was 30600 (8.5h one-wave, v24's
                                       # 0.65). Original one-wave note:
                                       # 28.08 (user): 8.5 h, one-wave config --
                                       # no per-game extensions (draws off), so
                                       # this IS the per-game ceiling too. Rides
                                       # closer to the platform kill than the
                                       # old 8h+1h-margin: the 12h rule and
                                       # v23's ~9.6h elapsed rerun say it fits.
                                       # Margin left covers setup (dataset
                                       # mount, wheelhouse
                                       # install if not cached, vLLM start +
                                       # smoke test measured at ~5 min alone).
                                       # Raised from 7.5h/1.19-score run
                                       # 23.08 -- user's call, more play time
                                       # over more safety margin.
# 24.08: explicit, named ceiling for atlas_fit_game_cap() -- replaces reading
# bm.solver.max_runtime_s_per_game (7920s, the bundle's own undocumented
# default), which silently absorbed the whole budget increase above for any
# n_games <= 42 (v3/v5/v6/v8/v10/v12's real behavior). Raised 7920 -> 8500
# 28.08 (user's call: "так подними его"): raised 8500 -> 14400 (4h/game)
# alongside the planned Phase B concurrency increase. Context: the hidden
# Phase B set is ~110 games (55 semi-private + 55 fully-private per the
# ARC-AGI-3 technical report), so at concurrency 20 the affordable cap was
# only 80 min/game -- the leading explanation for v20's real 0.82 vs 1.43
# on the stand. With concurrency ~55, affordable becomes 14400s (110 games,
# 2 waves) or 28800s (55 games, 1 wave); this ceiling admits the 4h/game
# the 28.08 rehearsals validated while still bounding any single game to
# half the 8h budget. History: a v7 regression (0.06) came from removing
# the ceiling ENTIRELY -- keep it firm and named, never delete it.
ATLAS_SUBMISSION_GAME_CAP_CEILING_S = __ATLAS_GAME_CAP_CEILING__  # substituted by the BUILDER (env ATLAS_GAME_CAP_CEILING_BUILD, default 8500)
                                       # 29.08 (option B): v23's ceiling
                                       # (was 30600 for one-wave). At 55 games /
                                       # 3 waves the affordable cap is 9600s ->
                                       # this ceiling binds at 8500s (~140 min),
                                       # the regime that scored 0.92.
ATLAS_MIN_GAME_CAP_S = 1800.0

# 31.08 (backlog 27): dynamic budget re-planning. The static fit below sizes
# the per-game cap ONCE, from the wave arithmetic; with draws disabled, the
# time a stalled game returns is then never taken by anyone. Measured on
# every Phase A length we have (25 min, 50 min, 2.2 h): 23 of 25 games
# self-terminate at 50% of their cap, so roughly half the wall budget is
# handed back and evaporates. When this is on, the solver recomputes a
# per-game window from the ABSOLUTE deadline on every game completion and
# only ever raises it, so survivors inherit the freed hours while the run
# still cannot outlive ATLAS_SUBMISSION_BUDGET_S.
# Substituted by the BUILDER (env ATLAS_DYNAMIC_BUDGET_BUILD, default "0") --
# an env read left inside the cell would execute on Kaggle, where it is
# unset, which is exactly how v25 silently ran a 4 h Phase A.
# Armed ONLY inside atlas_fit_game_cap (the submission branch): arming it in
# Phase A would stretch a 30-minute calibration to the full 8-hour budget.
ATLAS_DYNAMIC_BUDGET = __ATLAS_DYNAMIC_BUDGET__

print("atlas: solver config as it came from the bundle:")
for _key, _value in ATLAS_PRISTINE.items():
    print(f"atlas:   {_key} = {_value}")

bm.solver.analyzer_timeout = ATLAS_ANALYZER_TIMEOUT_S
bm.solver.concurrency = ATLAS_CONCURRENCY
if getattr(bm.solver, "max_runtime_s_per_game", None) is None:
    bm.solver.max_runtime_s_per_game = ATLAS_FALLBACK_GAME_CAP_S
    print(
        "atlas: bundle carried no per-game runtime cap; applied "
        f"{ATLAS_FALLBACK_GAME_CAP_S:.0f}s"
    )

import inference.agent.tool_agent as _atlas_tool_agent
_atlas_tool_agent._LOCAL_ANALYZER_MAX_OUTPUT = ATLAS_ANALYZER_MAX_OUTPUT_TOKENS
print(f"atlas: patched tool_agent._LOCAL_ANALYZER_MAX_OUTPUT = {ATLAS_ANALYZER_MAX_OUTPUT_TOKENS}")

# atlas 28.08 (one-wave config): module-attribute patches, NOT env vars --
# these constants are read at import, and the modules are already imported
# by the unpickle cell above, so os.environ would be silently too late
# (the exact trap the max-output comment above documents). Both patches are
# hasattr-guarded so this one notebook works with either dataset build
# (v23-agent hybrid or the full probes build).
import inference.framework.solver as _atlas_solver_mod
if hasattr(_atlas_solver_mod, "_ATLAS_TIME_BANK_DRAWS_ENABLED"):
    # 30.08: build-time substituted (env ATLAS_DRAWS_BUILD, default "1").
    # Draws ON = v23 regime (waves-of-20, 8500s ceiling). Draws OFF (user's
    # 30.08 V39 call): 14400s ceiling x 2 waves exactly fills the 28800s
    # budget -- a draw on top would blow it; the time-bank stays ENABLED so
    # the stall EARLY EXIT (0 levels past 50% of cap -> end, deposit time,
    # stop competing for the LLM) keeps working.
    _atlas_solver_mod._ATLAS_TIME_BANK_DRAWS_ENABLED = bool(__ATLAS_DRAWS__)
    print(f"atlas: solver._ATLAS_TIME_BANK_DRAWS_ENABLED = {bool(__ATLAS_DRAWS__)} (build-time)")
if hasattr(_atlas_tool_agent, "_ATLAS_LLM_REQUEST_GATE"):
    import threading as _atlas_threading
    _atlas_tool_agent._ATLAS_LLM_MAX_CONCURRENT = 25
    _atlas_tool_agent._ATLAS_LLM_REQUEST_GATE = _atlas_threading.Semaphore(25)
    print("atlas: patched tool_agent LLM request gate = 25 (110 concurrent games)")
if hasattr(_atlas_tool_agent, "_ATLAS_LLM_ZOMBIE_GATE"):
    # 29.08 (round 6, D3): level-1 games with no progress get only 10 of
    # the 25 request slots -- level-2+ games keep priority (zombie cull).
    _atlas_tool_agent._ATLAS_LLM_ZOMBIE_SLOTS = 10
    _atlas_tool_agent._ATLAS_LLM_ZOMBIE_GATE = _atlas_threading.Semaphore(10)
    print("atlas: patched tool_agent zombie gate = 10 (level-1 no-progress cull)")

# ---------------------------------------------------------------------------
# 01.09: THE COERCION STACK, as one build-time switch (candidate 2).
# Measured on calib_1 (25 games, one wave, 2h cap): 851 checkpoint injections
# against 449 real actions -- 1.9 coercions per move, explore-first alone
# firing 394 times. Meanwhile every game ended `gave_up` having used ~25 of
# its 120 available minutes, at 18 actions/game against stock's 130.
# The hypothesis this arm tests: our own machinery, not the games, is what
# throttles tempo, and tempo is what drags level-1 conversion (r = 0.78).
#
# Principle: remove what the HARNESS FORCES, keep what the MODEL CHOOSES.
# plan_real, verify_theory, memo, rollback all stay available as tools --
# the model called plan_real 108 times on its own and that is untouched.
#
# Implemented as module-attribute patches, not env vars: these constants are
# read at import and the modules are already imported by the unpickle cell
# (same trap the _LOCAL_ANALYZER_MAX_OUTPUT comment above documents). Every
# patch is hasattr-guarded and every constant was checked to be used ONLY in
# comparisons -- _ATLAS_ROLLBACK_LOOP_WINDOW is deliberately NOT touched
# because it also sizes a deque, and a huge value there leaks memory.
ATLAS_CHECKPOINTS = __ATLAS_CHECKPOINTS__
if not ATLAS_CHECKPOINTS:
    _atlas_off = 10 ** 9  # "never fires", safe: comparison-only thresholds
    _atlas_coercions = {
        "_ATLAS_EXPLORE_NUDGE_AFTER_CALLS": _atlas_off,      # fired 394x
        "_ATLAS_THEORY_FORCE_AFTER_CALLS": _atlas_off,       # fired 170x
        "_ATLAS_FORCE_ACT_AFTER_CALLS": _atlas_off,          # fired  96x
        "_ATLAS_THEORY_CHECKPOINT_ENABLED": False,           # fired  50x
        "_ATLAS_THEORY_NAG_AFTER_CALLS": _atlas_off,
        "_ATLAS_PLAN_FORCE_AFTER_CALLS": _atlas_off,         # fired  43x
        "_ATLAS_PLAN_NAG_EVERY": _atlas_off,                 # fired  11x
        "_ATLAS_MEMO_NUDGE_AFTER_CALLS": _atlas_off,         # fired  29x
        "_ATLAS_EXTRACT_NUDGE_AFTER_CALLS": _atlas_off,
        "_ATLAS_GOAL_RECONSIDER_AFTER_CALLS": _atlas_off,    # fired   3x
        "_ATLAS_ROLLBACK_STALL_AFTER_CALLS": _atlas_off,     # 18 of 21 triggers
        "_ATLAS_PROBE_RATION_FREE": _atlas_off,              # fired  17x
        "_ATLAS_PLAN_REAL_STALL_AFTER_ACTIONS": _atlas_off,
        "_ATLAS_CONTEXT_SANITIZE_EVERY_CALLS": _atlas_off,   # 23x, 19 empty
        "_ATLAS_PLAN_REAL_PROACTIVE": False,                 # 26x, all fruitless
    }
    _atlas_missing = [k for k in _atlas_coercions if not hasattr(_atlas_tool_agent, k)]
    for _k, _v in _atlas_coercions.items():
        if hasattr(_atlas_tool_agent, _k):
            setattr(_atlas_tool_agent, _k, _v)
    print(f"atlas: COERCION STACK OFF -- {len(_atlas_coercions) - len(_atlas_missing)} "
          f"of {len(_atlas_coercions)} knobs patched")
    if _atlas_missing:
        # Loud, not fatal: the source dataset may predate a constant. The arm
        # is still valid, but the log must say which part did not come off.
        print(f"atlas: WARNING -- absent from this build, NOT disabled: {_atlas_missing}")
    print("atlas: residual by design -- the rollback ping-pong trigger "
          "(3 of 21 fires on calib_1) stays, its window also sizes a deque")
else:
    print("atlas: coercion stack ON (baseline arm)")

# ---------------------------------------------------------------------------
# 01.09: MAXIMAL ABLATION -- every HOST-SIDE mechanism we ever added, off.
#
# Naming honesty: this is NOT byte-identical to stock Duck and cannot be.
# `plan_real` and `rollback` are our additions woven into the STATIC prompt,
# and no module attribute removes a tool from a prompt. True stock already
# exists and is already measured: the separate `arc3-duck-baseline` kernel,
# which scored 1.61 on 01.09. So this switch answers a different, still-open
# question -- is the damage in our HOST BEHAVIOURS, or in the prompt/dataset
# layer that survives them?
#
# Implies ATLAS_CHECKPOINTS=0 (the coercion stack is a host behaviour too).
# Every value below was checked to be a flag or a comparison operand.
# _ATLAS_ACTION_EFFECT_HISTORY_WINDOW is deliberately NOT touched: it slices
# a list, so a huge value allocates instead of disabling -- the summary is
# silenced through MIN_TRANSITIONS instead. Same trap as ROLLBACK_LOOP_WINDOW.
ATLAS_STOCK = __ATLAS_STOCK__
if ATLAS_STOCK:
    _atlas_never = 10 ** 9
    _atlas_ablation = {
        "_atlas_tool_agent": {
            # the search layer: the A* heuristic is trained on the same 25
            # public games and on unseen ones loses to a naive mean (MAE 18.6
            # vs 15.7, measured 30.08). Weight 0 makes it inert whatever the
            # cache holds; plan_real itself stays, the model calls it freely.
            "_ATLAS_ASTAR_WEIGHT": 0.0,
            "_ATLAS_ASTAR_MODEL_PATH": "",
            "_ATLAS_PLAN_REAL_PROACTIVE": False,
            # LLM rationing: stock has no gate at all.
            "_ATLAS_LLM_MAX_CONCURRENT": 0,
            "_ATLAS_LLM_REQUEST_GATE": None,
            "_ATLAS_LLM_ZOMBIE_SLOTS": 0,
            "_ATLAS_LLM_ZOMBIE_GATE": None,
            "_ATLAS_ZOMBIE_AFTER_ACTIONS": _atlas_never,
            "_ATLAS_ZOMBIE_ENTROPY_AFTER": _atlas_never,
            # host automation
            "_ATLAS_L1_STOCK_SPRINT": False,
            "_ATLAS_MECHANIC_HANDOFF": False,
            "_ATLAS_LEVEL_AUTO_REPLAY": False,
            "_ATLAS_DRAFT_SPEEDRUN": False,
            # the biggest single prompt injection we own: 716 of 818 lines in
            # calib_1 were the action-effect summary, more than every
            # checkpoint combined, and the coercion arm does NOT remove it.
            "_ATLAS_ACTION_EFFECT_MIN_TRANSITIONS": _atlas_never,
        },
        "_atlas_solver_mod": {
            "_ATLAS_TIME_BANK_ENABLED": False,
            "_ATLAS_TIME_BANK_DRAWS_ENABLED": False,
            "_ATLAS_SPEEDRUN_ENABLED": False,
            "_ATLAS_DYNAMIC_BUDGET_ENABLED": False,
        },
    }
    _atlas_absent = []
    for _mod_name, _patches in _atlas_ablation.items():
        _mod = globals().get(_mod_name)
        for _k, _v in _patches.items():
            if _mod is not None and hasattr(_mod, _k):
                setattr(_mod, _k, _v)
            else:
                _atlas_absent.append(f"{_mod_name}.{_k}")
    # the analyzer timeout goes back to the bundle's own 900s. The OUTPUT cap
    # deliberately stays at 8000: v1 shipped it unbounded and the comment on
    # ATLAS_ANALYZER_MAX_OUTPUT_TOKENS calls that "the real bug" -- reverting
    # a fix is not ablation, it is reintroducing a known defect.
    bm.solver.analyzer_timeout = 900.0
    _atlas_total = sum(len(v) for v in _atlas_ablation.values())
    print(f"atlas: MAXIMAL ABLATION -- {_atlas_total - len(_atlas_absent)} of "
          f"{_atlas_total} host mechanisms off, analyzer_timeout back to 900s")
    if _atlas_absent:
        print(f"atlas: WARNING -- absent from this build, NOT disabled: {_atlas_absent}")
    print("atlas: still OURS and NOT removable this way -- plan_real and "
          "rollback live in the static prompt; true stock is the "
          "arc3-duck-baseline kernel (1.61 on 01.09)")

# 30.08: A* heuristic for plan_real (pod A/B on the 25 publics: 5 levels /
# RHAE 0.69 with it vs 1 / 0.11 without, matched 1h arms). The path env is
# read at IMPORT time (the same too-late trap as everything above), so the
# module attribute is patched directly; the loader stays lazy + fail-safe
# (any load error falls back to novelty ordering). sklearn ships in the
# Kaggle image; h_model.pkl rides in the source dataset.
import glob as _atlas_glob
_astar_hits = sorted(_atlas_glob.glob("/kaggle/input/*/h_model.pkl")) or sorted(
    _atlas_glob.glob("/kaggle/input/**/h_model.pkl", recursive=True))
if _astar_hits and hasattr(_atlas_tool_agent, "_ATLAS_ASTAR_MODEL_PATH"):
    _atlas_tool_agent._ATLAS_ASTAR_MODEL_PATH = _astar_hits[0]
    _atlas_tool_agent._ATLAS_ASTAR_CACHE = {"loaded": False, "model": None}
    print(f"atlas: A* heuristic path patched -> {_astar_hits[0]}")
else:
    print("atlas: A* heuristic pkl not found in inputs -- novelty ordering stays")

# Phase A ONLY: shrink the per-game cap for a quick calibration check of the
# timeout/max-output change above, on the real 25-game set (no fabricated
# repeats -- repeats of one game would raise vLLM's prefix-cache hit rate and
# understate contention). Same solver config, same real games, just cut
# short. Phase B (true_submission) is untouched -- atlas_fit_game_cap() below
# still sizes its cap from ATLAS_SUBMISSION_BUDGET_S alone.
# 29.08 FIX (caught live by the user via "Diff: +0 -0"): this constant
# lives INSIDE the generated notebook cell, so reading the env var here
# meant reading it AT KAGGLE RUNTIME (where it is unset) -- v25 silently
# ran a 4h Phase A instead of 30 min. The builder now substitutes the
# BUILD-time env value as a literal into the cell (see build()).
ATLAS_CALIBRATION_CAP_S = __ATLAS_CALIBRATION_CAP_S__  # value substituted by the BUILDER (env ATLAS_CALIBRATION_CAP_S at build time)
                                   # 28.08 (kernel v24, user): 4h/game --
                                   # Phase A becomes the MATCHED CONTROL
                                   # against the probe-branch decisive run
                                   # (25 public games, 4h each, one wave at
                                   # the new concurrency) on the kernel's
                                   # own RTX Pro 6000. ~4.7h of the 45h
                                   # weekly quota. Prior value 750s (v23's
                                   # 30-min total calibration):
                                   # user set a 30-min TOTAL budget (not
                                   # per-game -- see the wave math below).
                                   # 25 games / concurrency 20 = 2 waves;
                                   # 2*750s=1500s (25min) + ~5min setup ~=
                                   # 30min budget, same math as v21. This
                                   # push carries the planforce+rollbackfix
                                   # build (7265cbd: ATLAS_PLAN_FORCE_
                                   # OVERRIDE + the wa30 rollback gate-
                                   # bypass fix; deliberately WITHOUT the
                                   # try_actions/plan_real snapshot probes,
                                   # which stay RunPod-experiment-only for
                                   # now) -- the version the user will
                                   # submit for real. On RunPod A100 it
                                   # measured mean 0.96 / 7-of-25 games
                                   # with a level at the 57-min mark,
                                   # against 0.11 / 1-of-25 for the
                                   # previous (theoryforce) build at a
                                   # matched window.
# 31.08: repetitions for Phase A. `n_passes` deep-copies every game N times and
# schedules all copies through the SAME concurrency semaphore (taaf/benchmark.py
# line ~149), so 25 games x 3 passes = 75 tasks at concurrency 28 = 3 waves.
# This is how the stock Duck reference was measured (25 games x 20 passes = 500
# game-runs) and it is the only affordable way to get repetition: the unit of
# observation becomes a game-pass, not a run, so one kernel run yields 75
# observations instead of 25. Phase A ONLY -- the submission branch sets
# n_passes = 1 itself and must never be multiplied.
ATLAS_PHASE_A_PASSES = __ATLAS_PASSES__  # substituted by the BUILDER (env ATLAS_PASSES_BUILD, default 1)

if not true_submission:
    bm.solver.max_runtime_s_per_game = ATLAS_CALIBRATION_CAP_S
    print(f"atlas: Phase A calibration cap -- max_runtime_s_per_game = {ATLAS_CALIBRATION_CAP_S:.0f}s")
    if ATLAS_PHASE_A_PASSES > 1:
        bm.n_passes = ATLAS_PHASE_A_PASSES
        print(
            f"atlas: Phase A repetitions -- n_passes = {bm.n_passes} "
            f"({len(bm.games)} games x {bm.n_passes} = {len(bm.games) * bm.n_passes} game-runs)"
        )


def atlas_fit_game_cap(n_games: int) -> None:
    """Shrink the per-game cap so every wave fits the notebook budget.

    Called from the submission branch, where the game list only becomes known
    after Kaggle's gateway answers.

    24.08: the ceiling is now the explicit ATLAS_SUBMISSION_GAME_CAP_CEILING_S
    constant, not whatever bm.solver.max_runtime_s_per_game happened to carry
    in from the bundle (7920s, undocumented) -- that implicit ceiling silently
    absorbed any budget increase for n_games<=42. This is still a firm
    ceiling (never removed outright, unlike the v7 regression) -- just a
    named, deliberately-chosen one instead of an inherited accident.
    """
    import json
    import math
    from datetime import datetime, timezone

    concurrency = max(1, int(bm.solver.concurrency))
    waves = max(1, math.ceil(max(1, int(n_games)) / concurrency))
    affordable = ATLAS_SUBMISSION_BUDGET_S / waves
    previous = float(bm.solver.max_runtime_s_per_game or ATLAS_FALLBACK_GAME_CAP_S)
    fitted = max(ATLAS_MIN_GAME_CAP_S, min(ATLAS_SUBMISSION_GAME_CAP_CEILING_S, affordable))
    print(
        f"atlas: {n_games} games / concurrency {concurrency} = {waves} wave(s); "
        f"per-game cap {previous:.0f}s -> {fitted:.0f}s (ceiling={ATLAS_SUBMISSION_GAME_CAP_CEILING_S:.0f}s)"
    )
    bm.solver.max_runtime_s_per_game = fitted

    # atlas 31.08: hand the planner the absolute deadline. Module attributes,
    # not env vars -- solver.py was imported long before this cell ran.
    if ATLAS_DYNAMIC_BUDGET and hasattr(_atlas_solver_mod, "_ATLAS_DYNAMIC_BUDGET_ENABLED"):
        _atlas_solver_mod._ATLAS_DYNAMIC_BUDGET_ENABLED = True
        _atlas_solver_mod._ATLAS_DYNAMIC_BUDGET_TOTAL_S = ATLAS_SUBMISSION_BUDGET_S
        _atlas_solver_mod._ATLAS_DYNAMIC_BUDGET_CEILING_S = ATLAS_SUBMISSION_GAME_CAP_CEILING_S
        _atlas_solver_mod._ATLAS_DYNAMIC_BUDGET_MIN_S = ATLAS_MIN_GAME_CAP_S
        print(
            "atlas: dynamic budget re-planning ARMED -- budget "
            f"{ATLAS_SUBMISSION_BUDGET_S:.0f}s, window ceiling "
            f"{ATLAS_SUBMISSION_GAME_CAP_CEILING_S:.0f}s, floor {ATLAS_MIN_GAME_CAP_S:.0f}s"
        )
    else:
        print("atlas: dynamic budget re-planning OFF -- static per-game cap only")

    # atlas: minimal_diagnostics=True on a real submission means the usual
    # summary.txt/transcripts never get written, and kernels output/logs may
    # not even reach a live competition rerun at all (unconfirmed -- the CLI
    # showed only the stale prior Phase A commit's files after 22.08's
    # submission finished). Write this anyway, on the chance it survives:
    # the one fact we actually want out of a real run is n_games itself.
    try:
        diagnostics_path = WORKING_DIR / "atlas_submission_diagnostics.json"
        diagnostics_path.write_text(
            json.dumps(
                {
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "n_games": int(n_games),
                    "concurrency": concurrency,
                    "waves": waves,
                    "submission_budget_s": ATLAS_SUBMISSION_BUDGET_S,
                    "per_game_cap_before_s": previous,
                    "per_game_cap_after_s": fitted,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"atlas: wrote {diagnostics_path}")
    except Exception as exc:
        print(f"atlas: could not write submission diagnostics: {exc!r}")


print("atlas: effective solver config:")
print(f"atlas:   analyzer_timeout      = {bm.solver.analyzer_timeout}")
print(f"atlas:   analyzer_max_output   = {_atlas_tool_agent._LOCAL_ANALYZER_MAX_OUTPUT}")
print(f"atlas:   concurrency           = {bm.solver.concurrency}")
print(f"atlas:   max_runtime_s_per_game = {bm.solver.max_runtime_s_per_game}")
print(f"atlas:   max_actions_per_game  = {bm.solver.max_actions_per_game}")
print(f"atlas:   n_passes              = {bm.n_passes}")
'''

# --- the two-line call added to the submission branch of the run cell ------
RUN_CELL_ANCHOR = """        bm.games = _competition_games()
        bm.n_passes = 1
        bm.game_weights = None"""

RUN_CELL_PATCH = """        bm.games = _competition_games()
        bm.n_passes = 1
        bm.game_weights = None
        import os as _atlas_os
        if __ATLAS_STOCK_PHASE_B__:
            # 01.09 (owner's call): Phase B is a BYTE-FOR-BYTE copy of stock
            # duck v1 -- the three lines above and nothing else. Rationale:
            # stock scored 1.61 on the hidden set against our median 0.85, and
            # its local mean (1.60, measured 30.08 over 20 passes) transferred
            # 1:1 while ours transferred x0.42. The hidden games are not
            # harder; our layer fails on unseen ones. So the control must
            # differ from stock ONLY in the agent layer under test.
            #
            # Concretely this drops two things:
            #  * atlas_fit_game_cap -- no budget division. The per-game cap
            #    stays whatever the bundle carries (7920s), exactly as stock.
            #    110 games / conc 28 = 4 rounds x 2.2h = 8.8h; stock itself
            #    just proved that survives (it passed 9.00h and finished).
            #  * the time bank -- stock has no such mechanism. Draws were
            #    already off, so this only removes the stall EARLY EXIT, which
            #    can end a game sooner but never later: it cannot cause a
            #    platform timeout, only forfeit time a stalled game had left.
            if hasattr(_atlas_solver_mod, "_ATLAS_TIME_BANK_ENABLED"):
                _atlas_solver_mod._ATLAS_TIME_BANK_ENABLED = False
            print(
                "atlas: Phase B = STOCK duck v1 parity -- per-game cap "
                f"{getattr(bm.solver, 'max_runtime_s_per_game', None)}s from the bundle, "
                "no budget fitting, time bank OFF",
                flush=True,
            )
        else:
            # atlas 29.08 (option B): v23 regime -- time-bank draws ON (waves of
            # 20; deposits from stalled games get re-drawn by progressing ones,
            # ceiling 8500s keeps a maxed draw far under the platform kill).
            _atlas_os.environ["ATLAS_TIME_BANK_DRAWS"] = "1" if __ATLAS_DRAWS__ else "0"
            # atlas: a rerun has no soft deadline, so size the per-game cap to fit.
            atlas_fit_game_cap(len(bm.games))"""


def _source_of(cell: dict) -> str:
    return "".join(cell["source"])


def _set_source(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def _new_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


# Build-time knobs: placeholder in ATLAS_CELL -> (env var, default, parser).
# An env read left INSIDE the cell would execute on Kaggle, where the env is
# unset -- that is exactly how v25 silently ran a 4 h Phase A instead of 30
# minutes. Everything is substituted as a literal here instead.
CELL_KNOBS = {
    "__ATLAS_CALIBRATION_CAP_S__": ("ATLAS_CALIBRATION_CAP_S", "14400", float),
    "__ATLAS_CHECKPOINTS__": ("ATLAS_CHECKPOINTS_BUILD", "1", _as_bool),
    "__ATLAS_CONCURRENCY__": ("ATLAS_CONCURRENCY_BUILD", "20", int),
    "__ATLAS_DRAWS__": ("ATLAS_DRAWS_BUILD", "1", _as_bool),
    "__ATLAS_DYNAMIC_BUDGET__": ("ATLAS_DYNAMIC_BUDGET_BUILD", "0", _as_bool),
    "__ATLAS_GAME_CAP_CEILING__": ("ATLAS_GAME_CAP_CEILING_BUILD", "8500", float),
    "__ATLAS_PASSES__": ("ATLAS_PASSES_BUILD", "1", int),
    "__ATLAS_STOCK_PHASE_B__": ("ATLAS_STOCK_PHASE_B_BUILD", "0", _as_bool),
    "__ATLAS_STOCK__": ("ATLAS_STOCK_BUILD", "0", _as_bool),
}


def substitute_knobs(text: str, env: dict | None = None, *, announce: bool = False) -> str:
    """Replace every build-time placeholder in `text` with a literal.

    31.08, found the hard way: build() used to substitute ONLY the atlas
    cell, while RUN_CELL_PATCH -- which also carries __ATLAS_DRAWS__ -- was
    spliced into the run cell raw. The placeholder sat on a line that runs
    exclusively in the competition-rerun branch, so every Phase A stayed
    green while three real submissions in a row died on a NameError about
    half an hour in. Substitution is now text-agnostic and build() asserts
    that no placeholder survives anywhere in the notebook."""
    source = os.environ if env is None else env
    for placeholder, (var, default, parse) in CELL_KNOBS.items():
        if placeholder not in text:
            continue
        value = parse(source.get(var, default))
        text = text.replace(placeholder, repr(value))
        if announce:
            print(f"builder: {placeholder.strip('_')} -> {value}")
    return text


def substituted_cell(env: dict | None = None, *, announce: bool = False) -> str:
    """ATLAS_CELL with every build-time knob replaced by a literal.

    Shared with scripts/test_atlas_*.py so the tests exercise the text that
    really ships: before this existed the tests exec'd the raw template and
    died on the first placeholder, which quietly disabled the guard around
    atlas_fit_game_cap -- the one function that only ever runs in the real
    submission branch (a hand-traced change to it once scored 0.06)."""
    text = substitute_knobs(ATLAS_CELL, env, announce=announce)
    for placeholder in CELL_KNOBS:
        assert placeholder not in text, placeholder
    return text


def build() -> None:
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    pickle_idx = next(
        i for i, c in enumerate(cells) if "benchmark_initial.pkl" in _source_of(c)
    )
    hook_idx = next(
        i for i, c in enumerate(cells) if "Inline customization hook" in _source_of(c)
    )
    run_idx = next(
        i for i, c in enumerate(cells) if RUN_CELL_ANCHOR in _source_of(c)
    )
    assert pickle_idx < hook_idx < run_idx, "unexpected notebook layout"

    _set_source(cells[pickle_idx], _source_of(cells[pickle_idx]).rstrip("\n") + "\n" + PRISTINE_CAPTURE)
    # The run-cell patch carries build-time knobs too -- substitute it with
    # the SAME values as the atlas cell, or the rerun branch dies on a
    # NameError that no Phase A can ever reach (three burnt submissions).
    _set_source(
        cells[run_idx],
        _source_of(cells[run_idx]).replace(RUN_CELL_ANCHOR, substitute_knobs(RUN_CELL_PATCH), 1),
    )

    dataset_cell_idx = next(
        i for i, c in enumerate(cells) if OLD_SOURCE_DATASET in _source_of(c)
    )
    _set_source(
        cells[dataset_cell_idx],
        _source_of(cells[dataset_cell_idx]).replace(OLD_SOURCE_DATASET, NEW_SOURCE_DATASET, 1),
    )

    atlas_cell_text = substituted_cell(announce=True)
    cells.insert(hook_idx + 1, _new_cell(atlas_cell_text))

    # Last line of defence: NOTHING may reach Kaggle with a placeholder left
    # in it, in any cell, whether or not Phase A would ever execute that line.
    leftover = [ph for ph in CELL_KNOBS if any(ph in _source_of(c) for c in cells)]
    assert not leftover, f"unsubstituted placeholders would ship: {leftover}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n"
    )

    meta = json.loads((ROOT / "notebooks_duck" / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KERNEL_SLUG
    meta["title"] = KERNEL_TITLE
    meta["dataset_sources"] = [
        NEW_SOURCE_DATASET if ref == OLD_SOURCE_DATASET else ref
        for ref in meta["dataset_sources"]
    ]
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"wrote {OUT_NB} ({len(cells)} cells)")
    print(f"wrote {OUT_DIR / 'kernel-metadata.json'} -> {KERNEL_SLUG}")


if __name__ == "__main__":
    build()

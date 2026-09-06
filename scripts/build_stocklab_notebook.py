"""Build the stock-lab kernel: UNMODIFIED Duck, playing OUR testbed.

Why a separate kernel. From 02.09 the base is stock, not our fork -- measured
that evening: with every switch of ours off and the prompt back to three
tools, we still cost 1647 generated tokens per model call against stock's 815.
Stock is a different kernel, not ours with the knobs down.

This kernel exists to put a number on stock under OUR conditions. It mounts
the upstream source dataset (so the agent is untouched Duck) plus our own
dataset (only for the 30 testbed games), and swaps Phase A's game list. It is
a MEASUREMENT kernel and is never meant to be submitted: the own-games swap is
guarded on `not true_submission`, exactly as duck's own public-eval block is.

`arc3-duck-baseline` stays untouched -- version 1 of it is our best submission
asset (1.61 on the hidden set) and must not be disturbed.

usage:
    python scripts/build_stocklab_notebook.py
    ATLAS_STOCKLAB_CAP_S=1500 ATLAS_STOCKLAB_CONC=30 python scripts/build_stocklab_notebook.py
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_NB = ROOT / "kernels" / "notebooks_duck" / "submission.ipynb"
OUT_DIR = ROOT / (os.environ.get("ATLAS_STOCKLAB_OUT", "").strip()
                  or ("kernels/notebooks_stockbattle"
                      if os.environ.get("ATLAS_STOCKLAB_BATTLE", "0") not in ("0", "", "false", "no")
                      else "kernels/notebooks_stocklab"))
OUT_NB = OUT_DIR / "submission.ipynb"

# 04.09: ATLAS_STOCKLAB_KERNEL overrides the kernel slug, so a pure-stock
# control or a long re-run of an old hint never becomes a version of the
# battle kernel (the competition page submits a kernel VERSION; keep them apart).
KERNEL_ID = (os.environ.get("ATLAS_STOCKLAB_KERNEL", "").strip()
             or ("sergueimakarov/arc3-stock-battle"
                 if os.environ.get("ATLAS_STOCKLAB_BATTLE", "0") not in ("0", "", "false", "no")
                 else "sergueimakarov/arc3-stock-lab"))
# 03.09 evening: the lab used to mount arc3-atlas-src for our_games/. That
# dataset carries our forked src/ AND the same bundle marker as Duck's, and on
# sessions where Kaggle failed to mount Duck's dataset the notebook's finder
# picked OURS -- three "stock" runs (stocklab_batch, stocklab_batch3, v7)
# measured our fork while the log said stock. Now the games come from a
# marker-free dataset and the cell below refuses any bundle but the upstream.
OUR_DATASET = "sergueimakarov/arc3-own-games"
UPSTREAM_BUNDLE = "jakobbrggen"

CAP_S = float(os.environ.get("ATLAS_STOCKLAB_CAP_S", "1500"))
CONC = int(os.environ.get("ATLAS_STOCKLAB_CONC", "30"))
# per-game cap plus slack: the field must end together, not trail off
DEADLINE_S = int(os.environ.get("ATLAS_STOCKLAB_DEADLINE_S", str(int(CAP_S) + 200)))
BATCH = os.environ.get("ATLAS_STOCKLAB_BATCH", "0") not in ("0", "", "false", "no")
# Which addition rides on top of stock: "batch" (03.09 morning, turn-budget
# hint -- battle v1 scored 1.28 against stock's 1.61, no effect shown),
# "observe" (03.09 evening, the board-reading protocol below), "both", or
# "none". ATLAS_STOCKLAB_BATCH=1 is kept as a synonym for "batch".
HINT = os.environ.get("ATLAS_STOCKLAB_HINT", "batch" if BATCH else "none").strip().lower()
if HINT not in ("batch", "observe", "both", "protocol", "protocol3", "lean", "leanshort", "core", "none"):
    raise SystemExit(f"ATLAS_STOCKLAB_HINT must be batch|observe|both|protocol|protocol3|lean|leanshort|core|none, got {HINT!r}")
# Explicit game list. 03.09: only 11 of the 30 own games were rebuilt onto the
# public-curve geometry (L1 ~ 30, 21x21 board); the other 19 still sit on the
# old 8x8 layouts and must not be mixed into a measurement. Empty = all games
# under our_games/, as before. Any listed game missing from the dataset RAISES
# -- a run on a partial set is a green run that means nothing.
GAMES = [g for g in os.environ.get("ATLAS_STOCKLAB_GAMES", "").replace(",", " ").split() if g]
# 05.09: extra knobs for test variants (see patch_lean4.py)
ENVS = [kv.split("=", 1) for kv in os.environ.get("ATLAS_STOCKLAB_ENV", "").split(";") if "=" in kv]
LEAN_HINT_VARIANT = os.environ.get("ATLAS_STOCKLAB_LEAN_HINT", "").strip().lower()
VLLM_REASONING = os.environ.get("ATLAS_STOCKLAB_VLLM_REASONING", "0") not in ("0", "", "false", "no")
# BATTLE mode: a submittable build. The hint moves OUTSIDE the
# `not true_submission` guard (otherwise a competition rerun skips it and
# ships plain stock), and the own-games swap is dropped entirely -- Phase A
# stays on duck's public 25, because a submission build must never depend on
# a code path the rerun does not take.
BATTLE = os.environ.get("ATLAS_STOCKLAB_BATTLE", "0") not in ("0", "", "false", "no")

# The one and only change we make to stock's own prompt, and the reason for it.
#
# Measured 02.09 on stock playing our 30-game testbed, 25-minute cap:
#   a game fits about TWELVE model calls, ~2 minutes each. That ceiling is set
#   by wall clock and cannot be raised from inside the agent.
#   games that cleared a level: 12 calls, 2.17 actions per call, 28 actions
#   games that scored zero:     11 calls, 0.68 actions per call,  8 actions
#   fl01 (best, 40 points):      9 calls, 5.11 actions per call, 46 actions
#   cl01 (worst):               34 calls, ZERO actions in the whole game
# Same number of calls either way. The whole difference is how many actions
# the model puts in one answer.
#
# Stock already mentions batching in one clause. This makes the turn budget
# explicit, because the model has no way to know it otherwise.
#
# Why the downside is bounded: a wrong batch costs real actions, and actions
# only enter the score through min(baseline/actions, 1.0) on levels you DO
# complete. We complete so few that the efficiency term is nearly free, while
# a level not reached is a flat zero.
BATCH_HINT = (
    "\n- YOUR TURN BUDGET IS SMALL AND FIXED. A game affords roughly a dozen "
    "`python` calls in total -- not hundreds. Every call that returns without "
    "an `action(...)` spends one of them for nothing.\n"
    "- Therefore prefer an ORDERED BATCH over a single step whenever your world "
    "model supports more than one move: `action([...])` takes a list and each "
    "entry costs you nothing extra in turns. If you are only sure about the "
    "first few moves of a plan, batch those and inspect afterwards -- that is "
    "still far better than spending a whole turn on one step.\n"
    "- Reserve single-step turns for the case where you genuinely cannot "
    "predict the next state until you have seen this one.\n"
)

# 03.09 evening. Three of the eight public games where stock takes level 1 in
# 11-20 plays out of 20 and level 2 in ZERO were decoded on the engine (su15,
# lp85, tn36; the owner guessed the rules, the engine confirmed them, and the
# levels fell in 14, 8 and 10 actions against human baselines of 42, 38 and
# 72). The model lost each one the same way: it never measured a control's
# reach, never noticed an unexpected merge or colour change, never read a
# shrinking bar as a budget, never matched a hollow outline with the filled
# block of the same colour, never told an example panel from its workspace,
# and mistook animation frames eating its clicks for "nothing happens". None
# of that is a rule of any one game. Stock already says "verify whether
# gameplay objects changed"; the model reads that and does not do it, so
# this is written as a numbered protocol with concrete actions, not advice.
OBSERVE_HINT = (
    '\n- NEW-LEVEL PROTOCOL. On the first turn of every level (and whenever the board layout changes in a way you did not cause) run these steps IN ORDER, before any plan:\n'
    '  1. INVENTORY from `segmentation`: group objects by colour and shape. Name explicitly: duplicated objects (pairs, identical dots); a hollow outline and a filled block of the SAME colour; a row of small squares of distinct colours in a corner (a legend); a long thin bar or a row of dots (a counter). For each structure write one candidate goal: same colour hollow+filled -> the filled block belongs inside the outline; duplicates -> combine them; legend -> an order or a colour mapping; counter -> a budget of actions. Like goes to like; a complement waits for its partner.\n'
    '  2. PANELS: if two similar regions exist (left/right, top/bottom), one is usually a read-only example and the other your workspace. Click one cell in each and keep the one that changes.\n'
    '  3. CALIBRATE every control ONCE before repeating it: click far away, click near, click on an object, click the same spot twice. Record the reach (the largest distance that still moved something) and what a click on an object does. Never move one cell at a time before you know the reach.\n'
    '  4. AFTER EVERY ACTION diff before/after. If an object vanished, merged, grew, or changed colour, that is a RULE of this level: write it into the world model before doing anything else. If NOTHING changed, either the click was out of reach or the game is still animating your previous command -- take one neutral action and look again before you change your hypothesis.\n'
    '  5. If a bar or a row of dots shrinks with each action, count the remaining actions and plan inside that number; do not spend it on exploration.\n'
    "  6. A new level is a new game: the previous level's goal is a hypothesis to re-check in step 1, not a fact.\n"
)


PROTOCOL_HINT = (
    '\n- MANDATORY NEW-LEVEL PROTOCOL (harness-enforced). Every `python` call MUST begin with a dict literal named PROTOCOL filled from `segmentation`, `previous_frame` and `last_action_result`; code without it is rejected without running. Fill it honestly -- it is your own working memory, not a form:\n'
    '  PROTOCOL = {\n'
    "    'inventory': '...',    # groups by colour/shape: UI bars or dot rows (budget counters), corner palettes (legends), hollow outline + filled shape of the same colour, duplicate groups, symmetric left/right or top/bottom panels\n"
    "    'diff': '...',         # previous_frame vs current_frame: what moved, merged, vanished, changed colour or size; 'zero' if nothing changed\n"
    "    'rule': '...',         # a state-transition rule you now believe, or 'none yet'\n"
    "    'budget': '...',       # remaining actions if a bar/counter shrinks per action, else 'unknown'\n"
    "    'hypothesis': '...',   # ONE goal hypothesis: hollow+filled of one colour -> destination; duplicates -> combine; legend -> mapping or order; two panels -> one is a read-only demo, the other your workspace\n"
    "    'next': '...',         # the single probe or the plan this call executes, and why\n"
    '  }\n'
    "  Rules that go with it: on turn 1 of a level, and whenever the layout changes in a way you did not cause, rebuild 'inventory' and 'hypothesis' from scratch -- a new level is a new game. Spend ONE probing click to learn a control's reach or what a click on an object does; never move one cell at a time before you know the reach. On a zero diff do NOT change your hypothesis: the click was out of reach or the game is still animating your previous command -- look again first. When a bar or dot row shrinks, every plan must finish inside the remaining count.\n"
)

# Protocol v3 (04.09, after all eight "wall" games were decoded on the engine --
# su15 lp85 tn36 r11l bp35 lf52 sp80 ka59, see docs/improvement_backlog_22.08.md).
# Same dict, two more keys and four rules. Each addition names one failure
# class that killed the stock agent on level 2 of a public game:
#   'win'   -- the win condition was never derived (lf52 solitaire, ka59 sockets)
#   'reach' -- a control's range was never measured (ka59 shove flies 5 cells
#              through walls; lp85/r11l manipulators; sp80 arrows remapped)
#   expensive probe -- a trial that costs one of N attempts (sp80: 4 spills)
#   lethal / hidden dynamics -- bp35 spikes and the tide that rises every 2 moves
#   chain -- output of one mechanic is input of the next (sp80, lf52)
PROTOCOL3_HINT = PROTOCOL_HINT.replace(
    "    'next': '...',",
    "    'win': '...',          # what exactly ends this level, derived BEFORE the first move: every X inside a Y of matching shape/colour; all cups filled; one piece left; N of M steps\n"
    "    'reach': '...',        # per control, measured by ONE probe: cells per press, what a click on an object does, how far a shove carries an object (may be several cells and THROUGH walls), whether arrows are remapped on this level\n"
    "    'next': '...',",
) + (
    "  Expensive probe: if a counter of attempts/lives shrinks on failure (a spill, a launch, a reset after a wrong move), do NOT probe by trying -- simulate the outcome in your code from the rule you hold, and only then act.\n"
    "  Lethal and moving things: list in 'inventory' every object that ends the game on contact (spikes, traps that look like the goal but differ in shape) and everything that moves without your input, with its period; every plan must avoid the former and beat the latter's deadline.\n"
    "  Chains: when one mechanic's output feeds another (a splitter feeding a splitter, a carrier that moves a piece to another zone), plan the whole chain end-to-end in 'next' before the first move; a single control never solves a level with several zones.\n"
    "  Keep the hypothesis until a diff contradicts it; a zero diff never does.\n"
)

# Harness-side enforcement for the 'protocol' hint. Applied at runtime by the
# notebook cell (the stock source is never edited), OUTSIDE any
# true_submission guard, so a competition rerun gets exactly what Phase A
# measured. Two wrappers:
#   * run_sandboxed_python: refuse code that does not start with the PROTOCOL
#     dict -- the model gets an error text back and spends one call, but no
#     action and no budget;
#   * ToolAgent._build_user_prompt: append a NOTICE when the previous call
#     changed nothing on the board (streak counted per agent), and a two-line
#     reminder of the PROTOCOL requirement right before the model answers --
#     Gemini round 17: instructions at the END of the context are the ones a
#     27B model at 30k tokens actually follows.
PROTOCOL_RUNTIME = r"""
import re as _pr_re
import inference.agent.tool_agent as _pr_agent
_pr_orig_run = _pr_agent.run_sandboxed_python
_PR_HEAD = _pr_re.compile(r"^\s*(?:#[^\n]*\n|\s*\n)*PROTOCOL\s*=\s*\{", _pr_re.S)
def _pr_run(*args, **kwargs):
    code = kwargs.get("code", "")
    if not _PR_HEAD.match(code or ""):
        return {"error": ("Rejected before execution: the python code must START with the "
                          "PROTOCOL = {...} dict (keys __KEYS__). "
                          "Fill it from segmentation/previous_frame/last_action_result, then put your "
                          "code after it. No action was spent."),
                "stdout": "", "action_results": []}
    return _pr_orig_run(*args, **kwargs)
_pr_agent.run_sandboxed_python = _pr_run
_pr_orig_prompt = _pr_agent.ToolAgent._build_user_prompt
def _pr_prompt(self, action_num, *args, **kwargs):
    text = _pr_orig_prompt(self, action_num, *args, **kwargs)
    summary = kwargs.get("previous_step_summary")
    streak = getattr(self, "_pr_zero_streak", 0)
    if summary:
        try:
            executed = int(summary.get("executed_count") or 0)
        except (TypeError, ValueError):
            executed = 0
        if executed > 0 and not summary.get("board_changed") and not summary.get("level_transition"):
            streak += 1
        elif executed > 0:
            streak = 0
    self._pr_zero_streak = streak
    extra = []
    if streak >= 1:
        extra.append(f"[NOTICE] The previous call executed actions that changed NOTHING on the board "
                     f"({streak} zero-diff call{'s' if streak > 1 else ''} in a row). Either the click was out of "
                     f"reach, the target was not interactive, or the game is still animating a previous command. "
                     f"Do not change your hypothesis on a zero diff; record it in PROTOCOL['diff'] and look again.")
    extra.append("[PROTOCOL] Your python code must START with the PROTOCOL = {...} dict "
                 "(__KEYS__); code without it is rejected unrun.")
    return text + "\n" + "\n".join(extra)
_pr_agent.ToolAgent._build_user_prompt = _pr_prompt
print("protocol: harness wrappers installed (PROTOCOL header required; zero-diff notice)")
"""

LEAN_HINT = (
    "\\n- HARNESS AIDS (read them, never recompute them). Each turn's prompt carries: [STATE] -- level, step, board size and the object table (letter, size, rows, cols) of the current board; [CONTROLS] -- on a new level the harness itself pressed each arrow once before your first call and lists what moved and by how much (these presses are already spent; do not repeat them); [DIFF] -- exactly what changed since your previous call (cells, colour transitions, components that appeared, vanished or moved), 'zero' means nothing changed: out of reach, not interactive, or still animating -- keep your hypothesis; [HUD] -- a bar that shrinks per action and the remaining count, every plan must finish inside it; [NOTICE] -- zero-diff streak; [SHORT] -- the previous call spent fewer than 5 actions.\\n"
    '- HELPERS preloaded in every python call (pure functions over `current_frame`, letters as in [STATE]): move_to(target, me=None) -> finds the path from your object (letter or (row, col); default: the object that moved last) to the target (letter, (row, col) or a list of cells) treating the large background letters as walkable and everything else as walls, then EXECUTES it in one action() batch and returns {path, executed, diff}; pass passable={letters} or step=N (cells per key press) when the default is wrong; find_path(start, goal, passable, step=1) -> the same BFS without executing; grid() -> list of row strings; cells(letter) -> [(row, col)]; objects(min_size=1, max_items=60) -> [{letter, size, r0, c0, r1, c1}] largest first; frame_diff() -> {changed, bbox, moved, appeared, vanished} between previous_frame and current_frame; move_until(action, max_steps=30) -> repeats one action until the board stops changing or the level or game ends. Never count cells in your head: locate with objects()/cells(), then move_to().\\n'
    '- EVENT PROTOCOL (harness-enforced). On the first turn of a level, after a zero-diff call and after a game over, the first python call that executes actions MUST begin with:\\n'
    '  PROTOCOL = {\\n'
    "    'win': '...',    # what exactly ends this level, derived before moving: every X inside a Y of matching shape or colour; all cups filled; one piece left; N of M\\n"
    "    'rule': '...',   # the mechanic you believe, incl. the reach of each control (cells per press, what a click on an object does, how far a shove carries -- possibly through walls), what kills, what moves by itself and its period, whether a failed try costs one of N attempts\\n"
    "    'plan': '...',   # the sequence this call executes and why\\n"
    '  }\\n'
    '  Inspection-only calls never need it. On routine turns just act.\\n'
    '- ACT IN BATCHES: a call that executes actions should run 5 or more of them (a find_path() result, or a loop that calls action(...) step by step and checks frame_diff() between steps); one or two actions per call is only right for a single probe of a control that [CONTROLS] did not cover, or right after a level ended.\\n'
    '- EXPENSIVE PROBES: if a failed try costs an attempt or a life (a counter that drops, a reset after a wrong move), simulate the outcome in code from the rule you hold before acting; never test by trying.\\n'
)

# Harness side of the 'lean' hint (Gemini round 18 + owner's 04.09 decision:
# helpers in the sandbox and a compact per-turn prompt). Wraps four points of
# the stock ToolAgent at runtime, OUTSIDE any true_submission guard:
#   * _build_user_prompt   -> drops the boilerplate sentences that repeat the
#     system prompt, appends [STATE] (object table), [DIFF] / [HUD] / [NOTICE]
#     / [SHORT] and, on event turns, [THINK] + the PROTOCOL requirement;
#   * _run_python_tool     -> on event turns rejects the first action-bearing
#     call that lacks the PROTOCOL header (no action spent);
#   * run_sandboxed_python -> prepends the helper functions to the model's code;
#   * build_chat_payload   -> enable_thinking only when the latest user message
#     carries [THINK] (stateless: safe with one agent per game thread).
LEAN_RUNTIME = r"""
import re as _ln_re, json as _ln_json
from collections import deque as _ln_deque, Counter as _ln_Counter
import inference.agent.tool_agent as _ln_agent

def _ln_stable_hash(s):
    h = 1469598103934665603
    for ch in s:
        h = ((h ^ ord(ch)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h
_LN_HEAD = _ln_re.compile(r"^\s*(?:#[^\n]*\n|\s*\n)*PROTOCOL\s*=\s*\{", _ln_re.S)
_LN_ACTS = _ln_re.compile(r"\baction\s*\(")
_LN_MIN_BATCH = 5
_LN_DROP = ("Only tool: `python`.", "Only letter-coded board views", "Keep tool output compact",
            "For the most recent change, compare", "Use Python to inspect the evidence",
            "Maintain a compact working world model", "Below you are provided with the current world model",
            "You may call `action(actions)` more than once", "Ground yourself in `current_frame`",
            "Focus on what changed most recently", "When ready, call `action(actions)`",
            "If you include assistant text before a tool call", "When calling `python`, emit exactly")

_LN_HELPERS = r'''
from collections import deque as _h_deque
_H_MOVES = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
def _h_stable_hash(s):
    h = 1469598103934665603
    for ch in s:
        h = ((h ^ ord(ch)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h
_h_orig_action = action
def action(acts, force=False):
    res = _h_orig_action(acts)
    try:
        _flag = _H_LOOP_BREAK          # defined by the harness only when ATLAS_LEAN_LOOP_BREAK=1 (globals() is not a sandbox builtin)
        seen = _H_SEEN
        if 'loop' not in res and _flag and current_frame is not None:
            hs = _h_stable_hash(str(current_frame.ascii))
            if hs in seen and not force:
                res['loop'] = True
                res['note'] = 'the board returned to a state seen earlier in this game; this sequence undoes itself -- stop and change the plan (pass force=True to override)'
            seen.add(hs)
    except Exception:
        pass
    return res
def grid(frame=None):
    f = current_frame if frame is None else frame
    return [] if f is None else str(f.ascii).split('\n')
def cells(letter, frame=None):
    g = grid(frame)
    return [(r, c) for r in range(len(g)) for c in range(len(g[r])) if g[r][c] == letter]
def _h_components(g):
    rows = len(g); seen = set(); out = []
    for r0 in range(rows):
        for c0 in range(len(g[r0])):
            if (r0, c0) in seen:
                continue
            ch = g[r0][c0]; q = _h_deque([(r0, c0)]); seen.add((r0, c0)); n = 0
            rmin = rmax = r0; cmin = cmax = c0
            while q:
                r, c = q.popleft(); n += 1
                if r < rmin: rmin = r
                if r > rmax: rmax = r
                if c < cmin: cmin = c
                if c > cmax: cmax = c
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < len(g[nr]) and (nr, nc) not in seen and g[nr][nc] == ch:
                        seen.add((nr, nc)); q.append((nr, nc))
            out.append({'letter': ch, 'size': n, 'r0': rmin, 'c0': cmin, 'r1': rmax, 'c1': cmax})
    return out
def objects(frame=None, min_size=1, max_items=60):
    comps = [o for o in _h_components(grid(frame)) if o['size'] >= min_size]
    comps.sort(key=lambda o: -o['size'])
    return comps[:max_items]
def frame_diff(before=None, after=None):
    a = grid(previous_frame if before is None else before); b = grid(current_frame if after is None else after)
    if not a or not b:
        return {'changed': None, 'note': 'no previous frame'}
    changed = [(r, c) for r in range(min(len(a), len(b))) for c in range(min(len(a[r]), len(b[r]))) if a[r][c] != b[r][c]]
    key = lambda o: (o['letter'], o['size'], o['r1'] - o['r0'], o['c1'] - o['c0'])
    ca = {}
    for o in _h_components(a):
        ca.setdefault(key(o), []).append((o['r0'], o['c0']))
    moved = []; appeared = []
    for o in _h_components(b):
        k = key(o); pos = (o['r0'], o['c0'])
        if k in ca and ca[k]:
            if pos in ca[k]:
                ca[k].remove(pos)
            else:
                src = ca[k].pop(0)
                moved.append({'letter': o['letter'], 'size': o['size'], 'from': src, 'to': pos, 'delta': (pos[0] - src[0], pos[1] - src[1])})
        else:
            appeared.append({'letter': o['letter'], 'size': o['size'], 'at': pos})
    vanished = [{'letter': k[0], 'size': k[1], 'at': p} for k, ps in ca.items() for p in ps]
    bbox = None
    if changed:
        bbox = (min(r for r, _ in changed), min(c for _, c in changed), max(r for r, _ in changed), max(c for _, c in changed))
    return {'changed': len(changed), 'bbox': bbox, 'moved': moved[:8], 'appeared': appeared[:8], 'vanished': vanished[:8]}
def probe(act):
    res = action([act] if isinstance(act, (str, dict)) else act)
    d = frame_diff()
    d['result'] = {k: res.get(k) for k in ('board_changed', 'level_completed', 'game_over', 'executed_count', 'stop_reason') if k in res}
    return d
def move_until(act, max_steps=30):
    steps = 0; last = None
    for _ in range(max_steps):
        res = action([act]); steps += 1; last = frame_diff()
        if res.get('game_over') or res.get('level_completed') or res.get('done') or not res.get('board_changed'):
            break
    return {'steps': steps, 'last_diff': last}
def _h_background(g, frac=0.15):
    from collections import Counter as _h_Counter
    n = sum(len(r) for r in g) or 1
    return {ch for ch, k in _h_Counter(ch for r in g for ch in r).items() if k >= frac * n}
def move_to(target, me=None, passable=None, step=1, max_len=200):
    g = grid()
    if me is None:
        d = frame_diff()
        mv = d.get('moved') if isinstance(d, dict) else None
        if not mv:
            return {'error': 'move_to: pass me=(row, col) or me=letter -- no object moved in the last diff'}
        me = mv[0]['to']; me_letter = mv[0]['letter']
    if isinstance(me, str):
        cs = cells(me)
        if not cs:
            return {'error': 'move_to: no cell with letter ' + me}
        me_letter = me; me = cs[0]
    else:
        me = tuple(me); me_letter = g[me[0]][me[1]]
    if isinstance(target, str):
        goals = cells(target); tl = target
    elif isinstance(target, (list, set, frozenset)) and target and isinstance(next(iter(target)), (tuple, list)):
        goals = [tuple(x) for x in target]; tl = None
    else:
        goals = [tuple(target)]; tl = None
    if not goals:
        return {'error': 'move_to: target not found on the board'}
    if tl is None:
        tl = g[goals[0][0]][goals[0][1]]
    ok = set(passable) if passable else _h_background(g)
    ok = ok | {me_letter, tl}
    path = find_path(me, goals, ok, step=step)
    if path is None:
        return {'error': 'move_to: no path', 'passable_used': sorted(ok), 'from': me, 'goals': goals[:5]}
    path = path[:max_len]
    res = action(path)
    return {'path': path, 'executed': res.get('executed_count', len(path)), 'result': {k: res.get(k) for k in ('board_changed', 'level_completed', 'game_over', 'stop_reason') if k in res}, 'diff': frame_diff()}
def find_path(start, goal, passable, step=1, moves=None, frame=None):
    g = grid(frame); rows = len(g)
    if callable(passable):
        ok = passable
    else:
        allowed = set(passable); ok = lambda ch: ch in allowed
    if isinstance(goal, (list, set, frozenset)) and goal and isinstance(next(iter(goal)), (tuple, list)):
        goals = set(tuple(x) for x in goal)
    else:
        goals = {tuple(goal)}
    mv = moves or _H_MOVES
    start = tuple(start); prev = {start: None}; q = _h_deque([start])
    while q:
        cur = q.popleft()
        if cur in goals:
            path = []
            while prev[cur] is not None:
                parent, m = prev[cur]; path.append(m); cur = parent
            return path[::-1]
        for name, (dr, dc) in mv.items():
            nr, nc = cur[0] + dr * step, cur[1] + dc * step
            if not (0 <= nr < rows and 0 <= nc < len(g[nr])):
                continue
            if any(not ok(g[cur[0] + dr * k][cur[1] + dc * k]) for k in range(1, step + 1)):
                continue
            if (nr, nc) not in prev:
                prev[(nr, nc)] = (cur, name); q.append((nr, nc))
    return None
'''

def _ln_rows(frame):
    if frame is None:
        return None
    a = getattr(frame, "ascii", None)
    if isinstance(a, str) and a:
        return a.split("\n")
    g = getattr(frame, "grid", None)
    if g is None:
        return None
    return ["".join("ABCDEFGHIJKLMNOP"[int(v) % 16] for v in r) for r in g]

def _ln_components(grid):
    rows = len(grid); seen = set(); comps = []
    for r0 in range(rows):
        for c0 in range(len(grid[r0])):
            if (r0, c0) in seen:
                continue
            col = grid[r0][c0]; seen.add((r0, c0))
            q = _ln_deque([(r0, c0)]); cells = 0; rmin = rmax = r0; cmin = cmax = c0
            while q:
                r, c = q.popleft(); cells += 1
                rmin = min(rmin, r); rmax = max(rmax, r); cmin = min(cmin, c); cmax = max(cmax, c)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < len(grid[nr]) and (nr, nc) not in seen and grid[nr][nc] == col:
                        seen.add((nr, nc)); q.append((nr, nc))
            comps.append((col, cells, rmin, cmin, rmax, cmax))
    return comps

def _ln_state_text(rows, level, step, max_items=24):
    if not rows:
        return "[STATE] no frame."
    comps = sorted(_ln_components(rows), key=lambda t: -t[1])
    counts = _ln_Counter(ch for r in rows for ch in r)
    head = f"[STATE] level {level}, step {step}, board {len(rows)}x{max(len(r) for r in rows)}; letters: " + \
           ", ".join(f"{k} {v}" for k, v in counts.most_common())
    items = [f"{col} {cells} r{rmin}-{rmax} c{cmin}-{cmax}" for col, cells, rmin, cmin, rmax, cmax in comps[:max_items]]
    more = f" (+{len(comps) - max_items} smaller)" if len(comps) > max_items else ""
    return head + f"\n  objects (letter size rows cols), largest first, {len(comps)} total{more}: " + "; ".join(items) + "."

def _ln_bar(prev, cur):
    out = []
    def runs(line):
        res = []; i = 0
        while i < len(line):
            j = i
            while j < len(line) and line[j] == line[i]:
                j += 1
            res.append((line[i], i, j - i)); i = j
        return res
    def check(name, idx, a, b):
        if a == b or len(a) != len(b):
            return
        ra, rb = runs(a), runs(b)
        if len(ra) != len(rb) or len(ra) > 4:
            return
        diffs = [(x, y) for x, y in zip(ra, rb) if x != y]
        if len(diffs) != 2:
            return
        (ca, sa, la), (cb, sb, lb) = diffs[0]
        (ca2, sa2, la2), (cb2, sb2, lb2) = diffs[1]
        if ca == cb and ca2 == cb2 and la > lb and la2 < lb2 and la + la2 == lb + lb2:
            out.append((name, idx, ca, la, lb))
    for r in range(min(len(prev), len(cur))):
        check("row", r, prev[r], cur[r])
    cols = min(max((len(r) for r in prev), default=0), max((len(r) for r in cur), default=0))
    for c in range(cols):
        a = tuple(row[c] for row in prev if c < len(row)); b = tuple(row[c] for row in cur if c < len(row))
        check("col", c, a, b)
    return out

def _ln_diff_text(prev, cur, executed):
    if prev is None or cur is None:
        return "[DIFF] no previous frame yet."
    changed = []
    for r in range(min(len(prev), len(cur))):
        pr, cr = prev[r], cur[r]
        for c in range(min(len(pr), len(cr))):
            if pr[c] != cr[c]:
                changed.append((r, c, pr[c], cr[c]))
    if not changed:
        return "[DIFF] zero -- no cell changed since your previous call."
    rs = [x[0] for x in changed]; cs = [x[1] for x in changed]
    trans = _ln_Counter((a, b) for _, _, a, b in changed).most_common(4)
    lines = [f"[DIFF] {len(changed)} cells changed in rows {min(rs)}-{max(rs)}, cols {min(cs)}-{max(cs)}; "
             f"colour transitions: " + ", ".join(f"{a}->{b} x{n}" for (a, b), n in trans) + "."]
    try:
        key = lambda t: (t[0], t[1], t[4] - t[2] + 1, t[5] - t[3] + 1)
        pc = _ln_Counter(key(t) for t in _ln_components(prev) if t[1] <= 400)
        cc = _ln_Counter(key(t) for t in _ln_components(cur) if t[1] <= 400)
        gone = list((pc - cc).elements()); new = list((cc - pc).elements())
        if gone or new:
            fmt = lambda k: f"{k[0]} {k[2]}x{k[3]} ({k[1]} cells)"
            parts = []
            if gone:
                parts.append("vanished: " + ", ".join(fmt(k) for k in gone[:5]) + (" ..." if len(gone) > 5 else ""))
            if new:
                parts.append("appeared: " + ", ".join(fmt(k) for k in new[:5]) + (" ..." if len(new) > 5 else ""))
            lines.append("  components " + "; ".join(parts) + ".")
        moved = []
        if gone and new:
            pmap = {}
            for t in _ln_components(prev):
                pmap.setdefault(key(t), []).append((t[2], t[3]))
            for t in _ln_components(cur):
                k = key(t)
                if k in pmap and pmap[k] and (t[2], t[3]) not in pmap[k]:
                    r0, c0 = pmap[k].pop(0)
                    moved.append(f"{t[0]} {k[2]}x{k[3]} from (r{r0},c{c0}) to (r{t[2]},c{t[3]})")
        if moved:
            lines.append("  moved: " + "; ".join(moved[:4]) + ".")
    except Exception:
        pass
    try:
        for name, idx, letter, la, lb in _ln_bar(prev, cur)[:2]:
            rate = (la - lb) / max(1, executed or 1)
            left = int(lb / rate) if rate > 0 else lb
            lines.append(f"[HUD] {name} {idx}: a {letter} bar shrank {la}->{lb} cells "
                         f"({la - lb} for {executed or '?'} actions) -> about {left} actions left if it is a budget.")
    except Exception:
        pass
    return "\n".join(lines)

_ln_orig_prompt = _ln_agent.ToolAgent._build_user_prompt
def _ln_prompt(self, action_num, *args, **kwargs):
    text = _ln_orig_prompt(self, action_num, *args, **kwargs)
    kept = [l for l in text.split("\n") if not any(l.startswith(pfx) for pfx in _LN_DROP)]
    text = "\n".join(kept)
    frame = kwargs.get("current_frame"); summary = kwargs.get("previous_step_summary") or {}
    rows = _ln_rows(frame); level = getattr(frame, "level", None); step = getattr(frame, "step", None)
    try:
        executed = int(summary.get("executed_count") or 0)
    except (TypeError, ValueError):
        executed = 0
    prev = getattr(self, "_ln_prev_rows", None); prev_level = getattr(self, "_ln_prev_level", None)
    new_level = prev_level is None or (level is not None and level != prev_level) or bool(summary.get("level_transition"))
    zero = executed > 0 and not summary.get("board_changed") and not summary.get("level_transition")
    streak = getattr(self, "_ln_zero_streak", 0)
    if zero:
        streak += 1
    elif executed > 0:
        streak = 0
    self._ln_zero_streak = streak
    extra = [_ln_state_text(rows, level, step)]
    if new_level:
        extra.append("[DIFF] new level -- rebuild win and rule from scratch; the previous level's rule is only a hypothesis here.")
    else:
        extra.append(_ln_diff_text(prev, rows, executed))
    if streak >= 1:
        extra.append(f"[NOTICE] {streak} zero-diff call{'s' if streak > 1 else ''} in a row: out of reach, not interactive, or still animating. Keep the hypothesis; look again before changing it.")
    ctrl = getattr(self, "_ln_controls", None)
    if ctrl and ctrl[0] == level:
        extra.append(ctrl[1])
    probed = getattr(self, "_ln_probe_just_ran", False); self._ln_probe_just_ran = False
    if executed and executed < _LN_MIN_BATCH and not summary.get("level_transition") and not summary.get("game_over") and not new_level and not probed:
        extra.append(f"[SHORT] the previous call spent only {executed} action{'s' if executed > 1 else ''}; batch 5+ (find_path + action) or loop with checks unless this was a single probe.")
    event = new_level or zero or bool(summary.get("game_over"))
    self._ln_protocol_required = event
    self._ln_event_turn = event
    try:
        hud_left = None
        for name, idx, letter, la, lb in (_ln_bar(prev, rows) if (prev is not None and rows is not None) else [])[:1]:
            rate = (la - lb) / max(1, executed or 1)
            hud_left = int(lb / rate) if rate > 0 else None
        self._ln_hud_left = hud_left
    except Exception:
        self._ln_hud_left = None
    try:
        if rows is not None:
            hsh = _ln_stable_hash("\n".join(rows))
            seen = getattr(self, "_ln_seen", None)
            if seen is None or new_level:
                seen = set(); self._ln_seen = seen; self._ln_seen_turn = {}
            if hsh in seen and not new_level and executed > 0:
                extra.append(f"[LOOP] the board is identical to a state you already saw {action_num - self._ln_seen_turn.get(hsh, action_num)} calls ago -- the last actions undid earlier ones; do not repeat that cycle.")
            seen.add(hsh); self._ln_seen_turn.setdefault(hsh, action_num)
    except Exception:
        pass
    if event:
        extra.append("[THINK] Event turn: the first python call that executes actions must start with PROTOCOL = {'win': ..., 'rule': ..., 'plan': ...}; think it through.")
    self._ln_prev_rows = rows; self._ln_prev_level = level
    return text + "\n" + "\n".join(extra)
_ln_agent.ToolAgent._build_user_prompt = _ln_prompt

_LN_PROBE_CODE = r'''
_ctrl = []
for _a in __ARROWS__:
    _r = action([_a])
    _d = frame_diff()
    _mv = _d.get('moved') or []
    if _r.get('game_over') or _r.get('level_completed') or _r.get('done'):
        _ctrl.append(_a + ': ' + ('GAME OVER' if _r.get('game_over') else 'level changed')); break
    if not _r.get('board_changed') and not _d.get('changed'):
        _ctrl.append(_a + ': zero diff')
    elif _mv:
        _ctrl.append(_a + ': ' + '; '.join('%s %d cells moved (%d,%d)' % (m['letter'], m['size'], m['delta'][0], m['delta'][1]) for m in _mv[:3]))
    else:
        _ctrl.append(_a + ': %d cells changed%s' % (_d.get('changed') or 0, (', appeared ' + ','.join(x['letter'] for x in (_d.get('appeared') or [])[:3])) if _d.get('appeared') else ''))
print('[CONTROLS] harness pressed each arrow once (these actions are spent): ' + ' | '.join(_ctrl))
'''
_ln_orig_analyze = _ln_agent.ToolAgent.analyze
def _ln_analyze(self, state_path, action_num, valid_actions=None, step_env=None, *args, **kwargs):
    import os as _ln_os2
    level = None
    try:
        if _ln_os2.environ.get("ATLAS_LEAN_AUTOPROBE", "1").strip().lower() not in ("0", "false", "no") and state_path.exists():
            self._ensure_session(state_path)
            self._step_env_callback = step_env
            self._current_valid_actions = _ln_agent._normalize_valid_actions(valid_actions)
            frame, _hist = _ln_agent.load_runtime_state(state_path)
            level = getattr(frame, "level", None)
            arrows = [a for a in ("UP", "DOWN", "LEFT", "RIGHT") if a in set(self._current_valid_actions)]
            if level is not None and getattr(self, "_ln_probe_level", None) != level and arrows and step_env is not None:
                self._ln_probe_level = level
                res = self._run_python_tool(state_path, {"code": _LN_PROBE_CODE.replace("__ARROWS__", repr(arrows))})
                text = getattr(res, "content", "") or ""
                m = _ln_re.search(r"\[CONTROLS\][^\n]*", text)
                self._ln_controls = (level, m.group(0) if m else "[CONTROLS] probe ran but produced no summary")
                self._ln_probe_just_ran = True
                self._ln_zero_streak = 0
    except Exception as exc:
        self._ln_controls = (level, "[CONTROLS] auto-probe failed: %s: %s" % (type(exc).__name__, str(exc)[:120]))
    return _ln_orig_analyze(self, state_path, action_num, valid_actions, step_env, *args, **kwargs)
_ln_agent.ToolAgent.analyze = _ln_analyze

_ln_orig_run = _ln_agent.ToolAgent._run_python_tool
def _ln_run(self, state_path, arguments):
    code = str((arguments or {}).get("code", "") or "")
    if getattr(self, "_ln_protocol_required", False) and _LN_ACTS.search(code):
        if not _LN_HEAD.match(code):
            return _ln_agent._ToolDispatchResult(_ln_json.dumps({"error": (
                "Rejected before execution: this is an event turn (new level, zero diff or game over), so the first "
                "python call that executes actions must START with PROTOCOL = {'win': ..., 'rule': ..., 'plan': ...}. "
                "Inspection-only code needs no header. No action was spent.")}, indent=2))
        self._ln_protocol_required = False
    import os as _ln_os3
    if _LN_ACTS.search(code):
        lits = _LN_ACT_LIST.findall(code)
        n_lit = sum(len(_LN_ACT_ITEM.findall(body)) for body in lits)
        has_loop = bool(_ln_re.search(r"^\s*(for|while)\b", code, _ln_re.M)) or "move_to(" in code or "move_until(" in code or "*" in "".join(lits)
        if _ln_os3.environ.get("ATLAS_LEAN_MIN_BATCH_HARD", "").strip() in ("1", "true", "yes") and lits and not has_loop and 0 < n_lit < _LN_MIN_BATCH and not getattr(self, "_ln_event_turn", False):
            return _ln_agent._ToolDispatchResult(_ln_json.dumps({"error": (
                f"Rejected before execution: this call would spend only {n_lit} action(s). Batch 5 or more (a computed path, a loop that "
                "checks frame_diff() between steps, or move_to), or explain in PROTOCOL why a single probe is needed. No action was spent.")}, indent=2))
        left = getattr(self, "_ln_hud_left", None)
        if _ln_os3.environ.get("ATLAS_LEAN_HUD_BLOCK", "").strip() in ("1", "true", "yes") and left is not None and lits and not has_loop and n_lit > left:
            return _ln_agent._ToolDispatchResult(_ln_json.dumps({"error": (
                f"Rejected before execution: the plan has {n_lit} actions but the [HUD] bar leaves about {left}. Plan inside the remaining budget. No action was spent.")}, indent=2))
    if _ln_os3.environ.get("ATLAS_LEAN_LOOP_BREAK", "").strip() in ("1", "true", "yes"):
        seen = sorted(getattr(self, "_ln_seen", set()))
        arguments = dict(arguments); arguments["code"] = "_H_SEEN = set(" + repr(seen) + ")\n_H_LOOP_BREAK = True\n" + code
    return _ln_orig_run(self, state_path, arguments)
_ln_agent.ToolAgent._run_python_tool = _ln_run
_LN_ACT_LIST = _ln_re.compile(r"action\s*\(\s*\[(.*?)\]\s*\)", _ln_re.S)
_LN_ACT_ITEM = _ln_re.compile(r"'[A-Z0-9_]+'|\"[A-Z0-9_]+\"|\{[^{}]*\}")

_ln_orig_sandbox = _ln_agent.run_sandboxed_python
def _ln_sandbox(*args, **kwargs):
    code = kwargs.get("code")
    if isinstance(code, str):
        kwargs["code"] = _LN_HELPERS + "\n" + code
    return _ln_orig_sandbox(*args, **kwargs)
_ln_agent.run_sandboxed_python = _ln_sandbox

_ln_orig_payload = _ln_agent.build_chat_payload
def _ln_payload(*args, **kwargs):
    msgs = kwargs.get("messages") or []
    think = False
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(str(part.get("text", "")) for part in c if isinstance(part, dict))
            think = "[THINK]" in str(c or "")
            break
    marker = think          # event turn = the prompt carries [THINK]; kept before the default override below
    import os as _ln_os
    # DEFAULT = variant B (pod 04.09: RHAE 5.44 vs 0.25 for events-only thinking): think on every turn.
    # ATLAS_LEAN_THINK_EVENTS_ONLY=1 restores variant A (thinking only on [THINK] turns) for experiments.
    if _ln_os.environ.get("ATLAS_LEAN_THINK_EVENTS_ONLY", "").strip().lower() not in ("1", "true", "yes"):
        think = True
    budget = _ln_os.environ.get("ATLAS_LEAN_THINK_BUDGET", "").strip()
    if budget.isdigit() and int(budget) > 0:
        # variant C (Gemini r.18 as written): think on every turn, but routine turns get a token budget;
        # event turns ([THINK] in the prompt) think without a cap. Needs vLLM started with --reasoning-config.
        event = marker; think = True
        payload = _ln_orig_payload(*args, **dict(kwargs, thinking=True))
        if not event:
            payload["thinking_token_budget"] = int(budget)
        return payload
    effort = _ln_os.environ.get("ATLAS_LEAN_REASONING_EFFORT", "").strip().lower()
    if effort in ("none", "low", "medium", "high"):
        event = marker
        payload = _ln_orig_payload(*args, **dict(kwargs, thinking=True))
        if not event:
            payload["reasoning_effort"] = effort
        return payload
    kwargs["thinking"] = bool(think)
    return _ln_orig_payload(*args, **kwargs)
_ln_agent.build_chat_payload = _ln_payload
print("lean: harness wrappers installed (compact prompt + [STATE]/[CONTROLS]/[DIFF]/[HUD]/[NOTICE]/[SHORT], sandbox helpers incl. move_to, auto-probe of arrows on new levels, event PROTOCOL)")
"""


# 'leanshort': replace the stock system prompt with a compact one. The stock
# ToolAgent builds its prompt once, in __init__, via the module-level
# _build_system_prompt -- so the override must be installed before the solver
# creates agents (the notebook cell / pod driver runs before the run starts).
LEANSHORT_RUNTIME = r"""
import inference.agent.tool_agent as _ls_agent
from inference.agent import prompts as _ls_prompts

_LS_HINT = __HINT__

def _ls_system(*, tool_output_tokens):
    parts = [
        "You are a coding agent solving a grid-based puzzle game.",
        "\n\nGame overview:\n"
        "- A multi-level grid puzzle; clear every level with as few in-game actions as reliably possible. Levels reuse mechanics but layouts and rules can change.\n"
        "- Each turn is one observe-plan-act cycle: read the harness blocks in the user prompt, decide, act through the `python` tool, re-evaluate next turn.\n"
        "- Boards are 64 x 64 grids of ARC colour letters. Legend: W=white, w=light gray, g=gray, G=dark gray, c=charcoal, B=black, M=magenta, P=pink, R=red, b=blue, S=sky blue, Y=yellow, O=orange, r=dark red, N=light green, p=purple.\n",
        "\nRuntime variables inside every `python` call:\n"
        "- `current_frame`: `.ascii` (newline-joined rows of colour letters), `.step`, `.level`, `.shape`, `.segmentation` (`{'nodes': [...], 'adjacency_list': [...]}`; a node has `id`, `color`, `hash` (shape+colour signature, position-free), `pixels`, `boundary`, `children`). The numeric grid is not exposed.\n"
        "- `previous_frame`: the frame before the most recent real action (None before any action). `history[-1].frame` equals `current_frame`; use `previous_frame` or `history[-2].frame` for the earlier board. `transitions` / `last_transition` expose `.before_frame`, `.after_frame`, `.action`, `.result`.\n"
        "- `action(actions)` executes real actions from a list like `['LEFT']` or `[{'action': 'MOUSE', 'row': 4, 'col': 7}]` and returns a result dict with `board_changed`, `done`, `level_completed`, `game_over`, `run_complete`, `reward`, `valid_actions`, `executed_count`; afterwards every runtime variable is refreshed, so `action` may be called several times in one snippet, including in loops. `last_action_result` keeps the last result. `valid_actions` lists the legal action names.\n"
        "- An action may play a short animation; `current_frame` is its final frame. If `last_action_result['animation']['board_unchanged']` is true together with `board_changed == False`, the effect was shown only mid-animation (a rejected click, a consumed attempt, a bounce): `animation()` returns that frame timeline, `animation(frame=k)` one frame cropped to the transient region; both cost no action.\n"
        "- For `MOUSE`, pass integer `row` (vertical) and `col` (horizontal).\n",
        "\nTool session rules:\n"
        "- You have exactly one tool: `python`. " + _ls_prompts.TOOL_CALL_FORMAT_GUIDANCE + "\n"
        "- Every call starts fresh (no state between calls). Importable modules: bisect, collections, copy, fractions, functools, heapq, itertools, json, math, operator, random, re, statistics, string.\n"
        "- You may call `python` several times per step; each call has a 30-second limit and its output is capped at about " + str(tool_output_tokens) + " tokens -- print compact summaries or assign `result`, never full boards.\n"
        "- If an action result reports `game_over`, `run_complete`, `level_completed` or `done`, stop acting and re-ground next turn.\n",
    ]
    try:
        if _ls_agent.current_grid_image_enabled():
            parts.append("- User turns also attach an image of the current grid; it shows the same board as `current_frame.ascii`.\n")
    except Exception:
        pass
    parts.append(_LS_HINT)
    return "".join(parts)
_ls_agent._build_system_prompt = _ls_system
print(f"leanshort: compact system prompt installed ({len(_ls_system(tool_output_tokens=1024))} chars incl. hint)")
"""


LEAN_HINT_B = "\n- HARNESS AIDS (read them, never recompute them). Each turn's prompt carries: [STATE] -- level, step, board size and the object table (letter, size, rows, cols) of the current board; [DIFF] -- exactly what changed since your previous call (cells, colour transitions, components that appeared, vanished or moved), 'zero' means nothing changed: out of reach, not interactive, or still animating -- keep your hypothesis; [HUD] -- a bar that shrinks per action and the remaining count, every plan must finish inside it; [NOTICE] -- zero-diff streak; [SHORT] -- the previous call spent fewer than 5 actions.\\n- HELPERS preloaded in every python call (pure functions over `current_frame`, letters as in [STATE]): grid() -> list of row strings; cells(letter) -> [(row, col)]; objects(min_size=1, max_items=60) -> [{letter, size, r0, c0, r1, c1}] largest first; find_path(start, goal, passable, step=1) -> list of UP/DOWN/LEFT/RIGHT moves or None (BFS over cells; passable = set of letters or a function of a letter; goal = (row, col) or a list of them; step = cells per key press); frame_diff() -> {changed, bbox, moved, appeared, vanished} between previous_frame and current_frame; probe(action) -> executes ONE action and returns its frame_diff (use it once per unknown control to learn its reach); move_until(action, max_steps=30) -> repeats one action until the board stops changing or the level or game ends. Never count cells in your head: locate with objects()/cells(), plan with find_path(), then action(path).\\n- EVENT PROTOCOL (harness-enforced). On the first turn of a level, after a zero-diff call and after a game over, the first python call that executes actions MUST begin with:\\n  PROTOCOL = {\\n    'win': '...',    # what exactly ends this level, derived before moving: every X inside a Y of matching shape or colour; all cups filled; one piece left; N of M\\n    'rule': '...',   # the mechanic you believe, incl. the reach of each control (cells per press, what a click on an object does, how far a shove carries -- possibly through walls), what kills, what moves by itself and its period, whether a failed try costs one of N attempts\\n    'plan': '...',   # the sequence this call executes and why\\n  }\\n  Inspection-only calls never need it. On routine turns just act.\\n- ACT IN BATCHES: a call that executes actions should run 5 or more of them (a find_path() result, or a loop that calls action(...) step by step and checks frame_diff() between steps); one or two actions per call is only right for a single probe() of an unknown control, or right after a level ended.\\n- EXPENSIVE PROBES: if a failed try costs an attempt or a life (a counter that drops, a reset after a wrong move), simulate the outcome in code from the rule you hold before acting; never test by trying.\\n\n"


CORE_HINT = "\n- Sandbox helpers: move_until(direction, stop=None, max_steps=30) repeats one action while stop(current_frame) is false (it also stops on two unchanged boards in a row, on a return to a board seen earlier in this game, on level change or game over) and returns {steps, reason}; action([...]) runs the list one action at a time and halts the rest with status '[BATCH HALTED: LOOP]' when the board returns to a state seen earlier in this game.\n"

# 'core' runtime: no prompt text, no protocol, no thinking toggles. Two wrappers:
#   * _build_user_prompt -> records the stable hash of the current board per game (nothing is added to the prompt);
#   * _run_python_tool   -> prepends the helpers and the set of seen hashes to the model's code.
CORE_RUNTIME = r"""
import inference.agent.tool_agent as _co_agent

def _co_stable_hash(s):
    h = 1469598103934665603
    for ch in s:
        h = ((h ^ ord(ch)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

_CO_HELPERS = r'''
def _h_stable_hash(s):
    h = 1469598103934665603
    for ch in s:
        h = ((h ^ ord(ch)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h
def _h_board():
    try:
        return str(current_frame.ascii) if current_frame is not None else ''
    except Exception:
        return ''
_h_orig_action = action
def action(acts, force=False):
    items = [acts] if isinstance(acts, (str, dict)) else list(acts)
    seen = _H_SEEN
    seen.add(_h_stable_hash(_h_board()))
    executed = 0; last = {}; halted = None
    for i, a in enumerate(items):
        last = _h_orig_action([a]); executed += int(last.get('executed_count', 1) or 1)
        if last.get('game_over') or last.get('level_completed') or last.get('done') or last.get('run_complete'):
            break
        hs = _h_stable_hash(_h_board())
        if hs in seen and not force:
            halted = '[BATCH HALTED: LOOP] the board returned to a state already seen in this game after action %d of %d (%s); the remaining %d action(s) were not sent. Change the plan.' % (i + 1, len(items), a if isinstance(a, str) else a.get('action'), len(items) - i - 1)
            break
        seen.add(hs)
    res = dict(last); res['executed_count'] = executed; res['requested_count'] = len(items)
    if halted:
        res['halted'] = halted; res['stopped_early'] = True; res['stop_reason'] = 'loop'
    return res
def move_until(direction, stop=None, max_steps=30):
    steps = 0; zero = 0; reason = 'max_steps'
    for _ in range(max(1, int(max_steps))):
        before = _h_board()
        r = action([direction])
        steps += int(r.get('executed_count', 1) or 1)
        if r.get('halted'):
            reason = 'loop'; break
        if r.get('game_over') or r.get('level_completed') or r.get('done') or r.get('run_complete'):
            reason = 'game_over' if r.get('game_over') else 'level_changed'; break
        zero = zero + 1 if _h_board() == before else 0
        if zero >= 2:
            reason = 'no_change_twice'; break
        try:
            if stop is not None and stop(current_frame):
                reason = 'stop_condition'; break
        except Exception as exc:
            reason = 'stop_error: ' + str(exc)[:80]; break
    return {'steps': steps, 'reason': reason, 'last': {k: r.get(k) for k in ('board_changed', 'level_completed', 'game_over', 'halted') if k in r}}
'''

_co_orig_prompt = _co_agent.ToolAgent._build_user_prompt
def _co_prompt(self, action_num, *args, **kwargs):
    frame = kwargs.get("current_frame")
    try:
        level = getattr(frame, "level", None)
        if getattr(self, "_co_level", None) != level:
            self._co_level = level; self._co_seen = set()
        a = getattr(frame, "ascii", None)
        if isinstance(a, str) and a:
            getattr(self, "_co_seen", None) is None and setattr(self, "_co_seen", set())
            self._co_seen.add(_co_stable_hash(a))
    except Exception:
        pass
    return _co_orig_prompt(self, action_num, *args, **kwargs)
_co_agent.ToolAgent._build_user_prompt = _co_prompt

_co_orig_run = _co_agent.ToolAgent._run_python_tool
def _co_run(self, state_path, arguments):
    code = str((arguments or {}).get("code", "") or "")
    seen = sorted(getattr(self, "_co_seen", set()) or set())
    arguments = dict(arguments); arguments["code"] = _CO_HELPERS + "\n_H_SEEN = set(" + repr(seen) + ")\n" + code
    return _co_orig_run(self, state_path, arguments)
_co_agent.ToolAgent._run_python_tool = _co_run
print("core: sandbox loop guard + move_until installed; prompt is stock")
"""


def hint_text() -> str:
    return {"batch": BATCH_HINT, "observe": OBSERVE_HINT, "both": OBSERVE_HINT + BATCH_HINT,
            "protocol": PROTOCOL_HINT, "protocol3": PROTOCOL3_HINT,
            "lean": (LEAN_HINT_B if LEAN_HINT_VARIANT == "b" else LEAN_HINT),
            "leanshort": (LEAN_HINT_B if LEAN_HINT_VARIANT == "b" else LEAN_HINT), "core": CORE_HINT, "none": ""}[HINT]


def runtime_text() -> str:
    if HINT == "core":
        return CORE_RUNTIME
    if HINT == "lean":
        return LEAN_RUNTIME
    if HINT == "leanshort":
        return LEAN_RUNTIME + "\n" + LEANSHORT_RUNTIME.replace("__HINT__", repr(LEAN_HINT))
    keys = {"protocol": "inventory, diff, rule, budget, hypothesis, next",
            "protocol3": "inventory, diff, rule, budget, hypothesis, win, reach, next"}.get(HINT)
    return PROTOCOL_RUNTIME.replace("__KEYS__", keys) if keys else ""


# The one cell we add. It runs AFTER duck's own public-evaluation block, so it
# overrides whatever that set -- and only outside a competition rerun.
SWAP_CELL = '''# =====================================================================
# stock-lab: play OUR 30-game testbed instead of the 25 public games.
#
# Everything above this cell is UNMODIFIED Duck. This cell touches only the
# game list and two solver knobs, so the number it produces is stock's,
# measured under the same conditions as our own runs (V49/V50): 30 games,
# concurrency {conc}, {cap:.0f}s per game.
#
# NEVER reaches a submission: guarded on `not true_submission`, and the run
# cell replaces bm.games from Kaggle's gateway for a real rerun anyway.
# =====================================================================
if not true_submission:
    import glob as _lab_glob
    import os as _lab_os
    import taaf.game_api as _lab_api

    # The agent MUST be Duck's own bundle. When Kaggle fails to mount it, the
    # finder in cell 3 takes whatever else carries the marker -- it took our
    # fork three times before this check existed.
    if "{upstream}" not in str(BUNDLE_DIR):
        raise RuntimeError(
            f"stock-lab: agent bundle is {{BUNDLE_DIR}}, not Duck's upstream "
            f"({upstream}) -- refusing to measure a non-stock agent as stock"
        )
    print(f"stock-lab: agent bundle verified upstream: {{BUNDLE_DIR}}")

    _own = sorted(_lab_glob.glob("/kaggle/input/*/our_games")) or sorted(
        _lab_glob.glob("/kaggle/input/**/our_games", recursive=True))
    if not _own:
        # fatal, never a silent fallback to the public 25 -- a green run on
        # the wrong game set is indistinguishable from a correct one until
        # the number comes back and quietly means nothing.
        raise RuntimeError(
            "stock-lab: no our_games/ under /kaggle/input -- refusing to "
            "fall back to the public game list"
        )
    _own_dir = _own[0]
    _ids = sorted(
        d for d in _lab_os.listdir(_own_dir)
        if _lab_os.path.isdir(_lab_os.path.join(_own_dir, d))
    )
    _want = {games!r}
    if _want:
        _missing = sorted(set(_want) - set(_ids))
        if _missing:
            raise RuntimeError(f"stock-lab: games not in the dataset: {{_missing}}")
        _ids = sorted(_want)
    if not _ids:
        raise RuntimeError(f"stock-lab: {{_own_dir}} holds no game directories")

    _spec = _lab_api.ArcadeSpec(environments_dir=_own_dir)
    bm.games = [_lab_api.GameAPI(env_name=_g, arcade_spec=_spec) for _g in _ids]
    bm.n_passes = 1
    bm.game_weights = None
    bm.label = f"{{bm.label}}-stocklab{{len(_ids)}}"
    if hasattr(bm.solver, "concurrency"):
        bm.solver.concurrency = {conc}
    if hasattr(bm.solver, "max_runtime_s_per_game"):
        bm.solver.max_runtime_s_per_game = {cap}

    # A GLOBAL deadline, without which the run has no defined end -- and that
    # is not cosmetic. Measured 03.09 across five runs: stock passes Phase A
    # `soft_end_time=None` (cell 7 reads max_runtime_s off an environment
    # object that has none, so the whole mechanism is inert here), so when
    # most games hit their own 1500s cap and stop, the few still going get the
    # GPU almost to themselves. Their answers come back several times faster,
    # they take several times more turns inside the SAME per-game cap, and
    # they score accordingly:
    #     run 1 of the hint arm: 5 stragglers averaged 22.17, the other 25 -> 2.61
    #     run 3 of the hint arm: 4 stragglers averaged 24.27, the other 26 -> 9.22
    # Two of five runs grew such a tail and ran 55 minutes; three ended
    # together at 25. The two with tails were the two best numbers we had, and
    # both were in the same arm -- which is how a scheduling artefact came to
    # look like a prompt working. Nothing was misread: the runs were simply
    # not comparable.
    #
    # Setting `soft_end` here works because cell 7 only computes it and the
    # run cell below reads it at call time (`await bm.run(soft_end_time=...)`).
    import datetime as _lab_dt
    soft_end = _lab_dt.datetime.now() + _lab_dt.timedelta(seconds={deadline})
    print(f"stock-lab: GLOBAL deadline {{soft_end:%H:%M:%S}} "
          f"({deadline}s) -- no game may outlive the field")

    if {batch}:
        import inference.agent.tool_agent as _lab_agent
        _before = len(_lab_agent.PYTHON_ADDENDUM)
        _lab_agent.PYTHON_ADDENDUM = _lab_agent.PYTHON_ADDENDUM + {hint!r}
        print(f"stock-lab: HINT '{hint_name}' added -- addendum {{_before}} -> "
              f"{{len(_lab_agent.PYTHON_ADDENDUM)}} chars")
{envs}        exec({runtime!r})

    print(f"stock-lab: UNMODIFIED Duck on {{len(_ids)}} own games from {{_own_dir}}")
    print(f"stock-lab:   concurrency={{bm.solver.concurrency}} "
          f"cap={{bm.solver.max_runtime_s_per_game}}s passes={{bm.n_passes}}")
    print(f"stock-lab:   {{', '.join(_ids)}}")
'''


BATTLE_CELL = '''# =====================================================================
# BATTLE build: stock Duck plus ONE addition -- the turn-budget hint.
#
# Measured on our 30-game testbed, three runs each, read through
# scripts/rhae.py (never summary.txt, which flushes on a timer and lied
# about two of these runs):
#     stock                RHAE 2.49, deepest 3 levels
#     stock + this hint    RHAE 5.87 / 5.17 / 11.22, deepest 5 -- one game
#                          cleared outright, which had never happened
# Threshold was written before the runs: median >= 4.0. Median came to 5.87.
#
# The hint is applied OUTSIDE any `true_submission` guard on purpose. In the
# lab kernel it sat inside one, next to the own-games swap, and would have
# been skipped by the competition rerun -- shipping plain stock while the log
# said otherwise. Phase A keeps duck's own public 25 for the same reason: a
# submission build must not depend on a path the rerun never takes.
#
# Honest limit: the MECHANISM is unknown. Batching did not rise (8-9% of
# action() calls carry 3+ moves either way), so whatever the hint does, it is
# not what its own wording asks for. Measured on our own games, which flatter
# any agent whose search cracks them; transfer to the hidden set is untested.
# =====================================================================
import inference.agent.tool_agent as _bat_agent
_bat_before = len(_bat_agent.PYTHON_ADDENDUM)
_bat_agent.PYTHON_ADDENDUM = _bat_agent.PYTHON_ADDENDUM + {hint!r}
print(f"battle: hint '{hint_name}' applied -- addendum {{_bat_before}} -> "
      f"{{len(_bat_agent.PYTHON_ADDENDUM)}} chars (applies in Phase A AND rerun)")
{envs}exec({runtime!r})

if not true_submission:
    if hasattr(bm.solver, "max_runtime_s_per_game"):
        bm.solver.max_runtime_s_per_game = {cap}
        print(f"battle: Phase A cap {{bm.solver.max_runtime_s_per_game}}s on duck's public 25")
{vllm_restart}'''


VLLM_RESTART = '    # ---- test variant: restart the stock vLLM with --reasoning-config (Phase A only; enables thinking_token_budget)\n    import json as _vj, os as _vo, signal as _vs, subprocess as _vp, sys as _vsys, time as _vt, urllib.request as _vu\n    _pidf = WORKING_DIR / "vllm-openai-server.pid"\n    try:\n        _pid = int(_pidf.read_text().strip()); _vo.kill(_pid, _vs.SIGTERM); print("battle: sent SIGTERM to stock vLLM pid", _pid)\n    except Exception as _e:\n        print("battle: could not kill stock vLLM:", _e)\n    for _ in range(90):\n        try:\n            _vu.urlopen("http://127.0.0.1:1234/v1/models", timeout=3); _vt.sleep(2)\n        except Exception:\n            break\n    _senv = _load_setup_env()\n    _model = _senv.get("TAAF_QWEN_MODEL_PATH") or _vo.environ.get("TAAF_QWEN_MODEL_PATH")\n    _served = _senv.get("TAAF_QWEN_SERVED_MODEL_NAME") or _vo.environ.get("TAAF_QWEN_SERVED_MODEL_NAME") or "Qwen/Qwen3.8-27B-FP8"\n    _cmd = [_vsys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(_model), "--served-model-name", _served,\n            "--host", "127.0.0.1", "--port", "1234", "--tensor-parallel-size", "1", "--enable-auto-tool-choice",\n            "--tool-call-parser", "qwen3_coder", "--generation-config", "vllm", "--enable-prefix-caching",\n            "--default-chat-template-kwargs", \'{"preserve_thinking": true}\', "--reasoning-parser", "qwen3",\n            "--max-model-len", "65536", "--reasoning-config", \'{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}\']\n    _vlog = open(WORKING_DIR / "vllm-openai-server-2.log", "w", encoding="utf-8")\n    _proc = _vp.Popen(_cmd, env=_command_env(), stdout=_vlog, stderr=_vp.STDOUT)\n    _pidf.write_text(str(_proc.pid))\n    _ok = False\n    for _ in range(180):\n        if _proc.poll() is not None:\n            break\n        try:\n            _vu.urlopen("http://127.0.0.1:1234/v1/models", timeout=5); _ok = True; break\n        except Exception:\n            _vt.sleep(5)\n    print("battle: vLLM restarted with --reasoning-config:", _ok)\n    if not _ok:\n        raise RuntimeError("vLLM restart with --reasoning-config failed; see vllm-openai-server-2.log")\n'


def build() -> None:
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))

    # place the swap right after duck's public-evaluation cell
    idx = next(
        i for i, c in enumerate(nb["cells"])
        if "Q38_P1_PUBLIC_GAME_IDS" in "".join(c["source"])
    )
    if BATTLE:
        if HINT == "none" and not os.environ.get("ATLAS_STOCKLAB_KERNEL", "").strip():
            raise SystemExit("a battle build with no hint is plain stock -- submit arc3-duck-baseline v1 instead; "
                             "for a pure-stock CONTROL run set ATLAS_STOCKLAB_KERNEL to a non-battle slug")
        env_lines = "".join(f"import os as _bat_os; _bat_os.environ[{k.strip()!r}] = {v.strip()!r}\n" for k, v in ENVS)
        source = BATTLE_CELL.format(cap=CAP_S, hint=hint_text(), hint_name=HINT, runtime=runtime_text(),
                                    envs=env_lines, vllm_restart=(VLLM_RESTART if VLLM_REASONING else ""))
    else:
        env_lines = "".join(f"        import os as _lab_os2; _lab_os2.environ[{k.strip()!r}] = {v.strip()!r}" + chr(10) for k, v in ENVS)
        source = SWAP_CELL.format(conc=CONC, cap=CAP_S, batch=(HINT != "none"), hint=hint_text(),
                              hint_name=HINT, deadline=DEADLINE_S, games=GAMES,
                              upstream=UPSTREAM_BUNDLE, runtime=runtime_text(), envs=env_lines)
    nb["cells"].insert(idx + 1, {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    meta = json.loads((ROOT / "kernels" / "notebooks_duck" / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = KERNEL_ID
    meta["title"] = KERNEL_ID.split("/")[-1].replace("-", " ")
    # The battle build must NOT mount our dataset. It needs nothing from it
    # (no own_games, no patch files), and duck's own public-eval block globs
    # /kaggle/input -- an extra mounted directory there is a way for Phase A
    # to quietly pick up the wrong game list. Fewer mounts, fewer paths a
    # competition rerun can take that we never rehearsed.
    if BATTLE:
        meta["dataset_sources"] = [d for d in meta["dataset_sources"] if d != OUR_DATASET]
    elif OUR_DATASET not in meta["dataset_sources"]:
        meta["dataset_sources"].append(OUR_DATASET)
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # sanity: the stock cells must be byte-identical to duck's
    orig = json.loads(SRC_NB.read_text(encoding="utf-8"))["cells"]
    built = [c for i, c in enumerate(nb["cells"]) if i != idx + 1]
    assert len(built) == len(orig), "cell count drifted"
    for a, b in zip(orig, built):
        assert "".join(a["source"]) == "".join(b["source"]), "a stock cell was modified"

    print(f"собрано: {OUT_NB}")
    print(f"  ячеек {len(nb['cells'])} (у стока {len(orig)}, добавлена одна)")
    print(f"  стоковые ячейки не тронуты: проверено побайтово")
    print(f"  игр {len(GAMES) if GAMES else 'все из our_games'}, конкурентность {CONC}, "
          f"потолок {CAP_S:.0f}с, общий дедлайн {DEADLINE_S}с")
    if GAMES:
        print(f"  список: {' '.join(GAMES)}")
    print(f"  подсказка: {HINT} ({len(hint_text())} символов)")
    print(f"  датасеты: {', '.join(meta['dataset_sources'])}")


if __name__ == "__main__":
    build()

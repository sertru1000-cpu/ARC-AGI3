# =============================================================================
# AGENTFIX — Scott Le Grand (Kaggle: scottlegrand), ARC Prize 2026 ARC-AGI-3.
# Source: https://www.kaggle.com/code/scottlegrand/taaf-flashnext-sheetu12b-0922 (cell 10), copied VERBATIM
# on 2026-09-26. Released publicly by its author; Kaggle notebooks are Apache 2.0 unless stated otherwise.
# Only change in nextfork: this header. It is a MODULE of our harness now (no notebook cell): imported once from
# the bottom of inference/framework/solver.py when NEXTFORK_AGENTFIX is on (default), i.e. when the notebook
# unpickles the benchmark — after the serving setup has exported MULTIMODAL_UPSCALE/MULTIMODAL_CONTEXT.
# Per-fix switches are the author's own AGENTFIX_* environment variables (setdefault below).
# =============================================================================
# ---------------------------------------------------------------------------
# AGENTFIX -- six measured agent-loop defects patched at runtime on the shipped bundle.
# Scheduling/engine untouched: this is c16's schedule with a repaired agent.
# ---------------------------------------------------------------------------
import os as _os
_os.environ.setdefault('AGENTFIX_DEDUP', '0')
_os.environ.setdefault('AGENTFIX_LEDGER', '0')
_os.environ.setdefault('AGENTFIX_IMAGES', '1')
_os.environ.setdefault('AGENTFIX_MEMORY', '1')
_os.environ.setdefault('AGENTFIX_LOOP', '0')
_os.environ.setdefault('AGENTFIX_ACTION7', '0')
_os.environ.setdefault('AGENTFIX_STALL', '0')
_os.environ.setdefault('AGENTFIX_RESULT', '0')
_os.environ.setdefault('AGENTFIX_NOIMPACT', '0')
_os.environ.setdefault('AGENTFIX_TIMING', '1')
_os.environ.setdefault('AGENTFIX_ANIM_SUMMARY', '0')
# cell defaults set BEFORE the module sources: switches read at exec time see them
# ===== begin agentfix.py (verbatim) =====
"""agentfix: runtime patches to the SHIPPED Duck agent (keithtyser bundle), applied from a kernel
cell before any ToolAgent is constructed. Every fix targets a defect measured in Kaggle transcripts
or a mechanism used by higher-scoring teams. Each is independently switchable via env.

  F1  AGENTFIX_IMAGES     history keeps only its text; the CURRENT prompt keeps its image. Images
                          are charged a flat token cost by the estimator instead of len(b64)//3.
                          (Measured: median 8 obsolete boards per request, ~26% of the budget; the
                          estimator over-charges images 5-8x so reasoning/tool results get evicted.)
  F2  AGENTFIX_RESULT     an action-executing tool call ALWAYS returns the structured action result
                          (board_changed / level_completed / score / stop_reason), even when the
                          model printed to stdout (81% of acting calls returned none). The next user
                          prompt also gets the harness's own one-line outcome description
                          (_describe_last_outcome, previously dead code).
  F3  AGENTFIX_MEMORY     the labeled-block parser accepts "World model (revised):" / "Plan (next):"
                          etc. (exact-prefix match dropped them); game_over no longer wipes the world
                          model (it is a level RETRY, not a new level).
  F4  AGENTFIX_LOOP       per-analysis-step bookkeeping that survives yield re-entries: an identical
                          python snippet re-run in the same step is refused with its prior verdict;
                          after AGENTFIX_LOOP_K tool calls with no executed action the model is told
                          to act. (sk48: 46 min, 35 calls, 0 actions on three identical snippets.)
  F5  AGENTFIX_NOIMPACT   per-action: count changed cells, mask the HUD bar (an edge row/col whose
                          cells flip with ONE fixed colour transition, a few cells per action, each
                          cell once -- the action-budget bar every public game draws; survey 09-15:
                          25/25 games, 1-2 cells/action, e.g. 9->3 along row 1), report board_changed_ex_hud,
                          changed_bbox, frames and distinct_frames (animation). A (level, board,
                          action) triple already proven to change nothing outside the HUD band is
                          EXECUTED again but tagged repeat_of_known_no_op=n (AGENTFIX_REFUSE=1 restores
                          the bounded refusal; the v3 regression hunt showed refusals misled the model).
  F7  AGENTFIX_LEDGER    harness-measured EFFECT of every single action (cells changed outside the HUD,
                          object shifts by colour/direction, near/away from a click), accumulated per level
                          into a ledger the user prompt shows: what each action has done here so far and
                          which valid actions are still UNTESTED on this level. Deterministic observation
                          half of predict-then-observe; the model's own summaries were the only memory before.
  F6  AGENTFIX_ACTION7    ACTION7 becomes executable (it was advertised and unmappable) and the
                          system prompt says it costs an action and is UNDO in several public games
                          (3-5 of the 6 that expose it, by two offline protocols); no test nudge.
"""
import hashlib, json, os, re, time, types

_ON = lambda k, d="1": os.environ.get(k, d).strip() not in ("0", "", "false", "False")
LOOP_K = int(os.environ.get("AGENTFIX_LOOP_K", "6"))
REFUSE = os.environ.get("AGENTFIX_REFUSE", "0").strip() not in ("0", "", "false", "False")   # default OFF since the v3 regression hunt
HUD_MIN_EVENTS = int(os.environ.get("AGENTFIX_HUD_MIN_EVENTS", "3"))   # ticks seen before an edge line counts as a HUD bar
HUD_MAX_STEP = int(os.environ.get("AGENTFIX_HUD_MAX_STEP", "3"))       # a bar advances at most this many cells per action per line (real max 3: sp80; vc33 border flashes 4)
HUD_CELL_MIN = int(os.environ.get("AGENTFIX_HUD_CELL_MIN", "5"))       # a band cell that flips on >= this many single actions ...
HUD_RATE = float(os.environ.get("AGENTFIX_HUD_RATE", "0.9"))           # ... and on >= this share of them since first seen is a counter/blinker
IMAGE_TOKENS = int(os.environ.get("AGENTFIX_IMAGE_TOKENS", "120"))

# ----------------------------------------------------------------------------- F1: images
IMAGE_KEEP = int(os.environ.get("AGENTFIX_IMAGE_KEEP", "2"))   # history user messages that keep their board image (newest first)

def _strip_images(messages):
    """Drop board images from history EXCEPT the newest IMAGE_KEEP image-bearing messages. v1-v3 stripped all of them;
    the regression hunt on v3 measured +51% reasoning per request once history existed and 29% fewer actions per 2 h."""
    out = []
    keep_idx = set()
    if IMAGE_KEEP > 0:
        for i in range(len(messages) - 1, -1, -1):
            c = messages[i].get("content")
            if isinstance(c, list) and any(isinstance(p, dict) and p.get("type") != "text" for p in c):
                keep_idx.add(i)
                if len(keep_idx) >= IMAGE_KEEP:
                    break
    for i, m in enumerate(messages):
        c = m.get("content")
        if isinstance(c, list) and i not in keep_idx:
            text = "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
            text = re.sub(r"\n*Current grid image:\s*$", "", text)
            m = dict(m); m["content"] = text
        out.append(m)
    return out

def _estimate_tokens_flat(value):
    def sub(v):
        if isinstance(v, dict):
            if "image_url" in v:
                return {"image_url": "<image>" + "x" * (IMAGE_TOKENS * 3)}
            return {k: sub(x) for k, x in v.items()}
        if isinstance(v, list):
            return [sub(x) for x in v]
        return v
    try:
        rendered = json.dumps(sub(value), ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        rendered = str(value)
    return max(1, (len(rendered) + 2) // 3)

# ----------------------------------------------------------------------------- F3: parser
_LABEL_RE_CACHE = {}
def _extract_labeled_blocks_tolerant(content, labels):
    """Like the shipped parser, but a label may carry a short qualifier before the colon:
    'World model (revised):', 'World model update:', 'Plan (next):'. Line-initial prose such as
    'Plan for the next move: left' is NOT a label (only parenthesised or one-word update qualifiers count)."""
    key = tuple(labels)
    if key not in _LABEL_RE_CACHE:
        alts = "|".join(re.escape(l.lower()) for l in sorted(labels, key=len, reverse=True))
        _LABEL_RE_CACHE[key] = re.compile(r"^(?:[-*]\s*)*(" + alts + r")\s*(?:\([^)\n]{0,30}\)|update[sd]?|revised|revision|now|v\d+)?\s*:\s*(.*)$", re.I)
    rx = _LABEL_RE_CACHE[key]
    canon = {l.lower(): l for l in labels}
    extracted = {l: [] for l in labels}
    current = None
    for raw in content.splitlines():
        stripped = raw.strip()
        m = rx.match(stripped)
        if m:
            current = canon[m.group(1).lower()]
            if m.group(2).strip():
                extracted[current].append(m.group(2).strip())
            continue
        if current is not None and stripped:
            extracted[current].append(stripped)
    return {l: " ".join("\n".join(v).split()) for l, v in extracted.items() if "\n".join(v).split()}

# ----------------------------------------------------------------------------- F5: no-impact
HUD_BORDER = int(os.environ.get("AGENTFIX_HUD_BORDER", "2"))   # HUD band = high-change cells within this many rows/cols of an edge

def _grid_of(state):
    """Current board as a list of lists of ints. Engine frames are 2-D np.int8 ndarrays."""
    try:
        f = state.raw.frame[-1]
        return f.tolist() if hasattr(f, "tolist") else [list(map(int, r)) for r in f]
    except Exception:
        return None

def _sig(grid):
    try:
        return hashlib.sha1(json.dumps(grid, separators=(",", ":")).encode()).hexdigest()[:16]
    except Exception:
        return None

def _frame_sigs(frames):
    out = []
    for f in frames:
        try:
            out.append(f.tobytes() if hasattr(f, "tobytes") else json.dumps(f, separators=(",", ":"), default=int))
        except Exception:
            out.append(repr(f))
    return out

def _hud_line(r, c, H, W):
    if r < HUD_BORDER or r >= H - HUD_BORDER:
        return ("r", r)
    if c < HUD_BORDER or c >= W - HUD_BORDER:
        return ("c", c)
    return None

def _hud_update(st, level, changed_vals, H, W, reset=False):
    """Learn HUD elements from ONE executed action's diff, border band only (interior cells are never HUD):
    (a) BAR: an edge line whose cells change with one fixed (from->to) transition, <= HUD_MAX_STEP cells per
        action, no cell twice (monotone advance); its reverse transition is the refill (level reset). Needs
        HUD_MIN_EVENTS ticks over >= HUD_MIN_EVENTS distinct cells. Survey of the 25 public games (c16 replay):
        every real HUD is exactly this shape (action-budget bar, 1-2 cells per action).
    (b) COUNTER/BLINKER: a single band cell that changes on >= HUD_CELL_MIN single actions and on >= HUD_RATE of
        the single actions since it was first seen (a ticking clock digit). Not seen in the public set; kept
        for hidden games."""
    hud = st["hud"].setdefault(level, {"_n": 0, "_cells": {}, "bars": {}})
    hud["_n"] += 1; n = hud["_n"]
    if reset:
        for e in hud["bars"].values():
            e["cells"] = set()
    groups = {}
    for r, c, f, t in changed_vals:
        line = _hud_line(r, c, H, W)
        if line is None:
            continue
        groups.setdefault((line, f, t), set()).add((r, c))
    bars = hud["bars"]
    mirror = {k for k in groups if (k[0], k[2], k[1]) in groups}   # f->t AND t->f on one line in ONE action = an object moving along the edge
    gained = {}
    for (line, f, t), cells in groups.items():
        gained[t] = gained.get(t, 0) + len(cells)
    for (line, f, t), cells in groups.items():
        if (line, f, t) in mirror or gained.get(f, 0) > 0:
            continue                                              # a moving object (its lost colour reappears elsewhere) is never a counter
        for rc in cells:
            rec = hud["_cells"].setdefault(rc, [0, n]); rec[0] += 1
    for (line, f, t), cells in groups.items():
        if (line, f, t) in mirror:
            bars.setdefault((line, f, t), {"cells": set(), "seen": set(), "events": 0, "bad": False})["bad"] = True
            continue
        rev = bars.get((line, t, f))
        if rev is not None and rev["events"] > 0 and (cells & rev["cells"] or len(cells) > HUD_MAX_STEP):
            if len(cells) > HUD_MAX_STEP:
                rev["cells"] = set()                              # wholesale refill of a known bar (all 40 real refills are > 4 cells)
            else:
                rev["bad"] = True                                 # a small reverse flip on the bar's own cells = a slider/toggle, not a bar
            continue
        e = bars.setdefault((line, f, t), {"cells": set(), "seen": set(), "events": 0, "bad": False})
        if e["bad"]:
            continue
        if cells & e["cells"]:
            e["bad"] = True                                       # a repeat: not a monotone bar
            continue
        if e["cells"]:                                            # a bar advances along the line: new cells sit next to its run
            pos = (lambda rc: rc[1]) if line[0] == "r" else (lambda rc: rc[0])
            if min(abs(pos(a) - pos(b)) for a in cells for b in e["cells"]) > HUD_MAX_STEP:
                e["bad"] = True                                   # a jump along the line: objects, not a bar
                continue
        if len(cells) > HUD_MAX_STEP and e["events"] < HUD_MIN_EVENTS:
            e["bad"] = True                                       # too big before qualification; on a qualified bar it is a penalty tick
            continue
        e["events"] += 1; e["cells"] |= cells; e["seen"] |= cells

def _hud_keys(st, level):
    """(qualified bar keys, qualified counter cells) for this level."""
    hud = st["hud"].get(level)
    if not hud:
        return [], set()
    keys = [k for k, e in hud["bars"].items() if not e["bad"] and e["events"] >= HUD_MIN_EVENTS and len(e["seen"]) >= HUD_MIN_EVENTS]
    n = hud["_n"]
    counters = {rc for rc, (k, first) in hud["_cells"].items() if k >= HUD_CELL_MIN and k / max(1, n - first + 1) >= HUD_RATE}
    return keys, counters

def _hud_mask(st, level, H, W, grid=None):
    """Cells covered by qualified HUD elements. Without a grid: the WHOLE edge line of each bar (used to notice
    that the mask changed). With a grid: only the line cells currently holding one of the bar's two colours, so a
    gameplay object sitting on the bar's line keeps its identity in the no-op signature (review 6, D3)."""
    keys, counters = _hud_keys(st, level)
    m = set(counters)
    bars = st["hud"].get(level, {}).get("bars", {})
    for (line, f, t) in keys:
        axis, idx = line
        cells = [(idx, c) for c in range(W)] if axis == "r" else [(r, idx) for r in range(H)]
        if grid is not None:
            ok = [r < len(grid) and c < len(grid[r]) and grid[r][c] in (f, t) for r, c in cells]
            e = bars.get((line, f, t), {}); seed = e.get("seen", set()) | e.get("cells", set())
            keep = [False] * len(cells); i = 0
            while i < len(cells):                                 # maximal runs of bar-coloured cells; keep a run iff it touches the bar
                if not ok[i]:
                    i += 1; continue
                j = i
                while j < len(cells) and ok[j]:
                    j += 1
                if any(cells[x] in seed for x in range(i, j)):
                    for x in range(i, j):
                        keep[x] = True
                i = j
            cells = [rc for rc, k in zip(cells, keep) if k]         # an object merely sharing a bar colour elsewhere stays visible
            # NOT capped to the bar's ticked extent: the cap would move with every tick and break the no-op key's
            # tick-invariance (review 8 fix 3 tried it: refusals 157 -> 101). A bar-coloured object SPAWNING adjacent
            # to the run is an accepted residual: bounded (2 of 3 refusals) and self-healing once it moves.
        m.update(cells)
    return frozenset(m)

def _hud_change(keys, counters, r, c, vb, va, H, W):
    """Is a changed cell a HUD change? Counter cell, or a bar-line cell flipping between the bar's two colours."""
    if (r, c) in counters:
        return True
    for (axis, idx), f, t in keys:                              # any qualified bar whose line holds this cell (corners belong to both)
        if (r if axis == "r" else c) == idx and vb in (f, t) and va in (f, t):
            return True
    return False

def _hud_absorb(st, level, changed_vals, H, W, reset=False):
    """A BATCH's diff is net and cannot teach a bar, but its ticks must still extend a known bar's run, otherwise
    the next single tick looks like a jump and kills the bar (review 6, D1). RESET inside the batch = refill (D2)."""
    hud = st["hud"].get(level)
    if not hud:
        return
    if reset:
        for e in hud["bars"].values():
            e["cells"] = set()
    for r, c, f, t in changed_vals:
        line = _hud_line(r, c, H, W)
        if line is None:
            continue
        e = hud["bars"].get((line, f, t))
        if e is not None and not e["bad"]:
            e["cells"].add((r, c)); e["seen"].add((r, c))
        rev = hud["bars"].get((line, t, f))
        if rev is not None and not rev["bad"]:
            rev["cells"].discard((r, c))                          # a refill of that cell inside the batch

def _refusal(self, level, n):
    cs = self.game.current_state
    try:
        state_name = cs.raw.state.name
    except Exception:
        state_name = None
    try:
        lvl = int(getattr(self, "_agentfix_last_level", level + 1))
    except Exception:
        lvl = level + 1
    out = {
        "executed": False, "action_num": getattr(self, "action_count", None), "level": lvl, "score": level,
        "reward": 0.0, "state": state_name, "valid_actions": list(getattr(self, "_agentfix_valid", []) or []),
        "board_changed": False, "board_changed_ex_hud": False, "done": False, "level_completed": False,
        "game_over": False, "run_complete": False, "requested_count": n, "executed_count": 0, "stopped_early": True,
        "stop_reason": "known_no_op",
        "stop_detail": "This exact action on this exact board was already executed and changed nothing outside the "
                       "HUD band. It was NOT executed again and cost no action. Do something different.",
    }
    try:
        out.update(self.timing_payload())
    except Exception:
        pass
    return out

def _step_env_core(self, orig, arguments, grid_fn, box):
    st = self.__dict__.setdefault("_agentfix", {"dead": set(), "hud": {}, "mask": {}})
    ac = getattr(self, "action_count", None)
    if ac is not None and st.get("count") is not None and st["count"] != ac:      # an action ran outside step_env (auto-reset): bars refilled
        for hud in st["hud"].values():
            for e in hud.get("bars", {}).values():
                e["cells"] = set()
    st["count"] = ac
    try:
        level = int(self.game.current_state.levels_completed)
    except Exception:
        level = -1
    before = grid_fn(self)
    H = len(before) if before else 0; W = len(before[0]) if before and before[0] else 0
    action_sig = hashlib.sha1(json.dumps(arguments, sort_keys=True, default=str).encode()).hexdigest()[:16]
    lines0 = _hud_mask(st, level, H, W)
    if st["mask"].get(level) != lines0:                  # the band changed -> earlier verdicts are stale
        st["mask"][level] = lines0; st["dead"] = {k for k in st["dead"] if k[0] != level}
    hud0 = _hud_mask(st, level, H, W, before) if before is not None else frozenset()
    masked = [[(0 if (r, c) in hud0 else v) for c, v in enumerate(row)] for r, row in enumerate(before)] if before is not None else None
    key = (level, _sig(masked), action_sig)
    try:
        n_req = len(arguments.get("actions") or [])
    except Exception:
        n_req = 1
    if key in st["dead"]:
        try:
            terminal = self.game.current_state.raw.state.name in ("GAME_OVER", "WIN") or bool(self.should_stop())
        except Exception:
            terminal = False
        cnt = st.setdefault("refused", {}).get(key, 0)
        st["refused"][key] = cnt + 1
        if REFUSE and not terminal and cnt % 3 != 2:   # refuse twice, then let the third attempt re-verify (hidden-state mechanics are never blocked for good)
            try:
                print(f"AGENTFIX_EVENT refusal level={level} n={cnt + 1}", flush=True)
            except Exception:
                pass
            return _refusal(self, level, n_req)
        box["repeat"] = cnt + 1                        # default: EXECUTE it (actions are not the scarce resource; wall clock is) and annotate
    box["called"] = True
    payload = orig(self, arguments)
    box["payload"] = payload
    if isinstance(payload, dict) and box.get("repeat") and payload.get("executed"):
        payload["repeat_of_known_no_op"] = box["repeat"]      # the regression hunt: refusals made cn04/g50t misread the harness as the game
    st["count"] = getattr(self, "action_count", None)
    if not isinstance(payload, dict) or not payload.get("executed"):
        return payload
    try:
        self._agentfix_last_level = payload.get("level"); self._agentfix_valid = list(payload.get("valid_actions") or [])
    except Exception:
        pass
    after = grid_fn(self)
    if before is None or after is None:
        return payload
    H = min(len(before), len(after)); W = min(len(before[0]) if before else 0, len(after[0]) if after else 0)
    executed = int(payload.get("executed_count") or 1)
    changed_vals = [(r, c, before[r][c], after[r][c]) for r in range(H) for c in range(W) if before[r][c] != after[r][c]]
    changed = [(r, c) for r, c, _, _ in changed_vals]
    try:
        acts = []
        for a in (arguments.get("actions") or []):
            if not isinstance(a, dict):
                continue
            nm = str(a.get("action") or a.get("id") or "").upper()
            if nm.startswith("ACTION") and _LEDGER_NORM is not None:
                try:
                    nm = str(_LEDGER_NORM(nm) or nm).upper()
                except Exception:
                    pass
            acts.append(nm)
    except Exception:
        acts = []
    act0 = acts[0] if acts else ""
    if executed == 1:                                          # learn bars from single actions only: a batch diff is NET
        _hud_update(st, level, changed_vals, H, W, reset=(act0 == "RESET"))
    else:
        _hud_absorb(st, level, changed_vals, H, W, reset=("RESET" in acts))
    keys, counters = _hud_keys(st, level)
    hud = _hud_mask(st, level, H, W, after)
    ex = [(r, c) for r, c, vb, va in changed_vals if not _hud_change(keys, counters, r, c, vb, va, H, W)]
    payload["changed_cells"] = len(changed); payload["changed_ex_hud"] = len(ex)
    payload["board_changed_ex_hud"] = bool(ex); payload["hud_cells"] = len(hud)
    payload["changed_bbox"] = [min(r for r, _ in ex), min(c for _, c in ex), max(r for r, _ in ex), max(c for _, c in ex)] if ex else None
    if executed == 1 and n_req == 1:
        try:
            exs = set(ex)
            freq, freq_b = {}, {}
            for row in after:
                for v in row:
                    freq[v] = freq.get(v, 0) + 1
            for row in before:
                for v in row:
                    freq_b[v] = freq_b.get(v, 0) + 1
            if _ON("AGENTFIX_LEDGER"):                          # model-visible only with F7's explaining sentence (review 18, D1)
                payload["effect"] = _effect_of(acts[0] if acts else "", (arguments.get("actions") or [{}])[0],
                                               [cv for cv in changed_vals if (cv[0], cv[1]) in exs], payload["changed_bbox"], freq, H, W, freq_b, before, after)
        except Exception:
            pass
    try:
        frames = list(self.game.current_state.raw.frame)
        payload["frames"] = len(frames); payload["distinct_frames"] = len(set(_frame_sigs(frames)))
    except Exception:
        pass
    # only a SINGLE action that changed nothing outside the band is a proven no-op; a batch may be net-zero yet useful
    if (executed == 1 and n_req == 1 and not ex and act0 != "RESET" and not payload.get("level_completed")
            and not payload.get("game_over") and not payload.get("run_complete")):
        st["dead"].add(key)
    return payload

def step_env_wrapped(self, orig, arguments, grid_fn):
    """F5 wrapper. FAIL-SAFE: any error in the bookkeeping falls through to the original step_env."""
    box = {}
    try:
        return _step_env_core(self, orig, arguments, grid_fn, box)
    except Exception as exc:
        if "payload" in box:                      # engine already ran; bookkeeping after it failed -> hand back its payload
            try:
                print(f"AGENTFIX F5 post-step bookkeeping failed, returning engine payload: {type(exc).__name__}: {exc}", flush=True)
            except Exception:
                pass
            return box["payload"]
        if box.get("called"):                     # the ORIGINAL raised: propagate exactly as stock would, never re-run the action
            raise
        try:
            print(f"AGENTFIX F5 fell back to stock step_env: {type(exc).__name__}: {exc}", flush=True)
        except Exception:
            pass
        return orig(self, arguments)

_EXTRA_RESULT_KEYS = ("changed_cells", "changed_ex_hud", "board_changed_ex_hud", "hud_cells", "changed_bbox", "frames", "distinct_frames", "effect", "repeat_of_known_no_op")

LEDGER_MAX_CHARS = int(os.environ.get("AGENTFIX_LEDGER_MAX_CHARS", "700"))
ENTRY_MAX_CHARS = int(os.environ.get("AGENTFIX_ENTRY_MAX_CHARS", "220"))

_LEDGER_NORM = None      # set by install(): engine action name -> model action name
_LEDGER_CALLABLE = None  # set by install(): the model-side action names the bundle can dispatch
_OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}
def _opposite(d):
    return "-".join(_OPP.get(x, x) for x in d.split("-"))

def _best_shift(L, G, min_frac=0.8):
    """The translation v carrying most of L onto G; None unless >= min_frac of L lands in G (partial overlap with
    other sprite parts or the ground is allowed). 0.8 not 0.6: at 0.6 the real-trajectory census reported 142 wrong
    directions on patterned sprites vs 11 at 0.8; a missed mover degrades to 'changed N cells', a wrong direction misleads."""
    if not L or not G or len(L) != len(G):
        return None
    if len(L) * len(G) <= 40000:
        votes = {}
        for r, c in L:
            for r2, c2 in G:
                d = (r2 - r, c2 - c); votes[d] = votes.get(d, 0) + 1
        v = max(votes, key=lambda d: (votes[d], -(abs(d[0]) + abs(d[1]))))   # ties -> the smallest translation (review 13, D1)
    else:
        v = (min(r for r, _ in G) - min(r for r, _ in L), min(c for _, c in G) - min(c for _, c in L))
    if v == (0, 0):
        return None
    hit = sum(1 for r, c in L if (r + v[0], c + v[1]) in G)
    return v if hit >= min_frac * len(L) else None

def _effect_of(act, action, ex_vals, bbox, freq=None, H=64, W=64, freq_before=None, before=None, after=None):
    """Deterministic description of ONE action's effect outside the HUD. ex_vals = [(r, c, before, after)].
    A colour 'shifts' only when its changed cells are a RIGID translation (gained == lost + v), its whole-board
    count is conserved, and it is not the ground: not the board's dominant colour, and not merely the cells a kept
    mover vacated/covered. On an exact two-colour swap the colour whose WHOLE cell set translates rigidly is the
    mover (review 11, DEF-1)."""
    eff = {"action": str(act or "").upper(), "cells": len(ex_vals), "kind": "none", "shifts": []}
    if not ex_vals:
        return eff
    if len(ex_vals) <= HUD_MAX_STEP and all(_hud_line(r, c, H, W) is not None for r, c, _, _ in ex_vals):
        eff["kind"] = "band"                                      # a few border-band cells only: most likely a not-yet-qualified HUD tick
        return eff
    lost, gained = {}, {}
    for r, c, vb, va in ex_vals:
        lost.setdefault(vb, set()).add((r, c)); gained.setdefault(va, set()).add((r, c))
    dominant = max(freq, key=freq.get) if freq else None
    cands = []
    for col, L in lost.items():
        G = gained.get(col)
        if not G or len(G) != len(L) or col == dominant:
            continue
        if freq is not None and freq_before is not None:
            fb, fa = freq_before.get(col, 0), freq.get(col, 0)
            if abs(fa - fb) > max(2, 0.02 * fb):
                continue                                          # the colour's population changed: recolour, not movement
        S = A = None
        if before is not None and after is not None:
            try:
                S = {(r, c) for r in range(len(before)) for c in range(len(before[r])) if before[r][c] == col}
                A = {(r, c) for r in range(len(after)) for c in range(len(after[r])) if after[r][c] == col}
            except Exception:
                S = A = None
        w = None
        if S and A and len(S) == len(A):
            w = (min(r for r, _ in A) - min(r for r, _ in S), min(c for _, c in A) - min(c for _, c in S))
            if w == (0, 0) or not all((r + w[0], c + w[1]) in A for r, c in S):
                w = None
        if w is not None:
            v, whole = w, True                                    # the colour's WHOLE cell set translated: certain vector (review 12)
        else:
            v = _best_shift(L, G)
            if v is None:
                continue                                          # not a (near-)rigid translation of the changed cells
            whole = bool(S) and A is not None and all((r + v[0], c + v[1]) in A for r, c in S)
        if not whole and freq:
            others = [o for o in set(lost) | set(gained) if o != col]
            oL = set().union(*[lost.get(o, set()) for o in others]) if others else set()
            oG = set().union(*[gained.get(o, set()) for o in others]) if others else set()
            if others and L <= oG and G <= oL and freq.get(col, 0) >= 2 * max(freq.get(o, 0) for o in others):
                continue                                          # the floor a (possibly non-conserved) sprite slid over: ground, not a mover
        d = ("down" if v[0] > 0 else "up" if v[0] < 0 else "") + ("-" if v[0] and v[1] else "") + ("right" if v[1] > 0 else "left" if v[1] < 0 else "")
        cands.append({"col": int(col), "d": d, "n": len(G), "L": L, "G": G, "whole": whole, "f": freq.get(col, 0) if freq else 0})
    cands.sort(key=lambda t: (not t["whole"], t["f"], t["col"]))    # whole-colour rigid movers first, then rarer colours
    kept = []
    for cd in cands:
        kg = set().union(*[k["G"] for k in kept]) if kept else set()
        kl = set().union(*[k["L"] for k in kept]) if kept else set()
        if kept and cd["L"] <= kg and cd["G"] <= kl:
            continue                                              # only the cells a kept mover covered/vacated: the ground
        kept.append(cd)
        if len(kept) == 2:
            break
    eff["shifts"] = [[k["col"], k["d"], k["n"]] for k in kept]
    eff["kind"] = "shift" if eff["shifts"] else "change"
    # object-level EVENTS for the model (deterministic perception of the diff): what appeared, vanished or recoloured,
    # excluding the cells already explained by a reported shift. Bounded to 6 lines. Gated with the ledger (F7/F8).
    try:
        if not _ON("AGENTFIX_LEDGER"):
            raise StopIteration
        shifted = {k["col"] for k in kept}
        events = []
        def bb(cells):
            rs = [r for r, _ in cells]; cs = [c for _, c in cells]
            return f"rows {min(rs)}-{max(rs)} cols {min(cs)}-{max(cs)}" if min(rs) != max(rs) or min(cs) != max(cs) else f"({min(rs)},{min(cs)})"
        pairs = {}
        for r, c, vb, va in ex_vals:
            pairs.setdefault((int(vb), int(va)), []).append((r, c))
        for (vb, va), cells in sorted(pairs.items(), key=lambda kv: -len(kv[1])):
            if vb in shifted or va in shifted:
                continue
            events.append(f"{len(cells)} cell{'s' if len(cells) > 1 else ''} colour {vb}->{va} at {bb(cells)}")
        events = [f"colour {col} object shifted {d} ({n} cells)" for col, d, n in eff["shifts"]] + events
        eff["events"] = events[:6] + ([f"+{len(events) - 6} more"] if len(events) > 6 else [])
    except Exception:
        pass
    if eff["action"] == "MOUSE" and bbox and isinstance(action, dict):
        try:
            r0, c0 = int(action.get("row")), int(action.get("col"))
            near = bbox[0] - 6 <= r0 <= bbox[2] + 6 and bbox[1] - 6 <= c0 <= bbox[3] + 6
            eff["where"] = "near click" if near else "away from click"
        except Exception:
            pass
    return eff

def _ledger_add(agent, compact):
    """Accumulate one executed single action's measured effect into the per-level ledger on the agent.
    Exactly ONE outcome per action (tallies sum to `tried`); MOUSE near/away counted separately."""
    eff = compact.get("effect")
    if not isinstance(eff, dict) or not eff.get("action") or eff["action"] == "RESET":
        return
    try:
        lvl = int(compact.get("level")) - (1 if compact.get("level_completed") else 0)   # level is reported AFTER the action
    except Exception:
        return
    led = agent.__dict__.setdefault("_agentfix_ledger", {})
    a = led.setdefault(lvl, {}).setdefault(eff["action"], {"tried": 0, "near": 0, "away": 0, "outcomes": {}, "last": ""})
    a["tried"] += 1
    if eff.get("where") == "near click":
        a["near"] += 1
    elif eff.get("where") == "away from click":
        a["away"] += 1
    if compact.get("game_over"):
        key, last = "GAME OVER", "GAME OVER"
    elif eff["kind"] == "none":
        key, last = "no change", "no change"
    elif eff["kind"] == "band":
        key, last = "border cells only", "only 1-3 border cells changed (probably HUD)"
    elif eff["kind"] == "shift":
        by_dir = {}
        for col, d, n in eff["shifts"]:
            by_dir.setdefault(d, []).append(int(col))
        key = " & ".join(f"colour {'+'.join(str(c) for c in sorted(cols))} {d}" for d, cols in sorted(by_dir.items())); last = key
    else:
        key, last = "changed cells", f"changed {eff['cells']} cells"
    a["outcomes"][key] = a["outcomes"].get(key, 0) + 1
    a["last"] = "COMPLETED THE LEVEL" if compact.get("level_completed") else last

def _ledger_lines(agent, level, valid_actions):
    """The prompt block: per-action measured effects on this level + untested valid actions. Empty if nothing measured."""
    led = getattr(agent, "_agentfix_ledger", {}) or {}
    try:
        lvl = int(level)
    except Exception:
        return ""
    entries = led.get(lvl, {})
    valid = []
    for v in (valid_actions or []):
        name = str(v).strip().upper()
        if _LEDGER_NORM is not None and name.startswith("ACTION"):
            try:
                name = str(_LEDGER_NORM(name) or name).upper()
            except Exception:
                pass
        if name and name not in valid:
            valid.append(name)
    untested = [v for v in valid if v not in entries and v not in ("RESET", "ACTION7") and (_LEDGER_CALLABLE is None or v in _LEDGER_CALLABLE)]
    if not entries and not untested:
        return ""
    parts = []
    for name, a in sorted(entries.items(), key=lambda kv: -kv[1]["tried"]):
        outs = sorted(a["outcomes"].items(), key=lambda kv: -kv[1])
        where = []
        if a.get("near") or a.get("away"):
            where.append(f"{a['near']} near click"); where.append(f"{a['away']} away from click")
            rest = a["tried"] - a["near"] - a["away"]
            if rest > 0:
                where.append(f"{rest} with no visible effect")
        head = f"{name}: {a['tried']} tried" + (f" ({', '.join(where)})" if where else "") + ": "
        def render(k_show, with_last):
            bits = [f"{n} {k}" for k, n in outs[:k_show]]
            if len(outs) > k_show:
                bits.append(f"+{sum(n for _, n in outs[k_show:])} tries in {len(outs) - k_show} other outcomes")
            return head + ", ".join(bits) + ((" -> last: " + a["last"]) if with_last and a["last"] else "")
        entry = render(3, True)
        for k_show, with_last in ((3, False), (2, False), (1, False)):   # fold outcomes into the bucket instead of slicing text
            if len(entry) <= ENTRY_MAX_CHARS:
                break
            entry = render(k_show, with_last)
        parts.append(entry)
    head = ""
    if untested:
        head = ("UNTESTED on this level: " + ", ".join(untested) + ". Test each once, on a board where its effect would be visible, "
                "before repeating an action that has changed nothing. ")
    prefix = "Harness-measured effects here (single actions, HUD excluded): "
    names = [name for name, _ in sorted(entries.items(), key=lambda kv: -kv[1]["tried"])]
    suffix = lambda i: (" | also tried here (not detailed): " + ", ".join(f"{n} x{entries[n]['tried']}" for n in names[i:])) if i < len(names) else ""
    room = LEDGER_MAX_CHARS - len(head) - len(prefix)
    shown = []
    for i, part in enumerate(parts):
        need = len(" | ".join(shown + [part])) + len(suffix(i + 1))
        if need > room:
            break
        shown.append(part)
    if not parts:
        body = "Harness-measured effects here: none yet."
    else:
        body = prefix + " | ".join(shown) + suffix(len(shown))
    text = head + body
    return text if len(text) <= LEDGER_MAX_CHARS else text[:LEDGER_MAX_CHARS - 3] + "..."

# ----------------------------------------------------------------------------- install
_STR_LIT = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')

def _norm_line(ln):
    indent = re.match(r"[ \t]*", ln).group(0); out = []; pos = 0
    for m in _STR_LIT.finditer(ln):
        out.append(re.sub(r"\s+", "", ln[pos:m.start()])); out.append(m.group(0)); pos = m.end()
    out.append(re.sub(r"\s+", "", ln[pos:]))
    return indent + "".join(out)

# F10 (stall triage, AGENTFIX_STALL_STEPS, default 40; 0 = off): actions on a level that is never cleared cost nothing
# under RHAE, but a stalled game's model requests take GPU share that the games still making progress would turn into
# analysis steps (v5 quality analysis: level yield per step is flat, so steps are the currency). After STALL_STEPS
# analysis steps on the same level, the game sleeps for STALL_FRAC x its last step duration before each further step
# (rate roughly halved at FRAC=1), capped at STALL_MAX_SLEEP seconds; the counter resets when the level changes.
# Model-invisible. should_stop is honoured during the sleep.
STALL_STEPS = int(os.environ.get("AGENTFIX_STALL_STEPS", "40"))
STALL_FRAC = float(os.environ.get("AGENTFIX_STALL_FRAC", "1.0"))
STALL_MAX_SLEEP = float(os.environ.get("AGENTFIX_STALL_MAX_SLEEP", "240"))

def _stall_tick(agent, should_stop=None):
    try:
        if STALL_STEPS <= 0 or not _ON("AGENTFIX_STALL"):
            return
        last = getattr(agent, "_last_action_result", None)
        lvl = last.get("level") if isinstance(last, dict) else None
        st = agent.__dict__.setdefault("_agentfix_stall", {"level": lvl, "steps": 0, "t_prev": None, "last_dt": None, "sleeps": 0})
        now = time.monotonic()
        if st["t_prev"] is not None:
            st["last_dt"] = now - st["t_prev"]
        st["t_prev"] = now
        if lvl != st["level"]:
            st["level"] = lvl; st["steps"] = 0
        st["steps"] += 1
        if st["steps"] > STALL_STEPS and st["last_dt"]:
            nap = min(max(0.0, st["last_dt"] * STALL_FRAC), STALL_MAX_SLEEP)
            if nap > 0:
                st["sleeps"] += 1
                try:
                    print(f"AGENTFIX_EVENT stall state={getattr(agent, '_agentfix_state', '?')} level={lvl} steps={st['steps']} sleep={nap:.0f}", flush=True)
                except Exception:
                    pass
                end = now + nap
                while time.monotonic() < end:
                    if should_stop is not None:
                        try:
                            if should_stop():
                                break
                        except Exception:
                            break
                    time.sleep(min(5.0, max(0.0, end - time.monotonic())))
                st["t_prev"] = time.monotonic()                              # the nap itself must not inflate the next last_dt
    except Exception:
        pass

# F11 (prompt de-duplication, AGENTFIX_DEDUP, default on): _build_user_prompt appends the same ~13 fixed instruction
# lines to EVERY user message, and those messages persist in history, so a request with k user turns carries k copies
# (strategy review 09-17: 36% of the model input is verbatim-repeated text; step time is prompt-proportional). Before
# each request, every user message EXCEPT the newest is sent with those fixed lines removed. The newest message, the
# dynamic head of older messages (executed actions / level / state / valid actions) and the world-model blocks are
# untouched; the stored history is not modified (trimming decisions unchanged), only the outgoing copy.
_DEDUP_FIXED = {
    "Only tool: `python`. It receives `current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`, `valid_actions`, `last_action_result`, and `action(actions)`.",
    "Only letter-coded board views and lightweight metadata are exposed; raw numeric color IDs are not available.",
    "Keep tool output compact: use `current_frame.segmentation` as the primary view, and `current_frame.ascii` only for a small specific region; never print full boards.",
    "For the most recent change, compare `previous_frame` to `current_frame`, or `last_transition.before_frame` to `last_transition.after_frame`; `history[-1].frame` is the current frame, not the previous one.",
    "Use Python to inspect the evidence, refine that world model from the newest history, and search or score candidate actions or short sequences against the current goal as you currently understand it.",
    "Maintain a compact working world model of what the current level seems to contain, what actions appear to do, what the goal seems to be, what is still uncertain, and what plan currently looks best.",
    "Below you are provided with the current world model from the previous turn. The default behavior is to copy it and add or remove things based on the evidence that you gathered. BEFORE EXECUTING NEW ACTIONS YOU MUST ALWAYS GIVE THE REVISED VERSION OF THE WORLD MODEL.",
    "You may call `action(actions)` more than once in one Python snippet if your search or control loop needs it, but stop immediately if a result reports `game_over`, `run_complete`, `level_completed`, or `done`.",
    "Ground yourself in `current_frame` before acting, but start with a compact structural summary rather than restating the full frame.",
    "Focus on what changed most recently in `history`, update the target environment change if needed, and separate gameplay-object changes from HUD-only changes.",
    "When ready, call `action(actions)` from inside the `python` tool with the best valid action or ordered batch selected by your code. If your code has found a reliable short sequence, prefer batching it in one call.",
    "You may call `action(actions)` more than once in one Python snippet if your search or control loop needs it.",
    "If you include assistant text before a tool call, keep it short and use it to update the world model. Helpful optional prefixes are `World model:`, `Goal model:`, `Action model:`, `Recent findings:`, `Open questions:`, `Plan:`, and `Cross-level notes:`.",
    "If you use MOUSE, include integer row and col arguments.",
}
_DEDUP_STATS = {"requests": 0, "chars_before": 0, "chars_after": 0}

def _dedup_text(text):
    kept = [ln for ln in text.split("\n") if ln.strip() not in _DEDUP_FIXED]
    return "\n".join(kept)

def _dedup_messages(messages):
    """Return a new message list; older user messages lose the fixed instruction lines. Never raises: on any error the
    original list is returned unchanged."""
    try:
        last_user = max((i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"), default=None)
        out = []; before = after = 0
        for i, m in enumerate(messages):
            if not (isinstance(m, dict) and m.get("role") == "user" and i != last_user):
                out.append(m); continue
            c = m.get("content")
            if isinstance(c, str):
                before += len(c); nc = _dedup_text(c); after += len(nc)
                out.append({**m, "content": nc})
            elif isinstance(c, list):
                parts = []
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                        before += len(part["text"]); nt = _dedup_text(part["text"]); after += len(nt)
                        parts.append({**part, "text": nt})
                    else:
                        parts.append(part)
                out.append({**m, "content": parts})
            else:
                out.append(m)
        _DEDUP_STATS["requests"] += 1; _DEDUP_STATS["chars_before"] += before; _DEDUP_STATS["chars_after"] += after
        return out
    except Exception:
        return messages

def _state_tag(state_path):
    """Short per-game tag for timing lines: the state directory name (the run stem), never the full path."""
    try:
        sp = str(state_path)
        parts = [x for x in sp.replace("\\", "/").split("/") if x]
        return parts[-2] if len(parts) >= 2 and "." in parts[-1] else (parts[-1] if parts else "?")
    except Exception:
        return "?"

def _acc_time(agent, key, t0):
    try:
        acc = agent.__dict__.setdefault("_agentfix_t", {"chat": 0.0, "nchat": 0, "ptok": 0, "ctok": 0, "tool": 0.0, "ntool": 0})
        acc[key] = acc.get(key, 0.0) + (time.monotonic() - t0); acc["n" + key] = acc.get("n" + key, 0) + 1
    except Exception:
        pass

def _timing_line(agent, kind, t0, **kv):
    """One AGENTFIX_TIMING log line. Never raises; off entirely unless AGENTFIX_TIMING is on."""
    try:
        if not _ON("AGENTFIX_TIMING"):
            return
        acc = getattr(agent, "_agentfix_t", None) or {}
        fields = {"state": getattr(agent, "_agentfix_state", "?"), "step": getattr(agent, "_agentfix_step", None),
                  "dt": f"{time.monotonic() - t0:.1f}"}
        if kind == "analyze":
            fields.update(nchat=acc.get("nchat", 0), chat=f"{acc.get('chat', 0.0):.1f}", ptok=acc.get("ptok", 0),
                          ctok=acc.get("ctok", 0), ntool=acc.get("ntool", 0), tool=f"{acc.get('tool', 0.0):.1f}")
        fields.update(kv)
        print("AGENTFIX_TIMING " + kind + " " + " ".join(f"{k}={v}" for k, v in fields.items()), flush=True)
    except Exception:
        pass

def install(ta_mod, an_mod, solv_mod):
    if getattr(ta_mod, "_agentfix_installed", False):
        return dict(ta_mod._agentfix_report, installed=True, note="already installed")
    TA = ta_mod.ToolAgent
    S = solv_mod._HarnessGameSession
    report = {"installed": True}

    if _ON("AGENTFIX_IMAGES"):
        _orig_phm = TA._persistent_history_messages
        def _persistent_history_messages(self, messages, *, tools=None):
            return _strip_images(_orig_phm(self, messages, tools=tools))
        TA._persistent_history_messages = _persistent_history_messages
        ta_mod._estimate_tokens = _estimate_tokens_flat
        # Stock over-charged ~442 est tokens per history image (~70-100 real), which was the only generation
        # headroom at the trimming steady state. With accurate estimates the real prompt reaches ~1.0x the
        # budget (review 4: r11l 32036/31744). Reserve room for the reply explicitly (read by ToolAgent.__init__).
        ta_mod._REQUEST_SAFETY_MARGIN_TOKENS = int(os.environ.get("AGENTFIX_SAFETY_MARGIN", "6144"))   # 4096 rejected by review 18: 6.4% of v3-length replies would not fit; cut only after v5's reply lengths are measured
        report["F1_images"] = True
        report["safety_margin"] = ta_mod._REQUEST_SAFETY_MARGIN_TOKENS

    if _ON("AGENTFIX_RESULT"):
        _orig_bup = TA._build_user_prompt
        def _build_user_prompt(self, action_num, **kw):
            text = _orig_bup(self, action_num, **kw)
            try:
                ps = kw.get("previous_step_summary")
                last = getattr(self, "_agentfix_last_exec", None) or getattr(self, "_last_action_result", None)
                hud_only = (isinstance(ps, dict) and isinstance(last, dict) and "board_changed_ex_hud" in last
                            and ps.get("executed_count") == 1 and last.get("executed_count", 1) == 1
                            and bool(ps.get("board_changed")) and not last.get("board_changed_ex_hud")
                            and not (ps.get("level_transition") or ps.get("run_complete") or ps.get("game_over")))
                line = self._describe_last_outcome(dict(ps, board_changed=False) if hud_only else ps)
                if line and hud_only:
                    line = line.rstrip() + " (only the HUD/border band changed; the play area did not.)"
            except Exception:
                line = ""
            if line:
                parts = text.split("\n", 1)
                text = parts[0] + "\n" + line + ("\n" + parts[1] if len(parts) > 1 else "")
            if _ON("AGENTFIX_LEDGER") and _ON("AGENTFIX_NOIMPACT"):
                try:
                    cf = kw.get("current_frame"); lvl = getattr(cf, "level", None)
                    if lvl is None and isinstance(kw.get("previous_step_summary"), dict):
                        lvl = kw["previous_step_summary"].get("level")
                    block = _ledger_lines(self, lvl, kw.get("valid_actions"))
                except Exception:
                    block = ""
                if block:
                    parts = text.split("\n", 2)
                    text = "\n".join(parts[:2]) + "\n" + block + ("\n" + parts[2] if len(parts) > 2 else "")
            return text
        TA._build_user_prompt = _build_user_prompt
        report["F2_result"] = True

    if _ON("AGENTFIX_MEMORY"):
        ta_mod._extract_labeled_blocks = _extract_labeled_blocks_tolerant
        def _update_summarized_knowledge_from_step_summary(self):
            summary = self._last_step_summary
            if not summary:
                return
            if summary.get("level_transition") or summary.get("run_complete"):
                for k in ("world_model", "goal_model", "action_model", "recent_findings", "open_questions", "current_plan"):
                    self._summarized_knowledge[k] = ""
        TA._update_summarized_knowledge_from_step_summary = _update_summarized_knowledge_from_step_summary
        report["F3_memory"] = True

    # F2 (always-return result) and F4 (loop bound) share the _run_python_tool wrapper
    _orig_rpt = TA._run_python_tool
    _orig_analyze = TA.analyze
    def analyze(self, state_path, action_num, *a, **kw):
        step = kw.get("analysis_step")
        if getattr(self, "_agentfix_step", None) != step:
            self._agentfix_step = step
            self._agentfix_noact = 0
            self._agentfix_codes = {}
            _stall_tick(self, kw.get("should_stop"))                      # F10: throttle a game stalled on one level
        _hm = getattr(self, "_history_messages", None)
        h0 = list(_hm) if isinstance(_hm, list) else _hm
        _t0 = time.monotonic(); self._agentfix_state = _state_tag(state_path)          # F9: per-step timing totals
        self._agentfix_t = {"chat": 0.0, "nchat": 0, "ptok": 0, "ctok": 0, "tool": 0.0, "ntool": 0}
        try:
            res = _orig_analyze(self, state_path, action_num, *a, **kw)
        except Exception:
            self._agentfix_codes = {}; self._agentfix_noact = 0      # retry of this step starts with discarded history
            _timing_line(self, "analyze", _t0, action=action_num, err=True)
            raise
        _timing_line(self, "analyze", _t0, action=action_num, executed=getattr(res, "step_executed", None))
        kept = getattr(res, "yielded_control", False) and getattr(self, "_history_messages", None) != h0
        if not kept:                                                  # plain return, or a yield whose history was discarded
            self._agentfix_codes = {}; self._agentfix_noact = 0
        return res
    TA.analyze = analyze

    def _run_python_tool(self, state_path, arguments):
        code = str(arguments.get("code", "")).rstrip()
        norm = "\n".join(_norm_line(ln) for ln in code.splitlines() if ln.strip())   # indentation + string literals kept, other spacing dropped
        h = hashlib.sha1(norm.encode()).hexdigest()
        codes = self.__dict__.setdefault("_agentfix_codes", {})
        if _ON("AGENTFIX_LOOP") and h in codes:
            prev = codes[h]
            self._agentfix_noact = getattr(self, "_agentfix_noact", 0) + 1     # a refusal is still a call with no action
            body = {"error": "You already ran this exact snippet in this turn; it was not run again.",
                    "previous_output_head": prev[:400],
                    "instruction": "Change the code, or execute an action now with your current best candidate."}
            if self._agentfix_noact >= LOOP_K:
                try:
                    print(f"AGENTFIX_EVENT notice noact={self._agentfix_noact}", flush=True)
                except Exception:
                    pass
                body["harness_notice"] = (f"{self._agentfix_noact} tool calls in this turn without executing an action. "
                                          "Execute your current best candidate action now; refine on the next turn.")
            return ta_mod._ToolDispatchResult(json.dumps(body, indent=2), step_executed=False)
        _tt = time.monotonic()
        res = (getattr(self, '_agentfix_orig_rpt', None) or _orig_rpt)(self, state_path, arguments)  # test hook
        _acc_time(self, 'tool', _tt)
        content = res.content
        extra = {}
        if _ON("AGENTFIX_RESULT") and res.step_executed:
            last = getattr(self, "_last_action_result", None)
            if isinstance(last, dict) and '"board_changed"' not in content:
                extra["last_action_result"] = {k: last.get(k) for k in ("executed", "action_num", "level", "score",
                        "board_changed", "board_changed_ex_hud", "changed_cells", "changed_bbox", "frames",
                        "distinct_frames", "effect", "repeat_of_known_no_op", "level_completed", "game_over",
                        "run_complete", "stop_reason", "stop_detail", "executed_count", "requested_count") if k in last}
        if _ON("AGENTFIX_LOOP"):
            codes[h] = content
            if res.step_executed:
                self._agentfix_noact = 0
            else:
                self._agentfix_noact = getattr(self, "_agentfix_noact", 0) + 1
                if self._agentfix_noact >= LOOP_K:
                    try:
                        print(f"AGENTFIX_EVENT notice noact={self._agentfix_noact}", flush=True)
                    except Exception:
                        pass
                    extra["harness_notice"] = (f"{self._agentfix_noact} tool calls in this turn without executing an action. "
                                               "Execute your current best candidate action now; refine on the next turn.")
        if extra:
            try:
                obj = json.loads(content)
                if isinstance(obj, dict):
                    obj.update(extra); content = json.dumps(obj, indent=2)      # ONE JSON object, renderer-safe
                else:
                    raise ValueError
            except Exception:
                content = content.rstrip() + "\n" + json.dumps(extra, indent=2)
            res = ta_mod._ToolDispatchResult(content, step_executed=res.step_executed)
        return res
    TA._run_python_tool = _run_python_tool
    report["F4_loop"] = _ON("AGENTFIX_LOOP")

    if _ON("AGENTFIX_NOIMPACT"):
        note5 = ("- Action results carry `board_changed_ex_hud` (did anything change outside the HUD/border band?) and, when "
                 "this exact action on this exact board was already executed and changed nothing outside the HUD, "
                 "`repeat_of_known_no_op: n` (it was executed again anyway; n = how many times before). Treat such a "
                 "repeat as wasted unless you expect a hidden counter.\n")
        if note5 not in ta_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM:
            ta_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM = ta_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM + note5
        _orig_step_env = S.step_env
        def step_env(self, arguments):
            return step_env_wrapped(self, _orig_step_env, arguments, lambda s: _grid_of(s.game.current_state))
        S.step_env = step_env
        _orig_car = TA._compact_action_result
        def _compact_action_result(self, payload):
            compact = _orig_car(self, payload)
            for k in _EXTRA_RESULT_KEYS:
                if k in payload:
                    compact[k] = payload[k]
            if payload.get("executed"):
                self._agentfix_last_exec = compact                 # last EXECUTED result (a refusal must not hide it from the outcome line)
                if (getattr(ta_mod, "_agentfix_report", None) or {}).get("F7_ledger"):
                    try:
                        _ledger_add(self, compact)
                    except Exception:
                        pass
            return compact
        TA._compact_action_result = _compact_action_result
        report["F5_noimpact"] = True

    # F9 (timing probe, AGENTFIX_TIMING, default on): wall clock of every model request, of the tool sandbox call, of
    # the engine step, and of the whole analysis step, plus the server's token usage, as AGENTFIX_TIMING log lines.
    # Pure observation: nothing model-visible changes, no control flow changes, every print is wrapped. Motivation:
    # the quality analysis of v5 (09-17) found 27 s per step unattributable because no per-phase timing was recorded.
    if _ON("AGENTFIX_DEDUP"):
        guidance = str(getattr(ta_mod, "TOOL_CALL_FORMAT_GUIDANCE", "") or "")
        for ln in guidance.split("\n"):
            if ln.strip():
                _DEDUP_FIXED.add(ln.strip())
        _orig_chat_d = TA._chat_completion
        def _chat_completion_dedup(self, messages, *a, **kw):
            return _orig_chat_d(self, _dedup_messages(messages), *a, **kw)
        TA._chat_completion = _chat_completion_dedup
        report["F11_dedup"] = True

    if _ON("AGENTFIX_TIMING"):
        _orig_chat = TA._chat_completion
        def _chat_completion(self, messages, *a, **kw):
            t0 = time.monotonic()
            try:
                res = _orig_chat(self, messages, *a, **kw)
            except BaseException as exc:
                _timing_line(self, "chat", t0, msgs=len(messages), err=type(exc).__name__)
                raise
            try:
                u = res.usage if isinstance(getattr(res, "usage", None), dict) else {}
                acc = self.__dict__.setdefault("_agentfix_t", {"chat": 0.0, "nchat": 0, "ptok": 0, "ctok": 0, "tool": 0.0, "ntool": 0})
                acc["chat"] += time.monotonic() - t0; acc["nchat"] += 1
                acc["ptok"] += int(u.get("prompt_tokens") or 0); acc["ctok"] += int(u.get("completion_tokens") or 0)
                _timing_line(self, "chat", t0, msgs=len(messages), ptok=u.get("prompt_tokens"), ctok=u.get("completion_tokens"),
                             finish=getattr(res, "finish_reason", ""))
            except Exception:
                pass
            return res
        TA._chat_completion = _chat_completion
        if report.get("F5_noimpact"):                       # engine-step timing rides on F5's step_env (keeps the kill-switch invariants)
            _orig_se_t = S.step_env
            def step_env(self, arguments):
                t0 = time.monotonic()
                try:
                    return _orig_se_t(self, arguments)
                finally:
                    try:
                        g = getattr(self, "game", None); r = getattr(g, "game_run", None)
                        gid = getattr(r, "game_id", None) or getattr(self, "game_index", "?")
                        print(f"AGENTFIX_TIMING env game={gid} dt={time.monotonic() - t0:.2f}", flush=True)
                    except Exception:
                        pass
            S.step_env = step_env
        report["F9_timing"] = True

    if _ON("AGENTFIX_ACTION7"):
        an_mod.ENGINE_TO_MODEL_ACTION["ACTION7"] = "ACTION7"
        an_mod.MODEL_TO_ENGINE_ACTION.clear()
        an_mod.MODEL_TO_ENGINE_ACTION.update({v: k for k, v in an_mod.ENGINE_TO_MODEL_ACTION.items()})
        note = ("- Some games list an extra action `ACTION7`. It takes no arguments (call `action(['ACTION7'])`) and "
                "costs an action. In several games it is UNDO (the board returns to its pre-move state). Use it when an "
                "undo would help; do not spend actions testing it for its own sake.\n")
        if note not in ta_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM:
            ta_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM = ta_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM + note
        report["F6_action7"] = True

    global _LEDGER_NORM, _LEDGER_CALLABLE
    _LEDGER_NORM = getattr(an_mod, "to_model_action", None)              # used by F5's action-name normaliser too (review 17)
    try:
        _LEDGER_CALLABLE = {str(k).upper() for k in getattr(an_mod, "MODEL_TO_ENGINE_ACTION", {})} or None
    except Exception:
        _LEDGER_CALLABLE = None
    if _ON("AGENTFIX_LEDGER") and _ON("AGENTFIX_RESULT") and _ON("AGENTFIX_NOIMPACT"):
        note7 = ("- The harness MEASURES the effect of every single action (cells changed outside the HUD, which colour "
                 "shifted which way, whether a click changed cells near it) and shows, per level, what each action has "
                 "done so far and which valid actions you have not tried on this level. Trust these measurements over "
                 "memory. Test every untested action once, early, on a board where its effect would be visible. Each "
                 "single action() result also carries `effect.events`: object shifts and every colour change with its "
                 "cell count and location (HUD excluded) -- read it before re-deriving the diff yourself.\n")
        if note7 not in ta_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM:
            ta_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM = ta_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM + note7
        report["F7_ledger"] = True
    report["F10_stall"] = bool(STALL_STEPS > 0 and _ON("AGENTFIX_STALL")); report["stall_steps"] = STALL_STEPS
    for k, env in (("F7_ledger","AGENTFIX_LEDGER"),("F9_timing","AGENTFIX_TIMING"),("F11_dedup","AGENTFIX_DEDUP"),("F1_images","AGENTFIX_IMAGES"),("F2_result","AGENTFIX_RESULT"),("F3_memory","AGENTFIX_MEMORY"),
                   ("F4_loop","AGENTFIX_LOOP"),("F5_noimpact","AGENTFIX_NOIMPACT"),("F6_action7","AGENTFIX_ACTION7")):
        report.setdefault(k, False); report[k + "_enabled"] = _ON(env)
    for fn in (TA._persistent_history_messages, TA._build_user_prompt, TA._update_summarized_knowledge_from_step_summary,
               TA.analyze, TA._run_python_tool, S.step_env, TA._compact_action_result, TA._chat_completion,
               ta_mod._estimate_tokens, ta_mod._extract_labeled_blocks):
        if getattr(fn, "__module__", "") not in ("inference.agent.tool_agent", "inference.framework.solver"):
            try: fn._agentfix = True
            except Exception: pass
    ta_mod._agentfix_installed = True; ta_mod._agentfix_report = dict(report)
    return report

# ===== end agentfix.py =====

# ===== begin agentfix_anim.py (verbatim) =====
"""agentfix_anim: F13, the animation-frame patch to the SHIPPED Duck agent (keithtyser bundle).

Applied from the kernel cell AFTER agentfix.install(...) has run, with the same modules:

    import agentfix, agentfix_anim
    rep = agentfix.install(ta, an, solv)
    agentfix_anim.install_anim(ta, an, solv, rep)

  F13 AGENTFIX_ANIM   arcengine returns a LIST of frames per action: `GameState.frame` is
                      `raw.frame[-1]` (the final board) and `GameState.animation_frames` is
                      `raw.frame[:-1]` -- every intermediate board of the move. The shipped
                      harness never reads them (`grep -rn "animation_frames\\|all_frames"` over
                      ARC3-Inference returns nothing), so the model only ever sees the final
                      board of a multi-frame action. Measured offline on the 25 public games
                      (first 40 c16 actions each) a large minority of actions animate, and the
                      games described by other teams as "not understandable without the
                      animation" (bp35 / sp80 / tn36) are among ours that never leave level 0.

                      This patch persists the intermediate boards (letter-coded ascii, the same
                      `Frame.ascii` encoding path as the final board) into the runtime state
                      next to the final frame, and exposes them inside the python tool as
                      `last_transition.frames` / `.frame_count` (also on `history[i]` and on the
                      frame views). No images are added; the numeric grid of an intermediate
                      frame is decoded inside the sandbox from its ascii so `.segmentation`
                      works on it.

Env switches
  AGENTFIX_ANIM=0             disables everything here; model-visible content is byte-identical
                              to stock (nothing is patched at all).
  AGENTFIX_ANIM_MAX_FRAMES=6  per action, how many intermediate boards are kept: first,
                              last-before-final and evenly spaced middles. `frame_count` always
                              reports the true total, including the dropped ones.
  AGENTFIX_ANIM_KEEP=3        how many of the most recent history entries carry their frames in
                              the runtime state / sandbox payload (0 = all of them). Bounded on
                              purpose: the state file is rewritten after every action and the
                              whole history is piped to the sandbox on every tool call, so
                              keeping animations for all N entries would grow both by
                              O(N * MAX_FRAMES * board) -- megabytes by mid-game.

F13b AGENTFIX_ANIM_SUMMARY    The frames above only reach the model if it calls the tool AND
                              inspects them; the transcripts say it rarely does.  This half
                              computes the motion in the harness (game thread, right after the
                              action) and gives it to the model for free as ONE line of the next
                              user prompt:

                                Animation: 4 intermediate frames; 'O' object moved (0,+1) x4
                                (cols 12->16); 3 'R' cells appeared only mid-animation at rows 20-22.

                              Per consecutive pair of boards (intermediates + the final board) the
                              largest changed 4-connected same-colour component is matched to the
                              nearest same-colour component of the next board; a pure translation
                              is reported as its (dr, dc), and runs of equal shifts collapse.
                              Cells holding a colour that is in NEITHER the board before the action
                              NOR the final board are counted separately -- several games put key
                              information only in mid-animation pixels.  The outer 2-cell HUD
                              border is ignored throughout.

  AGENTFIX_ANIM_SUMMARY=1     the line, the addendum sentence and the extra AGENTFIX_EVENT fields.
                              0 -> model-visible content is byte-identical to F13 alone.
  AGENTFIX_ANIM_SUMMARY_STRIP_OLD=1   older user messages of the OUTGOING copy lose their stale
                              copy of the line, so every request carries exactly one.
  AGENTFIX_ANIM_SUMMARY_CHARS=220     hard cap on the line (it is trimmed part by part to fit).
  AGENTFIX_ANIM_SUMMARY_WORK=60000    deterministic work budget per action (cells touched).  A
                              226-frame action on 64x64 stops early and says "analysis truncated"
                              rather than spending 100 ms in the game thread.
  AGENTFIX_ANIM_SUMMARY_OBJ=400       a flood-fill bigger than this is scenery, not an object.
  AGENTFIX_ANIM_SUMMARY_BORDER=2      HUD cells ignored on every edge.

Everything is fail-safe: any exception in the bookkeeping leaves the stock object untouched and
the stock value is returned.
"""
import json, os, threading

_ON = lambda k, d="1": os.environ.get(k, d).strip() not in ("0", "", "false", "False")


def _int_env(key, default):
    try:
        return int(str(os.environ.get(key, default)).strip() or default)
    except Exception:
        return int(default)


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


MAX_FRAMES = _int_env("AGENTFIX_ANIM_MAX_FRAMES", 6)
KEEP_ENTRIES = _int_env("AGENTFIX_ANIM_KEEP", 3)

ANIM_SUMMARY_ON = _ON("AGENTFIX_ANIM_SUMMARY")
ANIM_SUMMARY_STRIP_OLD = _ON("AGENTFIX_ANIM_SUMMARY_STRIP_OLD")
ANIM_LINE_CHARS = _int_env("AGENTFIX_ANIM_SUMMARY_CHARS", 220)
ANIM_WORK_CAP = _int_env("AGENTFIX_ANIM_SUMMARY_WORK", 60000)
ANIM_MAX_OBJ = _int_env("AGENTFIX_ANIM_SUMMARY_OBJ", 400)
ANIM_BORDER = _int_env("AGENTFIX_ANIM_SUMMARY_BORDER", 2)

ANIM_LINE_PREFIX = "Animation:"
ANIM_COLOR_CHARS = "WwgGcBMPRbSYOrNp"

# The one sentence the model is told. Appended to the system addendum exactly once.
ANIM_NOTE = (
    "- Some actions animate: `last_transition.frames` lists the intermediate boards (count in "
    "`last_transition.frame_count`); inspect them when an action's final board is confusing.\n"
)

# F13b's one sentence, appended once, after ANIM_NOTE.
ANIM_SUMMARY_NOTE = (
    "- The `Animation:` line summarises the intermediate frames of your last action; "
    "`last_transition.frames` has the boards.\n"
)

_ANIM_TL = threading.local()          # .map = the active session's {id(entry): (entry, frames, total)}
_ANIM_STATS = {"actions": 0, "animated": 0, "with_shift": 0, "with_midonly": 0, "truncated": 0,
          "errors": 0, "ms_max": 0.0, "ms_total": 0.0}


# --------------------------------------------------------------------------- capture helpers

def _keep_indices(total, cap):
    """Indices of the intermediate frames to keep: first, last-before-final, evenly spaced middles."""
    if total <= 0:
        return []
    if cap <= 0 or total <= cap:
        return list(range(total))
    if cap == 1:
        return [total - 1]
    return sorted({int(round(i * (total - 1) / float(cap - 1))) for i in range(cap)})


def _grid_rows(frame):
    """A taaf Frame (or raw np.int8 array) as a list of lists of ints."""
    data = getattr(frame, "data", frame)
    if hasattr(data, "tolist"):
        return data.tolist()
    return [[int(v) for v in row] for row in data]


def _animation_frames(state):
    try:
        frames = list(state.animation_frames)
        if frames:
            return frames
    except Exception:
        pass
    try:                                    # defensive: same thing straight off the raw payload
        return list(state.raw.frame)[:-1]
    except Exception:
        return []


def _frames_payload(rs_mod, state, step, level):
    """(payload list, true intermediate-frame count) for the action that produced `state`."""
    inter = _animation_frames(state)
    total = len(inter)
    if total <= 0:
        return [], 0
    out = []
    for i in _keep_indices(total, MAX_FRAMES):
        try:
            frame = rs_mod.Frame(grid=rs_mod.normalize_grid(_grid_rows(inter[i])), step=step, level=level)
            rows, cols = frame.shape
            out.append({"ascii": frame.ascii, "step": int(step), "level": int(level),
                        "shape": [int(rows), int(cols)], "index": int(i)})
        except Exception:
            continue
    return out, total


# ------------------------------------------------------------------- F13b: the animation summary
#
# All of this runs in the GAME thread, once per executed action, on plain python lists.  It is
# bounded by a deterministic work budget (cells touched) rather than a clock, so the same action
# always produces the same line -- a wall-clock budget would make the model-visible stream
# depend on machine load.

def _anim_char(value):
    try:
        return ANIM_COLOR_CHARS[max(0, min(15, int(value)))]
    except Exception:
        return "?"


def _anim_fmt_delta(value):
    value = int(value)
    return "0" if value == 0 else ("+%d" % value if value > 0 else "%d" % value)


def _anim_background(rows, H, W, b):
    """The most common interior colour -- treated as the floor, never as a moving object."""
    counts = {}
    for r in range(b, H - b):
        row = rows[r]
        for c in range(b, min(len(row), W - b)):
            value = row[c]
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _anim_diff_cells(a, b_rows, H, W, b, work):
    """Interior cells where two boards differ (whole-row compare first: the common case is cheap)."""
    out = []
    work[0] += H
    for r in range(b, H - b):
        row_a = a[r]
        row_b = b_rows[r]
        if row_a == row_b:
            continue
        hi = min(len(row_a), len(row_b), W - b)
        work[0] += hi - b
        for c in range(b, hi):
            if row_a[c] != row_b[c]:
                out.append((r, c))
    return out


def _anim_flood(rows, seed, colour, H, W, b, seen, work, max_obj):
    """(cells, complete) for the 4-connected same-colour component holding `seed`.

    `complete` is False once the component passes `max_obj` cells: it is scenery, and what was
    collected is only a FRAGMENT of it.  A fragment must never be treated as an object -- the
    first version of this module did, and reported the fragment's translation as an object move
    (10 of 232 pairs on the c16 replay, all of them fictional).
    """
    stack = [seed]
    seen.add(seed)
    cells = [seed]
    while stack:
        y, x = stack.pop()
        work[0] += 4
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if b <= ny < H - b and b <= nx < W - b and (ny, nx) not in seen:
                line = rows[ny]
                if nx < len(line) and line[nx] == colour:
                    seen.add((ny, nx))
                    stack.append((ny, nx))
                    cells.append((ny, nx))
                    if len(cells) > max_obj:
                        return cells, False
    return cells, True


def _anim_touches(cells, dead):
    for y, x in cells:
        if (y - 1, x) in dead or (y + 1, x) in dead or (y, x - 1) in dead or (y, x + 1) in dead:
            return True
    return False


def _anim_components(rows, seeds, colour, H, W, b, work, max_obj, cap):
    """Every WHOLE same-colour component of `rows` touching one of `seeds`, biggest first.

    Components that are (or touch) an oversized one are dropped: a later seed inside a component
    whose flood was abandoned would otherwise be fenced in by the cells the abandoned flood had
    already marked, and come back as a small "object" that does not exist.
    """
    seen = set()
    dead = set()
    out = []
    for cell in seeds:
        if cell in seen:
            continue
        if work[0] > cap:            # the caller discards a partial answer; stop paying for it
            break
        cells, complete = _anim_flood(rows, cell, colour, H, W, b, seen, work, max_obj)
        if not complete:
            dead.update(cells)
            continue
        out.append(cells)
    if dead:
        out = [cells for cells in out if not _anim_touches(cells, dead)]
    out.sort(key=lambda cells: -len(cells))
    return out


def _anim_bbox(cells):
    ys = [p[0] for p in cells]
    xs = [p[1] for p in cells]
    return (min(ys), min(xs), max(ys), max(xs))


def _anim_centre(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def anim_pair_shift(a, b_rows, bg, H, W, b, work, cap, max_obj):
    """(kind, colour, shift, from_bbox, to_bbox) for ONE consecutive pair of boards.

    kind: "same" (no interior change), "move" (the largest changed object is a pure translation
    of a same-colour component of the next board), "redraw" (it changed some other way),
    "budget" (the work cap stopped the analysis).
    """
    diff = _anim_diff_cells(a, b_rows, H, W, b, work)
    if work[0] > cap:
        return ("budget", None, None, None, None)
    if not diff:
        return ("same", None, (0, 0), None, None)
    was, now = {}, {}
    for (r, c) in diff:
        value_a = a[r][c]
        value_b = b_rows[r][c]
        if value_a != bg:
            was[value_a] = was.get(value_a, 0) + 1
        if value_b != bg:
            now[value_b] = now.get(value_b, 0) + 1
    # a moving object vacates cells AND occupies cells: its colour is in both tallies
    order = sorted(set(was) | set(now),
                   key=lambda v: (-(1 if (v in was and v in now) else 0),
                                  -(was.get(v, 0) + now.get(v, 0)), v))
    fallback = None
    for colour in order[:2]:
        seeds_a = [cell for cell in diff if a[cell[0]][cell[1]] == colour]
        if not seeds_a:
            continue
        comps_a = _anim_components(a, seeds_a, colour, H, W, b, work, max_obj, cap)
        if work[0] > cap:
            return ("budget", None, None, None, None)
        if not comps_a:
            continue
        obj = comps_a[0]
        box_a = _anim_bbox(obj)
        if fallback is None:
            fallback = (colour, box_a)
        seeds_b = [cell for cell in diff if b_rows[cell[0]][cell[1]] == colour]
        if not seeds_b:
            continue
        comps_b = _anim_components(b_rows, seeds_b, colour, H, W, b, work, max_obj, cap)
        if work[0] > cap:
            return ("budget", None, None, None, None)
        cy, cx = _anim_centre(box_a)
        target = set(obj)
        same_size = [cells for cells in comps_b if len(cells) == len(obj)]
        for cells in sorted(same_size, key=lambda cs: (abs(_anim_centre(_anim_bbox(cs))[0] - cy)
                                                       + abs(_anim_centre(_anim_bbox(cs))[1] - cx))):
            box_b = _anim_bbox(cells)
            dr, dc = box_b[0] - box_a[0], box_b[1] - box_a[1]
            if set((y + dr, x + dc) for y, x in target) == set(cells):
                return ("move", colour, (dr, dc), box_a, box_b)
    if fallback is not None:
        return ("redraw", fallback[0], None, fallback[1], None)
    return ("redraw", None, None, None, None)


def _anim_runs(values):
    """[a, a, b] -> [[a, 2], [b, 1]]"""
    out = []
    for value in values:
        if out and out[-1][0] == value:
            out[-1][1] += 1
        else:
            out.append([value, 1])
    return out


def anim_render_line(facts, chars=None):
    """The ONE prompt line for these facts, never longer than `chars`."""
    chars = ANIM_LINE_CHARS if chars is None else int(chars)
    if not facts.get("frames"):
        return "Animation: none (single frame)."
    head = "Animation: %d intermediate frame%s" % (facts["frames"], "" if facts["frames"] == 1 else "s")
    moves = facts["move_runs"]
    other = facts["still"] + facts["redraw"]
    line = ""
    for keep, with_span, with_other, with_rows in ((4, 1, 1, 1), (4, 0, 1, 1), (3, 0, 1, 1),
                                                   (2, 0, 1, 1), (2, 0, 0, 1), (1, 0, 0, 1),
                                                   (1, 0, 0, 0)):
        if moves:
            shown = moves[:keep]
            text = ", ".join(shown)
            if len(moves) > len(shown):
                text += " +%d more" % (len(moves) - len(shown))
            motion = ("; '%s' object moved %s" % (facts["colour"], text) if facts["colour"]
                      else "; moved %s" % text)
            if with_span and facts["span"]:
                motion += " " + facts["span"]
            if with_other and other:
                motion += "; %d frame%s without a clean shift" % (other, "" if other == 1 else "s")
        elif facts["redraw"]:
            motion = "; no single moving object (%d frame%s redrawn)" % (
                facts["redraw"], "" if facts["redraw"] == 1 else "s")
        else:
            motion = "; no interior change"
        tail = ""
        if facts["mid"]:
            tail = "; %d '%s' cell%s appeared only mid-animation" % (
                facts["mid"], facts["mid_colour"], "" if facts["mid"] == 1 else "s")
            if with_rows and facts["mid_rows"]:
                lo, hi = facts["mid_rows"]
                tail += (" at row %d" % lo) if lo == hi else (" at rows %d-%d" % (lo, hi))
        if facts["budget"]:
            tail += "; analysis truncated"
        line = head + motion + tail + "."
        if len(line) <= chars:
            return line
    line = line[:chars - 1]                  # nothing fits: cut on the last clause boundary
    cut = line.rfind("; ")
    if cut > len(head):
        line = line[:cut]
    return line.rstrip(" ,;") + "."


def anim_summarize_frames(inter, final_rows, before_rows, to_rows=None, border=None, cap=None,
                     max_obj=None, chars=None):
    """(line, facts) for one executed action.

    `inter` are the intermediate boards (raw engine frames when `to_rows` is given, else plain
    grids), `final_rows` the board the action ended on, `before_rows` the board it started from.
    Frames are materialised lazily and charged to the work budget, so a 226-frame action stops
    converting as soon as the budget is gone.
    """
    border = ANIM_BORDER if border is None else int(border)
    cap = ANIM_WORK_CAP if cap is None else int(cap)
    max_obj = ANIM_MAX_OBJ if max_obj is None else int(max_obj)
    facts = {"frames": len(inter), "colour": "", "move_runs": [], "span": "", "still": 0,
             "redraw": 0, "mid": 0, "mid_colour": "", "mid_rows": None, "budget": False,
             "work": 0, "kinds": [], "shift": "none"}
    if not inter or not final_rows:
        return "Animation: none (single frame).", facts
    H = len(final_rows)
    W = max(len(row) for row in final_rows)
    b = border if (H > 2 * border + 1 and W > 2 * border + 1) else 0
    work = [0]
    charge = max(1, (H * W) // 16)
    total = len(inter)
    cache = {}

    def frame(index):
        rows = cache.get(index)
        if rows is None:
            if index >= total:
                rows = final_rows
            else:
                rows = to_rows(inter[index]) if to_rows is not None else inter[index]
                work[0] += charge
            cache[index] = rows
        return rows

    bg = _anim_background(final_rows, H, W, b)
    work[0] += H * W
    shifts, colours = [], {}
    first_box = last_box = None
    for i in range(total):
        if work[0] > cap:
            facts["budget"] = True
            break
        kind, colour, shift, box_a, box_b = anim_pair_shift(frame(i), frame(i + 1), bg, H, W, b,
                                                       work, cap, max_obj)
        if kind == "budget":
            facts["budget"] = True
            break
        facts["kinds"].append((kind, shift, _anim_char(colour) if colour is not None else ""))
        if kind == "same":
            facts["still"] += 1
        elif kind == "move":
            colours[colour] = colours.get(colour, 0) + 1
            shifts.append(shift)
            if first_box is None:
                first_box = box_a
            last_box = box_b
        else:
            facts["redraw"] += 1
    mid = {}
    if not facts["budget"]:
        # cells holding a colour that is in NEITHER the board before the action nor the final
        # board: information that exists only while the animation is running
        for i in range(total):
            if work[0] > cap:
                facts["budget"] = True
                break
            rows = frame(i)
            work[0] += H
            for r in range(b, H - b):
                row = rows[r]
                if row == final_rows[r] or (r < len(before_rows) and row == before_rows[r]):
                    continue
                hi = min(len(row), W - b)
                work[0] += hi - b
                fin = final_rows[r]
                bef = before_rows[r] if r < len(before_rows) else ()
                for c in range(b, hi):
                    value = row[c]
                    if value != fin[c] and (c >= len(bef) or value != bef[c]):
                        mid[(r, c)] = value
    facts["work"] = work[0]
    if colours:
        facts["colour"] = _anim_char(max(colours.items(), key=lambda kv: kv[1])[0])
    facts["move_runs"] = ["(%s,%s)%s" % (_anim_fmt_delta(shift[0]), _anim_fmt_delta(shift[1]),
                                         "" if n == 1 else " x%d" % n)
                          for shift, n in _anim_runs(shifts)]
    facts["shift"] = ";".join(item.replace(" ", "") for item in facts["move_runs"]) or "none"
    if first_box and last_box and (first_box[0] != last_box[0] or first_box[1] != last_box[1]):
        if first_box[0] == last_box[0]:
            facts["span"] = "(cols %d->%d)" % (first_box[1], last_box[1])
        elif first_box[1] == last_box[1]:
            facts["span"] = "(rows %d->%d)" % (first_box[0], last_box[0])
        else:
            facts["span"] = "(r%d,c%d->r%d,c%d)" % (first_box[0], first_box[1],
                                                    last_box[0], last_box[1])
    if mid:
        tally = {}
        for value in mid.values():
            tally[value] = tally.get(value, 0) + 1
        facts["mid"] = len(mid)
        facts["mid_colour"] = _anim_char(max(tally.items(), key=lambda kv: kv[1])[0])
        rows_hit = sorted({cell[0] for cell in mid})
        facts["mid_rows"] = (rows_hit[0], rows_hit[-1])
    return anim_render_line(facts, chars), facts


# ------------------------------------------------------------ F13b: prompt plumbing (user text)

def _anim_user_text(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _anim_has_line(text):
    return any(item.strip().startswith(ANIM_LINE_PREFIX) for item in str(text).split("\n"))


def _anim_drop_line(text):
    kept = [item for item in str(text).split("\n") if not item.strip().startswith(ANIM_LINE_PREFIX)]
    new = "\n".join(kept)
    return new, len(text) - len(new)


def _anim_map_user_text(message, func):
    content = message.get("content")
    if isinstance(content, str):
        new, removed = func(content)
        return ({**message, "content": new}, removed) if removed else (message, 0)
    if isinstance(content, list):
        parts, total = [], 0
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                new, removed = func(part["text"])
                total += removed
                parts.append({**part, "text": new} if removed else part)
            else:
                parts.append(part)
        return ({**message, "content": parts}, total) if total else (message, 0)
    return message, 0


def _anim_strip_old_lines(messages):
    """Outgoing copy only: keep the LAST `Animation:` line, drop the stale ones."""
    last = None
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user" and _anim_has_line(_anim_user_text(message)):
            last = index
    if last is None:
        return messages
    out = []
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user" and index != last:
            out.append(_anim_map_user_text(message, _anim_drop_line)[0])
        else:
            out.append(message)
    return out


def _anim_session_summary(session):
    """Compute the line for the action just executed and park it on this game's analyzer."""
    import time
    state = session.game.current_state
    inter = _animation_frames(state)
    final = getattr(state, "frame", None)
    final_rows = _grid_rows(final) if final is not None else []
    entries = list(getattr(session, "history_entries", None) or [])
    before_rows = []
    if len(entries) >= 2:
        before_rows = [list(row) for row in (getattr(entries[-2].frame, "grid", ()) or ())]
    if not before_rows and inter:
        before_rows = _grid_rows(inter[0])
    started = time.perf_counter()
    line, facts = anim_summarize_frames(inter, final_rows, before_rows, to_rows=_grid_rows)
    elapsed = (time.perf_counter() - started) * 1000.0
    _ANIM_STATS["actions"] += 1
    _ANIM_STATS["ms_total"] = round(_ANIM_STATS["ms_total"] + elapsed, 3)
    if elapsed > _ANIM_STATS["ms_max"]:
        _ANIM_STATS["ms_max"] = round(elapsed, 3)
    if facts["frames"]:
        _ANIM_STATS["animated"] += 1
    if facts["move_runs"]:
        _ANIM_STATS["with_shift"] += 1
    if facts["mid"]:
        _ANIM_STATS["with_midonly"] += 1
    if facts["budget"]:
        _ANIM_STATS["truncated"] += 1
    try:
        setattr(session.analyzer, "_agentfix_anim_line", line)
    except Exception:
        pass
    return line, facts


# --------------------------------------------------------------------------- install

def install_anim(ta_mod, an_mod, solv_mod, report):
    report = report if isinstance(report, dict) else {}
    report.setdefault("F13_anim", False)
    report.setdefault("F13_summary", False)
    report["anim_max_frames"] = MAX_FRAMES
    report["anim_keep_entries"] = KEEP_ENTRIES
    report["F13_anim_enabled"] = _ON("AGENTFIX_ANIM")
    if not _ON("AGENTFIX_ANIM"):
        return report
    if getattr(ta_mod, "_agentfix_anim_installed", False):
        report["F13_anim"] = True
        report["note_anim"] = "already installed"
        return report

    import sys
    rs_mod = sys.modules.get("inference.agent.runtime_state")
    sb_mod = sys.modules.get("inference.agent.python_tool_sandbox")
    if rs_mod is None or sb_mod is None:
        try:
            import inference.agent.runtime_state as rs_mod            # noqa: F811
            import inference.agent.python_tool_sandbox as sb_mod      # noqa: F811
        except Exception as exc:
            report["anim_error"] = f"modules unavailable: {type(exc).__name__}: {exc}"
            return report

    try:            # the history entry that carries the frames (frozen dataclass + two fields)
        import dataclasses

        _AnimHistoryEntry = dataclasses.dataclass(frozen=True)(
            type("_AnimHistoryEntry", (rs_mod.HistoryEntry,),
                 {"__annotations__": {"frames": tuple, "frame_count": int},
                  "frames": (), "frame_count": 0})
        )
    except Exception as exc:                 # nothing patched yet -> stock behaviour
        report["anim_error"] = f"entry subclass: {type(exc).__name__}: {exc}"
        return report

    # -- 1. the sandbox source template (a module-level constant, read by run_sandboxed_python at
    #       Popen time -> patching the constant is enough, and it must happen before any spawn).
    try:
        patched = _anim_patch_sandbox_source(sb_mod._SANDBOX_BOOTSTRAP)
    except Exception as exc:
        report["anim_error"] = f"sandbox template: {type(exc).__name__}: {exc}"
        return report                                   # nothing patched yet -> stock behaviour
    sb_mod._SANDBOX_BOOTSTRAP = patched

    # -- 2. capture: every executed action's intermediate frames, recorded as the runtime state is
    #       written (the session writes it immediately after game.execute_action, so
    #       game.current_state is exactly the state of the last history entry).
    S = solv_mod._HarnessGameSession
    _orig_write = S.write_runtime_state

    def write_runtime_state(self):
        anim = None
        try:
            anim = _record_frames(self, rs_mod, solv_mod)
        except Exception:
            anim = None
        _ANIM_TL.map = anim
        try:
            return _orig_write(self)
        finally:
            _ANIM_TL.map = None

    S.write_runtime_state = write_runtime_state

    # the frames ride into the JSON through the entry serializer (a no-op unless _ANIM_TL.map is set)
    _orig_entry_payload = rs_mod.history_entry_to_payload

    def history_entry_to_payload(entry):
        payload = _orig_entry_payload(entry)
        try:
            table = getattr(_ANIM_TL, "map", None)
            if table:
                found = table.get(id(entry))
                if found is not None and found[0] is entry:
                    if found[1]:
                        payload["frames"] = found[1]
                    payload["frame_count"] = int(found[2])
        except Exception:
            pass
        return payload

    rs_mod.history_entry_to_payload = history_entry_to_payload

    # -- 3. load: carry the frames back out of the runtime state into the agent.
    _orig_load = ta_mod.load_runtime_state

    def load_runtime_state(path):
        try:
            if not path.exists():
                return None, []
            payload = json.loads(path.read_text(encoding="utf-8"))
            current = rs_mod.frame_from_payload(payload.get("current_frame"))
            entries = []
            for raw in payload.get("history", []) or []:
                entry = rs_mod.history_entry_from_payload(raw)
                if entry is None:
                    continue
                try:                       # one malformed entry must not cost the others their frames
                    frames = raw.get("frames") if isinstance(raw, dict) else None
                    count = raw.get("frame_count") if isinstance(raw, dict) else 0
                    if frames or count:
                        entry = _AnimHistoryEntry(
                            action=entry.action, frame=entry.frame,
                            frames=tuple(f for f in (frames or []) if isinstance(f, dict)),
                            frame_count=_safe_int(count),
                        )
                except Exception:
                    pass
                entries.append(entry)
            return current, entries
        except Exception:
            return _orig_load(path)

    ta_mod.load_runtime_state = load_runtime_state

    # -- 4. expose: the sandbox state payload carries the frames on the entry's frame view.
    _orig_frame_payload = ta_mod._ascii_frame_view_payload
    _orig_history_payload = ta_mod._ascii_history_view_payload

    def _ascii_history_view_payload(history_entries):
        try:
            payload = []
            for entry in history_entries:
                frame_payload = _orig_frame_payload(entry.frame)
                if frame_payload is None:
                    continue
                frames = list(getattr(entry, "frames", ()) or ())
                count = int(getattr(entry, "frame_count", 0) or 0)
                if frames:
                    frame_payload["frames"] = frames
                if count:
                    frame_payload["frame_count"] = count
                payload.append({"action": entry.action, "frame": frame_payload})
            return payload
        except Exception:
            return _orig_history_payload(history_entries)

    ta_mod._ascii_history_view_payload = _ascii_history_view_payload

    # -- 5. tell the model (one sentence, once).
    try:
        if ANIM_NOTE not in ta_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM:
            ta_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM = ta_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM + ANIM_NOTE
    except Exception as exc:
        report["anim_error"] = f"addendum: {type(exc).__name__}: {exc}"

    # -- 6. F13b: the harness-computed motion line (appended, so it composes with agentfix_mem's
    #       prepended block and agentfix_ewm's appended line in either install order).
    if ANIM_SUMMARY_ON:
        TA = ta_mod.ToolAgent
        _orig_knowledge_lines = TA._summarized_knowledge_lines

        def _summarized_knowledge_lines(self):
            stock = _orig_knowledge_lines(self)
            try:
                line = str(getattr(self, "_agentfix_anim_line", "") or "")
            except Exception:
                line = ""
            return [*stock, line] if line else stock

        TA._summarized_knowledge_lines = _summarized_knowledge_lines

        # exactly one copy per request: the older user messages of the OUTGOING copy lose their
        # stale line.  The stored history is untouched.
        if ANIM_SUMMARY_STRIP_OLD:
            _orig_chat = TA._chat_completion

            def _chat_completion(self, messages, *args, **kwargs):
                try:
                    messages = _anim_strip_old_lines(messages)
                except Exception:
                    pass
                return _orig_chat(self, messages, *args, **kwargs)

            TA._chat_completion = _chat_completion
            try:
                TA._chat_completion._agentfix = True
            except Exception:
                pass
        try:
            TA._summarized_knowledge_lines._agentfix = True
        except Exception:
            pass
        try:
            if ANIM_SUMMARY_NOTE not in ta_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM:
                ta_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM = (
                    ta_mod.STRUCTURED_RUNTIME_STATE_ADDENDUM + ANIM_SUMMARY_NOTE)
        except Exception as exc:
            report["anim_error"] = f"summary addendum: {type(exc).__name__}: {exc}"

    for fn in (S.write_runtime_state, ta_mod.load_runtime_state, ta_mod._ascii_history_view_payload,
               rs_mod.history_entry_to_payload):
        try:
            fn._agentfix = True
        except Exception:
            pass
    ta_mod._agentfix_anim_installed = True
    report["F13_anim"] = True
    report["F13_summary"] = bool(ANIM_SUMMARY_ON)
    report["F13_summary_strip_old"] = bool(ANIM_SUMMARY_ON and ANIM_SUMMARY_STRIP_OLD)
    report["anim_summary_chars"] = ANIM_LINE_CHARS
    report["anim_summary_work"] = ANIM_WORK_CAP
    report["anim_summary_obj"] = ANIM_MAX_OBJ
    report["anim_summary_border"] = ANIM_BORDER
    report["anim_summary_stats"] = _ANIM_STATS
    try:
        ta_mod._agentfix_report = dict(report)
    except Exception:
        pass
    return report


def _record_frames(session, rs_mod, solv_mod):
    """Stash the intermediate frames of the newest history entry; returns the id->frames table."""
    state = session.__dict__.setdefault("_agentfix_anim", {"map": {}, "ac": None})
    entries = getattr(session, "history_entries", None) or []
    if not entries:
        return state["map"]
    last = entries[-1]
    if not str(getattr(last, "action", "") or "").strip():
        return state["map"]                        # the seeded initial frame is not a transition
    try:
        count = int(session.action_count)
    except Exception:
        count = None
    if id(last) not in state["map"]:
        step = getattr(last.frame, "step", 0)
        level = getattr(last.frame, "level", 1)
        frames, total = _frames_payload(rs_mod, session.game.current_state, step, level)
        state["map"][id(last)] = (last, frames, total)
        extra = ""
        if ANIM_SUMMARY_ON:
            try:
                _line, facts = _anim_session_summary(session)
                extra = " shift=%s midonly=%d%s" % (facts["shift"], facts["mid"],
                                                    " truncated=1" if facts["budget"] else "")
            except Exception as exc:
                _ANIM_STATS["errors"] += 1
                extra = " shift=error:%s" % type(exc).__name__
        if count != state["ac"]:
            state["ac"] = count
            try:
                print("AGENTFIX_EVENT anim game=%s level=%s frames=%d%s"
                      % (getattr(session.game, "game_id", "?"), level, total, extra), flush=True)
            except Exception:
                pass
    if KEEP_ENTRIES > 0 and len(state["map"]) > KEEP_ENTRIES:
        live = {id(e) for e in entries[-KEEP_ENTRIES:]}
        for key in [k for k in state["map"] if k not in live]:
            state["map"].pop(key, None)
    return state["map"]


# --------------------------------------------------------------------------- sandbox template

_ANIM_SB_EDITS = (
    # helpers + FrameView carries .frames / .frame_count
    ("""class FrameView:
    def __init__(self, *, ascii, step, level, shape, grid):
        self.ascii = ascii""",
     """def _grid_from_ascii(text):
    try:
        index = {ch: i for i, ch in enumerate(COLOR_CHARS)}
        return [[index.get(ch, 0) for ch in line] for line in str(text).split("\\n") if line]
    except Exception:
        return []


def _anim_frame_view(payload):
    view = _frame_from_payload(payload)
    if view is not None:
        view.grid = [list(row) for row in (view._grid or [])]
        view.index = int(payload.get("index", 0) or 0)
    return view


class FrameView:
    def __init__(self, *, ascii, step, level, shape, grid, frames=(), frame_count=0):
        self.frames = [item for item in frames if item is not None]
        self.frame_count = int(frame_count or 0)
        self.ascii = ascii"""),
    # an intermediate frame ships as ascii only; decode it so .segmentation / .grid work
    ("""        self._grid = grid""",
     """        self._grid = grid if grid else _grid_from_ascii(ascii)"""),
    ("""class HistoryEntryView:
    def __init__(self, *, action, frame):
        self.action = action
        self.frame = frame""",
     """class HistoryEntryView:
    def __init__(self, *, action, frame):
        self.action = action
        self.frame = frame
        self.frames = list(getattr(frame, "frames", ()) or ())
        self.frame_count = int(getattr(frame, "frame_count", 0) or 0)"""),
    ("""        self.frame = after_frame
        self.result = dict(result) if isinstance(result, dict) else {}""",
     """        self.frame = after_frame
        self.frames = list(getattr(after_frame, "frames", ()) or ())
        self.frame_count = int(getattr(after_frame, "frame_count", 0) or 0)
        self.result = dict(result) if isinstance(result, dict) else {}"""),
    ("""        grid=payload.get("grid", []),
    )""",
     """        grid=payload.get("grid", []),
        frames=[_anim_frame_view(item) for item in (payload.get("frames") or []) if isinstance(item, dict)],
        frame_count=int(payload.get("frame_count", 0) or 0),
    )"""),
)


def _anim_patch_sandbox_source(source):
    """Rewrite the sandbox bootstrap so the frame views carry the animation frames.

    Raises if any anchor is not found exactly once -- the caller then leaves the stock template
    (and everything else) alone rather than spawning a broken sandbox.
    """
    text = str(source)
    if "self.frame_count" in text:
        return text                                   # already patched
    for old, new in _ANIM_SB_EDITS:
        found = text.count(old)
        if found != 1:
            raise RuntimeError(f"anchor found {found} times, expected 1: {old.splitlines()[0]!r}")
        text = text.replace(old, new)
    compile(text, "<sandbox>", "exec")                # never ship an unparsable child program
    return text


# The kernel inlines every agentfix module into ONE cell namespace, so a module-level name defined
# by two modules silently belongs to whichever source was pasted last.  Everything above is
# anim-private for that reason; this legacy alias is published only when no other module owns the
# name (agentfix_ewm.py defines its own `_patch_sandbox_source`).
if "_patch_sandbox_source" not in globals():
    _patch_sandbox_source = _anim_patch_sandbox_source

# ===== end agentfix_anim.py =====

# ===== begin agentfix_sheet.py (verbatim) =====
"""agentfix_sheet: F19, the ALL-FRAMES CONTACT SHEET for the SHIPPED Duck agent.

Applied from the kernel cell AFTER agentfix.install(...) and next to agentfix_anim / agentfix_win,
with the same modules:

    rep = install(_ta, _an, _solv)
    rep.update(install_anim(_ta, _an, _solv, rep) or {})
    rep.update(install_win(_ta, _an, _solv, rep) or {})
    rep.update(install_sheet(_ta, _an, _solv, rep) or {})

WHY
  F13 (agentfix_anim) is the only change that has ever moved the hidden leaderboard (3.20 ->
  3.71): it exposes the engine's intermediate frames in the SANDBOX.  A harness-authored
  per-turn TEXT summary of those same frames did not transfer (2.57).  Frames yes, narration no.
  F13's frames only reach the model if it writes python that reads `last_transition.frames`.
  A sibling harness instead ships EVERY frame of the current observation as ONE labelled
  contact-sheet IMAGE next to the board image, and never narrates it.  This module ports that
  algorithm (`codex_experiments/all_twenty_five_continuity_retry_candidate_20260916/
  chronological_sheet.py`, read, not imported) onto F13's existing capture.

WHAT
  When the last executed action produced >= 1 intermediate frame, ONE extra image is attached to
  the newest user message: a contact sheet of [intermediate frames ..., final board], in order,
  integer NEAREST upscale, row-major, one panel per frame, each panel labelled with its frame
  index, thin separator lines between tiles.  The stock current-grid image is NOT replaced and
  stays LAST in the message, so the final image the model sees is still the current board.
  Exactly one line of harness text rides with it -- coordinates and legend only, never narration:

      Frame sheet: 5 frames of the last action, panels left-to-right/top-to-bottom = frame 1..5
      (5 = final board); integer scale 4; act in ORIGINAL board coordinates, never sheet
      coordinates.

  (one physical line; when the sequence is longer than the sheet can carry it reads
  `Frame sheet: 7 of 227 frames shown of the last action, panels left-to-right/top-to-bottom =
  frames 1, 46, 91, 136, 181, 226, 227 (frame 227 = final board); integer scale 4; ...`).

  All-or-none: frames are never dropped, deduplicated, downsampled or reordered silently.  A
  sequence longer than AGENTFIX_SHEET_MAX_FRAMES is subsampled EXPLICITLY (first, the
  last-before-final, evenly spaced middles) and the line says `K of N frames shown` with the
  original frame indices as the panel labels.  If the sheet cannot be rendered at all (no PIL,
  no palette, does not fit the pixel bound at integer scale 1, anim not installed so no frames
  exist) there is NO sheet and NO line -- the turn is exactly the F13-only turn -- and the
  reason is recorded in the report as `sheet_unavailable` / on the AGENTFIX_EVENT line.

  Only the newest AGENTFIX_SHEET_KEEP TURNS (user messages) may carry a sheet: the OUTGOING copy
  of every request loses every stale sheet (image part AND line), including the case where the
  newest turn did not animate and the only sheet left describes an action that is no longer the
  last one.  The stored history is untouched, exactly like F13b's strip-old and F17's.

KV BUDGET (the binding constraint, measured on 64x64 boards)
  The stock current-grid image is 64*4 x 64*4 = 256x256 px ~ 120 estimated tokens
  (AGENTFIX_IMAGE_TOKENS, the F1 estimate).  A sheet is charged by pixel AREA:
      4 frames, scale 4 -> 2x2 panels, 528x560 px  ~ 541 est tokens (380 Qwen 28px units)
      5 frames, scale 4 -> 3x2 panels, 792x560 px  ~ 812 est tokens (580 Qwen 28px units)
      8 frames, scale 4 -> 3x3 panels, 792x840 px  ~ 1218 est tokens (870 Qwen 28px units)
  Prompts are already ~21.5k of a 32k window, so the default is scale 4 (NOT the sibling's 8)
  with ONE sheet kept.  AGENTFIX_SHEET_SCALE=2 quarters the cost if the budget bites.

Env switches
  AGENTFIX_SHEET=1              master switch.  0 -> nothing here is patched at all and the
                                model-visible stream is byte-identical to F13-only.
  AGENTFIX_SHEET_SCALE=4        preferred integer NEAREST upscale.  Reduced (4,3,2,1) only to
                                fit the pixel bound; never a non-integer or downward resample.
  AGENTFIX_SHEET_MAX_PIXELS=1.0 canvas bound.  <=100 means megapixels, >100 means raw pixels.
  AGENTFIX_SHEET_MAX_FRAMES=8   panels on one sheet (the final board is one of them).
  AGENTFIX_SHEET_KEEP=1         newest turns (user messages) of the OUTGOING copy allowed to
                                carry a sheet.  0 -> the sheet is stripped from every request.
  AGENTFIX_SHEET_GAP=4          padding around a panel inside its tile.
  AGENTFIX_SHEET_HEADER=16      label band above a panel.
  AGENTFIX_SHEET_EVENTS=1       the per-sheet AGENTFIX_EVENT line.
  AGENTFIX_IMAGE_TOKENS=120     the F1 per-image token estimate the cost model is scaled from.

Everything is fail-safe: any exception anywhere in here leaves the stock message untouched and
the stock stream is what goes out.  Composes with agentfix_anim / agentfix_win / agentfix_ewm in
any install order (it only wraps, never replaces, the methods those patch).  Idempotent.
"""
import base64, io, math, os, sys, time

_sheet_on = lambda k, d="1": os.environ.get(k, d).strip() not in ("0", "", "false", "False")


def _sheet_int_env(key, default):
    try:
        return int(str(os.environ.get(key, default)).strip() or default)
    except Exception:
        return int(default)


def _sheet_pixels_env(key, default):
    """`1.0` (and anything <= 100) is megapixels; a bigger number is raw pixels."""
    try:
        value = float(str(os.environ.get(key, default)).strip() or default)
    except Exception:
        value = float(default)
    if value <= 0:
        value = float(default)
    return int(value * 1_000_000) if value <= 100 else int(value)


SHEET_SCALE = max(1, _sheet_int_env("AGENTFIX_SHEET_SCALE", 4))
SHEET_MAX_PIXELS = _sheet_pixels_env("AGENTFIX_SHEET_MAX_PIXELS", 1.0)
SHEET_MAX_FRAMES = _sheet_int_env("AGENTFIX_SHEET_MAX_FRAMES", 8)
SHEET_KEEP = _sheet_int_env("AGENTFIX_SHEET_KEEP", 1)
SHEET_GAP = max(0, _sheet_int_env("AGENTFIX_SHEET_GAP", 4))
SHEET_HEADER = max(0, _sheet_int_env("AGENTFIX_SHEET_HEADER", 16))
SHEET_EVENTS = _sheet_on("AGENTFIX_SHEET_EVENTS")
SHEET_IMAGE_TOKENS = _sheet_int_env("AGENTFIX_IMAGE_TOKENS", 120)

SHEET_LINE_PREFIX = "Frame sheet:"
SHEET_BG = (224, 224, 224)            # padding, never a game colour (the palette has no such grey)
SHEET_SEP = (96, 96, 96)              # 1px separator between tiles, inside the padding
SHEET_LABEL = (0, 0, 0)
SHEET_PATCH = 28                      # Qwen vision patch-merge unit, for the cross-check estimate

_SHEET_STATS = {"turns": 0, "sheets": 0, "truncated": 0, "unavailable": 0, "errors": 0,
                "stripped": 0, "frames_max": 0, "panels_max": 0, "px_max": 0, "est_tokens_max": 0,
                "ms_max": 0.0, "ms_total": 0.0}
_SHEET_PALETTE = None
_SHEET_CHARS = None


# --------------------------------------------------------------------------- palette / decoding

def _sheet_palette():
    """The EXACT palette of the stock current-grid image (inference.agent.vision_context)."""
    global _SHEET_PALETTE
    if _SHEET_PALETTE is None:
        module = sys.modules.get("inference.agent.vision_context")
        if module is None:
            import inference.agent.vision_context as module      # noqa: F811
        _SHEET_PALETTE = dict(module.ARC_COLOR_MAP)
    return _SHEET_PALETTE


def _sheet_charmap():
    """char -> value, the inverse of inference.utils.grid_utils.format_grid_ascii."""
    global _SHEET_CHARS
    if _SHEET_CHARS is None:
        module = sys.modules.get("inference.utils.grid_utils")
        if module is None:
            import inference.utils.grid_utils as module          # noqa: F811
        _SHEET_CHARS = {ch: i for i, ch in enumerate(module.ARC_COLOR_CHARS)}
    return _SHEET_CHARS


def _sheet_rows_from_ascii(text):
    table = _sheet_charmap()
    rows = []
    for line in str(text).split("\n"):
        if not line:
            continue
        rows.append([table.get(ch, 0) for ch in line])
    return rows


def _sheet_rows(item):
    """A frame payload dict (F13 capture: `ascii`), a Frame, or raw rows -> list of list of int."""
    if isinstance(item, dict):
        grid = item.get("grid")
        if grid:
            return [[int(v) for v in row] for row in grid]
        return _sheet_rows_from_ascii(item.get("ascii") or "")
    grid = getattr(item, "grid", None)
    if grid is not None:
        return [[int(v) for v in row] for row in grid]
    if isinstance(item, (list, tuple)):
        return [[int(v) for v in row] for row in item]
    return []


def _sheet_keep_indices(total, cap):
    """Indices to keep: first, last-before-final, evenly spaced middles.  F13's rule, verbatim."""
    if total <= 0:
        return []
    if cap <= 0 or total <= cap:
        return list(range(total))
    if cap == 1:
        return [total - 1]
    return sorted({int(round(i * (total - 1) / float(cap - 1))) for i in range(cap)})


# --------------------------------------------------------------------------- the sheet itself

def _sheet_layout(count, cell_h, cell_w, scale, gap, header, max_pixels):
    """(width, height, columns, rows, integer scale) for `count` uniform tiles, or None."""
    for use in range(max(1, int(scale)), 0, -1):
        tile_w = cell_w * use + 2 * gap
        tile_h = cell_h * use + 2 * gap + header
        best = None
        for columns in range(1, count + 1):
            rows = int(math.ceil(count / float(columns)))
            width, height = columns * tile_w, rows * tile_h
            if width * height > max_pixels:
                continue
            key = (abs(width - height), width * height)
            if best is None or key < best[0]:
                best = (key, width, height, columns, rows)
        if best is not None:
            return best[1], best[2], best[3], best[4], use
    return None            # refuse explicitly rather than dropping or downsampling frames


_SHEET_GLYPHS = {   # 3x5 bitmap digits, rows top-to-bottom, 1 = ink
    "0": ("111", "101", "101", "101", "111"), "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"), "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"), "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"), "7": ("111", "001", "001", "001", "001"),
    "8": ("111", "101", "111", "101", "111"), "9": ("111", "101", "111", "001", "111"),
}


def _sheet_draw_label(canvas, text, left, top, header):
    """Label a panel with 3x5 bitmap digits scaled to the header band; pure Image ops, no ImageDraw."""
    from PIL import Image
    px = max(1, min((header - 2) // 5, 3))            # glyph pixel size: header 16 -> 2 px per cell
    y0 = top + max(0, (header - 5 * px) // 2)
    x = left
    ink = Image.new("RGB", (px, px), SHEET_LABEL)
    for ch in text:
        glyph = _SHEET_GLYPHS.get(ch)
        if glyph is None:
            x += 4 * px
            continue
        for r, row in enumerate(glyph):
            for c, bit in enumerate(row):
                if bit == "1":
                    canvas.paste(ink, (x + c * px, y0 + r * px))
        x += 4 * px                                   # 3 cells + 1 cell gap


def _sheet_render(frames_rows, labels, *, scale=None, max_pixels=None, gap=None, header=None):
    """One contact sheet.  Returns the data url + the panel geometry, or None if it cannot fit."""
    from PIL import Image        # ImageDraw is NOT used: the Kaggle base image ships a pillow mix whose
                                 # ImageDraw import fails (_Ink); separators and labels are plain pastes

    scale = SHEET_SCALE if scale is None else int(scale)
    max_pixels = SHEET_MAX_PIXELS if max_pixels is None else int(max_pixels)
    gap = SHEET_GAP if gap is None else int(gap)
    header = SHEET_HEADER if header is None else int(header)
    palette = _sheet_palette()
    background = palette[0]

    shapes = []
    for rows in frames_rows:
        height = len(rows)
        width = max((len(row) for row in rows), default=0)
        if height <= 0 or width <= 0:
            return None
        shapes.append((height, width))
    cell_h = max(h for h, _ in shapes)
    cell_w = max(w for _, w in shapes)

    layout = _sheet_layout(len(frames_rows), cell_h, cell_w, scale, gap, header, max_pixels)
    if layout is None:
        return None
    width, height, columns, row_count, used = layout
    tile_w = cell_w * used + 2 * gap
    tile_h = cell_h * used + 2 * gap + header

    canvas = Image.new("RGB", (width, height), SHEET_BG)
    for column in range(1, columns):                       # thin separators, inside the padding
        canvas.paste(Image.new("RGB", (1, height), SHEET_SEP), (column * tile_w, 0))
    for row in range(1, row_count):
        canvas.paste(Image.new("RGB", (width, 1), SHEET_SEP), (0, row * tile_h))

    panels = []
    for index, rows in enumerate(frames_rows):
        frame_h, frame_w = shapes[index]
        column, row = index % columns, index // columns
        left = column * tile_w + gap
        top = row * tile_h + gap + header
        original = Image.new("RGB", (frame_w, frame_h), background)
        pixels = []
        for line in rows:
            for col in range(frame_w):
                value = line[col] if col < len(line) else 0
                pixels.append(palette.get(int(value), background))
        original.putdata(pixels)
        enlarged = original if used == 1 else original.resize(
            (frame_w * used, frame_h * used), Image.Resampling.NEAREST)
        canvas.paste(enlarged, (left, top))
        if header:
            _sheet_draw_label(canvas, str(labels[index]), left, top - header, header)
        panels.append({"label": labels[index], "frame_shape": [frame_h, frame_w],
                       "box_xyxy": [left, top, left + enlarged.width, top + enlarged.height]})

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return {"url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
            "panels": panels, "scale": used, "columns": columns, "rows": row_count,
            "tile": [tile_w, tile_h], "canvas": [width, height], "pixels": width * height,
            "bytes": buffer.tell()}


def _sheet_line(shown, total, labels, scale):
    """The ONE line of harness text.  Coordinates and legend only -- never a narration."""
    if shown >= total:
        return (f"{SHEET_LINE_PREFIX} {total} frames of the last action, panels "
                f"left-to-right/top-to-bottom = frame 1..{total} ({total} = final board); "
                f"integer scale {scale}; act in ORIGINAL board coordinates, never sheet coordinates.")
    return (f"{SHEET_LINE_PREFIX} {shown} of {total} frames shown of the last action, panels "
            f"left-to-right/top-to-bottom = frames {', '.join(str(x) for x in labels)} "
            f"(frame {total} = final board); integer scale {scale}; act in ORIGINAL board "
            f"coordinates, never sheet coordinates.")


def _sheet_tokens(pixels, reference_pixels):
    """The F1 IMAGE_TOKENS estimate, scaled by pixel area (that is how a Qwen-style vision
    tokenizer charges).  `reference_pixels` is the stock current-grid image."""
    if reference_pixels <= 0:
        return 0
    return max(1, int(round(SHEET_IMAGE_TOKENS * (float(pixels) / float(reference_pixels)))))


def _sheet_patch_tokens(width, height):
    """Cross-check: Qwen counts one token per 28x28 patch-merge unit."""
    return int(math.ceil(width / float(SHEET_PATCH)) * math.ceil(height / float(SHEET_PATCH)))


# --------------------------------------------------------------------------- the turn's sequence

def _sheet_sequence(entry, current_frame):
    """[rows ...], [label ...], shown, total  for the newest history entry, or None."""
    if entry is None:
        return None
    if not str(getattr(entry, "action", "") or "").strip():
        return None                                   # the seeded initial frame is not a transition
    captured = [item for item in (getattr(entry, "frames", ()) or ()) if item is not None]
    if not captured:
        return None                                   # F13 not installed, or the action did not animate
    total_inter = int(getattr(entry, "frame_count", 0) or len(captured))
    if total_inter < len(captured):
        total_inter = len(captured)
    total = total_inter + 1                           # + the final board

    keep = _sheet_keep_indices(len(captured), max(1, SHEET_MAX_FRAMES - 1))
    rows, labels = [], []
    for position in keep:
        item = captured[position]
        grid = _sheet_rows(item)
        if not grid:
            return None        # all-or-none: a frame that will not decode kills the sheet, it is
                               # never quietly dropped from a sequence the line claims is complete
        index = item.get("index", position) if isinstance(item, dict) else position
        rows.append(grid)
        labels.append(int(index) + 1)
    if not rows:
        return None

    final = getattr(entry, "frame", None)
    final_rows = _sheet_rows(final) if final is not None else []
    if not final_rows and current_frame is not None:
        final_rows = _sheet_rows(current_frame)
    if not final_rows:
        return None
    rows.append(final_rows)
    labels.append(total)
    return rows, labels, len(rows), total


def _sheet_reference_pixels(current_frame):
    """Pixels of the stock current-grid image, the thing the 120-token estimate is calibrated on."""
    try:
        module = sys.modules.get("inference.agent.vision_context")
        if module is None:
            import inference.agent.vision_context as module      # noqa: F811
        scale = module.current_grid_image_upscale()
        grid = getattr(current_frame, "grid", None) or ()
        height = len(grid)
        width = max((len(row) for row in grid), default=0)
        return max(1, height * width * scale * scale)
    except Exception:
        return 64 * 64 * 4 * 4


def _sheet_build(agent, current_frame):
    """(line, image part, meta) for this turn, or None for 'no sheet, no line'."""
    entries = getattr(agent, "_sheet_entries", None) or []
    entry = entries[-1] if entries else None
    sequence = _sheet_sequence(entry, current_frame)
    if sequence is None:
        return None
    rows, labels, shown, total = sequence
    started = time.perf_counter()
    rendered = _sheet_render(rows, labels)
    elapsed = (time.perf_counter() - started) * 1000.0
    if rendered is None:
        _SHEET_STATS["unavailable"] += 1
        return None
    line = _sheet_line(shown, total, labels, rendered["scale"])
    reference = _sheet_reference_pixels(current_frame)
    meta = {"shown": shown, "total": total, "labels": list(labels), "line": line,
            "scale": rendered["scale"], "columns": rendered["columns"], "rows": rendered["rows"],
            "tile": rendered["tile"], "canvas": rendered["canvas"], "pixels": rendered["pixels"],
            "png_bytes": rendered["bytes"], "panels": rendered["panels"],
            "est_tokens": _sheet_tokens(rendered["pixels"], reference),
            "patch_tokens": _sheet_patch_tokens(*rendered["canvas"]),
            "reference_pixels": reference, "ms": round(elapsed, 3),
            "truncated": bool(shown < total)}
    part = {"type": "image_url", "image_url": {"url": rendered["url"]}}

    _SHEET_STATS["sheets"] += 1
    _SHEET_STATS["ms_total"] = round(_SHEET_STATS["ms_total"] + elapsed, 3)
    if elapsed > _SHEET_STATS["ms_max"]:
        _SHEET_STATS["ms_max"] = round(elapsed, 3)
    if meta["truncated"]:
        _SHEET_STATS["truncated"] += 1
    for key, value in (("frames_max", total), ("panels_max", shown), ("px_max", meta["pixels"]),
                       ("est_tokens_max", meta["est_tokens"])):
        if value > _SHEET_STATS[key]:
            _SHEET_STATS[key] = value
    if SHEET_EVENTS:
        try:
            print("AGENTFIX_EVENT sheet level=%s step=%s frames=%d/%d scale=%d panels=%dx%d "
                  "canvas=%dx%d px=%d png=%d est_tokens=%d patch_tokens=%d ms=%.1f"
                  % (getattr(current_frame, "level", "?"), getattr(current_frame, "step", "?"),
                     shown, total, meta["scale"], meta["columns"], meta["rows"],
                     meta["canvas"][0], meta["canvas"][1], meta["pixels"], meta["png_bytes"],
                     meta["est_tokens"], meta["patch_tokens"], meta["ms"]), flush=True)
        except Exception:
            pass
    return line, part, meta


# --------------------------------------------------------------------------- strip-old

def _sheet_is_line(text):
    return str(text).strip().startswith(SHEET_LINE_PREFIX)


def _sheet_message_has(message):
    content = message.get("content")
    if isinstance(content, str):
        return any(_sheet_is_line(item) for item in content.split("\n"))
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                if any(_sheet_is_line(item) for item in part["text"].split("\n")):
                    return True
    return False


def _sheet_strip_message(message):
    """Drop this message's sheet line AND the image part that follows it.  Returns (message, n)."""
    content = message.get("content")
    if isinstance(content, str):
        kept = [item for item in content.split("\n") if not _sheet_is_line(item)]
        if len(kept) == len(content.split("\n")):
            return message, 0
        return {**message, "content": "\n".join(kept).strip("\n")}, 1
    if not isinstance(content, list):
        return message, 0
    parts, removed, drop_next = [], 0, False
    for part in content:
        is_text = isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
        if drop_next and isinstance(part, dict) and not is_text:
            drop_next = False
            removed += 1
            continue
        drop_next = False
        if is_text:
            lines = part["text"].split("\n")
            kept = [item for item in lines if not _sheet_is_line(item)]
            if len(kept) != len(lines):
                removed += 1
                drop_next = True
                text = "\n".join(kept).strip("\n")
                if not text:
                    continue
                parts.append({**part, "text": text})
                continue
        parts.append(part)
    if not removed:
        return message, 0
    if len(parts) == 1 and isinstance(parts[0], dict) and parts[0].get("type") == "text":
        return {**message, "content": parts[0]["text"]}, removed
    return {**message, "content": parts}, removed


def _sheet_strip_old(messages):
    """Outgoing copy only: only the newest SHEET_KEEP TURNS (user messages) may carry a sheet.

    Turn-based, not sheet-based, on purpose: the line says `of the last action`, so a sheet that
    is no longer on the newest user message is a false statement about the current board and has
    to go even when it is the only sheet left in the request."""
    users = [index for index, message in enumerate(messages)
             if isinstance(message, dict) and message.get("role") == "user"]
    keep = set(users[len(users) - SHEET_KEEP:]) if SHEET_KEEP > 0 else set()
    out, removed = [], 0
    for index, message in enumerate(messages):
        if index in users and index not in keep and _sheet_message_has(message):
            message, count = _sheet_strip_message(message)
            removed += count
        out.append(message)
    if removed:
        _SHEET_STATS["stripped"] += removed
    return out


# --------------------------------------------------------------------------- install

def install_sheet(ta_mod, an_mod, solv_mod, report):
    report = report if isinstance(report, dict) else {}
    report.setdefault("F19_sheet", False)
    report["F19_sheet_enabled"] = _sheet_on("AGENTFIX_SHEET")
    if not _sheet_on("AGENTFIX_SHEET"):
        return report
    if getattr(ta_mod, "_agentfix_sheet_installed", False):
        report["F19_sheet"] = True
        report["note_sheet"] = "already installed"
        return report

    try:                                  # no PIL / no palette -> stock stream, explicitly marked
        from PIL import Image              # noqa: F401  (ImageDraw deliberately not required)
        _sheet_palette()
        _sheet_charmap()
    except Exception as exc:
        report["sheet_unavailable"] = f"{type(exc).__name__}: {exc}"
        return report

    TA = ta_mod.ToolAgent

    # -- 1. the history entries of THIS turn (they carry F13's frames; `_build_user_message` is
    #       not given them).  The stock loop calls `_build_user_prompt` immediately before
    #       `_build_user_message` with the entries it just loaded.
    _orig_prompt = TA._build_user_prompt

    def _build_user_prompt(self, action_num, *args, **kwargs):
        try:
            self._sheet_entries = list(kwargs.get("history_entries") or [])
        except Exception:
            pass
        return _orig_prompt(self, action_num, *args, **kwargs)

    TA._build_user_prompt = _build_user_prompt

    # -- 2. the sheet: ONE extra image on the newest user message, before the stock board image
    #       (so the LAST image the model sees is still the current board).
    _orig_message = TA._build_user_message

    def _build_user_message(self, user_prompt, current_frame):
        stock = _orig_message(self, user_prompt, current_frame)
        _SHEET_STATS["turns"] += 1
        try:
            content = stock.get("content") if isinstance(stock, dict) else None
            if not isinstance(content, list) or not content:
                return stock                      # images off -> nothing to sit next to
            built = _sheet_build(self, current_frame)
            if built is None:
                return stock
            line, part, meta = built
            self._sheet_last = meta
            return {**stock, "content": [{"type": "text", "text": line}, part, *content]}
        except Exception as exc:
            _SHEET_STATS["errors"] += 1
            if SHEET_EVENTS:
                try:
                    print("AGENTFIX_EVENT sheet error=%s: %s" % (type(exc).__name__, exc), flush=True)
                except Exception:
                    pass
            return stock
        finally:
            try:
                self._sheet_entries = None
            except Exception:
                pass

    TA._build_user_message = _build_user_message

    # -- 3. exactly one sheet per request: the older user messages of the OUTGOING copy lose
    #       theirs.  The stored history is untouched.
    _orig_chat = TA._chat_completion

    def _chat_completion(self, messages, *args, **kwargs):
        try:
            messages = _sheet_strip_old(messages)
        except Exception:
            _SHEET_STATS["errors"] += 1
        return _orig_chat(self, messages, *args, **kwargs)

    TA._chat_completion = _chat_completion

    for fn in (TA._build_user_prompt, TA._build_user_message, TA._chat_completion):
        try:
            fn._agentfix = True
        except Exception:
            pass
    ta_mod._agentfix_sheet_installed = True
    report["F19_sheet"] = True
    report["sheet_scale"] = SHEET_SCALE
    report["sheet_max_pixels"] = SHEET_MAX_PIXELS
    report["sheet_max_frames"] = SHEET_MAX_FRAMES
    report["sheet_keep"] = SHEET_KEEP
    report["sheet_stats"] = _SHEET_STATS
    try:
        ta_mod._agentfix_report = dict(report)
    except Exception:
        pass
    return report

# ===== end agentfix_sheet.py =====

import inference.agent.tool_agent as _ta, inference.agent.action_names as _an, inference.framework.solver as _solv
import os as _os   # defaults were set above, before the module sources
# v8: c16 + no-op awareness only (F2 result always returned + F5 HUD-aware board_changed / repeat tag); everything else OFF; timing lines ON
_rep = install(_ta, _an, _solv)
_rep.update(install_anim(_ta, _an, _solv, _rep) or {}) if callable(globals().get('install_anim')) else None
_rep.update(install_sheet(_ta, _an, _solv, _rep) or {}) if callable(globals().get('install_sheet')) else None
for _fn in (_ta.ToolAgent._run_python_tool, _ta.ToolAgent.analyze, _ta.ToolAgent._persistent_history_messages, _ta.ToolAgent._build_user_prompt,
            _ta.ToolAgent._update_summarized_knowledge_from_step_summary, _solv._HarnessGameSession.step_env, _ta.ToolAgent._compact_action_result, _ta.ToolAgent._chat_completion):
    if getattr(_fn, '__module__', '') not in ('inference.agent.tool_agent', 'inference.framework.solver'):
        try: _fn._agentfix = True
        except Exception: pass
assert _rep['installed'] is True, _rep
for _k in ('F1_images','F2_result','F3_memory','F4_loop','F5_noimpact','F6_action7','F9_timing'):
    assert bool(_rep.get(_k)) == bool(_rep.get(_k + '_enabled')), (_k, _rep)   # installed iff enabled (env kill switch)
assert bool(_rep.get('F7_ledger')) == (bool(_rep.get('F7_ledger_enabled')) and bool(_rep.get('F2_result')) and bool(_rep.get('F5_noimpact'))), _rep   # ledger rides on F2+F5
if _rep.get('F7_ledger'): assert 'MEASURES the effect' in _ta.STRUCTURED_RUNTIME_STATE_ADDENDUM and '_ledger_lines' in globals()
# each fix must be LIVE on the classes the solver will instantiate
_p = lambda f: getattr(f, '_agentfix', False) is True   # patched-by-agentfix marker (exec'd cells have no source for inspect)
_fresh = _rep.get('note') != 'already installed'         # re-executing this cell must not fail on identity vs the earlier namespace
if _rep['F1_images']: assert (not _fresh or _ta._estimate_tokens is _estimate_tokens_flat) and _p(_ta.ToolAgent._persistent_history_messages)
else: assert _ta.ToolAgent._persistent_history_messages.__module__ == 'inference.agent.tool_agent'
if _rep['F2_result']: assert _p(_ta.ToolAgent._build_user_prompt)
if _rep['F3_memory']: assert (not _fresh or _ta._extract_labeled_blocks is _extract_labeled_blocks_tolerant) and _p(_ta.ToolAgent._update_summarized_knowledge_from_step_summary)
assert _p(_ta.ToolAgent._run_python_tool) and _p(_ta.ToolAgent.analyze)   # F2/F4 shared wrapper is always installed; F4 gated inside by env
if _rep['F5_noimpact']: assert _p(_solv._HarnessGameSession.step_env) and _p(_ta.ToolAgent._compact_action_result)
else: assert _solv._HarnessGameSession.step_env.__module__ == 'inference.framework.solver'
if _rep['F6_action7']: assert _an.to_engine_action('ACTION7') == 'ACTION7' and 'ACTION7' in _ta.STRUCTURED_RUNTIME_STATE_ADDENDUM
if _rep['F3_memory']: assert _ta._extract_labeled_blocks('World model (revised): x', ['World model']) == {'World model': 'x'}
assert _rep.get('F10_stall') == (int(_os.environ.get('AGENTFIX_STALL_STEPS', '40')) > 0 and _os.environ.get('AGENTFIX_STALL', '1').strip() not in ('0', '', 'false', 'False')), _rep
print('AGENTFIX LIVE:', _rep, flush=True)

# ===== ARM P (upscale) block, verbatim from taaf-flashnext-armp-0908 cell 10 =====
import os, sys as _sys
os.environ.setdefault('ARMP_UPSCALE', '12')
_ARMP = os.environ.get('ARMP_UPSCALE', '').strip().lower()

if _ARMP in ('', 'off', '0', 'none'):
    print('ARM P: DISABLED (ARMP_UPSCALE off) -- stock c16 behaviour, bundle upscale 4', flush=True)
else:
    _ARMP_N = int(_ARMP)
    assert 1 <= _ARMP_N <= 16, f'unsupported upscale {_ARMP_N}'   # nextfork 27.09: 16 разрешено (препроцессор модели: патч 16, склейка 2, предел 16.7 Мп — 1024² проходит без сжатия)

    _prev = os.environ.get('MULTIMODAL_UPSCALE')
    _ctx = os.environ.get('MULTIMODAL_CONTEXT')
    assert _prev == '4', f'expected the bundle-persisted MULTIMODAL_UPSCALE=4 (the c16 control), got {_prev!r}'
    assert _ctx == 'current_grid', f'expected MULTIMODAL_CONTEXT=current_grid, got {_ctx!r}'
    os.environ['MULTIMODAL_UPSCALE'] = str(_ARMP_N)

    import base64 as _b64, io as _io, types as _ty
    from PIL import Image as _Im
    import inference.agent.vision_context as _vc

    # env is read lazily per frame, so the env line above is already sufficient; pin the accessor too
    # so nothing downstream can restore the setup env underneath this arm.
    _vc.current_grid_image_upscale = (lambda: _ARMP_N)

    # End-to-end proof through the exact call path the agent uses (tool_agent.py:1148).
    def _armp_render(grid):
        part = _vc.current_grid_image_part(_ty.SimpleNamespace(grid=grid))
        assert part is not None and part['type'] == 'image_url', part
        raw = _b64.b64decode(part['image_url']['url'].split(',', 1)[1])
        return _Im.open(_io.BytesIO(raw)).size, len(raw)

    _size2, _bytes2 = _armp_render([[1, 2], [3, 4]])
    assert _size2 == (2 * _ARMP_N, 2 * _ARMP_N), f'2x2 grid rendered {_size2}, expected {(2*_ARMP_N,)*2}'
    _size64, _bytes64 = _armp_render([[(r * 64 + c) % 16 for c in range(64)] for r in range(64)])
    assert _size64 == (64 * _ARMP_N, 64 * _ARMP_N), _size64

    import inference.agent.tool_agent as _ta
    assert _ta.current_grid_image_part is _vc.current_grid_image_part, 'tool_agent bound a different renderer'
    assert _ta.current_grid_image_enabled(), 'the current-grid image is not enabled'

    print(f'ARM P installed: MULTIMODAL_UPSCALE 4 -> {_ARMP_N} '
          f'(env={os.environ["MULTIMODAL_UPSCALE"]}, accessor={_vc.current_grid_image_upscale()})', flush=True)
    print(f'ARM P VERIFIED via tool_agent.current_grid_image_part: 2x2 grid -> {_size2} px; '
          f'64x64 board -> {_size64} px, {_bytes64} PNG bytes '
          f'(c16 control renders the same board at {(64*4, 64*4)})', flush=True)

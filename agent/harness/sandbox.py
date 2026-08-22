"""Sandbox for LLM-written Python + the runtime state it sees.

The model gets one tool: a Python snippet executed here. Inside the snippet
these names exist (Duck-style contract):

  current_frame   FrameView of the latest board (.ascii/.grid/.step/.level/
                  .shape/.segmentation)
  previous_frame  FrameView before the last real action (or None)
  history         list of (action_name, FrameView) after each real action
  valid_actions   list of currently legal action names
  action(acts)    execute real environment actions; acts is a list of names
                  ("UP") or dicts ({"action": "CLICK", "x": 3, "y": 7});
                  returns a result dict and refreshes all variables above
  print(...)      captured and returned to the model (size-capped)
  result          optional: assign a final value to report

Model-facing action names are remapped to be semantic-free-but-mnemonic:
UP/DOWN/LEFT/RIGHT/SPACE/CLICK/UNDO (the docs say ACTION1..4 are directional).
"""
from __future__ import annotations

import builtins as _builtins
import contextlib
import io
import os
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .perception import Segmentation, grid_diff, latest_grid, segment
from .toolbox import bfs_path, objects, reachable

# ── model-facing action vocabulary ────────────────────────────────────────
TO_ENGINE = {
    "UP": "ACTION1",
    "DOWN": "ACTION2",
    "LEFT": "ACTION3",
    "RIGHT": "ACTION4",
    "SPACE": "ACTION5",
    "CLICK": "ACTION6",
    "UNDO": "ACTION7",
    "RESET": "RESET",
}
FROM_ENGINE = {v: k for k, v in TO_ENGINE.items()}

ARC_CHARS = "0123456789ABCDEF"  # color index -> ascii symbol


class FrameView:
    """Read-only, model-friendly view of one frame."""

    def __init__(self, grid: np.ndarray, step: int, level: int):
        self.grid = grid
        self.step = step
        self.level = level
        self._seg: Segmentation | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.grid.shape)  # type: ignore[return-value]

    @property
    def ascii(self) -> str:
        return "\n".join("".join(ARC_CHARS[c % 16] for c in row) for row in self.grid.tolist())

    @property
    def segmentation(self) -> dict:
        if self._seg is None:
            self._seg = segment(self.grid)
        nodes = [
            {
                "id": o.id,
                "color": ARC_CHARS[o.color % 16],
                "pixels": o.cells,
                "bbox": o.bbox,
                "centroid": (round(o.centroid[0], 1), round(o.centroid[1], 1)),
                "hash": o.shape_hash,
                "touches_border": o.touches_border,
                # Chollet-style relational priors (20.08): which OTHER objects
                # this one borders, and which objects nest directly inside it
                # — spatial relationships the model would otherwise have to
                # recompute in sandbox code every turn.
                "adjacent_to": self._seg.adjacency.get(o.id, []),
                "children": self._seg.children.get(o.id, []),
            }
            for o in self._seg.objects
        ]
        return {"background": ARC_CHARS[self._seg.background % 16], "nodes": nodes}


@dataclass
class SandboxResult:
    output: str
    error: str | None = None
    actions_executed: int = 0
    interrupted: str | None = None  # "WIN" | "BUDGET" | None


class GameOverInterrupt(BaseException):
    """Raised inside sandbox code to unwind when the run must stop.

    BaseException on purpose: model code wrapping action() in a bare
    `except Exception:` (a solver loop with error handling) must NOT be able
    to swallow a WIN/GAME_OVER/budget/turn-cap unwind and keep stepping."""

    def __init__(self, reason: str):
        self.reason = reason


# Whitelisted imports for model code.
ALLOWED_MODULES = {
    "math", "collections", "itertools", "functools", "heapq", "bisect",
    "json", "re", "random", "statistics", "string", "copy", "operator",
    "numpy",
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(f"import of '{name}' is not allowed; allowed: {sorted(ALLOWED_MODULES)}")
    return _builtins.__import__(name, globals, locals, fromlist, level)


@dataclass
class Sandbox:
    """Executes model code against a live environment adapter.

    `env_step(engine_action_name, payload|None) -> FrameData` performs a real
    action. `budget_left()` returns how many actions may still be spent.
    """

    env_step: Callable[[str, dict | None], Any]
    budget_left: Callable[[], int]
    max_output_chars: int = 4000
    timeout_s: float = 30.0

    history: list[tuple[str, FrameView]] = field(default_factory=list)
    current: FrameView | None = None
    previous: FrameView | None = None
    valid_actions: list[str] = field(default_factory=list)
    last_result: dict = field(default_factory=dict)
    step_counter: int = 0
    # Survives across run_code calls: the model's own scratch memory.
    memo: dict = field(default_factory=dict)
    # Full transition log for the world-model verifier:
    # (grid_before, action_name, data|None, grid_after)
    transition_log: list = field(default_factory=list)
    # Result of the most recent verify_theory call (None until first use).
    last_verify: dict | None = None
    # Total verify_theory calls this episode (for the action gate).
    verify_attempts: int = 0
    # Set when a run_code times out: the abandoned thread must stop acting.
    _cancelled: bool = False
    # Max real actions ONE code block may execute (21.08: a teacher wrote a
    # `while True: solve()` loop of short action() calls -- the verify gate
    # only limits a single call >3 actions -- and burned all 800 actions of
    # the episode on turn 5). Stops the block (not the episode) with a clear
    # error the model sees next turn. 0 = off.
    turn_action_cap: int = field(
        default_factory=lambda: int(os.getenv("MY_AGENT_TURN_ACTION_CAP", "120")))
    _turn_start_step: int = 0
    # Engine calls made by the CURRENT code block -- counted independently of
    # step_counter, which only advances for frames that carry a grid. Frames
    # without a grid (some games after GAME_OVER) let a model loop spend the
    # whole episode budget while step_counter stood still (tr87, 21.08 r2+r4:
    # 1470 hidden engine calls in one turn). The cap must see every call.
    _turn_env_calls: int = 0
    # True while the env sits in GAME_OVER: any further non-RESET action is a
    # wasted budget unit, so action() refuses it with a clear error instead.
    game_over: bool = False
    # Verify-gate escape hatch: long batches also unlock after this many
    # verify_theory attempts regardless of accuracy (so a model that can't
    # write a good predict() isn't stuck forever in competition). Teacher
    # rounds set it very high: expert data should come from VERIFIED theories
    # (round 2, 21.08: once the hatch opened, Flash flooded 800 actions).
    gate_escape_attempts: int = field(
        default_factory=lambda: int(os.getenv("MY_AGENT_VERIFY_GATE_ATTEMPTS", "3")))

    # ── state fed from the agent ──────────────────────────────────────────
    def update_frame(self, frame_data: Any, action_name: str | None = None,
                     data: dict | None = None) -> None:
        self.game_over = str(getattr(frame_data, "state", "")).split(".")[-1] == "GAME_OVER"
        grid = latest_grid(frame_data)
        if grid is None:
            return
        self.previous = self.current
        self.step_counter += 1 if action_name else 0
        view = FrameView(grid, self.step_counter, int(frame_data.levels_completed or 0))
        self.current = view
        if action_name:
            name = FROM_ENGINE.get(action_name, action_name)
            self.history.append((name, view))
            if self.previous is not None and name != "RESET":
                self.transition_log.append((self.previous.grid, name, data, grid))
        raw = frame_data.available_actions or []
        names = []
        for a in raw:
            n = getattr(a, "name", None) or f"ACTION{int(a)}" if not isinstance(a, str) else a
            if n == "ACTION0":
                n = "RESET"
            names.append(FROM_ENGINE.get(n, n))
        self.valid_actions = names

    # ── world-model verifier ──────────────────────────────────────────────
    def _plan_with_theory(self, predict, goal, actions: Any = None,
                          max_depth: int = 6, max_nodes: int = 1500,
                          min_accuracy: float = 0.6,
                          time_budget: float = 8.0) -> dict:
        """Search for an action sequence using the model's OWN verified theory.

        Until now a theory was only ever a key to the action-batch gate: the
        model tuned predict() until verify_theory hit 0.6, the gate opened,
        and the verified world model was thrown away. This turns it into a
        planner. Search is pure simulation -- it spends ZERO engine actions,
        which is exactly the trade we want, since actions are scored
        quadratically and thinking time is our scarcest resource.

            predict(grid, action, data) -> next grid   (same as verify_theory)
            goal(grid) -> bool                         (what you are aiming at)

        Returns {'plan': [...] | None, ...}. The plan is a list of specs that
        action() accepts verbatim, so the usual next line is action(res['plan']).

        The theory is re-verified here first and planning is REFUSED below
        min_accuracy: planning over an unverified theory is just fantasy with
        extra steps, and the refusal keeps the intended order theorise ->
        verify -> plan.
        """
        import time as _t

        # The internal check must NOT count as a gate attempt. verify_gate_open
        # also unlocks on `verify_attempts >= gate_escape_attempts`, so without
        # this, three plan_with_theory calls carrying a throwaway predict would
        # open the action gate on effort alone -- a second door to exactly the
        # dodge we scrubbed out of the training data. A genuinely accurate
        # theory still opens the gate, because last_verify is left to update.
        _attempts_before = self.verify_attempts
        check = self._verify_theory(predict, actions)
        self.verify_attempts = _attempts_before
        acc = check.get("accuracy")
        if acc is None or acc < min_accuracy:
            return {"plan": None, "verified_accuracy": acc,
                    "reason": (f"theory not good enough to plan with: accuracy {acc} "
                               f"< {min_accuracy} on {check.get('transitions_tested')} "
                               "transitions. Refine predict() against the "
                               "counterexamples from verify_theory first."),
                    "counterexamples": check.get("counterexamples")}

        if self.current is None:
            return {"plan": None, "reason": "no current frame"}

        # Default branching set: the directional/interact actions. CLICK is
        # excluded unless the caller names targets explicitly -- 64x64 click
        # positions would blow up the branching factor to 4096.
        if actions:
            specs = list(actions)
        else:
            specs = [a for a in self.valid_actions if a.upper() not in ("CLICK", "RESET")]
        if not specs:
            return {"plan": None, "reason": "no candidate actions to search over"}

        def _split(spec):
            if isinstance(spec, str):
                return spec.upper(), None
            if isinstance(spec, dict):
                name = str(spec.get("action", "")).upper()
                data = {k: int(v) for k, v in spec.items() if k in ("x", "y")} or None
                return name, data
            raise ValueError(f"bad action spec for planning: {spec!r}")

        start = self.current.grid
        try:
            if goal(start.copy()):
                return {"plan": [], "verified_accuracy": acc, "nodes_expanded": 0,
                        "reason": "already at the goal"}
        except Exception as exc:
            return {"plan": None, "reason": f"goal() raised on the current grid: {exc!r}"}

        deadline = _t.monotonic() + time_budget
        seen = {start.tobytes()}
        frontier = [(start, [])]
        nodes = pred_errors = 0
        while frontier:
            nxt = []
            for grid, path in frontier:
                if len(path) >= max_depth:
                    continue
                for spec in specs:
                    if nodes >= max_nodes or _t.monotonic() > deadline:
                        return {"plan": None, "verified_accuracy": acc,
                                "nodes_expanded": nodes, "predict_errors": pred_errors,
                                "reason": (f"budget spent ({nodes} states, depth "
                                           f"<= {len(path) + 1}) without reaching the goal. "
                                           "Try a looser goal(), a larger max_depth, or a "
                                           "smaller action set.")}
                    name, data = _split(spec)
                    nodes += 1
                    try:
                        pred = np.asarray(predict(grid.copy(), name,
                                                  dict(data) if data else None),
                                          dtype=np.int8)
                    except Exception:
                        pred_errors += 1
                        continue
                    if pred.shape != start.shape:
                        pred_errors += 1
                        continue
                    key = pred.tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    plan = path + [spec]
                    try:
                        if goal(pred.copy()):
                            return {"plan": plan, "depth": len(plan),
                                    "verified_accuracy": acc, "nodes_expanded": nodes,
                                    "predict_errors": pred_errors, "reason": None}
                    except Exception as exc:
                        return {"plan": None, "nodes_expanded": nodes,
                                "reason": f"goal() raised: {exc!r}"}
                    nxt.append((pred, plan))
            frontier = nxt
        return {"plan": None, "verified_accuracy": acc, "nodes_expanded": nodes,
                "predict_errors": pred_errors,
                "reason": (f"exhausted {nodes} reachable states within depth {max_depth} "
                           "without reaching the goal -- the goal may be unreachable "
                           "with these actions, or the theory may miss the mechanic "
                           "that matters.")}

    def _verify_theory(self, predict, actions: Any = None) -> dict:
        """Test a transition theory against every recorded real transition.

        predict(grid, action, data) -> predicted next grid (numpy array).
        Optional `actions`: restrict testing to these action names.
        Returns accuracy + up to 3 counterexamples. Costs zero env actions.
        """
        wanted = {a.upper() for a in actions} if actions else None
        tested = matched = errors = 0
        mismatches: list[dict] = []
        for before, name, data, after in self.transition_log:
            if wanted and name not in wanted:
                continue
            tested += 1
            try:
                pred = predict(before.copy(), name, dict(data) if data else None)
                pred = np.asarray(pred, dtype=np.int8)
                if pred.shape == after.shape and bool((pred == after).all()):
                    matched += 1
                elif len(mismatches) < 3:
                    if pred.shape != after.shape:
                        mismatches.append({"action": name, "error": f"shape {pred.shape} != {after.shape}"})
                    else:
                        wrong = np.argwhere(pred != after)
                        sample = [(int(r), int(c), int(pred[r, c]), int(after[r, c]))
                                  for r, c in wrong[:4]]
                        mismatches.append({
                            "action": name, "data": data,
                            "wrong_cells": int(len(wrong)),
                            "sample (row,col,predicted,actual)": sample,
                        })
            except Exception as exc:
                errors += 1
                if len(mismatches) < 3:
                    mismatches.append({"action": name, "error": repr(exc)[:200]})
        out = {
            "transitions_tested": tested,
            "exact_matches": matched,
            "accuracy": round(matched / tested, 3) if tested else None,
            "predict_errors": errors,
            "counterexamples": mismatches,
        }
        self.verify_attempts += 1
        # Keep the BEST attempt for the gate: a later worse theory must not
        # revoke an already-earned unlock.
        if (self.last_verify is None
                or (out["accuracy"] or 0) >= (self.last_verify.get("accuracy") or 0)):
            self.last_verify = {"accuracy": out["accuracy"], "tested": tested}
        return out

    def verify_gate_open(self) -> bool:
        """Long action() batches unlock at decent accuracy or honest effort."""
        if self.last_verify is None:
            return False
        return ((self.last_verify.get("accuracy") or 0) >= 0.6
                or self.verify_attempts >= self.gate_escape_attempts)

    def actions_on_current_level(self) -> int:
        """How many real actions were spent since the current level began."""
        if self.current is None:
            return 0
        lvl = self.current.level
        n = 0
        for _name, fv in reversed(self.history):
            if fv.level != lvl:
                break
            n += 1
        return n

    # ── the `action()` the model calls ────────────────────────────────────
    def _action(self, acts: Any) -> dict:
        if isinstance(acts, (str, dict)):
            acts = [acts]
        if not isinstance(acts, list) or not acts:
            raise ValueError("action() expects a non-empty list of action names or dicts")
        # Verification gate: with enough recorded transitions, long batches
        # are gambling unless the theory demonstrably works (accuracy >= 0.6)
        # or the model has made 3 honest verify attempts (escape hatch for
        # genuinely hard-to-model games). Short probes stay allowed so
        # exploration is not starved.
        if (not self.verify_gate_open() and len(acts) > 3
                and len(self.transition_log) >= 10):
            lv = self.last_verify
            status = ("you never called verify_theory" if lv is None else
                      f"best accuracy so far {lv.get('accuracy')} after "
                      f"{self.verify_attempts} attempt(s)")
            raise RuntimeError(
                f"action() batch of {len(acts)} blocked: {len(self.transition_log)} "
                f"transitions are recorded and {status}. Probes of up to 3 actions "
                "are still allowed. Long batches unlock at verify_theory accuracy "
                ">= 0.6" + (f", or after {self.gate_escape_attempts} attempts"
                            if self.gate_escape_attempts < 100 else "")
                + " — refine predict(grid, action, data) "
                "against the counterexamples and call verify_theory(predict) again.")
        # Duck stops a batch immediately on level_completed, not just
        # game_over/win (tool_agent.py's terminal-stop instruction + host
        # enforcement) -- we only checked game_over/win. Without this, a
        # level-up mid-batch let the REMAINING actions execute blind against
        # the new level's (different) layout: pure waste, quantifiable in
        # today's stand data (e.g. vc33: 800 actions in 54 turns, levels=0).
        level_at_batch_start = self.current.level if self.current is not None else 0
        result: dict = {}
        for spec in acts:
            if self._cancelled:
                # A timed-out thread must never keep stepping the real env.
                raise GameOverInterrupt("CANCELLED")
            if self.budget_left() <= 0:
                raise GameOverInterrupt("BUDGET")
            if self.turn_action_cap and self._turn_env_calls >= self.turn_action_cap:
                raise GameOverInterrupt("TURN_CAP")
            if isinstance(spec, str):
                name, payload = spec.upper(), None
            elif isinstance(spec, dict):
                name = str(spec.get("action", "")).upper()
                payload = {k: int(v) for k, v in spec.items() if k in ("x", "y")} or None
            else:
                raise ValueError(f"bad action spec: {spec!r}")
            engine = TO_ENGINE.get(name)
            if engine is None:
                raise ValueError(f"unknown action {name!r}; use one of {sorted(TO_ENGINE)}")
            if engine == "ACTION6" and (payload is None or "x" not in payload or "y" not in payload):
                raise ValueError("CLICK needs {'action':'CLICK','x':int,'y':int} with 0..63 coords")

            if self.game_over and engine != "RESET":
                raise RuntimeError(
                    "the game is in GAME_OVER: every further action is wasted budget. "
                    "Call action(['RESET']) first (it restarts the level), then continue.")
            before = self.current.grid if self.current is not None else None
            self._turn_env_calls += 1
            frame = self.env_step(engine, payload)
            self.update_frame(frame, engine, payload)
            after = self.current.grid if self.current is not None else None

            state = str(getattr(frame, "state", "")).split(".")[-1]
            diff = grid_diff(before, after) if before is not None and after is not None else None
            result = {
                "action": name,
                "state": state,
                "level": int(frame.levels_completed or 0),
                "board_changed": bool(diff.changed) if diff else True,
                "hud_only_change": bool(diff.border_only) if diff and diff.changed else False,
                "changed_cells": diff.changed_cells if diff else -1,
                "game_over": state == "GAME_OVER",
                "win": state == "WIN",
                "budget_left": self.budget_left(),
            }
            self.last_result = result
            if state == "WIN":
                raise GameOverInterrupt("WIN")
            if state == "GAME_OVER":
                break  # let the model see it and decide to RESET
            if result["level"] > level_at_batch_start:
                result["level_completed"] = True
                break  # new level = different layout; remaining actions in
                       # this batch would fire blind against it (Duck parity)
        return result

    def _refresh_scope(self, scope: dict) -> None:
        scope.update(
            current_frame=self.current,
            previous_frame=self.previous,
            history=list(self.history[-50:]),
            valid_actions=list(self.valid_actions),
            last_action_result=dict(self.last_result),
        )

    # ── execution ─────────────────────────────────────────────────────────
    def run_code(self, code: str) -> SandboxResult:
        stdout = io.StringIO()
        allowed_builtins = (
            "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
            "callable", "chr", "complex", "dict", "divmod", "enumerate", "filter",
            "float", "format", "frozenset", "getattr", "hasattr", "hash", "hex",
            "id", "int", "isinstance", "issubclass", "iter", "len", "list", "map",
            "max", "min", "next", "object", "oct", "ord", "pow", "print", "range",
            "repr", "reversed", "round", "set", "setattr", "slice", "sorted",
            "str", "sum", "tuple", "type", "vars", "zip",
            "Exception", "BaseException", "ArithmeticError", "AssertionError",
            "AttributeError", "IndexError", "KeyError", "LookupError",
            "NameError", "NotImplementedError", "OverflowError", "RuntimeError",
            "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
            "True", "False", "None",
        )
        scope = {
            "__builtins__": {
                **{k: getattr(_builtins, k) for k in allowed_builtins},
                "__import__": _safe_import,
            },
            "action": None,  # bound below so it can refresh this very scope
            "memo": self.memo,
            "np": np,
            "bfs_path": bfs_path,
            "reachable": reachable,
            "objects": objects,
            "verify_theory": self._verify_theory,
            "plan_with_theory": self._plan_with_theory,
            "transition_count": len(self.transition_log),
            "result": None,
        }
        self._refresh_scope(scope)

        def _action_live(acts: Any) -> dict:
            res = self._action(acts)
            # Keep the model's view honest: after a real action, the injected
            # variables must point at the NEW state (the prompt promises this).
            self._refresh_scope(scope)
            return res

        scope["action"] = _action_live

        interrupted: str | None = None
        error: str | None = None
        executed_before = self.step_counter
        self._turn_start_step = self.step_counter
        self._turn_env_calls = 0

        def _target():
            nonlocal interrupted, error
            try:
                with contextlib.redirect_stdout(stdout):
                    exec(compile(code, "<model_code>", "exec"), scope)  # noqa: S102
            except GameOverInterrupt as gi:
                interrupted = gi.reason
            except Exception:
                error = traceback.format_exc(limit=3)

        self._cancelled = False
        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(self.timeout_s)
        if t.is_alive():
            self._cancelled = True  # abandoned thread stops at its next action()
            error = (
                f"TimeoutError: code exceeded {self.timeout_s}s and was cancelled. "
                "Split the work into smaller snippets."
            )

        if interrupted == "TURN_CAP":
            interrupted = None
            error = (
                f"TurnActionCap: this code block executed {self.turn_action_cap} real "
                "actions and was stopped (per-turn limit). Your loop was spending the "
                "episode's action budget blind -- inspect the board, re-plan, and act "
                "in smaller verified steps next turn."
            )
        out = stdout.getvalue()
        if scope.get("result") is not None:
            out += f"\nresult = {scope['result']!r}"
        if len(out) > self.max_output_chars:
            out = out[: self.max_output_chars] + f"\n...[output truncated at {self.max_output_chars} chars]"
        return SandboxResult(
            output=out,
            error=error,
            actions_executed=self.step_counter - executed_before,
            interrupted=interrupted,
        )

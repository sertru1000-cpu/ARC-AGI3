"""Lightweight isolated runner for analyzer Python tool calls."""
from __future__ import annotations

import inspect
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import textwrap
import time
from typing import Any, Callable

from inference.utils import segmentation as _segmentation
from inference.utils.grid_utils import ARC_COLOR_CHARS


_SANDBOX_BOOTSTRAP = textwrap.dedent(
    r"""
    import builtins
    import contextlib
    import io
    import json
    import os
    import sys
    import time
    import traceback

    try:
        import resource
    except ImportError:  # pragma: no cover
        resource = None

    COLOR_CHARS = ""

    __SEGMENTATION_SOURCE__

    HOST_STDOUT = sys.stdout

    SAFE_MODULES = {
        "bisect",
        "collections",
        "copy",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
    }
    SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "Exception",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "TypeError",
        "type",
        "ValueError",
        "RuntimeError",
        "zip",
    }


    def _send(payload):
        HOST_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
        HOST_STDOUT.flush()


    def _recv():
        line = sys.stdin.readline()
        if not line:
            raise EOFError("sandbox input closed")
        return json.loads(line)


    class FrameView:
        def __init__(self, *, ascii, step, level, shape, grid):
            self.ascii = ascii
            self.step = step
            self.level = level
            self.shape = tuple(shape)
            # Raw color-index grid (list of rows of int), for verify_theory /
            # plan_with_theory predict() functions. segmentation stays the
            # documented primary view; this is additive, not a replacement.
            self.grid = grid
            self._grid = grid
            self._segmentation = None

        @property
        def segmentation(self):
            if self._segmentation is None:
                self._segmentation = segment_layer(self._grid, COLOR_CHARS)
            return self._segmentation

        def __str__(self):
            rows, cols = self.shape
            return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

        __repr__ = __str__


    class HistoryEntryView:
        def __init__(self, *, action, frame):
            self.action = action
            self.frame = frame

        def __str__(self):
            return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"

        __repr__ = __str__


    class TransitionView:
        def __init__(self, *, action, before_frame, after_frame, result):
            self.action = action
            self.before_frame = before_frame
            self.after_frame = after_frame
            self.frame = after_frame
            self.result = dict(result) if isinstance(result, dict) else {}

        def __str__(self):
            return (
                "ActionTransitionView("
                f"action={self.action!r}, "
                f"before_frame={self.before_frame}, "
                f"after_frame={self.after_frame})"
            )

        __repr__ = __str__


    def _frame_from_payload(payload):
        if not isinstance(payload, dict):
            return None
        return FrameView(
            ascii=str(payload.get("ascii", "")),
            step=int(payload.get("step", 0)),
            level=int(payload.get("level", 0)),
            shape=payload.get("shape", [0, 0]),
            grid=payload.get("grid", []),
        )


    def _history_from_payload(payload):
        items = []
        for entry in payload or []:
            if not isinstance(entry, dict):
                continue
            items.append(
                HistoryEntryView(
                    action=str(entry.get("action", "")),
                    frame=_frame_from_payload(entry.get("frame")),
                )
            )
        return items


    def _transitions_from_history(history, last_action_result):
        transitions = []
        for index, entry in enumerate(history):
            action = str(getattr(entry, "action", "") or "").strip()
            if not action:
                continue
            before_frame = history[index - 1].frame if index > 0 else None
            transitions.append(
                TransitionView(
                    action=action,
                    before_frame=before_frame,
                    after_frame=entry.frame,
                    result={},
                )
            )
        if transitions and isinstance(last_action_result, dict):
            transitions[-1].result = dict(last_action_result)
        return transitions


    def _json_safe(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return str(value)


    def _sanitize_exception(exc):
        extracted = traceback.extract_tb(exc.__traceback__)
        user_frames = [frame for frame in extracted if frame.filename == "<python_tool>"]
        lines = ["Traceback (most recent call last):"]
        for frame in user_frames or extracted[-1:]:
            lines.append(f'  File "<python_tool>", line {frame.lineno}, in {frame.name}')
        lines.append(f"{exc.__class__.__name__}: {exc}")
        return "\n".join(lines)


    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name or "").split(".", 1)[0]
        if root not in SAFE_MODULES:
            raise ImportError(f"Module '{name}' is not allowed in the sandbox.")
        return builtins.__import__(name, globals, locals, fromlist, level)


    def _set_limits(timeout_seconds):
        if resource is None:
            return
        cpu_limit = max(1, int(timeout_seconds)) + 1
        for limit, value in (
            (getattr(resource, "RLIMIT_CPU", None), cpu_limit),
            (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),
            (getattr(resource, "RLIMIT_NOFILE", None), 32),
        ):
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (value, value))
            except (OSError, ValueError):
                pass


    def _normalize_actions(actions):
        if isinstance(actions, str):
            items = [actions]
        elif isinstance(actions, dict):
            items = [actions]
        elif isinstance(actions, (list, tuple)):
            items = list(actions)
        else:
            raise TypeError(
                "action(actions) expects a string, an action object, or a list of action strings/objects."
            )
        if not items:
            raise ValueError("action(actions) requires at least one action.")

        normalized = []
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
                    raise ValueError(
                        f"Action {index} uses legacy MOUSE x/y fields; use row and col."
                    )
                if "row" in item:
                    entry["row"] = item.get("row")
                if "col" in item:
                    entry["col"] = item.get("col")
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        return normalized


    def main():
        initial = _recv()
        global COLOR_CHARS
        COLOR_CHARS = str(initial.get("color_chars") or "")
        timeout_seconds = max(1, int(initial.get("timeout_seconds", 30)))
        sandbox_cwd = str(initial.get("sandbox_cwd", "")).strip()
        if sandbox_cwd:
            os.chdir(sandbox_cwd)
        _set_limits(timeout_seconds)

        action_results = []
        stdout = io.StringIO()
        runtime_globals = {
            "__builtins__": {
                name: getattr(builtins, name)
                for name in SAFE_BUILTINS
            },
            "result": None,
        }
        runtime_globals["__builtins__"]["__import__"] = _safe_import
        # atlas: persistent scratch memory for THIS episode. Set once here,
        # from what the host round-tripped from the end of the previous
        # turn -- deliberately NOT touched by _refresh_state below, which
        # re-fires after every action() call. current_frame/history SHOULD
        # be replaced after each action (they describe the live game state);
        # memo is the model's own variable, mutated in place by its code, and
        # resetting it mid-turn on every action() call would silently erase
        # whatever it just stored.
        initial_memo = (initial.get("state") or {}).get("memo")
        runtime_globals["memo"] = initial_memo if isinstance(initial_memo, dict) else {}

        def _refresh_state(state_payload):
            current_frame = _frame_from_payload(state_payload.get("current_frame"))
            history = _history_from_payload(state_payload.get("history"))
            last_action_result = state_payload.get("last_action_result")
            action_result = (
                dict(last_action_result) if isinstance(last_action_result, dict) else {}
            )
            transitions = _transitions_from_history(history, action_result)
            last_transition = transitions[-1] if transitions else None

            runtime_globals["current_frame"] = current_frame
            runtime_globals["latest_frame"] = current_frame
            runtime_globals["history"] = history
            runtime_globals["transitions"] = transitions
            runtime_globals["last_transition"] = last_transition
            runtime_globals["previous_frame"] = (
                last_transition.before_frame if last_transition is not None else None
            )
            runtime_globals["last_action_frame"] = (
                last_transition.after_frame if last_transition is not None else None
            )
            runtime_globals["last_action"] = last_transition.action if last_transition is not None else None
            runtime_globals["valid_actions"] = [str(item) for item in state_payload.get("valid_actions", [])]
            runtime_globals["last_action_result"] = action_result

        def action(actions):
            normalized_actions = _normalize_actions(actions)
            _send({"type": "action", "actions": normalized_actions})
            reply = _recv()
            if reply.get("type") == "action_error":
                raise RuntimeError(str(reply.get("error", "action failed")))
            if reply.get("type") != "action_result":
                raise RuntimeError("Invalid action response from sandbox host.")
            action_result = reply.get("action_result") or {}
            action_results.append(action_result)
            _refresh_state(reply.get("state") or {})
            return action_result

        def animation(action_num=None, frame=None, region=None):
            _send(
                {
                    "type": "animation",
                    "request": {
                        "action_num": action_num,
                        "frame": frame,
                        "region": list(region) if region is not None else None,
                    },
                }
            )
            reply = _recv()
            if reply.get("type") == "animation_error":
                raise RuntimeError(str(reply.get("error", "animation failed")))
            if reply.get("type") != "animation_result":
                raise RuntimeError("Invalid animation response from sandbox host.")
            return reply.get("animation") or {}

        def _action_display(spec):
            if isinstance(spec, str):
                return spec.strip()
            if isinstance(spec, dict):
                name = str(spec.get("action", "")).strip().upper()
                if name == "MOUSE" and "row" in spec and "col" in spec:
                    return f"MOUSE(row={int(spec['row'])}, col={int(spec['col'])})"
                return name
            raise ValueError(f"bad action spec: {spec!r}")

        def verify_theory(predict, actions=None):
            '''Test a transition theory against every recorded real transition.

            predict(grid, action) -> predicted next grid (list of rows of int).
            `action` is the same display string seen on `transitions[i].action`
            (e.g. "UP", "MOUSE(row=4, col=7)"). Optional `actions` restricts
            testing to transitions whose action name (before any "(") is in
            that list. Returns accuracy + up to 3 counterexamples. Costs zero
            environment actions -- it only replays what already happened.
            '''
            wanted = {str(a).strip().upper() for a in actions} if actions else None
            tested = matched = errors = 0
            mismatches = []
            for t in runtime_globals.get("transitions") or []:
                base_name = t.action.split("(", 1)[0].strip().upper()
                if wanted and base_name not in wanted:
                    continue
                before = t.before_frame
                after = t.after_frame
                if before is None or after is None:
                    continue
                tested += 1
                try:
                    pred = predict([list(row) for row in before.grid], t.action)
                except Exception as exc:
                    errors += 1
                    if len(mismatches) < 3:
                        mismatches.append({"action": t.action, "error": repr(exc)[:200]})
                    continue
                actual = after.grid
                if (
                    not isinstance(pred, list)
                    or len(pred) != len(actual)
                    or any(len(pr) != len(ar) for pr, ar in zip(pred, actual))
                ):
                    errors += 1
                    if len(mismatches) < 3:
                        mismatches.append({"action": t.action, "error": "predict() returned the wrong shape"})
                    continue
                wrong = [
                    (r, c)
                    for r in range(len(actual))
                    for c in range(len(actual[r]))
                    if pred[r][c] != actual[r][c]
                ]
                if not wrong:
                    matched += 1
                elif len(mismatches) < 3:
                    sample = [(r, c, pred[r][c], actual[r][c]) for r, c in wrong[:4]]
                    mismatches.append({
                        "action": t.action,
                        "wrong_cells": len(wrong),
                        "sample (row,col,predicted,actual)": sample,
                    })
            return {
                "transitions_tested": tested,
                "accuracy": round(matched / tested, 3) if tested else None,
                "predict_errors": errors,
                "counterexamples": mismatches,
            }

        def plan_with_theory(predict, goal, actions=None, max_depth=6, max_nodes=1500,
                              min_accuracy=0.6, time_budget=8.0):
            '''Search for an action sequence using the model's OWN verified theory.

            Pure simulation against predict() -- spends ZERO environment
            actions, which matters because actions are scored quadratically.
            The theory is re-verified here first and planning is REFUSED below
            min_accuracy: planning over an unverified theory is guessing with
            extra steps. Returns {'plan': [...] | None, ...}; the plan is a
            list of specs `action()` accepts verbatim (whatever you passed in
            `actions`, or the current `valid_actions` by default), so the
            usual next line is `action(res['plan'])`.

            MOUSE is excluded from the default candidate set (64x64 targets
            would blow up branching); pass explicit `{'action': 'MOUSE',
            'row': r, 'col': c}` specs in `actions` to plan toward a click.
            '''
            # `actions` may hold dict specs (e.g. a MOUSE target), which
            # verify_theory's filter cannot match directly -- reduce each to
            # its base action name first ("MOUSE(row=4, col=7)" -> "MOUSE").
            verify_actions = (
                [_action_display(a).split("(", 1)[0] for a in actions] if actions else None
            )
            check = verify_theory(predict, verify_actions)
            acc = check.get("accuracy")
            if acc is None or acc < min_accuracy:
                return {
                    "plan": None,
                    "verified_accuracy": acc,
                    "reason": (
                        f"theory not good enough to plan with: accuracy {acc} < {min_accuracy} "
                        f"on {check.get('transitions_tested')} transitions. Refine predict() "
                        "against the counterexamples from verify_theory first."
                    ),
                    "counterexamples": check.get("counterexamples"),
                }

            current = runtime_globals.get("current_frame")
            if current is None:
                return {"plan": None, "reason": "no current frame"}

            specs = list(actions) if actions else [
                a for a in (runtime_globals.get("valid_actions") or [])
                if str(a).strip().upper() not in ("MOUSE", "RESET")
            ]
            if not specs:
                return {"plan": None, "reason": "no candidate actions to search over"}

            start = [list(row) for row in current.grid]
            try:
                if goal([list(row) for row in start]):
                    return {"plan": [], "verified_accuracy": acc, "nodes_expanded": 0,
                            "reason": "already at the goal"}
            except Exception as exc:
                return {"plan": None, "reason": f"goal() raised on the current grid: {exc!r}"}

            def _key(grid):
                return tuple(tuple(row) for row in grid)

            deadline = time.monotonic() + time_budget
            seen = {_key(start)}
            frontier = [(start, [])]
            nodes = pred_errors = 0
            while frontier:
                nxt = []
                for grid, path in frontier:
                    if len(path) >= max_depth:
                        continue
                    for spec in specs:
                        if nodes >= max_nodes or time.monotonic() > deadline:
                            return {
                                "plan": None, "verified_accuracy": acc,
                                "nodes_expanded": nodes, "predict_errors": pred_errors,
                                "reason": (
                                    f"budget spent ({nodes} states, depth <= {len(path) + 1}) "
                                    "without reaching the goal. Try a looser goal(), a larger "
                                    "max_depth, or a smaller action set."
                                ),
                            }
                        nodes += 1
                        try:
                            pred = predict([list(row) for row in grid], _action_display(spec))
                        except Exception:
                            pred_errors += 1
                            continue
                        if (
                            not isinstance(pred, list)
                            or len(pred) != len(start)
                            or any(len(row) != len(start[0]) for row in pred)
                        ):
                            pred_errors += 1
                            continue
                        key = _key(pred)
                        if key in seen:
                            continue
                        seen.add(key)
                        plan = path + [spec]
                        try:
                            if goal([list(row) for row in pred]):
                                return {
                                    "plan": plan, "depth": len(plan), "verified_accuracy": acc,
                                    "nodes_expanded": nodes, "predict_errors": pred_errors,
                                    "reason": None,
                                }
                        except Exception as exc:
                            return {"plan": None, "nodes_expanded": nodes,
                                    "reason": f"goal() raised: {exc!r}"}
                        nxt.append((pred, plan))
                frontier = nxt
            return {
                "plan": None, "verified_accuracy": acc, "nodes_expanded": nodes,
                "predict_errors": pred_errors,
                "reason": (
                    f"exhausted {nodes} reachable states within depth {max_depth} without "
                    "reaching the goal -- the goal may be unreachable with these actions, or "
                    "the theory may miss the mechanic that matters."
                ),
            }

        runtime_globals["action"] = action
        runtime_globals["verify_theory"] = verify_theory
        runtime_globals["plan_with_theory"] = plan_with_theory
        if initial.get("animation_enabled"):
            runtime_globals["animation"] = animation
        _refresh_state(initial.get("state") or {})

        try:
            compiled = compile(str(initial.get("code", "")), "<python_tool>", "exec")
            with contextlib.redirect_stdout(stdout):
                exec(compiled, runtime_globals, runtime_globals)
            _send(
                {
                    "type": "final",
                    "stdout": stdout.getvalue(),
                    "result": _json_safe(runtime_globals.get("result")),
                    "action_results": _json_safe(action_results),
                    "memo": _json_safe(runtime_globals.get("memo")),
                }
            )
        except Exception as exc:
            _send(
                {
                    "type": "error",
                    "error": _sanitize_exception(exc),
                    "stdout": stdout.getvalue(),
                    "action_results": _json_safe(action_results),
                    # atlas: round-trip memo even on a raised exception -- a
                    # crash midway through a turn's code should not erase
                    # whatever the model had already stored via memo[...]=...
                    # before the line that raised.
                    "memo": _json_safe(runtime_globals.get("memo")),
                }
            )


    if __name__ == "__main__":
        main()
    """
).replace("__SEGMENTATION_SOURCE__\n", inspect.getsource(_segmentation))


def _sanitize_host_error_text(text: str) -> str:
    if not str(text or "").strip():
        return "Sandbox process exited unexpectedly."
    return "Sandbox process exited unexpectedly."


def _sandbox_env() -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PATH": os.environ.get("PATH", ""),
    }


def _send_json_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        try:
            process.kill()
        except OSError:
            pass


def _wait_for_process_exit(process: subprocess.Popen[str], *, timeout: float = 1.0) -> None:
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
    except OSError:
        return

    try:
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        pass


def run_sandboxed_python(
    *,
    code: str,
    timeout_seconds: int,
    initial_state: dict[str, Any],
    action_handler: Callable[[list[dict[str, Any]]], dict[str, Any]],
    animation_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rgb_python_tool_") as sandbox_dir:
        host_action_results: list[dict[str, Any]] = []
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _SANDBOX_BOOTSTRAP],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                cwd=sandbox_dir,
                env=_sandbox_env(),
                start_new_session=True,
            )
        except OSError:
            return {
                "error": "Sandbox process could not start.",
                "stdout": "",
                "action_results": [],
            }
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        stdout_queue: queue.Queue[str | None] = queue.Queue()

        def _stdout_reader() -> None:
            for raw_line in process.stdout:
                stdout_queue.put(raw_line)
            stdout_queue.put(None)

        threading.Thread(target=_stdout_reader, daemon=True).start()

        _send_json_line(
            process.stdin,
            {
                "code": code,
                "timeout_seconds": timeout_seconds,
                "sandbox_cwd": sandbox_dir,
                "state": initial_state,
                "color_chars": ARC_COLOR_CHARS,
                "animation_enabled": animation_handler is not None,
            },
        )

        deadline = time.monotonic() + max(1, int(timeout_seconds))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": f"Tool timed out after {timeout_seconds}s",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                line = stdout_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                stderr = process.stderr.read()
                _wait_for_process_exit(process)
                return {
                    "error": _sanitize_host_error_text(stderr),
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                stderr = process.stderr.read()
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": "Sandbox process returned an invalid response.",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            msg_type = str(message.get("type", "")).strip()
            if msg_type == "action":
                try:
                    action_result_payload = action_handler(list(message.get("actions") or []))
                except Exception:  # noqa: BLE001
                    _send_json_line(
                        process.stdin,
                        {
                            "type": "action_error",
                            "error": "action failed in sandbox host.",
                        },
                    )
                    continue
                raw_action_result = action_result_payload.get("action_result") or {}
                if isinstance(raw_action_result, dict):
                    host_action_results.append(dict(raw_action_result))
                _send_json_line(
                    process.stdin,
                    {
                        "type": "action_result",
                        "action_result": raw_action_result,
                        "state": action_result_payload.get("state") or {},
                    },
                )
                continue

            if msg_type == "animation":
                if animation_handler is None:
                    _send_json_line(
                        process.stdin,
                        {"type": "animation_error", "error": "animation() is not available."},
                    )
                    continue
                try:
                    animation_payload = animation_handler(dict(message.get("request") or {}))
                except Exception:  # noqa: BLE001
                    _send_json_line(
                        process.stdin,
                        {"type": "animation_error", "error": "animation failed in sandbox host."},
                    )
                    continue
                _send_json_line(
                    process.stdin,
                    {"type": "animation_result", "animation": animation_payload},
                )
                continue

            if msg_type in {"final", "error"}:
                _wait_for_process_exit(process)
                return {
                    "stdout": str(message.get("stdout", "") or ""),
                    "result": message.get("result"),
                    "error": str(message.get("error", "") or ""),
                    "action_results": list(message.get("action_results") or host_action_results),
                    "memo": message.get("memo"),
                }

            _wait_for_process_exit(process)
            return {
                "error": "Sandbox process returned an unknown message type.",
                "stdout": "",
                "action_results": list(host_action_results),
            }

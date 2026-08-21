"""LLM policy: the observe -> code -> execute -> feedback loop.

One `play_turn()` = one LLM completion + execution of its code block in the
sandbox. The policy owns the message history (with context trimming) and
tracks stall turns (no real action taken) to nudge the model forward.
"""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field

from .llm import LLMBackend, Message
from .perception import salient_click_targets
from .prompts import (
    NUDGE_NO_ACTION,
    NUDGE_NO_CODE,
    SYSTEM_PROMPT,
    THEORY_CHECKPOINT,
    TOOL_LOOP_ADDENDUM,
    initial_user_message,
    turn_user_message,
)
from .sandbox import Sandbox
from .trace import TraceWriter

# Native tool-calling schema for the tool-loop (backlog #11 v2, 21.08): code
# is submitted as a `run_python` function call instead of a ```python fence,
# so the wire format matches what Duck uses and what vLLM's OpenAI-compatible
# server natively parses/validates (no more regex-scraping fenced code out of
# free-text replies for this path).
RUN_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute python in the game sandbox: inspect state or call "
            "action([...]). This is the ONLY way to act or observe -- there "
            "is no other tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute."},
            },
            "required": ["code"],
        },
    },
}

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
WORLD_MODEL_RE = re.compile(
    r"WORLD_MODEL:\s*\n(.*?)(?=```|\Z)", re.DOTALL | re.IGNORECASE
)
WM_LINE_RE = re.compile(
    r"^\s*(controls|prior|goal|plan|recent_findings|open_questions|cross_level_notes)"
    r"\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def _approx_tokens(text: str) -> int:
    """Cheap, deliberately conservative token estimate (no tokenizer on
    hand here). Real ratio on our grid-heavy data is ~2.5-2.7 chars/token
    (build_sft_dataset.py) -- dividing by 2 overestimates, erring toward
    trimming too much rather than overflowing the context window."""
    return max(1, len(text) // 2)


@dataclass
class LLMPolicy:
    backend: LLMBackend
    sandbox: Sandbox
    game_id: str
    win_levels: int
    # Token budget for kept history, not a fixed pair count — a handful of
    # verbose turns (long sandbox output, segmentation dumps) can blow the
    # context even with few pairs kept (backlog item 6, 20.08).
    max_context_tokens: int = field(
        default_factory=lambda: int(os.getenv("MY_AGENT_CONTEXT_TOKENS", "16000")))
    max_tokens: int = 2048
    temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.6")))
    # Text nudges (NUDGE_NO_ACTION at stall_turns>=2) get ignored by the
    # no-thinking student (20.08 stand: wa30 hit stall_turns=40+, 53 identical
    # inspection-only replies). At this threshold the harness stops asking
    # and acts for the model, cf. the v8 lesson: enforce structurally.
    stall_force_threshold: int = field(
        default_factory=lambda: int(os.getenv("MY_AGENT_STALL_FORCE", "4")))
    # Duck-style tool-loop (backlog #11, adopted 20.08): 0 = off (default,
    # one LLM call = one turn, unchanged). N>0 = up to N rounds of fresh
    # LLM calls WITHIN one outer turn, each seeing the previous round's real
    # sandbox result before deciding the next code block -- instead of
    # planning a whole batch blind in one shot. Ported as our own protocol
    # (text code-blocks, not OpenAI-native tool-calling) to avoid depending
    # on vLLM tool-call support we haven't verified for this model.
    tool_loop_steps: int = field(
        default_factory=lambda: int(os.getenv("MY_AGENT_TOOL_LOOP_STEPS", "0")))

    messages: list[Message] = field(default_factory=list)
    stall_turns: int = 0
    no_code_strikes: int = 0
    turns: int = 0
    trace: TraceWriter | None = None
    # Cycles through salient click targets on repeated forced CLICKs instead
    # of hammering the same point (20.08 stand: a CLICK-only game got forced
    # 9x in a row, every single one at the grid center — zero new info).
    forced_click_idx: int = 0

    def _frame_seen(self) -> str | None:
        """ASCII of the board the model is looking at for THIS call. Stored
        in the trace so a multimodal SFT dataset (student is a VL model) can
        regenerate the exact PNG the teacher saw -- not reconstructible from
        sandbox output otherwise."""
        cur = self.sandbox.current
        return cur.ascii if cur is not None else None

    def start(self, probe_summary: str = "", cross_note: str = "") -> None:
        if self.trace is None:
            self.trace = TraceWriter(self.game_id)
        level = self.sandbox.current.level if self.sandbox.current else 0
        system_content = SYSTEM_PROMPT + (TOOL_LOOP_ADDENDUM if self.tool_loop_steps > 0 else "")
        self.messages = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": initial_user_message(
                    self.game_id, level, self.win_levels,
                    self.sandbox.valid_actions, probe_summary, cross_note,
                ),
            },
        ]
        if self.trace:
            # cross_note carries the cross-game journal AND any human hint —
            # store it so SFT reconstruction can rebuild the exact prompt.
            self.trace.write({"turn": 0, "probe_summary": probe_summary or None,
                              "cross_note": cross_note or None,
                              # Which backend produced this trace (e.g. a
                              # Vision(...) wrapper) -- lets later analysis
                              # split multimodal vs text-only runs.
                              "backend": self.backend.name})

    def note(self, text: str) -> None:
        """Inject an out-of-band user note (e.g. level-up reground)."""
        self.messages.append({"role": "user", "content": text})
        self._trim()

    # ── context management ────────────────────────────────────────────────
    def _trim(self) -> None:
        """Keep system + first user + as many recent exchange pairs as fit
        inside max_context_tokens, walking backward from the newest turn.

        Always keeps at least the single most recent pair, even if it alone
        exceeds the budget -- dropping it would strand the model without the
        feedback for what it just did.
        """
        from .prompts import memo_digest

        head, tail = self.messages[:2], self.messages[2:]
        kept: list[Message] = []
        used = 0
        i = len(tail)
        while i >= 2:
            pair = tail[i - 2:i]
            pair_tokens = sum(_approx_tokens(m["content"]) for m in pair)
            if kept and used + pair_tokens > self.max_context_tokens:
                break
            kept = pair + kept
            used += pair_tokens
            i -= 2
        dropped = tail[:i]
        if dropped:
            head.append(
                {
                    "role": "user",
                    "content": f"[context note] {len(dropped) // 2} older exchange(s) "
                    f"were trimmed (~{used} of {self.max_context_tokens} token budget kept)."
                    + (memo_digest(self.sandbox.memo)
                       or " Your accumulated understanding should live in memo."),
                }
            )
        self.messages = head + kept

    # ── forced probe (structural stall-break) ───────────────────────────────
    def _pick_forced_action(self) -> tuple[str, dict | None]:
        """Random valid action, avoiding blind CLICK unless it's the only option."""
        pool = [a for a in self.sandbox.valid_actions if a != "CLICK"] or list(self.sandbox.valid_actions)
        if not pool:
            return "SPACE", None
        name = random.choice(pool)
        if name == "CLICK":
            return name, self._pick_click_target()
        return name, None

    def _pick_click_target(self) -> dict:
        """Cycle through likely-interactive points instead of one fixed spot."""
        grid = self.sandbox.current.grid if self.sandbox.current is not None else None
        targets = salient_click_targets(grid) if grid is not None else []
        if targets:
            x, y = targets[self.forced_click_idx % len(targets)]
            self.forced_click_idx += 1
            return {"x": x, "y": y}
        h, w = grid.shape if grid is not None else (64, 64)
        return {"x": random.randint(0, w - 1), "y": random.randint(0, h - 1)}

    def _forced_turn(self) -> dict:
        """Skip the LLM this turn and execute one real action directly.

        No LLM call is spent — the model was ignoring NUDGE_NO_ACTION anyway,
        so asking again would just burn tokens on the same dead-end reply.
        """
        name, payload = self._pick_forced_action()
        code = (f"action([{{'action': {name!r}, 'x': {payload['x']}, 'y': {payload['y']}}}])"
                if payload else f"action([{name!r}])")
        res = self.sandbox.run_code(code)

        if res.actions_executed == 0:
            self.stall_turns += 1
        else:
            self.stall_turns = 0

        level = self.sandbox.current.level if self.sandbox.current else 0
        lv = self.sandbox.last_verify
        wm_status = (f"VERIFIED, accuracy {lv['accuracy']} on {lv['tested']} transitions"
                     if lv else "NOT VERIFIED yet")
        where = f" at ({payload['x']},{payload['y']})" if payload else ""
        content = (
            f"[harness note] {self.stall_force_threshold} turns passed with no "
            f"game action, so the harness executed one probe directly "
            f"({name}{where}) instead of asking again. Continuing from here.\n\n"
        ) + turn_user_message(
            res.output, res.error, level, self.win_levels,
            self.sandbox.budget_left(), self.sandbox.valid_actions,
            self.sandbox.actions_on_current_level(), wm_status,
        )
        self.messages.append({
            "role": "assistant",
            "content": "(forced probe — stall threshold reached; harness took over for one action)",
        })
        self.messages.append({"role": "user", "content": content})
        self._trim()

        if self.trace:
            self.trace.write({
                "turn": self.turns, "reply": None, "code": code, "forced": True,
                "sandbox_output": res.output, "sandbox_error": res.error,
                "actions_executed": res.actions_executed, "interrupted": res.interrupted,
                "level": level, "budget_left": self.sandbox.budget_left(),
                "valid_actions": list(self.sandbox.valid_actions),
                "stall_turns": self.stall_turns,
            })

        return {"actions": res.actions_executed, "win": res.interrupted == "WIN", "error": res.error}

    # ── tool-loop turn (Duck-style multi-round, native tool-calling) ────────
    def _play_turn_toolloop(self) -> dict:
        """Up to tool_loop_steps rounds of (LLM call -> code exec -> observe)
        within this ONE outer turn, using native OpenAI-style tool-calling
        (`RUN_CODE_TOOL`) instead of a text ```python fence -- matches what
        Duck does and what vLLM's OpenAI-compatible server parses/validates
        natively (21.08 rework; was our own fence-scraping protocol before).

        Ends early when: the model's reply has no tool call (its deliberate
        signal that it's done deciding for now -- mirrors Duck exiting when
        the tool stops being called), a WIN/GAME_OVER fires, the round cap is
        hit, or the action budget runs out. A no-call FIRST round is a
        genuine failure (counts as a strike, same as the single-round path);
        a no-call round 2+ is a normal, expected exit. If the model requests
        more than one tool call in a round, only the first is executed (the
        model sees the result and can call again next round)."""
        total_actions = 0
        win = False
        error: str | None = None
        round_no = 0
        while round_no < self.tool_loop_steps:
            round_no += 1
            frame_seen = self._frame_seen()
            resp = self.backend.chat_tools(
                self.messages, tools=[RUN_CODE_TOOL],
                max_tokens=self.max_tokens, temperature=self.temperature,
            )
            content = resp.get("content") or ""
            tool_calls = resp.get("tool_calls") or []

            if content:
                wm_match = WORLD_MODEL_RE.search(content)
                if wm_match:
                    parsed = {k.lower(): v.strip() for k, v in WM_LINE_RE.findall(wm_match.group(1))}
                    if parsed:
                        self.sandbox.memo.setdefault("world_model", {}).update(parsed)

            assistant_msg: Message = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                    }
                    for tc in tool_calls
                ]
            self.messages.append(assistant_msg)

            if not tool_calls:
                if round_no == 1:
                    # Genuine failure (same as the single-round path) -- MUST
                    # give the same corrective nudge, or the model repeats the
                    # same mistake every turn with no signal why (found live
                    # 20.08 on the old fence protocol: 5 strikes in a row with
                    # no explanation; kept here since a model can still just
                    # not call the tool on round 1).
                    self.no_code_strikes += 1
                    error = "no_code"
                    self.messages.append({"role": "user", "content": NUDGE_NO_CODE})
                    self._trim()
                else:
                    self.no_code_strikes = 0  # deliberate end-of-turn, not a failure
                if self.trace:
                    self.trace.write({"turn": self.turns, "reply": content, "code": None,
                                      "tool_loop_round": round_no, "tool_loop_final": True,
                                      "no_code_strikes": self.no_code_strikes,
                                      "frame_seen": frame_seen})
                break

            self.no_code_strikes = 0
            tc = tool_calls[0]
            code = tc["arguments"].get("code", "") if isinstance(tc["arguments"], dict) else ""
            res = self.sandbox.run_code(code)
            total_actions += res.actions_executed
            error = res.error
            self.stall_turns = 0 if res.actions_executed else self.stall_turns + 1

            level = self.sandbox.current.level if self.sandbox.current else 0
            lv = self.sandbox.last_verify
            wm_status = (f"VERIFIED, accuracy {lv['accuracy']} on {lv['tested']} transitions"
                         if lv else "NOT VERIFIED yet")
            tool_result = turn_user_message(
                res.output, res.error, level, self.win_levels,
                self.sandbox.budget_left(), self.sandbox.valid_actions,
                self.sandbox.actions_on_current_level(), wm_status,
            ) + (f"\n\n[tool-loop] round {round_no}/{self.tool_loop_steps} within this turn. "
                 "Call run_python again to react to this result, or reply with plain "
                 "text and NO tool call to end the turn.")
            self.messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": tc["name"],
                "content": tool_result,
            })
            self._trim()

            if self.trace:
                self.trace.write({
                    "turn": self.turns, "reply": content, "code": code, "tool_loop_round": round_no,
                    "world_model": self.sandbox.memo.get("world_model"),
                    "sandbox_output": res.output, "sandbox_error": res.error,
                    "actions_executed": res.actions_executed, "interrupted": res.interrupted,
                    "level": level, "budget_left": self.sandbox.budget_left(),
                    "valid_actions": list(self.sandbox.valid_actions),
                    "stall_turns": self.stall_turns,
                    "frame_seen": frame_seen,
                })

            if res.interrupted == "WIN":
                win = True
                break
            if res.interrupted == "GAME_OVER" or self.sandbox.budget_left() <= 0:
                break
        return {"actions": total_actions, "win": win, "error": error}

    # ── one turn ──────────────────────────────────────────────────────────
    def play_turn(self) -> dict:
        """Returns {'actions': int, 'win': bool, 'error': str|None}."""
        self.turns += 1
        if self.stall_turns >= self.stall_force_threshold:
            return self._forced_turn()
        if self.tool_loop_steps > 0:
            return self._play_turn_toolloop()
        frame_seen = self._frame_seen()
        reply = self.backend.chat(self.messages, max_tokens=self.max_tokens, temperature=self.temperature)
        self.messages.append({"role": "assistant", "content": reply})

        # Harvest the mandatory WORLD_MODEL section into persistent memory —
        # enforced structurally because the model ignores polite suggestions.
        wm_match = WORLD_MODEL_RE.search(reply)
        if wm_match:
            parsed = {k.lower(): v.strip() for k, v in WM_LINE_RE.findall(wm_match.group(1))}
            if parsed:
                self.sandbox.memo.setdefault("world_model", {}).update(parsed)

        m = CODE_BLOCK_RE.search(reply)
        if not m:
            self.no_code_strikes += 1
            # A lone opening fence = the reply hit max_tokens mid-code. Asking
            # for "a code block" again reproduces the same overlong reply and
            # spirals into 5 strikes; ask for SHORTER code instead.
            truncated = "```python" in reply and reply.count("```") % 2 == 1
            nudge = (
                "Your reply was CUT OFF mid-code (output token limit) so no "
                "code ran. Resend a MUCH SHORTER python block: fewer comments, "
                "split the work over several turns if needed."
            ) if truncated else NUDGE_NO_CODE
            # Replace the degenerate reply in-context: leaving it verbatim
            # teaches the model to keep emitting code-less replies.
            self.messages[-1] = {
                "role": "assistant",
                "content": "(world model noted; reply lacked the required python block)",
            }
            self.messages.append({"role": "user", "content": nudge})
            self._trim()
            if self.trace:
                self.trace.write({
                    "turn": self.turns, "reply": reply, "code": None,
                    "no_code_strikes": self.no_code_strikes,
                    "frame_seen": frame_seen,
                })
            return {"actions": 0, "win": False, "error": "no_code"}

        self.no_code_strikes = 0
        code = m.group(1)
        res = self.sandbox.run_code(code)

        if res.actions_executed == 0:
            self.stall_turns += 1
        else:
            self.stall_turns = 0

        level = self.sandbox.current.level if self.sandbox.current else 0
        lv = self.sandbox.last_verify
        wm_status = (f"VERIFIED, accuracy {lv['accuracy']} on {lv['tested']} transitions"
                     if lv else "NOT VERIFIED yet")
        content = turn_user_message(
            res.output,
            res.error,
            level,
            self.win_levels,
            self.sandbox.budget_left(),
            self.sandbox.valid_actions,
            self.sandbox.actions_on_current_level(),
            wm_status,
        )
        if self.stall_turns >= 2:
            content += "\n\n" + NUDGE_NO_ACTION
        # Structural enforcement: enough facts + gate still closed -> demand a
        # (better) theory EVERY turn until accuracy >= 0.6 or 3 attempts spent
        # (every-3rd was ignored for 15+ turns; a 0.0-accuracy dummy attempt
        # must not silence the nudge either).
        if (not self.sandbox.verify_gate_open()
                and len(self.sandbox.transition_log) >= 6):
            content += "\n\n" + THEORY_CHECKPOINT
        if not wm_match:
            content += (
                "\n\nFormat reminder: begin every reply with the WORLD_MODEL "
                "section (controls/prior/goal/plan lines), then one python block."
            )
        self.messages.append({"role": "user", "content": content})
        self._trim()

        if self.trace:
            self.trace.write({
                "turn": self.turns,
                "reply": reply,
                "world_model": self.sandbox.memo.get("world_model"),
                "code": code,
                "sandbox_output": res.output,
                "sandbox_error": res.error,
                "actions_executed": res.actions_executed,
                "interrupted": res.interrupted,
                "level": level,
                "budget_left": self.sandbox.budget_left(),
                "valid_actions": list(self.sandbox.valid_actions),
                "stall_turns": self.stall_turns,
                "frame_seen": frame_seen,
            })

        return {
            "actions": res.actions_executed,
            "win": res.interrupted == "WIN",
            "error": res.error,
        }

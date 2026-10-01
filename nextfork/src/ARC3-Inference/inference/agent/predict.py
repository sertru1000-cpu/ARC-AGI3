"""Предсказание перед каждым ходом + страница заметок (NEXTFORK_PREDICT=1, 30.09) — в коде, ВЫКЛЮЧЕНО по умолчанию.

Идея — навык arc-skill (github.com/pbshgthm/arc-skill: Claude Code на Opus 5 прошёл 25/25 публичных игр, RHAE 100):
«скажи, что сделает ход, прежде чем его тратить». Код свой (у репозитория нет лицензии), язык утверждений в
координатах нашей обвязки (строка, столбец; цвет — буква из current_frame.ascii).

Что делает:
  * в песочнице action(actions, predict=...) — ход без предсказания отклоняется (ход не тратится); после каждого
    хода печатается «PREDICT ✓/✗ ...» с разбором каждого утверждения и коротким рассказом о переходе (что
    изменилось, без клеток, меняющихся на каждом ходу — счётчики); пачка останавливается на первом промахе;
  * notes(text) — одна страница заметок на игру (Verified / Assumed / Plan); обвязка вынимает её из вывода и
    показывает в каждом следующем сообщении — переживает обрезку истории;
  * в системный промпт — правило и словарь утверждений.
Замеры в самой игре (у нас) ещё не делались; на Opus 5 у автора одиночные пробы промахивались в 37%, ходы плана — в 2.9%.
Ставится из inference/framework/solver.py после AGENTFIX и памяти функций.
"""
from __future__ import annotations

import re
import sys
import threading
from pathlib import Path

import inference.agent.python_tool_sandbox as _sb
import inference.agent.tool_agent as _ta

SANDBOX = Path(__file__).with_name("predict_sandbox.py").read_text(encoding="utf-8")
_NOTES_RE = re.compile(r"@@NOTES@@\n(.*?)\n@@END_NOTES@@\n?", re.S)
_TLS = threading.local()
STATS = {"ok": 0, "miss": 0, "refused": 0, "notes": 0}

ADDENDUM = (
    "\n\nPREDICT-BEFORE-YOU-ACT RULE (enforced by the harness):\n"
    "- Every `action(...)` call must carry `predict=`: a falsifiable claim of what the move will do to the board. "
    "Without it the call is refused and no move is spent.\n"
    "- Claim forms (join several with ';'): `noop` | `change` | `cell R,C=X` (cell at row R, col C becomes color letter X) | "
    "`move R,C DR,DC` (the object covering cell R,C shifts by DR rows, DC cols) | `vanish R,C` | "
    "`region R0:R1,C0:C1` (something inside this box changes) | `level+1` | `win` | `gameover`.\n"
    "- One move: `action(['LEFT'], predict='move 12,5 0,-1')`. Several moves: pass a LIST of predictions, one per move; "
    "the batch stops at the first miss so a wrong theory cannot waste the rest.\n"
    "- After each move the harness prints `PREDICT ✓` or `PREDICT ✗` with what actually changed. A ✗ is the most "
    "valuable signal: reality just corrected your model of the game — fix your understanding before the next move.\n"
    "- Prefer one specific claim over a vague one. Early levels are cheap tutorials: act early, learn from every grade. "
    "Batch only moves you can predict exactly; never batch exploration.\n"
    "- Keep ONE page of notes with `notes(text)`: sections `Verified` (with the move that proved it), `Assumed / open`, "
    "`Plan`. It is shown back to you every turn and survives history truncation; rewrite it after every ✗ and at every "
    "new level (treat old Verified items as Assumed until they survive one test on the new board).\n"
)

# --- системный промпт ---
_orig_system = _ta._build_system_prompt


def _system(*a, **kw):
    return _orig_system(*a, **kw) + ADDENDUM


_ta._build_system_prompt = _system

# --- песочница: обёртка action и notes перед кодом модели; вынуть заметки из вывода ---
_orig_sandbox = _ta.run_sandboxed_python


def _sandbox(*, code, **kw):
    res = _orig_sandbox(code=SANDBOX + "\n" + str(code), **kw)
    try:
        agent = getattr(_TLS, "agent", None)
        out = str(res.get("stdout", "") or "")
        found = _NOTES_RE.findall(out)
        if found:
            if agent is not None:
                agent._pn_notes = found[-1].strip()
            STATS["notes"] += 1
            res["stdout"] = _NOTES_RE.sub("(notes saved: shown to you every turn)\n", out)
            out = res["stdout"]
        STATS["ok"] += out.count("PREDICT ✓")
        STATS["miss"] += out.count("PREDICT ✗")
        if "REFUSED (no move spent)" in str(res.get("error", "") or ""):
            STATS["refused"] += 1
        tot = STATS["ok"] + STATS["miss"]
        if tot and tot % 200 == 0:
            print("NEXTFORK PREDICT: предсказаний %d, промахов %d (%.0f%%), отказов %d, заметок %d" % (
                tot, STATS["miss"], 100 * STATS["miss"] / tot, STATS["refused"], STATS["notes"]), file=sys.__stderr__, flush=True)
    except Exception as exc:  # noqa: BLE001
        print("NEXTFORK PREDICT: сбой разбора вывода: %r" % (exc,), file=sys.__stderr__, flush=True)
    return res


_ta.run_sandboxed_python = _sandbox

_orig_run = _ta.ToolAgent._run_python_tool


def _run(self, state_path, arguments):
    key = str(getattr(state_path, "parent", state_path))
    if self.__dict__.get("_pn_game") != key:
        self._pn_game = key
        self._pn_notes = ""
    _TLS.agent = self
    try:
        return _orig_run(self, state_path, arguments)
    finally:
        _TLS.agent = None


_ta.ToolAgent._run_python_tool = _run

# --- страница заметок в каждом сообщении ---
_orig_prompt = _ta.ToolAgent._build_user_prompt


def _prompt(self, action_num, **kw):
    text = _orig_prompt(self, action_num, **kw)
    page = str(self.__dict__.get("_pn_notes", "") or "")
    if page:
        return text + "\n\nYOUR NOTES (update with notes(text) after every ✗ and at each new level):\n" + page
    return text + "\n\nYou have no notes yet: after your first moves, write one page with notes(text) (Verified / Assumed / Plan)."


_ta.ToolAgent._build_user_prompt = _prompt
print("NEXTFORK PREDICT: предсказание перед каждым ходом и страница заметок", flush=True)

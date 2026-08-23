"""Exercise the atlas A4 action-effect summary in tool_agent.py.

Backlog A4: the transition history already contains everything needed to
answer "what does each action change" -- nobody was aggregating it, so the
model re-derives the obvious (which cells are HUD, which action moves what)
by trial and error every episode. This computes the summary from
history_entries directly (host-side, zero model/tool cost) and checks it
against a synthetic board with a known HUD cell, a known-effect action, a
no-op action, and MOUSE clicks with a known relative effect.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "tufa-arc-agi-framework" / "src"))
sys.path.insert(0, str(ROOT / "atlas_src" / "src" / "ARC3-Inference"))

from inference.agent.runtime_state import Frame, HistoryEntry  # noqa: E402
from inference.agent.tool_agent import _atlas_action_effect_summary  # noqa: E402


def _fail(name: str, detail: str) -> None:
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def _ok(name: str) -> None:
    print(f"ok   {name}")


def _grid(base: int, hud: int) -> list[list[int]]:
    g = [[base] * 4 for _ in range(4)]
    g[0][0] = hud  # HUD cell: toggles every transition regardless of action
    return g


def _frame(grid: list[list[int]], step: int, level: int = 1) -> Frame:
    return Frame(grid=tuple(tuple(row) for row in grid), step=step, level=level)


def main() -> None:
    current = _grid(0, 0)
    entries = [HistoryEntry(action="", frame=_frame(current, step=0))]
    hud_toggle = 0
    up_toggle = 0
    step = 1

    def _push(action: str, mutate=lambda g: None) -> None:
        nonlocal hud_toggle, step
        mutate(current)
        hud_toggle = 1 - hud_toggle
        current[0][0] = hud_toggle
        entries.append(HistoryEntry(action=action, frame=_frame(current, step=step)))
        step += 1

    # 1. Too few transitions -> empty summary.
    early = entries[:2]
    if _atlas_action_effect_summary(early) != []:
        _fail("too few transitions", str(_atlas_action_effect_summary(early)))
    _ok("stays silent below the minimum transition count")

    def _toggle_up(g: list[list[int]]) -> None:
        nonlocal up_toggle
        up_toggle = 1 - up_toggle
        g[2][2] = 9 if up_toggle else 0

    # UP: genuinely flips cell (2,2) every single call -- a real, repeated effect.
    for _ in range(3):
        _push("UP", _toggle_up)

    # DOWN: a true no-op besides the HUD toggle -- (2,2) is left exactly as UP left it.
    for _ in range(3):
        _push("DOWN")

    # MOUSE clicks: each one flips the cell exactly at the click point, nothing else.
    for row, col in ((1, 1), (2, 2), (1, 3)):
        _push(f"MOUSE(row={row}, col={col})", lambda g, r=row, c=col: g.__setitem__(r, g[r][:c] + [5] + g[r][c + 1:]))

    lines = _atlas_action_effect_summary(entries)
    text = "\n".join(lines)
    if not lines:
        _fail("summary produced", "expected non-empty summary once enough transitions exist")
    _ok(f"produces a summary once enough transitions exist ({len(lines)} lines)")

    if "UP (3x): avg 1.0 cell(s) change, mostly within rows 2..2, cols 2..2." not in text:
        _fail("UP effect detected", text)
    _ok("UP's consistent effect on (2,2) alone is captured, HUD cell correctly excluded")

    if "DOWN (3x): no cell ever changed" not in text:
        _fail("DOWN is a true no-op once HUD is excluded", text)
    _ok("DOWN is reported as a genuine no-op once the HUD cell is excluded")

    if "MOUSE (3 clicks)" not in text or "rows 0..0, cols 0..0" not in text:
        _fail("MOUSE relative effect", text)
    _ok("MOUSE clicks report a tight RELATIVE bbox (each click only moved its own cell)")

    if "likely a HUD/timer element" not in text or "excluded from the stats above" not in text or "rows [0]" not in text:
        _fail("HUD cell flagged", text)
    _ok("the always-toggling cell (0,0) is flagged as likely HUD, not gameplay")

    # Invariants (Г): the union of every cell that EVER changed, across all
    # actions including HUD, is (0,0), (2,2), (1,1), (1,3) on a 4x4=16 board
    # -> bbox rows 0..2, cols 0..3; 16-4=12 cells (75%) never changed at all.
    if "12/16 cell(s) (75%) never changed" not in text or "rows 0..2, cols 0..3" not in text:
        _fail("invariant region reported", text)
    _ok("reports the static/invariant region as the bbox of everything that ever changed")

    print("\nAll atlas action-effect summary checks passed.")


if __name__ == "__main__":
    main()

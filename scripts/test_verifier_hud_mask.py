"""The verifier must not fail a correct theory over a ticking HUD (22.08).

Found in Kaggle v35: vc33 scored accuracy 0.0 over 27 transitions while every
one of its 15 counterexamples was one or two cells wide and every single wrong
cell sat in row 0 -- the HUD strip that ticks once per action. The model had
the mechanics right and the clock wrong, and exact-grid matching called that a
total failure.

That is not a cosmetic score problem. Zero accuracy keeps verify_gate_open()
false, the gate rejects action() batches longer than 3, and the agent is forced
into one-probe-per-turn -- the passivity we spent the day attributing to the
model. It also keeps THEORY_CHECKPOINT nagging "refine your theory" when there
is nothing left to refine.

Run:  .venv/Scripts/python.exe scripts/test_verifier_hud_mask.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

from agent.harness.sandbox import Sandbox

SIZE = 16
DOT = 3


def check(cond: bool, msg: str) -> None:
    print(f"[{'OK' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def build(n_transitions: int = 10, hud: bool = True) -> Sandbox:
    """A dot that steps right, plus (optionally) a HUD cell ticking every turn."""
    sb = Sandbox(env_step=lambda *a: None, budget_left=lambda: 100)
    grid = np.zeros((SIZE, SIZE), dtype=np.int8)
    grid[8, 0] = DOT
    if hud:
        grid[0, 15] = 4
    for i in range(n_transitions):
        after = grid.copy()
        after[8, i] = 0
        after[8, i + 1] = DOT
        if hud:
            # Cycle through colours that are never DOT: otherwise the HUD cell
            # occasionally equals the dot's colour and np.where(g == DOT) picks
            # it up first, which breaks the fixture rather than the code.
            after[0, 15] = (i % 5) + 4
        sb.transition_log.append((grid, "RIGHT", None, after))
        grid = after
    return sb


def theory_ignoring_hud(g, action, data=None):
    """Correct about the game, silent about the clock."""
    out = g.copy()
    rs, cs = np.where(g == DOT)
    r, c = int(rs[0]), int(cs[0])
    out[r, c] = 0
    out[r, min(c + 1, SIZE - 1)] = DOT
    return out


def main() -> None:
    # --- the exact v35 situation --------------------------------------------
    sb = build(hud=True)
    res = sb._verify_theory(theory_ignoring_hud)
    check(res["accuracy"] == 1.0,
          f"a theory right about the game scores 1.0 despite the ticking HUD (got {res['accuracy']})")
    check(res.get("ignored_ticking_cells", 0) >= 1, "the ticking cell is reported as ignored")
    check(res.get("ignored_rows") == [0], f"and its row is named: {res.get('ignored_rows')}")
    check("do not try to predict them" in res.get("note", ""), "the model is told not to model it")
    check(sb.verify_gate_open(), "so the action gate finally opens")

    # --- without a HUD nothing changes --------------------------------------
    sb = build(hud=False)
    res = sb._verify_theory(theory_ignoring_hud)
    check(res["accuracy"] == 1.0, "a clean board still scores 1.0")
    check("ignored_ticking_cells" not in res, "and nothing is reported as ignored")

    # --- a genuinely WRONG theory is still caught ---------------------------
    def wrong(g, action, data=None):
        return g  # claims the dot never moves

    sb = build(hud=True)
    res = sb._verify_theory(wrong)
    check(res["accuracy"] == 0.0, "the mask does not rescue a theory that is actually wrong")
    check(res["counterexamples"], "and counterexamples are still produced")
    ce = res["counterexamples"][0]
    cells = [c for c in ce.get("sample (row,col,predicted,actual)", [])]
    check(all(r != 0 for r, *_ in cells),
          f"counterexamples no longer point at the forgiven clock: {cells}")

    # --- too little evidence: no mask, old behaviour ------------------------
    sb = build(n_transitions=3, hud=True)
    res = sb._verify_theory(theory_ignoring_hud)
    check("ignored_ticking_cells" not in res,
          "under 6 transitions nothing is masked -- too little evidence to call a cell a clock")

    # --- a board where EVERYTHING moves must not be masked away -------------
    sb = Sandbox(env_step=lambda *a: None, budget_left=lambda: 100)
    rng = np.random.default_rng(0)
    g = rng.integers(0, 9, (SIZE, SIZE), dtype=np.int8)
    for _ in range(10):
        after = rng.integers(0, 9, (SIZE, SIZE), dtype=np.int8)
        sb.transition_log.append((g, "RIGHT", None, after))
        g = after
    check(sb._clocklike_cells() is None,
          "a board where every cell churns yields no mask (else any theory would look perfect)")

    print("\nThe verifier forgives clocks and still catches wrong theories.")


if __name__ == "__main__":
    main()

def test_depleting_bar() -> None:
    """The vc33 case: a timer BAR whose cells each change exactly once.

    No "changes every turn" mask can catch it -- cell (0,63) flips 7->4 once
    and never again -- yet it failed every transition under exact matching and
    held the action gate shut across v35 and v36. Tolerating a couple of wrong
    cells needs no guess about where a game draws its HUD.
    """
    sb = Sandbox(env_step=lambda *a: None, budget_left=lambda: 100)
    grid = np.zeros((SIZE, SIZE), dtype=np.int8)
    grid[8, 0] = DOT
    grid[0, 8:] = 7                       # a full timer bar
    for i in range(10):
        after = grid.copy()
        after[8, i] = 0
        after[8, i + 1] = DOT
        after[0, SIZE - 1 - i] = 4        # the bar depletes one cell per action
        sb.transition_log.append((grid, "RIGHT", None, after))
        grid = after

    check(sb._clocklike_cells() is None,
          "no cell ticks every turn, so the mask correctly finds nothing")
    res = sb._verify_theory(theory_ignoring_hud)
    check(res["exact_accuracy"] == 0.0, "strict matching still calls it 0.0 -- as it did on vc33")
    check(res["accuracy"] == 1.0, f"tolerance recognises the theory as working (got {res['accuracy']})")
    check(res.get("ignored_ticking_cells", 0) >= 1, "the bar cells are reported as ignored")
    check(sb.verify_gate_open(), "so the action gate opens and the agent may act in batches")

    # A theory wrong about the MECHANICS is still caught, bar or no bar.
    def wrong(g, action, data=None):
        out = g.copy()
        out[0, SIZE - 1] = 4              # only ever touches the HUD
        return out

    res = sb._verify_theory(wrong)
    check(res["accuracy"] == 0.0, "a theory that ignores the dot is still rejected")


test_depleting_bar()


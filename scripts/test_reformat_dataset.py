"""Unit test for build_sft_dataset.py's reformatting filters (22.08 backlog item):
drop fruitless-inspection turns + gate-dodge dummy predicts, upweight decisive turns.
See docs/student_vs_teacher_analysis.md #5.2 for the rationale.

Run:  .venv/Scripts/python.exe scripts/test_reformat_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_sft_dataset import is_dummy_predict, is_fruitless_inspection, is_decisive  # noqa: E402


def check(cond: bool, msg: str) -> None:
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        raise SystemExit(1)


def main() -> None:
    # is_dummy_predict: bare no-op body -> dummy; body with real logic -> not dummy.
    check(is_dummy_predict("def predict(grid, act, data):\n    return grid.copy()\n"),
          "bare 'return grid.copy()' flagged as dummy")
    check(is_dummy_predict("def predict(g, act, data):\n    # comment\n    return g\n"),
          "bare 'return g' (with comment) flagged as dummy")
    check(not is_dummy_predict(
        "def predict(grid, act, data):\n"
        "    g = grid.copy()\n"
        "    if act == 'UP':\n"
        "        g[:-1] = g[1:]\n"
        "    return g\n"),
        "predict with real conditional logic NOT flagged as dummy")
    check(not is_dummy_predict("print('no predict here')\n"),
          "code without a predict() def is not flagged")

    turns = [
        {"actions_executed": 0, "level": 0},  # 0: inspection, next also 0 actions -> fruitless
        {"actions_executed": 0, "level": 0},  # 1: inspection, next acts -> not fruitless
        {"actions_executed": 1, "level": 0},  # 2: single action, no level-up soon, not decisive
        {"actions_executed": 3, "level": 0},  # 3: batch action -> decisive
        {"actions_executed": 1, "level": 0},  # 4: level-up 2 turns later -> decisive
        {"actions_executed": 1, "level": 0},  # 5: filler
        {"actions_executed": 1, "level": 1},  # 6: level-up happened here
    ]
    check(is_fruitless_inspection(turns, 0), "turn 0 (0 actions, next also 0 actions) is fruitless")
    check(not is_fruitless_inspection(turns, 1), "turn 1 (0 actions, next acts) is NOT fruitless")
    check(not is_fruitless_inspection(turns, 2), "turn 2 (acted) is never fruitless")

    trailing_zero = turns + [{"actions_executed": 0, "level": 1}]
    check(is_fruitless_inspection(trailing_zero, len(trailing_zero) - 1),
          "trailing 0-action turn with no next turn is fruitless")

    check(not is_decisive(turns, 2), "single action, no level-up within 2 turns -> not decisive")
    check(is_decisive(turns, 3), "batch action (>=2) -> decisive")
    check(is_decisive(turns, 4), "level-up 2 turns later -> decisive")
    check(is_decisive(turns, 5), "level-up 1 turn later -> decisive")

    print("All reformat-dataset filter tests passed.")


if __name__ == "__main__":
    main()

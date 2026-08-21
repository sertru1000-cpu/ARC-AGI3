"""Unit test for segment()'s adjacency/children priors (backlog item 4, 20.08).

Run:  .venv/Scripts/python.exe scripts/test_perception_structure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.harness.perception import segment


def obj_at(seg, r, c):
    """Find the object id owning cell (r, c) via bbox+color match (test helper)."""
    for o in seg.objects:
        r0, c0, r1, c1 = o.bbox
        if r0 <= r <= r1 and c0 <= c <= c1 and (r, c) in o.mask_rc:
            return o.id
    raise AssertionError(f"no object found at ({r},{c})")


def main() -> None:
    # ── adjacency: two touching blocks + one isolated block ────────────────
    grid = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [0, 0, 0, 0],
        [3, 3, 0, 0],
        [3, 3, 0, 0],
    ], dtype=np.int8)
    seg = segment(grid)
    a1 = obj_at(seg, 0, 0)  # color 1
    a2 = obj_at(seg, 0, 2)  # color 2, touches color 1 at the col1/col2 border
    a3 = obj_at(seg, 3, 0)  # color 3, separated from 1/2 by a background row

    assert a2 in seg.adjacency[a1], f"expected obj {a2} adjacent to {a1}: {seg.adjacency}"
    assert a1 in seg.adjacency[a2], "adjacency must be symmetric"
    assert a3 not in seg.adjacency[a1], "obj3 is separated by background, must not be adjacent to obj1"
    assert a2 not in seg.adjacency[a3], "obj3 is separated by background, must not be adjacent to obj2"
    print("adjacency OK:", {k: v for k, v in seg.adjacency.items() if k in (a1, a2, a3)})

    # ── children: two levels of nesting (frame -> ring -> center dot) ──────
    grid2 = np.array([
        [5, 5, 5, 5, 5],
        [5, 0, 0, 0, 5],
        [5, 0, 3, 0, 5],
        [5, 0, 0, 0, 5],
        [5, 5, 5, 5, 5],
    ], dtype=np.int8)
    seg2 = segment(grid2)
    frame = obj_at(seg2, 0, 0)   # color 5, outer perimeter
    ring = obj_at(seg2, 1, 1)    # color 0, inner ring
    dot = obj_at(seg2, 2, 2)     # color 3, center cell

    assert ring in seg2.children[frame], f"expected {ring} as direct child of frame {frame}: {seg2.children}"
    assert dot in seg2.children[ring], f"expected {dot} as direct child of ring {ring}: {seg2.children}"
    assert dot not in seg2.children[frame], (
        "dot's DIRECT parent is the ring, not the frame -- frame's children "
        f"list must not skip the intermediate level: {seg2.children}")
    print("children OK: frame ->", seg2.children[frame], "; ring ->", seg2.children[ring])

    print("\nALL PERCEPTION-STRUCTURE CHECKS PASSED")


if __name__ == "__main__":
    main()

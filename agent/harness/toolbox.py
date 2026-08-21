"""Optional solver helpers injected into the model's sandbox.

Kept deliberately small and generic (Duck's lesson: heavy bespoke tooling
hurts — the model should stay free to improvise). These three cover the
patterns that burn the most model-written code: pathfinding, reachability,
object lookup.
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Iterable

import numpy as np

from .perception import segment

# Move name <-> row/col delta (matches ACTION1..4 semantics: UP/DOWN/LEFT/RIGHT).
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


def _passable_fn(passable) -> Callable[[int], bool]:
    if callable(passable):
        return passable
    allowed = {int(c) for c in (passable if isinstance(passable, Iterable) else [passable])}
    return lambda color: int(color) in allowed


def bfs_path(grid: np.ndarray, start_rc, goal_rc, passable) -> list[str] | None:
    """Shortest 4-dir path start->goal over cells whose color is passable.

    passable: a color, an iterable of colors, or a function color->bool.
    Returns a list like ['UP','UP','RIGHT'] or None if unreachable.
    Start/goal cells themselves are not tested for passability.
    """
    h, w = grid.shape
    ok = _passable_fn(passable)
    sr, sc = int(start_rc[0]), int(start_rc[1])
    gr, gc = int(goal_rc[0]), int(goal_rc[1])
    prev: dict[tuple[int, int], tuple[tuple[int, int], str]] = {}
    seen = {(sr, sc)}
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            path: list[str] = []
            cur = (r, c)
            while cur != (sr, sc):
                cur, move = prev[cur][0], prev[cur][1]
                path.append(move)
            return path[::-1]
        for move, (dr, dc) in MOVES.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen:
                if (nr, nc) == (gr, gc) or ok(grid[nr, nc]):
                    seen.add((nr, nc))
                    prev[(nr, nc)] = ((r, c), move)
                    q.append((nr, nc))
    return None


def reachable(grid: np.ndarray, start_rc, passable) -> set[tuple[int, int]]:
    """All cells reachable from start via 4-dir moves over passable colors."""
    h, w = grid.shape
    ok = _passable_fn(passable)
    sr, sc = int(start_rc[0]), int(start_rc[1])
    seen = {(sr, sc)}
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in MOVES.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and ok(grid[nr, nc]):
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def objects(grid: np.ndarray) -> list[dict]:
    """Connected same-color components with masks (superset of
    current_frame.segmentation nodes: adds 'cells' list of (row, col))."""
    seg = segment(grid)
    out = []
    for o in seg.objects:
        out.append({
            "id": o.id, "color": int(o.color), "pixels": o.cells,
            "bbox": o.bbox, "centroid": o.centroid, "hash": o.shape_hash,
            "touches_border": o.touches_border, "cells": list(o.mask_rc),
        })
    return out

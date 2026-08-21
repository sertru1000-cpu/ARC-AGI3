"""Grid perception utilities: diffs, hashing, connected-component segmentation.

Pure functions over 2-D numpy int grids (values 0..15). A FrameData.frame is a
list of grids (a short animation); callers should pass the *last* grid of the
frame via `latest_grid`.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def latest_grid(frame_data: Any) -> np.ndarray | None:
    """Extract the final animation grid of a FrameData as a numpy array."""
    if not getattr(frame_data, "frame", None):
        return None
    return np.asarray(frame_data.frame[-1], dtype=np.int8)


def frame_hash(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes()).hexdigest()[:16]


def background_color(grid: np.ndarray) -> int:
    """Most frequent color — the usual (but not guaranteed) background."""
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


@dataclass
class GridDiff:
    """Compact summary of what changed between two grids."""

    changed_cells: int
    bbox: tuple[int, int, int, int] | None  # (r0, c0, r1, c1) inclusive
    color_moves: list[tuple[int, int, int]]  # (from_color, to_color, count)
    border_only: bool  # True if all changes hug the outer 2-cell border (HUD?)

    @property
    def changed(self) -> bool:
        return self.changed_cells > 0


def grid_diff(before: np.ndarray, after: np.ndarray) -> GridDiff:
    if before.shape != after.shape:
        return GridDiff(int(after.size), (0, 0, *after.shape), [], False)
    mask = before != after
    n = int(mask.sum())
    if n == 0:
        return GridDiff(0, None, [], False)
    rows, cols = np.nonzero(mask)
    bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))
    moves = Counter(zip(before[mask].tolist(), after[mask].tolist()))
    color_moves = [(int(f), int(t), c) for (f, t), c in moves.most_common(8)]
    h, w = before.shape
    border = 2
    border_only = bool(
        (rows.min() >= h - border or rows.max() < border)
        or (cols.min() >= w - border or cols.max() < border)
    )
    return GridDiff(n, bbox, color_moves, border_only)


@dataclass
class GridObject:
    """One 4-connected same-color component."""

    id: int
    color: int
    cells: int
    bbox: tuple[int, int, int, int]  # (r0, c0, r1, c1) inclusive
    centroid: tuple[float, float]
    shape_hash: str  # position-invariant: same shape+color anywhere → same hash
    touches_border: bool
    mask_rc: list[tuple[int, int]] = field(repr=False, default_factory=list)


@dataclass
class Segmentation:
    objects: list[GridObject]
    background: int
    # Chollet-style structural priors, computed alongside the components so
    # the model doesn't have to re-derive them in sandbox code every turn.
    adjacency: dict[int, list[int]] = field(default_factory=dict)  # id -> touching object ids
    children: dict[int, list[int]] = field(default_factory=dict)  # id -> directly-nested object ids

    def by_color(self, color: int) -> list[GridObject]:
        return [o for o in self.objects if o.color == color]

    def non_background(self) -> list[GridObject]:
        return [o for o in self.objects if o.color != self.background]


def _adjacency(labels: np.ndarray, n_objects: int) -> dict[int, list[int]]:
    """Which objects share a 4-connected border, regardless of color."""
    h, w = labels.shape
    pairs: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            oid = int(labels[r, c])
            if oid < 0:
                continue  # unlabeled: only happens past the max_objects cutoff
            for nr, nc in ((r + 1, c), (r, c + 1)):  # each edge counted once
                if nr < h and nc < w:
                    nid = int(labels[nr, nc])
                    if nid != oid and nid >= 0:
                        pairs.add((min(oid, nid), max(oid, nid)))
    adj: dict[int, list[int]] = {i: [] for i in range(n_objects)}
    for a, b in pairs:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _children(objects: list[GridObject]) -> dict[int, list[int]]:
    """Direct nesting: obj B is a direct child of A if A's bbox strictly
    contains B's bbox and no smaller object's bbox also contains B's."""
    children: dict[int, list[int]] = {o.id: [] for o in objects}
    for obj in objects:
        r0, c0, r1, c1 = obj.bbox
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        best_parent, best_area = None, None
        for other in objects:
            if other.id == obj.id:
                continue
            or0, oc0, or1, oc1 = other.bbox
            other_area = (or1 - or0 + 1) * (oc1 - oc0 + 1)
            contains = or0 <= r0 and oc0 <= c0 and or1 >= r1 and oc1 >= c1
            if contains and other_area > area and (best_area is None or other_area < best_area):
                best_parent, best_area = other.id, other_area
        if best_parent is not None:
            children[best_parent].append(obj.id)
    return children


def segment(grid: np.ndarray, max_objects: int = 256) -> Segmentation:
    """4-connected flood-fill segmentation, top-left scan order."""
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    objects: list[GridObject] = []
    bg = background_color(grid)

    for r in range(h):
        for c in range(w):
            if labels[r, c] != -1:
                continue
            color = int(grid[r, c])
            oid = len(objects)
            stack = [(r, c)]
            labels[r, c] = oid
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == -1 and grid[nr, nc] == color:
                        labels[nr, nc] = oid
                        stack.append((nr, nc))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            norm = tuple(sorted((p[0] - r0, p[1] - c0) for p in cells))
            shash = hashlib.md5(f"{color}:{norm}".encode()).hexdigest()[:12]
            objects.append(
                GridObject(
                    id=oid,
                    color=color,
                    cells=len(cells),
                    bbox=(r0, c0, r1, c1),
                    centroid=(sum(rs) / len(rs), sum(cs) / len(cs)),
                    shape_hash=shash,
                    touches_border=(r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1),
                    mask_rc=cells,
                )
            )
            if len(objects) >= max_objects:
                return Segmentation(objects, bg, _adjacency(labels, len(objects)), _children(objects))
    return Segmentation(objects, bg, _adjacency(labels, len(objects)), _children(objects))


def salient_click_targets(grid: np.ndarray, limit: int = 12) -> list[tuple[int, int]]:
    """Candidate (x, y) click points: centers of small, rare-colored objects.

    Heuristic saliency: small non-background components whose color is scarce
    on the board rank first — these are the most likely interactive elements.
    """
    seg = segment(grid)
    color_area = Counter()
    for o in seg.objects:
        color_area[o.color] += o.cells
    total = grid.size

    def score(o: GridObject) -> float:
        rarity = 1.0 - color_area[o.color] / total
        smallness = 1.0 / (1 + o.cells)
        edge_penalty = 0.5 if o.touches_border else 1.0
        return rarity * smallness * edge_penalty

    ranked = sorted(seg.non_background(), key=score, reverse=True)
    out: list[tuple[int, int]] = []
    for o in ranked[:limit]:
        r, c = int(round(o.centroid[0])), int(round(o.centroid[1]))
        # ACTION6 payload is {"x": col, "y": row}
        out.append((c, r))
    return out


@dataclass
class Zone:
    """One structural region of the board (play area, panel, legend...)."""

    label: str
    bbox: tuple[int, int, int, int]  # (r0, c0, r1, c1) inclusive
    colors: list[int]                # distinct non-background colors inside
    filled_cells: int

    def contains(self, r: int, c: int) -> bool:
        r0, c0, r1, c1 = self.bbox
        return r0 <= r <= r1 and c0 <= c <= c1


def detect_zones(grid: np.ndarray, max_zones: int = 8) -> list[Zone]:
    """Split the board into rectangular zones along separator bands.

    A separator is a full row/column that is entirely background. Games in
    this suite leave background gutters between play area / control panel /
    reference legend, so this cheap geometry recovers the screen layout
    without any learning. (Solid non-background lines are NOT separators:
    a solid content block would otherwise dissolve into them.)
    """
    bg = background_color(grid)

    def is_sep(line: np.ndarray) -> bool:
        return bool((line == bg).all())

    def bands(axis_len: int, take) -> list[tuple[int, int]]:
        out, start = [], None
        for i in range(axis_len):
            if is_sep(take(i)):
                if start is not None:
                    out.append((start, i - 1))
                    start = None
            elif start is None:
                start = i
        if start is not None:
            out.append((start, axis_len - 1))
        return out

    zones: list[Zone] = []
    for r0, r1 in bands(grid.shape[0], lambda i: grid[i, :]):
        sub = grid[r0:r1 + 1, :]
        for c0, c1 in bands(grid.shape[1], lambda i, s=sub: s[:, i]):
            block = grid[r0:r1 + 1, c0:c1 + 1]
            colors = [int(c) for c in np.unique(block) if c != bg]
            filled = int((block != bg).sum())
            if colors and filled >= 4:  # skip empty/near-empty slivers
                zones.append(Zone("", (r0, c0, r1, c1), colors, filled))
    zones.sort(key=lambda z: -z.filled_cells)
    zones = zones[:max_zones]
    for i, z in enumerate(zones):
        z.label = chr(ord("A") + i)
    return zones


def zones_summary(zones: list[Zone]) -> str:
    if len(zones) < 2:
        return ""
    lines = [f"Board layout: {len(zones)} zones (biggest first):"]
    for z in zones:
        r0, c0, r1, c1 = z.bbox
        lines.append(
            f"- zone {z.label}: rows {r0}-{r1}, cols {c0}-{c1} "
            f"({r1 - r0 + 1}x{c1 - c0 + 1}), colors {z.colors}")
    lines.append(
        "The biggest zone is usually the play area; small side/bottom zones "
        "are usually control panels, references/legends, or progress bars — "
        "figure out which is which and how actions in one affect another.")
    return "\n".join(lines)

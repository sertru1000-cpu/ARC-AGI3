"""Структурный разбор последнего перехода — то, что в опыте Б кладётся во ВХОД промпта.

ЗАЧЕМ. 36% рассуждения модели — пересказ доски словами (координаты, узлы, что сдвинулось),
а входной токен дешевле выходного в 2317 раз. Разбор строится нашим кодом ТОЧНО по записанным
доскам до и после исполненной пачки действий, и модель просят не пересказывать его прозой.

ЧТО В РАЗБОРЕ. Сколько клеток изменилось и в какой рамке; объекты (4-связные одноцветные
компоненты, фон — самый частый цвет) сопоставляются по подписи «цвет + форма»: сдвинулся
(на сколько), появился, исчез, изменил форму. Цвета — буквами из легенды системного промпта,
которые модель видит в `current_frame.ascii`; соответствие букв числам сверяется с
`board_ascii` из журнала, при расхождении скрипт падает.
"""
from __future__ import annotations

import collections
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_replay_parse import parse  # noqa: E402

LETTERS = "WwgGcBMPRbSYOrNp"   # легенда системного промпта в порядке цветов 0..15
MAX_OBJ = 8
INSTRUCTION = (
    "Use this diff directly as ground truth for what the last action sequence changed. "
    "Do NOT restate coordinates, re-list objects, or re-describe the board in your reasoning — "
    "spend reasoning only on hypotheses about the mechanics and on choosing the next action.")


def comps(board, bg):
    h, w = len(board), len(board[0])
    seen = [[False] * w for _ in range(h)]
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or board[r][c] == bg:
                continue
            col = board[r][c]
            stack, cells = [(r, c)], []
            seen[r][c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and board[ny][nx] == col:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            r0 = min(y for y, _ in cells); c0 = min(x for _, x in cells)
            shape = tuple(sorted((y - r0, x - c0) for y, x in cells))
            hh = max(y for y, _ in cells) - r0 + 1; ww = max(x for _, x in cells) - c0 + 1
            out.append({"col": col, "shape": shape, "pos": (r0, c0), "size": (hh, ww),
                        "cells": set(cells)})
    return out


def describe(o):
    return "%s %dx%d%s" % (LETTERS[o["col"]], o["size"][0], o["size"][1],
                           "" if len(o["shape"]) == o["size"][0] * o["size"][1] else " shape")


def diff(before, after, executed, level_changed=False):
    changed = [(r, c) for r in range(len(after)) for c in range(len(after[0])) if before[r][c] != after[r][c]]
    lines = ["HARNESS DIFF (computed exactly from the recorded frames before/after your last action sequence):",
             "- executed: %s" % (", ".join(executed) if executed else "none")]
    if level_changed:
        lines.append("- level changed: the board was replaced by a new level, per-object diff omitted")
        return "\n".join(lines + [INSTRUCTION])
    if not changed:
        lines.append("- board changed: NO — the sequence was a complete no-op on the board")
        return "\n".join(lines + [INSTRUCTION])
    rs = [r for r, _ in changed]; cs = [c for _, c in changed]
    lines.append("- board changed: %d cells, rows %d-%d, cols %d-%d" % (len(changed), min(rs), max(rs), min(cs), max(cs)))
    flat = [v for row in after for v in row]
    bg = collections.Counter(flat).most_common(1)[0][0]
    ch = set(changed)
    A = [o for o in comps(before, bg) if o["cells"] & ch]
    B = [o for o in comps(after, bg) if o["cells"] & ch]
    moved, appeared, gone = [], [], []
    used = set()
    for a in A:
        j = next((i for i, b in enumerate(B) if i not in used and b["col"] == a["col"] and b["shape"] == a["shape"]), None)
        if j is not None:
            used.add(j); b = B[j]
            dr, dc = b["pos"][0] - a["pos"][0], b["pos"][1] - a["pos"][1]
            if (dr, dc) != (0, 0):
                moved.append("%s at (%d,%d) -> (%d,%d) [%+d,%+d]" % (describe(a), *a["pos"], *b["pos"], dr, dc))
        else:
            gone.append("%s at (%d,%d)" % (describe(a), *a["pos"]))
    for i, b in enumerate(B):
        if i not in used:
            appeared.append("%s at (%d,%d)" % (describe(b), *b["pos"]))
    for label, items in (("moved", moved), ("appeared or reshaped", appeared), ("disappeared or reshaped", gone)):
        if items:
            extra = " (+%d more)" % (len(items) - MAX_OBJ) if len(items) > MAX_OBJ else ""
            lines.append("- %s: %s%s" % (label, "; ".join(items[:MAX_OBJ]), extra))
    return "\n".join(lines + [INSTRUCTION])


def boards_for(snapshot_path):
    """(доска до пачки, доска после, исполненные действия, сменился ли уровень) для снимка."""
    snap = parse(snapshot_path)
    gid = Path(snapshot_path).name[:4]
    run = Path(snapshot_path).parent.parent
    ev = [json.loads(l) for l in open(glob.glob(str(run / "artifacts" / f"{gid}*_events.jsonl"))[0], encoding="utf-8")]
    init = next(d for d in ev if d.get("type") == "initial")
    acts = {d["action_num"]: d for d in ev if d.get("type") == "action"}
    last_user = snap["messages"][-1]["content"]
    k = re.search(r"code executed (\d+) action", last_user)
    k = int(k.group(1)) if k else 0
    names = re.search(r"Executed actions(?: \(first 10\))?: (.*)\.", last_user)
    executed = [s.strip() for s in names.group(1).split(",")] if names and names.group(1) != "none" else []
    a_now = int(snap["meta"]["action"]) - 1
    after = acts[a_now] if a_now in acts else (acts[max(acts)] if acts else init)
    b_idx = a_now - k
    before = acts.get(b_idx, init) if b_idx >= 1 else init
    # сверка букв с board_ascii — модель видит именно эти буквы
    asc = after.get("board_ascii")
    if asc:
        rows = asc.split("\n")
        for r in range(0, 64, 9):
            for c in range(0, 64, 9):
                if r < len(rows) and c < len(rows[r]):
                    assert rows[r][c] == LETTERS[after["board"][r][c]], "буквы цветов не совпали с board_ascii"
    return snap, before["board"], after["board"], executed, before.get("level") != after.get("level")


if __name__ == "__main__":
    tot = 0
    for f in sorted(glob.glob("runs/flash_v1_phaseA/prompts/*.log")):
        snap, b, a, ex, lvl = boards_for(f)
        d = diff(b, a, ex, lvl)
        tot += len(d)
        print("=" * 8, Path(f).name[:4], "знаков %d" % len(d))
        print(d.split("\n- ", 1)[1].split(INSTRUCTION)[0].strip()[:300])
    print("\nсредний размер разбора: %.0f знаков (~%.0f токенов входа)" % (tot / 25, tot / 25 / 3.5))

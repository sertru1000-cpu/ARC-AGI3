"""Deterministic opening probe: press each simple action once, diff the board.

Runs BEFORE the first LLM turn. A handful of cheap engine actions buys the
model a ready-made "control legend" (what moved, what did nothing, what is
HUD noise), saving several expensive LLM turns of blind poking. Aborts early
on level change or game over; the speedrun phase erases the cost on wins.
"""
from __future__ import annotations

from typing import Any, Callable

from .perception import detect_zones, grid_diff, latest_grid, zones_summary
from .sandbox import FROM_ENGINE

PROBE_ORDER = [1, 2, 3, 4, 5, 7]  # engine ids: directions, interact, undo


def probe_actions(
    env_step: Callable[[int, dict | None], Any],
    first_frame: Any,
    available_ids: set[int],
) -> tuple[str, Any]:
    """Returns (summary_text, last_frame). Empty summary if nothing to probe."""
    ids = [a for a in PROBE_ORDER if a in available_ids]
    if not ids:
        return "", first_frame

    lines: list[str] = []
    prev_grid = latest_grid(first_frame)
    frame = first_frame
    level = int(first_frame.levels_completed or 0)
    zones = detect_zones(prev_grid) if prev_grid is not None else []

    def zone_tag(bbox: tuple[int, int, int, int] | None) -> str:
        if not bbox or not zones:
            return ""
        r0, c0, r1, c1 = bbox
        hit = sorted({z.label for z in zones
                      if z.contains(r0, c0) or z.contains(r1, c1)
                      or z.contains((r0 + r1) // 2, (c0 + c1) // 2)})
        return f" [zone {'/'.join(hit)}]" if hit else ""

    for aid in ids:
        frame = env_step(aid, None)
        grid = latest_grid(frame)
        state = str(getattr(frame, "state", "")).split(".")[-1]
        name = FROM_ENGINE.get(f"ACTION{aid}", f"ACTION{aid}")
        if prev_grid is not None and grid is not None:
            d = grid_diff(prev_grid, grid)
            if not d.changed:
                effect = "no visible change"
            elif d.border_only:
                effect = f"only border/HUD changed ({d.changed_cells} cells) — likely a timer tick"
            else:
                r0, c0, r1, c1 = d.bbox
                effect = (f"{d.changed_cells} cells changed in rows {r0}-{r1}, "
                          f"cols {c0}-{c1}{zone_tag(d.bbox)}")
        else:
            effect = "?"
        new_level = int(frame.levels_completed or 0)
        lines.append(f"- {name}: {effect}" + (f" [LEVEL UP -> {new_level}]" if new_level > level else "")
                     + (f" [{state}]" if state == "GAME_OVER" else ""))
        prev_grid = grid
        if state == "GAME_OVER":
            frame = env_step(0, None)  # reset the level, stop probing
            prev_grid = latest_grid(frame)
            lines.append("  (probe caused GAME_OVER; level was reset — treat that action as dangerous here)")
            break
        if new_level > level:
            break  # probe accidentally solved something; stop touching things

    head = zones_summary(zones)
    summary = (
        (head + "\n\n" if head else "")
        + "Automatic opening probe (each listed action was pressed once already; "
        "these actions are spent):\n" + "\n".join(lines)
    )
    return summary, frame

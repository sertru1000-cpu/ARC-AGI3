"""Карта эффектов на локальном движке: из старта уровня каждый ход алфавита (стрелки + клики по центрам объектов)
— что меняется на доске, кроме «часов» (клеток, меняющихся при любом ходе). usage: effect_map_local.py <game4>"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents")); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from replay_battle_local import play
from agent.harness.perception import segment
import logging; logging.disable(logging.WARNING)
import arc_agi
from arc_agi import OperationMode
g = sys.argv[1]
arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
gid = [e.game_id.split("-")[0] for e in arc.get_environments() if e.game_id.startswith(g)][0]
start = play(arc, gid, [{"name": "RESET", "payload": None}])[-1][0]
objs = segment(start).non_background()[:40]
alts = [{"name": n, "payload": None, "label": n} for n in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")]
alts += [{"name": "ACTION6", "payload": {"x": int(round(o.centroid[1])), "y": int(round(o.centroid[0]))}, "label": "click(%d,%d) color %d px %d" % (round(o.centroid[0]), round(o.centroid[1]), o.color, o.cells)} for o in objs]
diffs = []
for a in alts:
    tr = play(arc, gid, [{"name": "RESET", "payload": None}, {"name": a["name"], "payload": a["payload"]}])
    after = tr[-1][0]; d = (after != start)
    diffs.append((a["label"], d, tr[-1][1], tr[-1][2]))
# часы: клетки, меняющиеся в >= 80% ходов, у которых что-то менялось
changed = [d for _, d, _, _ in diffs if d.any()]
clock = np.zeros_like(start, dtype=bool)
if changed:
    freq = np.mean([d for d in changed], axis=0); clock = freq >= 0.8
print(f"{g}: старт уровня, действий проверено {len(alts)}, объектов {len(objs)}; клеток-часов {int(clock.sum())} (строки {sorted(set(np.where(clock)[0].tolist()))[:10]})")
for label, d, lvl, state in diffs:
    real = d & ~clock
    if real.any() or lvl > 0 or "GAME_OVER" in state:
        rows = sorted(set(np.where(real)[0].tolist())); cols = sorted(set(np.where(real)[1].tolist()))
        print(f"   {label:32s} изменилось клеток {int(real.sum()):4d} (строки {rows[:6]}, столбцы {cols[:6]}) уровень {lvl} {state.split('.')[-1]}")
print("   остальные действия меняют только часы или ничего")

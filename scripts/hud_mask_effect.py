"""Скрытые петли за счётчиками в кадре (26.09).

Клетки, меняющиеся почти на каждом ходу независимо от действия (таймер, шкала шагов), делают каждое
состояние «новым». Определяем такие клетки на уровне: доля переходов, где клетка изменилась, >= 0.6.
Считаем долю новых состояний до и после маскировки, и долю ходов, где доска изменилась ТОЛЬКО в них.
"""
import hashlib, json, logging, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"vendor/ARC-AGI-3-Agents"))
logging.disable(logging.ERROR)
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
def grid(fr): return None if fr is None or not fr.frame else np.asarray(fr.frame[-1],dtype=np.int16)
arc=arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT/"environment_files"))
bench=json.loads((ROOT/"runs/flash_v1_phaseA/benchmark.json").read_text())["game_runs"]
print("%-6s %6s %9s %11s %11s %13s" % ("игра","ходов","HUD-клеток","новых: до","новых: после","ходов «только HUD»"))
tot_before=tot_after=tot=0; hud_only=0
for gr in bench:
    env=arc.make(gr["game_id"]); fr=env.reset(); g=grid(fr)
    if g is None: continue
    lv=getattr(fr,"levels_completed",0) or 0; frames=[g]
    for rec in (gr.get("history") or []):
        a=rec.get("action") or {}
        if not a.get("id") or a["id"]=="RESET": continue
        try: fr2=env.step(GameAction[a["id"]], data=a.get("data"))
        except Exception: break
        g2=grid(fr2); lv2=(getattr(fr2,"levels_completed",lv) or lv) if fr2 is not None else lv
        if g2 is None or lv2!=lv or g2.shape!=g.shape: break      # только первый уровень
        frames.append(g2); g=g2
    if len(frames)<15: continue
    F=np.stack(frames); ch=(F[1:]!=F[:-1])                        # (T, H, W)
    rate=ch.mean(axis=0); hud=rate>=float(sys.argv[1]) if len(sys.argv)>1 else rate>=0.6
    def newfrac(mask):
        seen=set(); new=0
        for f in F:
            k=hashlib.blake2b((np.where(mask,0,f)).tobytes(),digest_size=8).hexdigest()
            new+= k not in seen; seen.add(k)
        return new/len(F)
    none=np.zeros_like(hud)
    b=newfrac(none); a2=newfrac(hud)
    only=int(sum(1 for t in range(ch.shape[0]) if ch[t].any() and not (ch[t]&~hud).any()))
    tot_before+=b*len(F); tot_after+=a2*len(F); tot+=len(F); hud_only+=only
    if hud.sum() or b-a2>0.05:
        print("%-6s %6d %9d %10.0f%% %10.0f%% %13d" % (gr["game_id"][:4], len(F)-1, int(hud.sum()), 100*b, 100*a2, only))
print("\nвсего (первые уровни, 25 игр): новых состояний до маски %.0f%%, после %.0f%%; ходов, менявших ТОЛЬКО HUD: %d из %d"
      % (100*tot_before/tot, 100*tot_after/tot, hud_only, tot-25))

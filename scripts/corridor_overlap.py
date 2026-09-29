"""Застрявшие прогоны: в ТЕХ ЖЕ состояниях, что успешные, или в другой области? (26.09, по критику)

Коридор игры = множество хешей доски, которые посещали УСПЕШНЫЕ прогоны на первом уровне.
Для застрявшего прогона считаем долю его состояний первого уровня, лежащих в коридоре, и первый
ход входа. Контроль: тот же счёт для каждого успешного прогона против коридора ОСТАЛЬНЫХ
успешных — без него нельзя понять, «мало» это или «как у всех» (хеш строгий, HUD-счётчики
разводят даже одинаковые по сути состояния).
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
RUNS=["flash_v1_phaseA","public_flash_tufa","public_flash_keithtyser","precond_v2","nostop_v2",
      "flash_combined","flash_goalrule","night_nextfork-b1","night_stock-flash","night_nextfork-b2",
      "night_stock-base2","night_nextfork-b4","night_stock-base4"]
UNSTABLE={"sk48","tn36","g50t","m0r0","sp80","dc22","sc25","cn04","ls20"}
def grid(fr): return None if fr is None or not fr.frame else np.asarray(fr.frame[-1],dtype=np.int16)
def H(g): return hashlib.blake2b(g.tobytes(),digest_size=8).hexdigest()
arc=arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT/"environment_files"))
traj=defaultdict(list)   # игра -> [(успех?, [хеши состояний на уровне 1], прогон)]
for n in RUNS:
    p=ROOT/"runs"/n/"benchmark.json"
    if not p.is_file(): continue
    for gr in json.loads(p.read_text())["game_runs"]:
        k=gr["game_id"][:4]
        if k not in UNSTABLE: continue
        env=arc.make(gr["game_id"]); fr=env.reset(); g=grid(fr)
        if g is None: continue
        lv=getattr(fr,"levels_completed",0) or 0; hs=[H(g)]
        for rec in (gr.get("history") or []):
            a=rec.get("action") or {}
            if not a.get("id"): continue
            try: fr2=env.step(GameAction[a["id"]] if a["id"]!="RESET" else GameAction.RESET, data=a.get("data"))
            except Exception: break
            g2=grid(fr2); lv2=(getattr(fr2,"levels_completed",lv) or lv) if fr2 is not None else lv
            if g2 is None or lv2!=lv: break          # первый уровень кончился
            hs.append(H(g2)); g=g2
        traj[k].append(((gr.get("levels_completed") or 0)>=1, hs, n))
print("%-6s %7s %6s | %-22s | %-22s | %s" % ("игра","успехов","провал","контроль: успех vs др.успехи","провал vs успехи","первый вход (провал)"))
print("%-6s %7s %6s | %-22s | %-22s |" % ("","","","доля состояний в коридоре","доля состояний в коридоре"))
for k in sorted(traj):
    ok=[t for t in traj[k] if t[0]]; bad=[t for t in traj[k] if not t[0]]
    if len(ok)<2 or not bad: continue
    ctrl=[]
    for i,(_,hs,_) in enumerate(ok):
        cor=set().union(*(set(o[1]) for j,o in enumerate(ok) if j!=i))
        ctrl.append(sum(1 for h in hs[:60] if h in cor)/min(len(hs),60))
    cor=set().union(*(set(o[1]) for o in ok))
    bads=[]; firsts=[]
    for _,hs,_ in bad:
        bads.append(sum(1 for h in hs[:60] if h in cor)/min(len(hs),60))
        f=next((i for i,h in enumerate(hs) if i>0 and h in cor), None); firsts.append(f)
    fs=[f for f in firsts if f is not None]
    print("%-6s %7d %6d | %22.0f%% | %22.0f%% | %s" % (k, len(ok), len(bad), 100*np.median(ctrl), 100*np.median(bads),
          ("ход %d (у %d из %d)" % (np.median(fs), len(fs), len(bad))) if fs else "НИКОГДА"))
print("\n(первые 60 состояний уровня 1; хеш побитовый, поэтому контроль калибрует, что считать «много»)")

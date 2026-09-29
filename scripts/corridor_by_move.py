"""В1.2 второго критика: «блуждание по графу» — удаляются ли застрявшие от золотого пути монотонно?
Доля состояний в коридоре (посещённых успешными) по номеру хода, застрявшие против контроля
(успех против остальных успехов). Монотонное падение у застрявших при плоском контроле = уход в
другую область; одинаковый профиль = колея внутри той же области."""
import hashlib, json, logging, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"vendor/ARC-AGI-3-Agents"))
logging.disable(logging.ERROR)
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
RUNS=["flash_v1_phaseA","public_flash_tufa","public_flash_keithtyser","precond_v2","nostop_v2","flash_combined","flash_goalrule",
      "night_nextfork-b1","night_stock-flash","night_nextfork-b2","night_stock-base2","night_nextfork-b4","night_stock-base4"]
UNSTABLE={"sk48","tn36","g50t","m0r0","sp80","dc22","sc25","cn04","ls20"}
BINS=[(1,10),(11,20),(21,40),(41,80),(81,160)]
def grid(fr): return None if fr is None or not fr.frame else np.asarray(fr.frame[-1],dtype=np.int16)
def H(g): return hashlib.blake2b(g.tobytes(),digest_size=8).hexdigest()
arc=arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT/"environment_files"))
traj=defaultdict(list)
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
            if not a.get("id") or a["id"]=="RESET": continue
            try: fr2=env.step(GameAction[a["id"]], data=a.get("data"))
            except Exception: break
            g2=grid(fr2); lv2=(getattr(fr2,"levels_completed",lv) or lv) if fr2 is not None else lv
            if g2 is None or lv2!=lv: break
            hs.append(H(g2)); g=g2
        traj[k].append(((gr.get("levels_completed") or 0)>=1, hs))
acc={"stuck":defaultdict(list), "ctrl":defaultdict(list)}
for k,ts in traj.items():
    ok=[hs for t,hs in ts if t]; bad=[hs for t,hs in ts if not t]
    if len(ok)<2 or not bad: continue
    for i,hs in enumerate(ok):
        cor=set().union(*(set(o) for j,o in enumerate(ok) if j!=i))
        for lo,hi in BINS:
            seg=hs[lo:hi+1]
            if seg: acc["ctrl"][(lo,hi)].append(np.mean([h in cor for h in seg]))
    cor=set().union(*(set(o) for o in ok))
    for hs in bad:
        for lo,hi in BINS:
            seg=hs[lo:hi+1]
            if seg: acc["stuck"][(lo,hi)].append(np.mean([h in cor for h in seg]))
print("Доля состояний в коридоре успешных, по номеру хода (медиана по прогонам, 9 неустойчивых игр)\n")
print("%-10s %12s %12s" % ("ходы","контроль","застрявшие"))
for lo,hi in BINS:
    c=acc["ctrl"][(lo,hi)]; s=acc["stuck"][(lo,hi)]
    print("%-10s %11.0f%% %11.0f%%   (n=%d/%d)" % ("%d-%d"%(lo,hi), 100*np.median(c) if c else float('nan'), 100*np.median(s) if s else float('nan'), len(c), len(s)))

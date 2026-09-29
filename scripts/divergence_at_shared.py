"""Расхождение в ОБЩИХ состояниях: что делает застрявший там, где успешные делали иначе (26.09).

Коридор показал: застрявшие проходят через те же состояния, что успешные. Тогда колея — это
выбор действия из знакомого состояния. Здесь по хешу состояния сопоставляем действие застрявшего
с множеством действий, которые успешные прогоны делали из ТОГО ЖЕ состояния (уровень 1).
Считаем: долю общих состояний, где застрявший сделал то, чего ни один успешный не делал;
первый такой ход; и чем именно отличается выбор (тип действия). Контроль: успех против других
успехов тем же способом.
"""
import hashlib, json, logging, sys
from collections import defaultdict, Counter
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
NAMES={"ACTION1":"UP","ACTION2":"DOWN","ACTION3":"LEFT","ACTION4":"RIGHT","ACTION5":"5th","ACTION6":"клик","ACTION7":"7th"}
def grid(fr): return None if fr is None or not fr.frame else np.asarray(fr.frame[-1],dtype=np.int16)
def H(g): return hashlib.blake2b(g.tobytes(),digest_size=8).hexdigest()
def key(a):
    d=a.get("data"); n=a.get("id") or "?"
    return "%s(%s,%s)"%(n,d["x"],d["y"]) if isinstance(d,dict) and "x" in d else n
arc=arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT/"environment_files"))
traj=defaultdict(list)   # игра -> [(успех?, [(хеш_до, действие)], прогон)]
for n in RUNS:
    p=ROOT/"runs"/n/"benchmark.json"
    if not p.is_file(): continue
    for gr in json.loads(p.read_text())["game_runs"]:
        k=gr["game_id"][:4]
        if k not in UNSTABLE: continue
        env=arc.make(gr["game_id"]); fr=env.reset(); g=grid(fr)
        if g is None: continue
        lv=getattr(fr,"levels_completed",0) or 0; seq=[]
        for rec in (gr.get("history") or []):
            a=rec.get("action") or {}
            if not a.get("id"): continue
            h=H(g)
            try: fr2=env.step(GameAction[a["id"]] if a["id"]!="RESET" else GameAction.RESET, data=a.get("data"))
            except Exception: break
            seq.append((h, key(a), a["id"]))
            g2=grid(fr2); lv2=(getattr(fr2,"levels_completed",lv) or lv) if fr2 is not None else lv
            if g2 is None or lv2!=lv: break
            g=g2
        traj[k].append(((gr.get("levels_completed") or 0)>=1, seq, n))
def analyse(seq, ok_runs):
    acts=defaultdict(set); types=defaultdict(set)
    for _,s,_ in ok_runs:
        for h,a,t in s: acts[h].add(a); types[h].add(t)
    shared=div=div_type=0; first=None; contrast=Counter()
    for i,(h,a,t) in enumerate(seq[:60]):
        if h not in acts: continue
        shared+=1
        if a not in acts[h]:
            div+=1
            if first is None: first=i+1
            if t not in types[h]:
                div_type+=1
                contrast[(NAMES.get(t,t), "/".join(sorted(NAMES.get(x,x) for x in types[h])))]+=1
    return shared, div, div_type, first, contrast
print("%-6s %6s %6s | %-24s | %-24s | %s" % ("игра","успех","провал","контроль (успех vs др.)","провал vs успехи","первый разошедшийся ход"))
print("%-6s %6s %6s | %-24s | %-24s |" % ("","","","общих / разошлись / по типу","общих / разошлись / по типу"))
allc=Counter()
for k in sorted(traj):
    ok=[t for t in traj[k] if t[0]]; bad=[t for t in traj[k] if not t[0]]
    if len(ok)<2 or not bad: continue
    c=[analyse(s, [o for j,o in enumerate(ok) if j!=i]) for i,(_,s,_) in enumerate(ok)]
    b=[analyse(s, ok) for _,s,_ in bad]
    def fmt(rs):
        sh=np.median([r[0] for r in rs]); d=np.median([r[1] for r in rs]); dt=np.median([r[2] for r in rs])
        return "%3.0f / %3.0f / %3.0f" % (sh, d, dt)
    firsts=[r[3] for r in b if r[3]]
    for r in b: allc.update(r[4])
    print("%-6s %6d %6d | %-24s | %-24s | %s" % (k, len(ok), len(bad), fmt(c), fmt(b),
          ("ход %.0f" % np.median(firsts)) if firsts else "нет"))
print("\nчем отличается выбор застрявших в общих состояниях (застрявший -> что делали успешные):")
for (t, ok_t), n in allc.most_common(8): print("   %-6s -> %-22s %d раз" % (t, ok_t, n))

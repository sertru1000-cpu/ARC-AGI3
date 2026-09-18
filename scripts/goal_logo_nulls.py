import json,random,sys
from pathlib import Path
import numpy as np
ROOT=Path('/Users/sergeimakarov/Projects/ARC-AGI-3')
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'vendor/ARC-AGI-3-Agents')); sys.path.insert(0,str(ROOT/'scripts'))
import logging; logging.disable(logging.ERROR)
import arc_agi
from arc_agi import OperationMode
from goal_cross_level_replay import load_events, replay_collect, DEFAULT_RUNS
from goal_logo_test import feats, check, candidates_from, separating, predict
arc=arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT/'environment_files'))
rng=random.Random(0); per_game={}
for e in arc.get_environments():
    gid=e.game_id; best={}
    for run in DEFAULT_RUNS:
        evs=load_events(run,gid)
        if not evs: continue
        for k,v in replay_collect(arc,gid,evs).items():
            if k not in best or len(v['negs'])>len(best[k]['negs']): best[k]=v
    lv={}
    for k,v in best.items():
        goal=v['goal']; negs=[g for g in v['negs'][-200:] if g.shape==goal.shape]
        if not negs: continue
        fg=feats(goal); fn=[feats(g) for g in negs]
        lv[k]={'f_goal':fg,'f_negs':fn,'sep':separating(fg,fn,candidates_from(fg,fn[0]))}
    if lv: per_game[gid[:4]]=lv
pairs=[(g,k) for g,lv in per_game.items() for k in sorted(lv) if k+1 in lv and lv[k]['sep']]
def cover(src_sep, tgt, mode='direction'):
    preds=[]
    for p in src_sep:
        for q in predict(p,mode):
            if q not in preds: preds.append(q)
    ok=[i for i,q in enumerate(preds) if check(q,tgt['f_goal']) and not any(check(q,x) for x in tgt['f_negs'])]
    return (1 if ok and ok[0]==0 else 0, 1 if ok and min(ok)<3 else 0)
c1=c3=n=0; n1=n3=0
for g,k in pairs:
    tgt=per_game[g][k+1]
    a,b=cover(per_game[g][k]['sep'],tgt); c1+=a; c3+=b; n+=1
    # нулевая: предикаты СЛУЧАЙНОГО уровня ЧУЖОЙ игры
    others=[(gg,kk) for gg,lv in per_game.items() if gg!=g for kk in lv if lv[kk]['sep']]
    tot1=tot3=0; T=5
    for _ in range(T):
        gg,kk=rng.choice(others); a2,b2=cover(per_game[gg][kk]['sep'],tgt); tot1+=a2; tot3+=b2
    n1+=tot1/T; n3+=tot3/T
pass
pass

# решающий контроль: переносим ПСЕВДОЦЕЛЬ уровня k (случайное не-целевое состояние), а не настоящую цель
pc1=pc3=0; m=0
for g,k in pairs:
    src=per_game[g][k]; tgt=per_game[g][k+1]
    t1=t3=0; T=5
    for _ in range(T):
        i=rng.randrange(len(src['f_negs']))
        pseudo=src['f_negs'][i]; rest=[x for j,x in enumerate(src['f_negs']) if j!=i]
        sep=separating(pseudo,rest,candidates_from(pseudo,rest[0] if rest else None))
        if not sep: continue
        a,b=cover(sep,tgt); t1+=a; t3+=b
    pc1+=t1/T; pc3+=t3/T; m+=1
print('пар %d'%n)
print('перенос НАСТОЯЩЕЙ цели своего уровня:   Coverage@1 %.0f%%, @3 %.0f%%'%(100*c1/n,100*c3/n))
print('перенос ПСЕВДОЦЕЛИ того же уровня:      Coverage@1 %.0f%%, @3 %.0f%%'%(100*pc1/m,100*pc3/m))
print('перенос цели ЧУЖОЙ игры:                Coverage@1 %.0f%%, @3 %.0f%%'%(100*n1/n,100*n3/n))

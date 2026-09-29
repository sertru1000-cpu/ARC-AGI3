"""Depth-3 контрфактический тест по второму критику (26.09).

Оракул расстояния до уровня — успешные траектории: для каждого состояния коридора d(S) = минимум
по успешным прогонам числа ходов от S до взятия уровня 1. Из ОБЩИХ состояний первых 12 ходов
застрявшего прогона ветвим движком: реальное действие застрявшего, действие успешного из того же
состояния, все клавиатурные альтернативы на глубину 1..3 (для мыши — клики, которые успешные
делали из этого состояния). Продолжение оцениваем лучшей достигнутой d в коридоре; не попал в
коридор за k ходов — «неизвестно».

Сценарии критика: best3 >> success >> stuck — короткий горизонт; всё ≈ — искать скрытую фазу;
stuck < success уже на глубине 1 — одноходовый фильтр; разницы нет — траекторный уровень.
Кандидат 2 (история как скрытая фаза): насколько S, S+1, S+2 предыдущих действия сужают множество
успешных продолжений.
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
FIRST=12; KEYS=["ACTION1","ACTION2","ACTION3","ACTION4","ACTION5","ACTION7"]
def grid(fr): return None if fr is None or not fr.frame else np.asarray(fr.frame[-1],dtype=np.int16)
def H(g): return hashlib.blake2b(g.tobytes(),digest_size=8).hexdigest()
def act_of(a):
    d=a.get("data"); return (a["id"], (d["x"],d["y"]) if isinstance(d,dict) and "x" in d else None)
arc=arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT/"environment_files"))
gid_full={}
traj=defaultdict(list)     # игра -> [(успех?, [(hash_before, action)], took_level?)]
for n in RUNS:
    p=ROOT/"runs"/n/"benchmark.json"
    if not p.is_file(): continue
    for gr in json.loads(p.read_text())["game_runs"]:
        k=gr["game_id"][:4]
        if k not in UNSTABLE: continue
        gid_full[k]=gr["game_id"]
        env=arc.make(gr["game_id"]); fr=env.reset(); g=grid(fr)
        if g is None: continue
        lv=getattr(fr,"levels_completed",0) or 0; seq=[]; took=False
        for rec in (gr.get("history") or []):
            a=rec.get("action") or {}
            if not a.get("id") or a["id"]=="RESET": continue
            h=H(g)
            try: fr2=env.step(GameAction[a["id"]], data=a.get("data"))
            except Exception: break
            seq.append((h, act_of(a)))
            g2=grid(fr2); lv2=(getattr(fr2,"levels_completed",lv) or lv) if fr2 is not None else lv
            if lv2!=lv: took=True; break
            if g2 is None: break
            g=g2
        traj[k].append(((gr.get("levels_completed") or 0)>=1 and took, seq))
def step(env, act):
    aid, xy = act
    return env.step(GameAction[aid], data=({"x":xy[0],"y":xy[1]} if xy else None))
_ENV={}
def replay(k, seq, upto):
    """Одна среда на игру: env.reset() восстанавливает начальное состояние точно (проверено по хешу),
    и реплей 12 шагов стоит 4 мс против 1.09 с на arc.make — первая версия пересоздавала среду на
    каждую ветку и ползла часами."""
    env=_ENV.get(k)
    if env is None: env=_ENV[k]=arc.make(gid_full[k])
    fr=env.reset()
    for h,a in seq[:upto]: fr=step(env,a)
    return env, fr
res={"stuck":[], "success":[], "best1":[], "best3":[]}; hist_pred={0:[],1:[],2:[]}
for k in sorted(traj):
    ok=[t for t in traj[k] if t[0]]; bad=[t for t in traj[k] if not t[0]]
    if not ok or not bad: continue
    # оракул: d(S) и множество успешных продолжений из S (с историей)
    dist={}; succ=defaultdict(set); succ_h=defaultdict(lambda: defaultdict(set))
    for _,seq in ok:
        L=len(seq)
        for i,(h,a) in enumerate(seq):
            dist[h]=min(dist.get(h,10**9), L-i); succ[h].add(a)
            for kk in (1,2):
                key=(h, tuple(x[1] for x in seq[max(0,i-kk):i])); succ_h[kk][key].add(a)
    # кандидат 2: сколько успешных продолжений остаётся при знании истории
    for h in succ:
        hist_pred[0].append(len(succ[h]))
    for kk in (1,2):
        for key,s in succ_h[kk].items(): hist_pred[kk].append(len(s))
    game_keys=sorted({a[0] for _,seq in ok for _,a in seq if a[1] is None})   # клавиши, которыми играли успешные
    for _,seq in bad:
        for i,(h,a_st) in enumerate(seq[:FIRST]):
            if h not in dist or h not in succ: continue
            d0=dist[h]
            def eval_seq(first_act, depth):
                """лучшая d после first_act и до depth-1 клавиатурных шагов; None = коридор не достигнут"""
                best=None
                def rec(env_seq, dleft):
                    nonlocal best
                    env,fr=replay(k, seq, i)
                    for a in env_seq: fr=step(env,a)
                    g=grid(fr); hh=H(g) if g is not None else None
                    if hh in dist: best=min(best if best is not None else 10**9, dist[hh])
                    if dleft>0:
                        for kaid in game_keys:
                            rec(env_seq+[(kaid,None)], dleft-1)
                rec([first_act], depth-1)
                return best
            s1=eval_seq(a_st,1); res["stuck"].append((d0, s1))
            a_ok=next(iter(succ[h])); res["success"].append((d0, eval_seq(a_ok,1)))
            alts=[(x,None) for x in game_keys]+[x for x in succ[h] if x[1]]
            b1=[eval_seq(a,1) for a in alts]; b1=[x for x in b1 if x is not None]
            res["best1"].append((d0, min(b1) if b1 else None))
            b3=[eval_seq(a,3) for a in alts]; b3=[x for x in b3 if x is not None]
            res["best3"].append((d0, min(b3) if b3 else None))
    print("игра %s: общих состояний %d" % (k, sum(1 for _,seq in bad for h,_ in seq[:FIRST] if h in dist)), flush=True)
print("\nΔd = d(до) − лучшая d(после) по оракулу коридора; «в коридоре» = продолжение попало в известное состояние")
print("%-10s %8s %12s %14s %14s" % ("вариант","n","в коридоре","медиана Δd","доля Δd>0"))
for name in ("stuck","success","best1","best3"):
    r=res[name]; known=[(d0,d1) for d0,d1 in r if d1 is not None]
    dd=[d0-d1 for d0,d1 in known]
    print("%-10s %8d %11.0f%% %14s %13s" % (name, len(r), 100*len(known)/max(len(r),1),
          ("%+.1f" % np.median(dd)) if dd else "—", ("%.0f%%" % (100*np.mean([x>0 for x in dd]))) if dd else "—"))
print("\nкандидат 2 — сколько РАЗНЫХ успешных продолжений допускает состояние:")
for kk in (0,1,2):
    v=hist_pred[kk]; print("  S + история %d: медиана %.1f, доля однозначных %.0f%%" % (kk, np.median(v), 100*np.mean([x==1 for x in v])))

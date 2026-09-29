"""Тест 7 третьего критика: в точке расхождения — ПАМЯТЬ или СУЖДЕНИЕ? (26.09)

Для каждого общего состояния S (побитовый хеш, уровень 1), где застрявший прогон сделал ход,
которого ни один успешный из S не делал, сравниваем ФАКТЫ, доступные к этому моменту из истории:
множество пар (действие -> меняло ли доску) и множество (действие, состояние_до -> состояние_после).
Вопрос: был ли у успешного прогона к его ходу из S факт, которого у застрявшего к его ходу из S нет?
Если в большинстве расхождений — да, это отказ памяти (обвязка может её дать: ledger, 0 токенов).
Если оба знают одно и то же — отказ суждения (память не источник +3).
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
RUNS=["flash_v1_phaseA","public_flash_tufa","public_flash_keithtyser","precond_v2","nostop_v2","flash_combined","flash_goalrule",
      "night_nextfork-b1","night_stock-flash","night_nextfork-b2","night_stock-base2","night_nextfork-b4","night_stock-base4",
      "night_nextfork-b5","night_stock-base5","night_nextfork-b3","night_stock-base3"]
UNSTABLE={"sk48","tn36","g50t","m0r0","sp80","dc22","sc25","cn04","ls20"}
def grid(fr): return None if fr is None or not fr.frame else np.asarray(fr.frame[-1],dtype=np.int16)
def H(g): return hashlib.blake2b(g.tobytes(),digest_size=8).hexdigest()
def key(a):
    d=a.get("data"); n=a.get("id") or "?"
    return "%s(%s,%s)"%(n,d["x"],d["y"]) if isinstance(d,dict) and "x" in d else n
arc=arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT/"environment_files"))
traj=defaultdict(list)   # игра -> [(успех?, [(h_before, action_key, type, changed, h_after)])]
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
            if not a.get("id") or a["id"]=="RESET": continue
            h=H(g)
            try: fr2=env.step(GameAction[a["id"]], data=a.get("data"))
            except Exception: break
            g2=grid(fr2); lv2=(getattr(fr2,"levels_completed",lv) or lv) if fr2 is not None else lv
            ch=(g2 is None) or (g2.shape!=g.shape) or bool(np.any(g2!=g)) or lv2!=lv
            seq.append((h, key(a), a["id"], bool(ch), H(g2) if g2 is not None else None))
            if g2 is None or lv2!=lv: break
            g=g2
        traj[k].append(((gr.get("levels_completed") or 0)>=1, seq))
def facts(seq, upto):
    """факты к моменту upto: типы действий с наблюдённым эффектом; конкретные переходы"""
    eff=set(); trans=set()
    for h,a,t,ch,h2 in seq[:upto]:
        eff.add((t, ch)); trans.add((h, a, h2))
    return eff, trans
tot=mem=judge=0; per=defaultdict(lambda:[0,0])
for k,ts in traj.items():
    ok=[s for t,s in ts if t]; bad=[s for t,s in ts if not t]
    if not ok or not bad: continue
    # успешные: из состояния h -> (действие, факты к тому моменту)
    ok_at=defaultdict(list)
    for s in ok:
        for i,(h,a,t,ch,h2) in enumerate(s): ok_at[h].append((a, facts(s,i)))
    for s in bad:
        for i,(h,a,t,ch,h2) in enumerate(s[:40]):
            if h not in ok_at: continue
            ok_actions={x[0] for x in ok_at[h]}
            if a in ok_actions: continue                      # не расхождение
            tot+=1
            # СТРОГО (поправка 26.09): смотрим только на ТИП выигрышного действия (что выбрал успешный
            # из этого состояния). Пробовал ли застрявший этот тип действия раньше в этой партии?
            #   пробовал и видел эффект, но выбрал другое -> СУЖДЕНИЕ;
            #   никогда не пробовал                        -> РАЗВЕДКА (память тут ни при чём: факта нет ни у кого до пробы).
            # Прежнее определение («любой переход») давало 89% «памяти» тривиально: разные пути = разные переходы.
            win_types={x[0].split("(")[0] for x in ok_at[h]}
            tried_types={t2 for _,_,t2,_,_ in s[:i]}
            if win_types & tried_types: judge+=1; per[k][1]+=1
            else: mem+=1; per[k][0]+=1
print("Расхождений в общих состояниях (первые 40 ходов): %d" % tot)
print("  застрявший НИКОГДА не пробовал тип выигрышного действия (РАЗВЕДКА): %d (%.0f%%)" % (mem, 100*mem/max(tot,1)))
print("  пробовал, видел эффект, но выбрал другое (СУЖДЕНИЕ):         %d (%.0f%%)" % (judge, 100*judge/max(tot,1)))
print("\nпо играм (разведка/суждение):")
for k in sorted(per): print("  %-6s %3d / %3d" % (k, per[k][0], per[k][1]))
print("\n«факт» = (тип действия -> меняло ли доску) или конкретный переход (состояние, действие -> состояние), наблюдённый ранее в этой партии")

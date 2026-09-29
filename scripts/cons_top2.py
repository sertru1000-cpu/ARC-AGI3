"""Top-1 / Top-2 по логу прогона cons против оракула коридора (тест третьего критика, 26.09).

Лог cons печатает на первых 4 ходах каждой игры: «CONSENSUS: move N, A=[...], B=[...] -> agree|DISAGREE».
Оракул: «выигрышное действие» из состояния = действия, которые делали УСПЕШНЫЕ прогоны той же игры из
того же состояния (побитовый хеш доски), по 17 записанным прогонам. Состояние хода N восстанавливается
реплеем первых N ходов прогона cons на локальном движке.

Считаем по состояниям, для которых оракул существует:
  Top-1 = доля, где A (исполняемый ответ) совпал с выигрышным;
  Top-2 = доля, где выигрышное есть среди {A, B};
  согласие A==B; и Top-1 внутри согласных / несогласных.
Критерий критика: Top-2 заметно выше Top-1 -> узкое место в ВЫБОРЕ (правильная ветка у модели есть).
usage: .venv/bin/python scripts/cons_top2.py runs/night_nextfork-cons
"""
import ast, hashlib, json, logging, re, sys
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
def grid(fr): return None if fr is None or not fr.frame else np.asarray(fr.frame[-1],dtype=np.int16)
def H(g): return hashlib.blake2b(g.tobytes(),digest_size=8).hexdigest()
def key(a):
    d=a.get("data"); n=a.get("id") or "?"
    return "%s(%s,%s)"%(n,d["x"],d["y"]) if isinstance(d,dict) and "x" in d else n
def key_from_cons(item):
    """действие из лога cons: {'action': 'UP'} или {'action': 'MOUSE','row':r,'col':c} -> ключ оракула"""
    from inference.agent.action_names import to_engine_action
    name=str(item.get("action",""))
    eng=to_engine_action(name) or name
    if eng=="ACTION6" or name.upper()=="MOUSE":
        return "ACTION6(%s,%s)" % (item.get("col"), item.get("row"))     # data x=col, y=row
    return eng
def main(run_dir):
    sys.path.insert(0, str(ROOT/"nextfork/src/ARC3-Inference"))
    arc=arc_agi.Arcade(operation_mode=OperationMode.NORMAL, environments_dir=str(ROOT/"environment_files"))
    # оракул: игра -> hash -> {действия успешных}
    oracle=defaultdict(lambda: defaultdict(set)); gid_full={}
    for n in RUNS:
        p=ROOT/"runs"/n/"benchmark.json"
        if not p.is_file(): continue
        for gr in json.loads(p.read_text())["game_runs"]:
            if (gr.get("levels_completed") or 0)<1: continue
            k=gr["game_id"][:4]; gid_full[k]=gr["game_id"]
            env=arc.make(gr["game_id"]); fr=env.reset(); g=grid(fr); lv=getattr(fr,"levels_completed",0) or 0
            for rec in (gr.get("history") or []):
                a=rec.get("action") or {}
                if not a.get("id") or a["id"]=="RESET": continue
                oracle[k][H(g)].add(key(a))
                try: fr2=env.step(GameAction[a["id"]], data=a.get("data"))
                except Exception: break
                g2=grid(fr2); lv2=(getattr(fr2,"levels_completed",lv) or lv) if fr2 is not None else lv
                if g2 is None or lv2!=lv: break
                g=g2
    # лог cons
    log=next(Path(run_dir).glob("*.log"))
    txt=log.read_text(errors="ignore")
    ev=re.findall(r"CONSENSUS: move (\d+), A=(\[.*?\]), B=(\[.*?\]) -> (agree|DISAGREE)", txt)
    print("событий согласования в логе: %d" % len(ev))
    # какой игре принадлежит событие — по порядку в логе нельзя; берём по транскриптам: в них тот же маркер? нет.
    # поэтому сопоставляем через benchmark: для каждой игры реплеим первые 4 хода и берём хеши; событие с move N
    # относим к игре, чей ход N (первые ходы) совпадает с A по действию — грубо, но для Top-2 достаточно.
    bench=json.loads((Path(run_dir)/"benchmark.json").read_text())["game_runs"]
    states={}   # (игра, N) -> hash
    for gr in bench:
        k=gr["game_id"][:4]; env=arc.make(gr["game_id"]); fr=env.reset(); g=grid(fr)
        acts=[a for a in ((r.get("action") or {}) for r in (gr.get("history") or [])) if a.get("id") and a["id"]!="RESET"][:4]
        for N,a in enumerate(acts):
            states[(k,N)]=(H(g), key(a))
            try: fr2=env.step(GameAction[a["id"]], data=a.get("data")); g=grid(fr2)
            except Exception: break
    top1=top2=n=agree=0; t1_agree=[0,0]; t1_dis=[0,0]
    for N,A,B,verdict in ev:
        N=int(N); A=ast.literal_eval(A); B=ast.literal_eval(B)
        ka=key_from_cons(A[0]) if A else None; kb=key_from_cons(B[0]) if B else None
        # игра: та, у которой на ходу N исполнено действие ka (cons исполняет A или C; при agree — A)
        cands=[(k,h) for (k,nn),(h,kk) in states.items() if nn==N and kk==ka]
        if not cands: continue
        for k,h in cands:
            win=oracle.get(k,{}).get(h)
            if not win: continue
            n+=1; hit1= ka in win; hit2= hit1 or (kb in win)
            top1+=hit1; top2+=hit2; ag=(verdict=="agree"); agree+=ag
            (t1_agree if ag else t1_dis)[0]+=hit1; (t1_agree if ag else t1_dis)[1]+=1
            break
    if not n: print("оракул не покрыл ни одного события"); return
    print("состояний с оракулом: %d | согласие A==B: %.0f%%" % (n, 100*agree/n))
    print("Top-1 (A совпал с выигрышным): %.0f%%" % (100*top1/n))
    print("Top-2 (выигрышное среди A,B):  %.0f%%" % (100*top2/n))
    for lab,(h,c) in (("при согласии",t1_agree),("при расхождении",t1_dis)):
        if c: print("  Top-1 %-16s %.0f%% (n=%d)" % (lab, 100*h/c, c))
    print("\nкритерий критика: Top-2 >> Top-1 -> узкое место в выборе; Top-2 ≈ Top-1 -> правильной ветки у модели нет")
if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "runs/night_nextfork-cons")

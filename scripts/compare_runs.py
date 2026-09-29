"""Сравнение прогонов: балл по всем 25 и по тихим 21 против среднего стоковой базы и форка v3 (26.09).
usage: .venv/bin/python scripts/compare_runs.py night_nextfork-twin2 [...]   (имена папок в runs/)
Для прогонов с потолком 1 ч сравнивать с обрезкой: scripts/truncate_run.py runs/<база> --seconds 3600."""
import json,sys,statistics as st
from pathlib import Path
R=Path('runs'); NOISY={'ft09','re86','ar25','r11l'}
def sc(d):
    g=json.load(open(R/d/'score.json'))['games']; return {k[:4]:v['score'] for k,v in g.items()}
def m(s,excl=False): v=[x for k,x in s.items() if not(excl and k in NOISY)]; return sum(v)/len(v)
base=["flash_v1_phaseA","public_flash_tufa","public_flash_keithtyser","night_stock-flash","night_stock-base2","night_stock-base4","night_stock-base5"]
fork=["night_nextfork-b1","night_nextfork-b2","night_nextfork-b4","night_nextfork-b5"]
for name,grp in (("база",base),("форк v3",fork)):
    xs=[];ys=[]
    for d in grp:
        try: s=sc(d); xs.append(m(s)); ys.append(m(s,True))
        except Exception as e: print(d,e)
    print(f"{name}: n={len(xs)} все25 {st.mean(xs):.2f} (sd {st.stdev(xs):.2f}) {[round(x,2) for x in xs]} | тихие21 {st.mean(ys):.2f}")
for d in sys.argv[1:]:
    s=sc(d); print(f"{d}: все25 {m(s):.2f} | тихие21 {m(s,True):.2f} | игр {len(s)}")

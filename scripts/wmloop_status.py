"""Сводка по логам цикла «модель думает, алгоритм ходит». usage: wmloop_status.py <log glob>"""
import glob, re, sys, ast
rows = []
for p in sorted(glob.glob(sys.argv[1])):
    done = {}
    for line in open(p, encoding="utf-8", errors="replace"):
        m = re.search(r"WARNING:root:(\w{4}): wmloop (\{.*\})", line)
        if m:
            try: d = ast.literal_eval(m.group(2))
            except Exception: continue
            pl = d.get("planner") or {}
            done[m.group(1)] = (d.get("phase"), d.get("calls"), d.get("repairs"), pl.get("states"), pl.get("executed"), pl.get("pred_ok"), pl.get("pred_wrong"), d.get("synth_fail"))
        m2 = re.search(r"\] (\w{4}): levels\s+(\d+)/(\d+)\s+actions\s+(\d+)", line)
        if m2 and m2.group(1) in done:
            done[m2.group(1)] = done[m2.group(1)] + (m2.group(2) + "/" + m2.group(3), m2.group(4))
    for g, v in done.items():
        rows.append((p.split("/")[-1].replace("wmloop_api_", "").replace(".log", ""), g) + v)
print("лог\tигра\tфаза\tвызовов\tпочинок\tсостояний\tпроверок\tpred_ok\tpred_wrong\tсбоев\tуровни\tходов")
for r in rows: print("\t".join(str(x) for x in r))
lv = sum(int(r[10].split("/")[0]) for r in rows if len(r) > 10); print("итого игр", len(rows), "уровней", lv)

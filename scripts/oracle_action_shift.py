"""KL-сдвиг распределения ходов до/после оракул-инъекции (раунд 4 критика, утверждение 3), без квоты.
Окна: 50 ходов до инъекции и 50 после; контроль — те же окна тех же игр в базе flash_v1_phaseA."""
import json, math, glob, os, collections, sys
INJ = {"g50t": 51, "cn04": 60, "sp80": 50, "tn36": 53, "sk48": 59, "wa30": 50, "ls20": 53, "lf52": 54, "bp35": 51}
W = 50
def acts(run, game):
    p = glob.glob(f"runs/{run}/artifacts/{game}-*_p0_events.jsonl")
    if not p: return []
    out = []
    for l in open(p[0], encoding="utf-8"):
        e = json.loads(l)
        if e.get("type") != "action": continue
        out.append((int(e["action_num"]), str(e.get("action_display") or e.get("action_name")).split("(")[0].strip().upper(),
                    e.get("board_changed") in (True, "True"), e.get("level_completed") in (True, "True")))
    return out
def dist(seq):
    c = collections.Counter(a for _, a, _, _ in seq); n = sum(c.values()); return c, n
def kl(after, before, keys):
    ca, na = dist(after); cb, nb = dist(before); k = len(keys)
    return sum(((ca[x] + 1) / (na + k)) * math.log(((ca[x] + 1) / (na + k)) / ((cb[x] + 1) / (nb + k))) for x in keys)
rows = []
for run in ("flash_oracle_v1", "flash_v1_phaseA"):
    for g, at in INJ.items():
        s = acts(run, g)
        before = [x for x in s if at - W <= x[0] < at]; after = [x for x in s if at <= x[0] < at + W]
        keys = sorted(set(a for _, a, _, _ in before + after))
        if not before or not after: rows.append((run, g, len(before), len(after), None, None, None, None, None)); continue
        ch_b = sum(1 for x in before if x[2]) / len(before); ch_a = sum(1 for x in after if x[2]) / len(after)
        lv_a = sum(1 for x in after if x[3]); lv_rest = sum(1 for x in s if x[0] >= at and x[3])
        top_b = dist(before)[0].most_common(1)[0]; top_a = dist(after)[0].most_common(1)[0]
        rows.append((run, g, len(before), len(after), round(kl(after, before, keys), 3), f"{top_b[0]} {top_b[1]}", f"{top_a[0]} {top_a[1]}", f"{ch_b:.2f}->{ch_a:.2f}", f"{lv_a}/{lv_rest}"))
print("run\tgame\tn_before\tn_after\tKL(after||before)\ttop_before\ttop_after\tboard_changed\tlevels(after50/after_all)")
for r in rows: print("\t".join(str(x) for x in r))
for run in ("flash_oracle_v1", "flash_v1_phaseA"):
    v = [r[4] for r in rows if r[0] == run and r[4] is not None]; print(run, "median KL", round(sorted(v)[len(v)//2], 3), "mean", round(sum(v)/len(v), 3), "n", len(v))

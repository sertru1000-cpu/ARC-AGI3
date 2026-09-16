"""Сводка теста №2 (критик, раунд 6): точный перебор против макроходов и гибрида на 25 оригиналах.
usage: bfs2_compare.py <dir с bfs2_orig_{exact,macro,both}[_keep|_redo].json>"""
import json, os, sys
S = sys.argv[1]; modes = ("exact", "macro", "both"); R = {}
for m in modes:
    d = {}
    for suf in ("", "_keep", "_redo"):
        p = f"{S}/bfs2_orig_{m}{suf}.json"
        if os.path.exists(p):
            d.update({k[:4]: v for k, v in json.load(open(p)).items()})
    R[m] = d
games = sorted(set().union(*[set(d) for d in R.values()]))
def cell(r):
    if not r: return "—"
    if r.get("error"): return "ошибка"
    return f"{'✔' if r.get('solved') else '✘'} {r.get('states')}s/{r.get('depth_moves')}d/{r.get('moves')}m/{r.get('seconds')}с" + ("⏱" if r.get("timeout") else "")
print(f"{'игра':5s} | {'exact':34s} | {'macro':34s} | {'both':34s}")
tot = {m: 0 for m in modes}
for g in games:
    row = [cell(R[m].get(g)) for m in modes]
    for m in modes:
        tot[m] += bool((R[m].get(g) or {}).get("solved"))
    print(f"{g:5s} | " + " | ".join(f"{c:34s}" for c in row))
print("уровень 1 взят:", tot)
# сокращение состояний на играх, решённых обоими
for m in ("macro", "both"):
    pairs = [(g, R["exact"][g]["states"], R[m][g]["states"]) for g in games if R["exact"].get(g, {}).get("solved") and R[m].get(g, {}).get("solved")]
    if pairs:
        ratios = [e / max(1, x) for _, e, x in pairs]
        print(f"{m}: решены обоими {len(pairs)}; отношение состояний exact/{m}: медиана {sorted(ratios)[len(ratios)//2]:.1f}, " + ", ".join(f"{g} {e}/{x}" for g, e, x in pairs))

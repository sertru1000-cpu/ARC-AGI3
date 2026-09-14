"""Классификация повторов (раунд 4, результат 2): «повторные неудачные переходы» против «покрытия новых классов
действий». По журналам событий прогонов, без квоты.
  повтор неудачный  = ход (хеш доски, действие) уже делался раньше в этой игре И тогда доску не изменил
  повтор удачный    = пара уже делалась, и тогда доска менялась (возврат в круг)
  новый класс       = класс действия (стрелка/SPACE/клик по объекту-центру) впервые на этом уровне
usage: .venv/bin/python scripts/repeat_classes.py runs/flash_v1_phaseA runs/flash_loop_v2 ..."""
import json, glob, os, sys, hashlib, re, collections
def load(run):
    out = {}
    for p in sorted(glob.glob(f"{run}/artifacts/*_p0_events.jsonl")):
        g = os.path.basename(p)[:4]; seq = []; prev_board = None
        for l in open(p, encoding="utf-8"):
            e = json.loads(l)
            b = e.get("board_ascii") or json.dumps(e.get("board"))
            h = hashlib.blake2b(str(b).encode(), digest_size=8).hexdigest()
            if e.get("type") == "action":
                name = str(e.get("action_display") or e.get("action_name"))
                m = re.match(r"MOUSE\(row=(\d+), col=(\d+)\)", name)
                cls = "CLICK" if m else name.split("(")[0]
                seq.append({"before": prev_board, "after": h, "act": name, "cls": cls, "changed": e.get("board_changed") in (True, "True"),
                            "level": int(e.get("level") or 0), "lvl_done": e.get("level_completed") in (True, "True")})
            prev_board = h
        out[g] = seq
    return out
def classify(seq):
    seen = {}; classes = set(); n = len(seq); fail_rep = ok_rep = new_cls = 0; lvl = None
    for s in seq:
        if s["level"] != lvl:
            lvl = s["level"]; classes = set()
        key = (s["before"], s["act"])
        if key in seen:
            if seen[key]: ok_rep += 1
            else: fail_rep += 1
        else:
            seen[key] = s["changed"]
        if s["cls"] not in classes:
            classes.add(s["cls"]); new_cls += 1
    return n, fail_rep, ok_rep, new_cls
if __name__ == "__main__":
    for run in sys.argv[1:]:
        d = load(run); tot = collections.Counter(); rows = []
        for g, seq in d.items():
            n, fr, okr, nc = classify(seq); lv = sum(1 for s in seq if s["lvl_done"])
            tot.update({"moves": n, "fail_rep": fr, "ok_rep": okr, "levels": lv}); rows.append((g, n, fr, okr, nc, lv))
        print(f"== {run}: ходов {tot['moves']}, повторов неудачных {tot['fail_rep']} ({tot['fail_rep']/max(1,tot['moves']):.1%}), "
              f"повторов удачных {tot['ok_rep']} ({tot['ok_rep']/max(1,tot['moves']):.1%}), уровней {tot['levels']}")
        stuck = [r for r in rows if r[5] == 0]; ok = [r for r in rows if r[5] > 0]
        for name, grp in (("без уровней", stuck), ("с уровнями", ok)):
            if grp:
                m = sum(r[1] for r in grp); f = sum(r[2] for r in grp); o = sum(r[3] for r in grp)
                print(f"   {name}: игр {len(grp)}, ходов {m}, неудачных повторов {f/max(1,m):.1%}, удачных повторов {o/max(1,m):.1%}")

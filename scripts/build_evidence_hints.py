"""Знание как УЛИКА из собственного журнала игры, а не как утверждение извне.

ЗАЧЕМ. 12.09 опыт «цепочка» показал, где рвётся путь от выданного правила к ходу: модель
правило читает и пересказывает (27 ответов из 27), но в 4 играх из 9 ОТКРЫТО НЕ СОГЛАШАЕТСЯ,
ссылаясь на собственные наблюдения («история кадров показывает, что SPACE меняет ориентацию
фигуры», «я отбросил SPACE как бездействующий»). То есть внешнее утверждение не перебивает
гипотезу, построенную на её же данных. Здесь то же знание подаётся в том виде, которому
она доверяет: конкретные ходы из её журнала с номерами шагов.

ЧЕСТНОСТЬ ВЫВОДА. Улики берутся из того же прогона, из которого собрана нагрузка
(`runs/flash_v1_phaseA`), и ТОЛЬКО из ходов ДО записанного момента (`analysis_step` записи).
Иначе модель увидела бы улики о ходах, которых в её истории ещё нет, и опыт измерял бы
подсказку из будущего.

ЧТО СЧИТАЕТСЯ. Для каждого типа действия (стрелки, SPACE, мышь): сколько раз применялось,
сколько раз изменило доску и на каких шагах; отдельно выделяются строки-индикаторы — те,
что меняются почти на каждом ходу (в tn36 так менялась верхняя полоса при 367 ходах из 561,
и агент принимал это за успех).

usage:  .venv/bin/python scripts/build_evidence_hints.py
"""
import glob
import json
from pathlib import Path

RUN = Path("runs/flash_v1_phaseA")
PAYLOAD = "kernels/notebooks_prompt_replay/payload.json.xz"
OUT = "docs/oracle_hints_evidence.json"


def kind(name):
    n = (name or "").upper()
    if "MOUSE" in n:
        return "MOUSE clicks"
    if "SPACE" in n:
        return "SPACE"
    if any(d in n for d in ("UP", "DOWN", "LEFT", "RIGHT")):
        return "arrow keys"
    return "other actions"


def rows_of(board):
    if board is None:
        return None
    if isinstance(board, str):
        return board.split("\n")
    return ["".join(str(v) for v in r) for r in board]


def evidence_for(gid, cutoff_step):
    f = glob.glob(str(RUN / "artifacts" / (gid + "*_events.jsonl")))
    if not f:
        return None
    evs = [json.loads(l) for l in open(f[0], encoding="utf-8")]
    init = next((e for e in evs if e.get("type") == "initial"), None)
    acts = [e for e in evs if e.get("type") == "action"
            and int(e.get("analysis_step", 0)) < cutoff_step]
    if not acts:
        return None

    prev = rows_of((init or {}).get("board_ascii") or (init or {}).get("board"))
    stats, changed_rows_count = {}, {}
    for e in acts:
        cur = rows_of(e.get("board_ascii") or e.get("board"))
        k = kind(e.get("action_display"))
        s = stats.setdefault(k, {"n": 0, "changed": [], "rows": set()})
        s["n"] += 1
        if prev and cur:
            diff = [i for i in range(min(len(prev), len(cur))) if prev[i] != cur[i]]
            if diff:
                s["changed"].append((int(e.get("action_num", 0)), tuple(diff)))
                s["rows"].update(diff)
                for r in diff:
                    changed_rows_count[r] = changed_rows_count.get(r, 0) + 1
        prev = cur

    # строки-индикаторы: меняются почти на каждом действии
    hud = {r for r, c in changed_rows_count.items() if c >= 0.8 * len(acts)}
    lines = []
    for k, s in sorted(stats.items(), key=lambda x: -x[1]["n"]):
        real = [(num, rs) for num, rs in s["changed"] if set(rs) - hud]
        if real:
            steps = ", ".join(str(n) for n, _ in real[:3])
            lines.append("%s: used %d times, changed the play area on moves %s (%d of %d)."
                         % (k, s["n"], steps, len(real), s["n"]))
        elif s["changed"] and hud:
            lines.append("%s: used %d times; every change was confined to rows %s, which change on "
                         "almost every move and are therefore a meter, not the puzzle."
                         % (k, s["n"], ",".join(str(r) for r in sorted(hud))))
        else:
            lines.append("%s: used %d times, never changed the board." % (k, s["n"]))
    head = ("Evidence from your own log in this game, moves you already played (verified by the "
            "harness against the frames before and after each move):")
    tail = ("Treat these as observations of the same kind you rely on yourself. If they contradict "
            "your current hypothesis, the observations are the ones to keep.")
    return "\n".join([head] + ["- " + l for l in lines] + [tail]), hud, len(acts)


def main():
    import lzma
    d = json.loads(lzma.open(PAYLOAD).read().decode("utf-8"))
    hints = json.load(open("docs/oracle_hints.json", encoding="utf-8"))
    out = {}
    for it in d["items"]:
        gid = it["game"]
        if gid not in hints:
            continue
        res = evidence_for(gid, int(it["analysis_step"]))
        if not res:
            print("%-6s журнала нет" % gid)
            continue
        text, hud, n = res
        out[gid] = text
        print("=== %s (ходов до записанного момента: %d, строки-индикаторы: %s)"
              % (gid, n, ",".join(str(r) for r in sorted(hud)) or "нет"))
        for l in text.splitlines()[1:-1]:
            print("   " + l)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print()
    print("собрано игр: %d -> %s" % (len(out), OUT))


if __name__ == "__main__":
    main()

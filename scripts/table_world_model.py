"""Табличная модель мира БЕЗ языковой модели: проверка на уже записанных прогонах.

Идея (вопрос владельца 08.09 «а синтезатор можно на С++ написать?»): цена синтезатора —
не процессор, а обращения к карте. Если модель переходов выводить ПРОГРАММОЙ из наблюдений,
она стоит ноль запросов и не может испортить темп игры.

Строим ровно то, что у OPINE-World называется объектной моделью, но без языковой модели:
  * объекты — связные области одного цвета (аналог segmentation бандла);
  * тип объекта — цвет + форма без привязки к позиции (аналог их hash);
  * правило — для каждой пары (тип, действие) сдвиг по осям и изменение размера;
  * приём ТОЧНЫЙ, как в статье: таблица принимается, только если воспроизводит КАЖДЫЙ
    записанный переход.

Запуск: .venv/bin/python scripts/table_world_model.py runs/flash_wm_v5 [ещё каталоги]
"""
import json, sys, glob, os
from collections import deque, defaultdict


def objects(board):
    """Связные области одного цвета. Фон — самый частый цвет — не объект."""
    h, w = len(board), len(board[0])
    cnt = defaultdict(int)
    for row in board:
        for c in row:
            cnt[c] += 1
    bg = max(cnt, key=cnt.get)
    seen = [[False] * w for _ in range(h)]
    out = []
    for y in range(h):
        for x in range(w):
            if seen[y][x] or board[y][x] == bg:
                continue
            col = board[y][x]; q = deque([(y, x)]); seen[y][x] = True; cells = []
            while q:
                cy, cx = q.popleft(); cells.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and board[ny][nx] == col:
                        seen[ny][nx] = True; q.append((ny, nx))
            ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
            y0, x0 = min(ys), min(xs)
            shape = frozenset((c[0] - y0, c[1] - x0) for c in cells)
            out.append({"t": (col, shape), "y": y0, "x": x0, "n": len(cells)})
    return out


def state(board):
    """Состояние: объекты, сгруппированные по типу и упорядоченные по позиции."""
    by = defaultdict(list)
    for o in objects(board):
        by[o["t"]].append(o)
    for t in by:
        by[t].sort(key=lambda o: (o["y"], o["x"]))
    return by


def transitions(path):
    """Переходы (доска до, действие, доска после) из записанного прогона."""
    prev = None; out = []
    for line in open(path):
        try: e = json.loads(line)
        except Exception: continue
        b = e.get("board")
        if not b: continue
        if e.get("type") == "action" and prev is not None:
            a = str(e.get("action_name") or e.get("action_display") or "").strip()
            if a: out.append((prev, a, b))
        if e.get("type") in ("action", "initial"):
            prev = b
    return out


def fit(trs):
    """Обучение таблицы: (тип, действие) -> наблюдённый эффект. Противоречие -> отказ строки."""
    table = {}; bad = set()
    for before, act, after in trs:
        sb, sa = state(before), state(after)
        for t, objs in sb.items():
            others = sa.get(t, [])
            if len(others) != len(objs):
                bad.add((t, act)); continue      # объекты появились или исчезли — строка не табличная
            for o, n in zip(objs, others):
                eff = (n["y"] - o["y"], n["x"] - o["x"], n["n"] - o["n"])
                key = (t, act)
                if key in table and table[key] != eff:
                    bad.add(key)
                table[key] = eff
    for k in bad:
        table.pop(k, None)
    return table, bad


def exact(trs, table):
    """Точный приём: таблица обязана воспроизвести КАЖДЫЙ переход целиком."""
    ok = 0
    for before, act, after in trs:
        sb, sa = state(before), state(after)
        good = True
        for t, objs in sb.items():
            key = (t, act)
            if key not in table: good = False; break
            dy, dx, dn = table[key]
            others = sa.get(t, [])
            if len(others) != len(objs): good = False; break
            for o, n in zip(objs, others):
                if (n["y"], n["x"], n["n"]) != (o["y"] + dy, o["x"] + dx, o["n"] + dn):
                    good = False; break
            if not good: break
        if good and set(sa) == set(sb):
            ok += 1
    return ok


def main(dirs):
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "artifacts", "*_events.jsonl")))
        if not files:
            print("нет записей в", d); continue
        tot_tr = full = 0; rows = []
        for f in files:
            g = os.path.basename(f).split("-")[0]
            trs = transitions(f)
            if not trs: rows.append((g, 0, 0, 0.0)); continue
            table, bad = fit(trs)
            ok = exact(trs, table)
            tot_tr += len(trs); full += (ok == len(trs))
            rows.append((g, len(trs), ok, ok / len(trs)))
        print("\n=== %s: игр %d, переходов %d" % (d, len(files), tot_tr))
        print("   игр, где таблица воспроизводит ВСЕ переходы точно: %d из %d" % (full, len(files)))
        cov = [r[3] for r in rows if r[1]]
        if cov:
            cov.sort()
            print("   доля воспроизведённых переходов: медиана %.2f, среднее %.2f" % (cov[len(cov)//2], sum(cov)/len(cov)))
        for g, n, ok, frac in sorted(rows, key=lambda r: -r[3])[:8]:
            print("     %-6s переходов %3d, воспроизведено %3d  (%.0f%%)" % (g, n, ok, 100 * frac))


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/flash_wm_v5"])

"""Табличная модель мира v2: расщепление строк по обстановке — без языковой модели.

Первая версия (scripts/table_world_model.py) дала 1% воспроизведённых переходов, и разбор
показал две причины, обе в представлении, а не в идее:
  * тип был привязан к точной форме — 91% переходов меняли набор типов, потому что объекты
    перерисовываются;
  * правило было одно на пару (тип, действие), без признака обстановки — строки противоречили
    друг другу чаще, чем совпадали.

Здесь исправлено ровно это, по разделам 3.1 и 3.4 статьи OPINE-World:
  * объект — связная область одного цвета; ТИП = цвет, форма ушла в атрибуты;
  * сопоставление кадров — по ближайшему центру внутри цвета, форма меняться может;
  * строка правила — (цвет, действие, ПРИЗНАК ОБСТАНОВКИ), признак подбирается жадно из
    набора кандидатов, пока строка не перестанет противоречить себе;
  * цвет, который не удаётся объяснить ни одним признаком, объявляется УКРАШЕНИЕМ и
    выбрасывается из состояния — это автоматический аналог того, что у них делает языковая
    модель, выбирая представление;
  * приём ТОЧНЫЙ: таблица обязана воспроизвести каждый переход по всем оставленным цветам.

Запуск: .venv/bin/python scripts/table_world_model2.py runs/flash_wm_v3
"""
import json, sys, glob, os
from collections import deque, defaultdict

DIRS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


def components(board):
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
            out.append({"c": col, "y": y0, "x": x0, "n": len(cells),
                        "h": max(ys) - y0 + 1, "w": max(xs) - x0 + 1,
                        "cy": sum(ys) / len(ys), "cx": sum(xs) / len(xs),
                        "shape": frozenset((cc[0] - y0, cc[1] - x0) for cc in cells)})
    return out, bg


def match(before, after):
    """Сопоставление объектов между кадрами по ближайшему центру внутри цвета."""
    pairs = []; born = []; gone = []
    by_c_b = defaultdict(list); by_c_a = defaultdict(list)
    for o in before: by_c_b[o["c"]].append(o)
    for o in after: by_c_a[o["c"]].append(o)
    for c in set(by_c_b) | set(by_c_a):
        bs = sorted(by_c_b.get(c, []), key=lambda o: (o["cy"], o["cx"]))
        as_ = sorted(by_c_a.get(c, []), key=lambda o: (o["cy"], o["cx"]))
        used = set()
        for o in bs:
            best = None; bd = None
            for i, n in enumerate(as_):
                if i in used: continue
                d = (o["cy"] - n["cy"]) ** 2 + (o["cx"] - n["cx"]) ** 2
                if bd is None or d < bd: bd, best = d, i
            if best is None: gone.append(o)
            else: used.add(best); pairs.append((o, as_[best]))
        born.extend(n for i, n in enumerate(as_) if i not in used)
    return pairs, born, gone


def ctx_none(o, objs, board, bg, act): return 0

def ctx_ahead(o, objs, board, bg, act):
    """Что стоит вплотную в сторону хода: стена, другой объект или пусто."""
    d = DIRS.get(act)
    if d is None: return "n/a"
    dy, dx = d; h, w = len(board), len(board[0])
    seen = set()
    for (ry, rx) in o["shape"]:
        y, x = o["y"] + ry + dy, o["x"] + rx + dx
        if not (0 <= y < h and 0 <= x < w): seen.add("край"); continue
        v = board[y][x]
        if v != o["c"]: seen.add(v)
    return tuple(sorted(str(s) for s in seen))

def ctx_size(o, objs, board, bg, act): return o["n"]
def ctx_shape(o, objs, board, bg, act): return hash(o["shape"]) % 100000
def ctx_unique(o, objs, board, bg, act): return sum(1 for x in objs if x["c"] == o["c"]) == 1
def ctx_pos(o, objs, board, bg, act): return (o["y"] % 2, o["x"] % 2)

CTX = [("без обстановки", ctx_none), ("что впереди", ctx_ahead), ("размер", ctx_size),
       ("форма", ctx_shape), ("единственный своего цвета", ctx_unique), ("чётность позиции", ctx_pos)]


def observations(trs):
    """Наблюдения: для каждого объекта — эффект хода, вместе со всеми признаками обстановки."""
    obs = []
    for before, act, after in trs:
        ob, bg = components(before); oa, _ = components(after)
        pairs, born, gone = match(ob, oa)
        for o, n in pairs:
            eff = ("move", n["y"] - o["y"], n["x"] - o["x"], n["n"] - o["n"])
            feats = {name: fn(o, ob, before, bg, act) for name, fn in CTX}
            obs.append((o["c"], act, feats, eff))
        for o in gone:
            obs.append((o["c"], act, {name: fn(o, ob, before, bg, act) for name, fn in CTX}, ("gone",)))
        for n in born:
            obs.append((n["c"], act, {name: "born" for name, _ in CTX}, ("born",)))
    return obs


def fit(obs):
    """Для каждой пары (цвет, действие) подобрать ПЕРВЫЙ признак, снимающий противоречия."""
    rows = defaultdict(list)
    for col, act, feats, eff in obs:
        rows[(col, act)].append((feats, eff))
    chosen = {}; unexplained = set()
    for key, items in rows.items():
        for name, _ in CTX:
            groups = defaultdict(set)
            for feats, eff in items:
                groups[feats[name]].add(eff)
            if all(len(v) == 1 for v in groups.values()):
                chosen[key] = (name, {k: next(iter(v)) for k, v in groups.items()})
                break
        else:
            unexplained.add(key[0])
    return chosen, unexplained


def check(trs, chosen, drop):
    """Точный приём по оставленным цветам."""
    ok = 0
    for before, act, after in trs:
        ob, bg = components(before); oa, _ = components(after)
        ob = [o for o in ob if o["c"] not in drop]; oa = [o for o in oa if o["c"] not in drop]
        pairs, born, gone = match(ob, oa)
        good = True
        for o, n in pairs:
            key = (o["c"], act)
            if key not in chosen: good = False; break
            name, table = chosen[key]
            f = dict((nm, fn(o, ob, before, bg, act)) for nm, fn in CTX)[name]
            pred = table.get(f)
            if pred is None or pred[0] != "move": good = False; break
            if (n["y"], n["x"], n["n"]) != (o["y"] + pred[1], o["x"] + pred[2], o["n"] + pred[3]):
                good = False; break
        if good and not born and not gone:
            ok += 1
    return ok


def main(dirs):
    import importlib.util
    spec = importlib.util.spec_from_file_location("twm", os.path.join(os.path.dirname(__file__), "table_world_model.py"))
    twm = importlib.util.module_from_spec(spec); spec.loader.exec_module(twm)
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "artifacts", "*_events.jsonl")))
        full = 0; tot = 0; rows = []
        for f in files:
            g = os.path.basename(f).split("-")[0]
            trs = twm.transitions(f)
            if not trs: continue
            obs = observations(trs)
            chosen, drop = fit(obs)
            ok = check(trs, chosen, drop)
            tot += len(trs); full += (ok == len(trs))
            picks = defaultdict(int)
            for name, _ in chosen.values(): picks[name] += 1
            rows.append((g, len(trs), ok, len(chosen), len(drop), dict(picks)))
        print("\n=== %s: игр %d, переходов %d" % (d, len(rows), tot))
        print("   игр, где таблица воспроизводит ВСЕ переходы: %d" % full)
        cov = [r[2] / r[1] for r in rows if r[1]]
        cov.sort()
        if cov: print("   доля воспроизведённых переходов: медиана %.2f, среднее %.2f" % (cov[len(cov)//2], sum(cov)/len(cov)))
        allpicks = defaultdict(int)
        for r in rows:
            for k, v in r[5].items(): allpicks[k] += v
        print("   какие признаки обстановки выбирались:", dict(sorted(allpicks.items(), key=lambda kv: -kv[1])))
        for g, n, ok, nrows, ndrop, picks in sorted(rows, key=lambda r: -(r[2] / max(1, r[1])))[:8]:
            print("     %-6s переходов %3d, воспроизведено %3d (%3.0f%%), строк %3d, цветов-украшений %d"
                  % (g, n, ok, 100 * ok / n, nrows, ndrop))


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/flash_wm_v3"])

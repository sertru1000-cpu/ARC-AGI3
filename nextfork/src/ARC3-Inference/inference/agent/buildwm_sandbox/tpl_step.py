# --- a8b: шаблоны правил (rule_learner_probe.Learner, 27.09) как заготовка tpl_step; в песочнице a8b разрешены numpy, hashlib, классы ---
import numpy as np
import hashlib
import json
from collections import Counter, defaultdict
KEYS = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}
_TPL_NAMES = {"UP": "ACTION1", "DOWN": "ACTION2", "LEFT": "ACTION3", "RIGHT": "ACTION4"}


def comps(g):
    """Связные одноцветные компоненты (4-связность): список (color, cells[np.array Nx2], hash)."""
    H, W = g.shape; seen = np.zeros_like(g, dtype=bool); out = []
    for r in range(H):
        for c in range(W):
            if seen[r, c]:
                continue
            col = g[r, c]; stack = [(r, c)]; seen[r, c] = True; cells = []
            while stack:
                y, x = stack.pop(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == col:
                        seen[ny, nx] = True; stack.append((ny, nx))
            a = np.array(cells); rel = a - a.min(0)
            h = hashlib.blake2b(bytes([int(col)]) + np.ascontiguousarray(rel[np.lexsort((rel[:, 1], rel[:, 0]))]).tobytes(), digest_size=8).hexdigest()
            out.append((int(col), a, h))
    return out


class Learner:
    def __init__(self):
        self.moves = defaultdict(Counter)       # action -> Counter(hash -> list of deltas) via key (hash, dy, dx)
        self.passable = Counter()               # цвета, на которые заезжали движущиеся объекты
        self.revealed = Counter()               # цвета, открывавшиеся под уехавшими
        self.recolor = defaultdict(Counter)     # (from_color) -> Counter(to_color) для кликов
        self.table = {}                         # (board hash, action) -> next board
        self.mode_score = defaultdict(lambda: [0, 0])   # action -> [очки фиксированного, очки скольжения]
        self.last_mode = {}
        self.under = {}                         # клетка -> цвет, который там был до того, как на неё заехал движущийся объект

    def key(self, g):
        return hashlib.blake2b(g.tobytes(), digest_size=8).hexdigest()

    def observe(self, g0, a, data, g1):
        self.table[(self.key(g0), a, json.dumps(data, sort_keys=True))] = g1.copy()
        if a in KEYS:
            c0 = {h: (col, cells) for col, cells, h in comps(g0) if len(cells) <= 400}
            for col, cells, h in comps(g1):
                if h in c0 and len(cells) <= 400:
                    d = tuple(cells.min(0) - c0[h][1].min(0))
                    if d != (0, 0):
                        self.moves[a][(h, d[0], d[1])] += 1
                        for y, x in cells:
                            self.passable[int(g0[y, x])] += 1
                            if g0[y, x] != col:
                                self.under[(int(y), int(x))] = int(g0[y, x])
                        old = set(map(tuple, c0[h][1])) - set(map(tuple, cells))
                        for y, x in old:
                            self.revealed[int(g1[y, x])] += 1
        elif a == "ACTION6" and data:
            y, x = int(data.get("y", -1)), int(data.get("x", -1))
            if 0 <= y < g0.shape[0] and 0 <= x < g0.shape[1] and g0[y, x] != g1[y, x]:
                self.recolor[int(g0[y, x])][int(g1[y, x])] += 1

    def _apply_moves(self, g, a, slide):
        """Двигается ГРУППА объектов, согласных с направлением хода (выучено из данных); упор проверяется целиком."""
        if not self.moves[a] or not self.revealed:
            return None
        dirs = Counter()
        for (h, dy, dx), n in self.moves[a].items():
            if (dy == 0) != (dx == 0):                            # только прямые сдвиги
                dirs[(int(np.sign(dy)), int(np.sign(dx)))] += n
        if not dirs:
            return None
        uy, ux = dirs.most_common(1)[0][0]
        mags = Counter()
        rules = defaultdict(Counter)
        for (h, dy, dx), n in self.moves[a].items():
            if (int(np.sign(dy)), int(np.sign(dx))) == (uy, ux) and (dy == 0) != (dx == 0):
                rules[h][abs(dy) + abs(dx)] += n; mags[abs(dy) + abs(dx)] += n
        step = mags.most_common(1)[0][0]
        cs = comps(g); cnt = Counter(h for _, _, h in cs)
        movers = [(col, cells) for col, cells, h in cs if h in rules and cnt[h] <= 2]
        if not movers:
            return None
        bg = self.revealed.most_common(1)[0][0]
        own = set((int(y), int(x)) for _, cells in movers for y, x in cells)
        def free(k):
            for _, cells in movers:
                for y, x in cells:
                    ny, nx = int(y) + uy * k, int(x) + ux * k
                    if not (0 <= ny < g.shape[0] and 0 <= nx < g.shape[1]):
                        return False
                    if (ny, nx) not in own and self.passable[int(g[ny, nx])] == 0:
                        return False
            return True
        if slide:
            k = 0
            while k < 64 and free(k + 1):
                k += 1
        else:
            k = step if free(step) else 0
        out = g.copy()
        if k == 0:
            return out
        for _, cells in movers:
            for y, x in cells:
                out[y, x] = self.under.get((int(y), int(x)), bg)   # проявляется то, что было под объектом
        for col, cells in movers:
            for y, x in cells:
                out[y + uy * k, x + ux * k] = col
        return out

    def predict(self, g, a, data):
        t = self.table.get((self.key(g), a, json.dumps(data, sort_keys=True)))
        if t is not None:
            return t.copy(), "table"
        if a in KEYS:
            fixed = self._apply_moves(g, a, slide=False); slide = self._apply_moves(g, a, slide=True)
            if fixed is None:
                return g.copy(), "identity"
            sf, ss = self.mode_score[a]
            self.last_mode[a] = (fixed, slide)
            return (slide if ss > sf else fixed), "rule"
        if a == "ACTION6" and data:
            y, x = int(data.get("y", -1)), int(data.get("x", -1))
            if 0 <= y < g.shape[0] and 0 <= x < g.shape[1] and self.recolor.get(int(g[y, x])):
                to = self.recolor[int(g[y, x])].most_common(1)[0][0]
                out = g.copy()
                for col, cells, h in comps(g):
                    if any((cy, cx) == (y, x) for cy, cx in map(tuple, cells)):
                        for cy, cx in cells:
                            out[cy, cx] = to
                        break
                return out, "rule"
        return g.copy(), "identity"


def _tpl_act(action):
    """Ход песочницы -> (ACTIONn, data) как в движке: 'RIGHT' -> ACTION4; 'MOUSE(row=R, col=C)' -> ACTION6, {x, y}."""
    s = str(action).strip()
    if s.upper().startswith("MOUSE"):
        nums = [int(t) for t in "".join(ch if ch.isdigit() else " " for ch in s).split()]
        if len(nums) >= 2:
            return "ACTION6", {"x": nums[1], "y": nums[0]}
    return _TPL_NAMES.get(s.upper(), s.upper()), {}


def _tpl_grid(rows):
    return np.array([[ord(ch) for ch in r] for r in rows], dtype=np.int16)


_TPL_CACHE = {}


def _tpl_learner():
    ts = level_transitions()
    key = (len(ts), ts[-1][1] if ts else "")
    if _TPL_CACHE.get("key") != key:
        L = Learner()
        for b, a, af in ts:
            g0, g1 = _tpl_grid(b), _tpl_grid(af)
            if g0.shape != g1.shape:
                continue
            act, data = _tpl_act(a)
            L.predict(g0, act, data)                              # как в зонде: сначала предсказать, потом счёт режимов
            if act in L.last_mode:
                fixed, slide = L.last_mode.pop(act)
                L.mode_score[act][0] += bool(np.array_equal(fixed, g1)); L.mode_score[act][1] += bool(np.array_equal(slide, g1))
            L.observe(g0, act, data, g1)
        _TPL_CACHE.update(key=key, L=L)
    return _TPL_CACHE["L"]


def tpl_step(rows, action):
    """Шаблоны правил, выученные на переходах ЭТОГО уровня: таблица увиденного, сдвиг группы объектов по ходу
    (с упором и скольжением), перекраска по клику. Если правила нет — base_step."""
    rows = [str(x) for x in rows]
    act, data = _tpl_act(action)
    p, kind = _tpl_learner().predict(_tpl_grid(rows), act, data)
    if kind == "identity":
        return base_step(rows, action)
    return ["".join(chr(int(v)) for v in r) for r in p]
# --- конец шаблонов a8b ---

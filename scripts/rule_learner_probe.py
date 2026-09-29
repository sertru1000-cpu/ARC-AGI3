"""Модель мира без языковой модели: алгоритм учит простые правила по ходу игры (27.09, офлайн, без квоты).

Вопрос владельца: можно ли достраивать модель мира параллельно игре, не тратя вызовы модели. Проверка: переходы
записанных прогонов подаются по одному, как в бою; перед каждым ходом t алгоритм, обученный на ходах < t той же игры,
предсказывает следующую доску; сверяем с настоящей (локальный движок повторяет бой 561/561).

Правила (все выводятся из наблюдений, ничего не задано под игру):
  движение — для клавиатурного хода объекты с данным «отпечатком» (цвет+форма, как segmentation) сдвигаются на
             выученный вектор; «проходимые» цвета — те, на которые движущиеся объекты уже заезжали; если место назначения
             занято непроходимым цветом — объект стоит (упёрся). Две гипотезы на ход: фиксированный шаг / скольжение
             единичными шагами до упора — выбирается та, что лучше объясняла прошлое;
             освободившиеся клетки заливаются цветом, который обычно открывается под уехавшим объектом;
  клик     — объект под кликом перекрашивается по выученному соответствию цветов (если такое видели);
  индикатор — строки/столбцы у края, которые менялись на большинстве ходов, при сверке исключаются
             (счётчик ходов предсказывать не нужно — отдельная метрика «с индикатором»).
Базлайны: «ничего не меняется»; «таблица» — этот же ход из этой же доски уже видели → повторить тот результат.
Метрики: доля точно предсказанных переходов (без индикатора); сходимость — первый ход, после которого 8 из
следующих 10 предсказаний точны.
usage: .venv/bin/python scripts/rule_learner_probe.py [runs/flash_v1_phaseA ...]
"""
from __future__ import annotations
import hashlib, json, logging, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
logging.disable(logging.ERROR)
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

KEYS = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}


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

    @staticmethod
    def key(g):
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

    def score_modes(self, a, g1, mask):
        if a in self.last_mode:
            fixed, slide = self.last_mode.pop(a)
            self.mode_score[a][0] += eq(fixed, g1, mask); self.mode_score[a][1] += eq(slide, g1, mask)


def hud_mask(trans, H, W):
    cnt = Counter()
    for g0, _, _, g1 in trans:
        d = np.argwhere(g0 != g1)
        rows = {int(r) for r, c in d if r <= 2 or r >= H - 3}; cols = {int(c) for r, c in d if c <= 2 or c >= W - 3}
        cnt.update(("r", r) for r in rows); cnt.update(("c", c) for c in cols)
    n = max(len(trans), 1)
    m = np.zeros((H, W), dtype=bool)
    for (k, i), v in cnt.items():
        if v >= 0.5 * n and n >= 4:
            if k == "r": m[i, :] = True
            else: m[:, i] = True
    return m


def eq(p, g, mask):
    return bool(p is not None and p.shape == g.shape and np.array_equal(p[~mask], g[~mask]))


def game_transitions(game_id, history, env_dir):
    env = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir).make(game_id)
    fr = env.reset(); g = np.asarray(fr.frame[-1], dtype=np.int16); lv = fr.levels_completed or 0; out = []
    for rec in history:
        a = rec.get("action") or {}
        if not a.get("id") or a["id"] == "RESET":
            continue
        fr = env.step(GameAction[a["id"]], data=a.get("data"))
        if fr is None or not fr.frame:
            break
        g1 = np.asarray(fr.frame[-1], dtype=np.int16); lv1 = fr.levels_completed or 0
        if lv1 == lv and g1.shape == g.shape:
            out.append((g, a["id"], a.get("data") or {}, g1))
        g, lv = g1, lv1
    return out


def main():
    runs = sys.argv[1:] or ["runs/flash_v1_phaseA"]
    env_dir = str(ROOT / "environment_files")
    tot = Counter(); rows = []
    for run in runs:
        for gr in json.loads((ROOT / run / "benchmark.json").read_text())["game_runs"]:
            tr = game_transitions(gr["game_id"], gr.get("history") or [], env_dir)
            if len(tr) < 5:
                continue
            L = Learner(); hits = []; kinds = Counter(); ident = table = 0
            for t, (g0, a, data, g1) in enumerate(tr):
                mask = hud_mask(tr[:t], *g0.shape)
                p, kind = L.predict(g0, a, data)
                ok = eq(p, g1, mask); hits.append(ok); kinds[(kind, ok)] += 1
                ident += eq(g0, g1, mask)
                table += (kind == "table" and ok)
                L.score_modes(a, g1, mask)
                L.observe(g0, a, data, g1)
            conv = next((t for t in range(len(hits) - 9) if sum(hits[t:t + 10]) >= 8), None)
            n = len(hits)
            rows.append((gr["game_id"][:4], n, sum(hits), ident, conv, kinds))
            tot["n"] += n; tot["hit"] += sum(hits); tot["ident"] += ident
            tot["rule_hit"] += kinds[("rule", True)]; tot["rule_n"] += kinds[("rule", True)] + kinds[("rule", False)]
            tot["table_hit"] += kinds[("table", True)]; tot["table_n"] += kinds[("table", True)] + kinds[("table", False)]
            tot["conv"] += conv is not None
    for g, n, h, i, conv, k in sorted(rows):
        print("%s переходов %4d | предсказано %3d%% | «ничего не меняется» %3d%% | правило %d/%d | таблица %d/%d | сошлось на ходу %s" % (
            g, n, 100 * h // max(n, 1), 100 * i // max(n, 1), k[("rule", True)], k[("rule", True)] + k[("rule", False)],
            k[("table", True)], k[("table", True)] + k[("table", False)], conv if conv is not None else "—"))
    print("\nИТОГО: переходов %d | предсказано точно %.0f%% (без индикатора) | «ничего не меняется» %.0f%% | "
          "правила %.0f%% из %d срабатываний | таблица %.0f%% из %d | игр сошлось (8 из 10 подряд) %d из %d" % (
              tot["n"], 100 * tot["hit"] / tot["n"], 100 * tot["ident"] / tot["n"], 100 * tot["rule_hit"] / max(tot["rule_n"], 1),
              tot["rule_n"], 100 * tot["table_hit"] / max(tot["table_n"], 1), tot["table_n"], tot["conv"], len(rows)))


if __name__ == "__main__":
    main()

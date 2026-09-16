"""Перебор перед моделью на стоковом Duck (16.09, слово владельца «перебор в бою — тоже попробуем»).

Замысел: в начале каждой игры, до первого вызова модели, обвязка сама ищет уровень 1 перебором по настоящей среде:
состояние = доска с маской «часов», раскрытие узла = RESET + повтор пути + ход (снимков среды в бою нет, режим ONLINE),
алфавит = стрелки/SPACE из valid_actions + «живые» клики (точки сетки, меняющие доску из старта). Бюджет на игру:
_BF_MOVES ходов (12000) и _BF_SECONDS секунд (600). Нашли уровень -- модель начинает с уровня 2; не нашли -- RESET, модель играет как база.
Ходы перебора идут через step_env(probe=True): зачётные, но не пишутся в историю модели. Маркер [[BFS]] в stdout на игру,
итог в /kaggle/working/bfs_stats.json.

Пороги, записанные ДО пуска, против базы runs/flash_v1_phaseA (10.25, 40 уровней): ИЗМЕРЕНО локально (docs/bfs2_compare_16_09.txt)
уровень 1 берётся перебором в 15/25 игр при неограниченном бюджете; в бюджет 12000 ходов (с повтором пути) укладываются примерно ar25, ft09, lf52, lp85, ls20,
sk48 (2849 ходов со снимками, с повтором пути -- нет), sp80, tu93, vc33, cd82 -- из них база не берёт sk48, sp80, cd82 (ур.1).
Ожидание по уровням: +2..3 над 40; по баллу RHAE: ≈ +0 (уровень за тысячи ходов стоит ~0), возможен вред от потраченного
времени (до 10 мин из 132 на игру). Польза -- дельта >= +4 балла и парный критерий p < 0.05; вред -- <= -4; иначе шум.
Механизм: строк [[BFS]] = 25, суммарные ходы перебора в логе. Только оффлайн (TRUE_SUBMISSION=False).
Проба: --probe ставит потолок игры 1800 с (сравнение с docs/base30_flash_v1_h115.json, 3.32).
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_lvfact_reset_notebook import build  # noqa: E402

PROBE_CAP_S = 1800.0

BFS_CELL = r'''
# =====================================================================
# ПЕРЕБОР ПЕРЕД МОДЕЛЬЮ (16.09): до первого вызова модели обвязка ищет уровень 1 перебором по настоящей среде
# (RESET + повтор пути + ход; состояние = доска с маской часов; клики -- «живые» точки сетки). Только оффлайн.
# =====================================================================
import time as _bf_time, json as _bf_json, random as _bf_random, atexit as _bf_atexit, os as _bf_os
from collections import deque as _bf_deque
import numpy as _bf_np
import inference.framework.solver as _bf_solver
_BF_MOVES, _BF_SECONDS, _BF_MAX_STATES, _BF_CLICK_STEP = __MOVES__, __SECONDS__, 4000, 4
_bf_stats = {"games": 0, "levels": 0, "moves": 0, "seconds": 0.0, "per_game": {}}
_BF_SIMPLE = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")
_BF_M2E = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT", "ACTION5": "SPACE"}


def _bf_info(sess):
    st = sess.game.current_state
    g = _bf_np.asarray(_bf_solver._grid_from_state(st), dtype=_bf_np.int16)
    if g.ndim != 2 or g.size == 0:
        g = _bf_np.zeros((64, 64), dtype=_bf_np.int16)
    return g, int(st.levels_completed or 0), str(st.raw.state).split(".")[-1], list(_bf_solver._engine_action_names(sess.game))


def _bf_do(sess, act, cnt):
    """act = ("RESET", None) | ("ACTIONk", None) | ("ACTION6", (row, col)). Возвращает False при ошибке."""
    cnt["moves"] += 1
    if act[0] == "RESET":
        sess._execute_auto_reset(); return True
    args = {"action": "MOUSE" if act[0] == "ACTION6" else _BF_M2E[act[0]], "probe": True}
    if act[1] is not None:
        args["row"], args["col"] = int(act[1][0]), int(act[1][1])
    p = sess.step_env(args)
    return isinstance(p, dict) and bool(p.get("executed"))


def _bf_replay(sess, path, cnt):
    if not _bf_do(sess, ("RESET", None), cnt):
        return False
    for act in path:
        if not _bf_do(sess, act, cnt):
            return False
    return True


def _bf_exhausted(sess, cnt):
    return cnt["moves"] >= _BF_MOVES or _bf_time.time() > cnt["deadline"] or sess.should_stop()


def _bf_active_clicks(sess, grid0, avail, cnt, mask, cap=48):
    """Клики по сетке, меняющие доску из старта; точки с ОДИНАКОВЫМ результатом (после маски часов) -- одно ребро
    (sp80/cd82: счётчик кликов делает «живой» любую точку). Потолок cap точек."""
    if "ACTION6" not in avail:
        return []
    pts = [(y, x) for y in range(_BF_CLICK_STEP // 2, 64, _BF_CLICK_STEP) for x in range(_BF_CLICK_STEP // 2, 64, _BF_CLICK_STEP) if y < grid0.shape[0] and x < grid0.shape[1]]
    k0 = grid0.copy(); k0[mask] = -1; k0 = k0.tobytes()
    active = []; effects = set()
    for (y, x) in pts:
        if _bf_exhausted(sess, cnt) or len(active) >= cap:
            break
        if not _bf_replay(sess, [("ACTION6", (y, x))], cnt):
            continue
        g, lvl, st, av = _bf_info(sess)
        if lvl > 0:
            active.append(("ACTION6", (y, x))); continue
        if g.shape != grid0.shape or not (g != grid0).any():
            continue
        k = g.copy(); k[mask] = -1; k = k.tobytes()
        if k == k0 or k in effects:
            continue
        effects.add(k); active.append(("ACTION6", (y, x)))
    return active


def _bf_object_clicks(g, cap=12):
    """НЕ ИСПОЛЬЗУЕТСЯ (16.09: lf52 не спасла, sk48/ar25 потеряли глубину). Центры крупнейших связных областей не-фонового цвета (4-связность, чистый python): цели, появившиеся
    после ходов (lf52: новые фигуры), которых нет в стартовом алфавите."""
    h, w = g.shape; vals, counts = _bf_np.unique(g, return_counts=True); bg = int(vals[counts.argmax()])
    seen = _bf_np.zeros_like(g, dtype=bool); comps = []
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0, x0] or int(g[y0, x0]) == bg:
                continue
            col = int(g[y0, x0]); stack = [(y0, x0)]; seen[y0, x0] = True; cells = []
            while stack:
                y, x = stack.pop(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and int(g[yy, xx]) == col:
                        seen[yy, xx] = True; stack.append((yy, xx))
            comps.append(cells)
    comps.sort(key=len, reverse=True); out = []
    for cells in comps[:cap]:
        ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
        out.append(("ACTION6", ((min(ys) + max(ys)) // 2, (min(xs) + max(xs)) // 2)))
    return out


def _bf_clock_mask(sess, grid0, acts, cnt, warm=40, thr=0.8):
    rng = _bf_random.Random(0); changes = _bf_np.zeros_like(grid0, dtype=_bf_np.int32); n = 0; prev = grid0; diffs = []
    row_ch = _bf_np.zeros(grid0.shape[0], dtype=_bf_np.int32); col_ch = _bf_np.zeros(grid0.shape[1], dtype=_bf_np.int32)
    _bf_replay(sess, [], cnt)
    for _ in range(warm):
        if _bf_exhausted(sess, cnt) or not acts:
            break
        _bf_do(sess, rng.choice(acts), cnt); g, lvl, st, av = _bf_info(sess)
        if st == "GAME_OVER" or lvl > 0:
            _bf_replay(sess, [], cnt); prev = _bf_info(sess)[0]; continue
        if g.shape == prev.shape:
            d = (g != prev)
            if d.any():
                changes += d; n += 1; row_ch += d.any(axis=1); col_ch += d.any(axis=0); diffs.append(d)
        prev = g
    mask = (changes >= thr * max(1, n)) & (changes >= 3)
    if n >= 5:
        for r in _bf_np.where(row_ch >= thr * n)[0]: mask[r, :] = True
        for c in _bf_np.where(col_ch >= thr * n)[0]: mask[:, c] = True
    return mask


def _bf_prephase(sess):
    t0 = _bf_time.time(); cnt = {"moves": 0, "deadline": t0 + _BF_SECONDS}
    gid = str(getattr(sess.game, "game_id", "") or "?")
    _bf_replay(sess, [], cnt); grid0, lvl0, st0, avail = _bf_info(sess)
    simple = [(a, None) for a in avail if a in _BF_SIMPLE]
    warm_acts = simple + ([("ACTION6", (y, x)) for y in range(4, 64, 8) for x in range(4, 64, 8)] if "ACTION6" in avail else [])
    mask = _bf_clock_mask(sess, grid0, warm_acts, cnt)
    clicks = _bf_active_clicks(sess, grid0, avail, cnt, mask)
    def key_of(g):
        k = g.copy()
        if k.shape == mask.shape: k[mask] = -1
        return k.tobytes()
    seen = {key_of(grid0): []}; q = _bf_deque([([], grid0, avail)]); found = None; expansions = 0; root_changed = False
    while q and found is None and len(seen) < _BF_MAX_STATES and not _bf_exhausted(sess, cnt):
        if expansions == 1 and len(seen) == 1 and root_changed and mask.any():
            # предохранитель (lp85): маска часов склеила всех детей корня в корень -- снимаем маску и начинаем заново
            mask[:] = False; seen = {key_of(grid0): []}; q = _bf_deque([([], grid0, avail)]); expansions = 0
        path, g, av = q.popleft(); expansions += 1
        for act in [(a, None) for a in av if a in _BF_SIMPLE] + clicks:
            if _bf_exhausted(sess, cnt):
                break
            if not _bf_replay(sess, path + [act], cnt):
                continue
            g2, lvl, st, av2 = _bf_info(sess)
            if lvl > lvl0:
                found = path + [act]; break
            if st == "GAME_OVER":
                continue
            if not path and g2.shape == g.shape and (g2 != g).any():
                root_changed = True
            k = key_of(g2)
            if k in seen:
                continue
            seen[k] = path + [act]; q.append((path + [act], g2, av2))
    if found is None:
        _bf_replay(sess, [], cnt)   # модель начинает со старта
    rec = {"game": gid, "found": found is not None, "path_len": len(found) if found else None, "states": len(seen), "expansions": expansions,
           "clicks": len(clicks), "moves": cnt["moves"], "seconds": round(_bf_time.time() - t0, 1), "stopped_by": ("found" if found else ("moves" if cnt["moves"] >= _BF_MOVES else ("time" if _bf_time.time() > cnt["deadline"] else "exhausted")))}
    _bf_stats["games"] += 1; _bf_stats["levels"] += int(found is not None); _bf_stats["moves"] += cnt["moves"]; _bf_stats["seconds"] += rec["seconds"]; _bf_stats["per_game"][gid] = rec
    print("[[BFS]] " + _bf_json.dumps(rec), flush=True)
    return rec


_bf_orig_play = _bf_solver._HarnessGameSession.play


def _bf_play(self):
    if not getattr(self, "_bf_done", False):
        self._bf_done = True
        try:
            _bf_prephase(self)
        except Exception as exc:
            print("[[BFS]] ошибка " + repr(exc)[:300], flush=True)
    return _bf_orig_play(self)


def _bf_dump():
    try:
        _bf_os.makedirs("/kaggle/working", exist_ok=True)
        _bf_json.dump(_bf_stats, open("/kaggle/working/bfs_stats.json", "w"), ensure_ascii=False, indent=1)
    except Exception:
        pass


if not TRUE_SUBMISSION:
    _bf_solver._HarnessGameSession.play = _bf_play
    _bf_atexit.register(_bf_dump)
    print("[[BFS]] слой установлен: ходов %d, секунд %.0f на игру" % (_BF_MOVES, _BF_SECONDS), flush=True)
'''


def cell(moves: int, seconds: float) -> str:
    return BFS_CELL.replace("__MOVES__", str(moves)).replace("__SECONDS__", repr(float(seconds)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="потолок игры %s с (проба 30 мин)" % PROBE_CAP_S)
    ap.add_argument("--moves", type=int, default=12000); ap.add_argument("--seconds", type=float, default=600.0)
    a = ap.parse_args()
    out = "kernels/notebooks_stockflash_bfs"
    build(cell(a.moves, a.seconds), out, "sergueimakarov/arc3-stock-flash-bfs", "arc3 stock flash bfs", "_bf_stats = ")
    if a.probe:
        p = os.path.join(out, "submission.ipynb")
        nb = json.load(open(p, encoding="utf-8"))
        c15 = "".join(nb["cells"][15]["source"])
        marker = "# Play the benchmark; watchdog stop and teardown run even if it raises."
        assert marker in c15
        c15 = c15.replace(marker, "if not TRUE_SUBMISSION:\n    bm.solver.max_runtime_s_per_game = %r    # проба вне боя\n\n" % PROBE_CAP_S + marker, 1)
        nb["cells"][15]["source"] = c15.splitlines(keepends=True)
        json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("ok   проба: потолок игры %s с" % PROBE_CAP_S)


if __name__ == "__main__":
    main()

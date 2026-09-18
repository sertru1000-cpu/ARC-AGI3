
# =====================================================================
# ПЕРЕБОР ПЕРЕД МОДЕЛЬЮ (16.09): до первого вызова модели обвязка ищет уровень 1 перебором по настоящей среде
# (RESET + повтор пути + ход; состояние = доска с маской часов; клики -- «живые» точки сетки). Только оффлайн.
# =====================================================================
import time as _bf_time, json as _bf_json, random as _bf_random, atexit as _bf_atexit, os as _bf_os
from collections import deque as _bf_deque
import numpy as _bf_np
import inference.framework.solver as _bf_solver
import arcengine as _bf_arcengine
_BF_MOVES, _BF_SECONDS, _BF_MAX_STATES, _BF_CLICK_STEP = 12000, 600.0, 4000, 4
_BF_MODE, _BF_TAIL_S, _BF_STALL_S = "pre", 600.0, 600.0   # tail: перебор в хвосте игры; pre: до первого вызова; stall: после застоя
_BF_MIN_LEFT = 600.0   # stall: перебор только если после него модели остаётся >= _BF_MIN_LEFT с
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
    """act = ("RESET", None) | ("ACTIONk", None) | ("ACTION6", (row, col)). Ход через taaf Game.execute_action
    (публичный API Duck): зачётный, учтён в action_count, но мимо истории модели/вьюера/диска. False при ошибке."""
    cnt["moves"] += 1
    gid = _bf_arcengine.GameAction.RESET if act[0] == "RESET" else _bf_arcengine.GameAction[act[0]]
    data = {"x": int(act[1][1]), "y": int(act[1][0])} if act[1] is not None else {}
    try:
        cnt["last_state"] = sess.game.execute_action(_bf_arcengine.ActionInput(id=gid, data=data), generated_tokens=0, uncached_input_tokens=0)
    except Exception:
        return False
    sess.last_engine_action = gid.name
    return True


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


def _bf_time_left(sess):
    try:
        r = sess.timing_payload().get("time_remaining_seconds")
    except Exception:
        r = None
    return 1e9 if r is None else float(r)


def _bf_prephase(sess, budget_s=None):
    t0 = _bf_time.time(); cnt = {"moves": 0, "deadline": t0 + (float(budget_s) if budget_s is not None else _BF_SECONDS)}
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
            if len(cnt.setdefault("samples", [])) < 200 and len(seen) % 2 == 0:
                cnt["samples"].append(g2)
    if found is None:
        _bf_replay(sess, [], cnt)   # модель начинает со старта
    else:
        try:   # кадры завершающего хода: последний принадлежит уже следующему уровню, предыдущие показывают ЦЕЛЬ
            _frames = [_bf_np.asarray(x, dtype=_bf_np.int16) for x in _bf_solver._raw_frames(cnt.get("last_state"))]
            sess._bf_goal_raw = {"level_done": lvl0 + 1, "goal": (_frames[:-1] or _frames)[-1] if _frames else None,
                                 "samples": list(cnt.get("samples", []))}
        except Exception as _e:
            print("[[BFS]] кадр цели не снят: %r" % (_e,), flush=True)
        try:   # путь в записи модели -- его читает слой carry, если он стоит
            disp = [("MOUSE(row=%d, col=%d)" % (a[1][0], a[1][1])) if a[0] == "ACTION6" else _BF_M2E.get(a[0], a[0]) for a in found]
            sess._bf_found = list(getattr(sess, "_bf_found", []) or []) + [{"level_done": lvl0 + 1, "path": disp, "moves": cnt["moves"]}]
        except Exception:
            pass
    rec = {"game": gid, "found": found is not None, "path_len": len(found) if found else None, "states": len(seen), "expansions": expansions,
           "clicks": len(clicks), "moves": cnt["moves"], "seconds": round(_bf_time.time() - t0, 1), "stopped_by": ("found" if found else ("moves" if cnt["moves"] >= _BF_MOVES else ("time" if _bf_time.time() > cnt["deadline"] else "exhausted")))}
    _bf_stats["games"] += 1; _bf_stats["levels"] += int(found is not None); _bf_stats["moves"] += cnt["moves"]; _bf_stats["seconds"] += rec["seconds"]; _bf_stats["per_game"][gid] = rec
    print("[[BFS]] " + _bf_json.dumps(rec), flush=True)
    return rec


_bf_orig_play = _bf_solver._HarnessGameSession.play


class _BfAnalyzerProxy:
    """Прокси анализатора одной сессии: перед каждым вызовом модели проверяет условие хвоста игры."""
    def __init__(self, inner, sess):
        self._inner = inner; self._sess = sess; self._done = False
        self._last_lvl = int(sess.game.current_state.levels_completed or 0); self._last_t = _bf_time.monotonic(); self._tried = set()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def analyze(self, *a, **k):
        sess = self._sess
        try:
            lvl = int(sess.game.current_state.levels_completed or 0); now = _bf_time.monotonic()
            if lvl != self._last_lvl:
                self._last_lvl = lvl; self._last_t = now
            left = _bf_time_left(sess)
            won = str(sess.game.current_state.raw.state).split(".")[-1] == "WIN"
            if _BF_MODE == "stall":
                if (not won and lvl not in self._tried and (now - self._last_t) >= _BF_STALL_S and left >= _BF_MIN_LEFT + 30.0):
                    self._tried.add(lvl)   # не больше одного перебора на уровень
                    _bf_prephase(sess, budget_s=max(30.0, min(_BF_SECONDS, left - _BF_MIN_LEFT)))
                    lvl2 = int(sess.game.current_state.levels_completed or 0)
                    if lvl2 != lvl:
                        self._last_lvl = lvl2; self._last_t = _bf_time.monotonic()
            elif (not self._done and left <= _BF_TAIL_S and (now - self._last_t) >= _BF_STALL_S and not won):
                self._done = True
                _bf_prephase(sess, budget_s=max(30.0, left - 20.0))
        except Exception as exc:
            print("[[BFS]] ошибка хвоста " + repr(exc)[:300], flush=True)
        return self._inner.analyze(*a, **k)


def _bf_play(self):
    if _BF_MODE == "pre":
        if not getattr(self, "_bf_done", False):
            self._bf_done = True
            try:
                _bf_prephase(self)
            except Exception as exc:
                print("[[BFS]] ошибка " + repr(exc)[:300], flush=True)
    else:
        if not isinstance(self.analyzer, _BfAnalyzerProxy):
            self.analyzer = _BfAnalyzerProxy(self.analyzer, self)
    return _bf_orig_play(self)


def _bf_dump():
    try:
        _bf_os.makedirs("/kaggle/working", exist_ok=True)
        _bf_json.dump(_bf_stats, open("/kaggle/working/bfs_stats.json", "w"), ensure_ascii=False, indent=1)
    except Exception:
        pass


if _bf_os.environ.get("BFS_LAYER", "1") != "0":
    _bf_solver._HarnessGameSession.play = _bf_play
    _bf_atexit.register(_bf_dump)
    print("[[BFS]] слой установлен (%s): ходов %d, секунд %.0f, хвост %.0f с, застой %.0f с, остаток модели %.0f с; TRUE_SUBMISSION=%s" % (_BF_MODE, _BF_MOVES, _BF_SECONDS, _BF_TAIL_S, _BF_STALL_S, _BF_MIN_LEFT, TRUE_SUBMISSION), flush=True)


# =====================================================================
# ЦЕЛЬ ИЗ КАДРА ВЗЯТИЯ УРОВНЯ -> ВО ВХОД МОДЕЛИ (18.09). Выключатель GOALHINT=0.
# =====================================================================
import os as _gh_os, json as _gh_json, atexit as _gh_atexit
import numpy as _gh_np
import inference.agent.tool_agent as _gh_wta
_gh_stats = {"games": 0, "goals": 0, "statements": [], "prompts": 0, "errors": 0}
_GH_MAX = 3   # сколько утверждений о цели показывать модели


def _gh_comps(mask):
    h, w = mask.shape; seen = _gh_np.zeros_like(mask, dtype=bool); out = []
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            st = [(y0, x0)]; seen[y0, x0] = True; cells = []
            while st:
                y, x = st.pop(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True; st.append((yy, xx))
            ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
            out.append({"n": len(cells), "shape": frozenset((y - min(ys), x - min(xs)) for y, x in cells)})
    return out


def _gh_feat(g):
    vals, counts = _gh_np.unique(g, return_counts=True)
    bg = int(vals[counts.argmax()]) if len(vals) else 0
    cnt = {int(v): int(c) for v, c in zip(vals, counts)}
    cm = {c: _gh_comps(g == c) for c in cnt if c != bg}
    return {"bg": bg, "cnt": cnt, "ncolors": len(cnt), "comps": cm,
            "shapes": {c: {x["shape"] for x in cs} for c, cs in cm.items()}}


def _gh_preds(f_goal):
    """(ранг, фраза, проверка) для словаря целей; параметры взяты из кадра ЦЕЛИ. Ранг = насколько утверждение
    содержательно: исчезновение цвета и совпадение форм информативнее, чем «ровно N клеток» у крупного цвета.
    Цвета, занимающие больше пятой части доски (фон и заливка), в утверждения о количестве не идут."""
    out = []; bg = f_goal["bg"]; cols = [c for c in f_goal["cnt"] if c != bg]
    total = max(1, sum(f_goal["cnt"].values())); big = {c for c in cols if f_goal["cnt"][c] > 0.2 * total}
    for c in range(16):
        if f_goal["cnt"].get(c, 0) == 0 and c not in (bg,):
            out.append((0, "no cells of colour %d are left on the board" % c, lambda f, c=c: f["cnt"].get(c, 0) == 0))
    for c in cols:
        cs = f_goal["comps"].get(c, []); m = len(cs)
        for d in cols:
            if c < d and (f_goal["shapes"].get(c, set()) & f_goal["shapes"].get(d, set())):
                out.append((1, "a shape of colour %d matches a shape of colour %d" % (c, d),
                            lambda f, c=c, d=d: bool(f["shapes"].get(c, set()) & f["shapes"].get(d, set()))))
        if m == 1:
            out.append((2, "all cells of colour %d are joined into one group" % c, lambda f, c=c: len(f["comps"].get(c, [])) == 1))
        elif c not in big:
            out.append((3, "colour %d forms exactly %d connected group(s)" % (c, m), lambda f, c=c, m=m: len(f["comps"].get(c, [])) == m))
        if cs and c not in big:
            n = max(x["n"] for x in cs)
            out.append((4, "the largest group of colour %d has exactly %d cells" % (c, n),
                        lambda f, c=c, n=n: bool(f["comps"].get(c)) and max(x["n"] for x in f["comps"][c]) == n))
            k = f_goal["cnt"][c]
            out.append((5, "exactly %d cells of colour %d remain" % (k, c), lambda f, c=c, k=k: f["cnt"].get(c, 0) == k))
    out.append((6, "exactly %d distinct colours are on the board" % f_goal["ncolors"],
                lambda f, k=f_goal["ncolors"]: f["ncolors"] == k))
    return sorted(out, key=lambda x: x[0])


def _gh_infer(goal, samples):
    """утверждения, истинные в кадре цели и ложные во всех образцах не-целевых состояний."""
    if goal is None:
        return []
    f_goal = _gh_feat(goal)
    f_neg = [_gh_feat(g) for g in samples if getattr(g, "shape", None) == goal.shape]
    out = []
    for _rank, text, fn in _gh_preds(f_goal):
        try:
            if not fn(f_goal) or any(fn(x) for x in f_neg):
                continue
        except Exception:
            continue
        out.append(text)
    return out[:_GH_MAX]


def _gh_block(level_done, stmts, same_kind):
    head = ("HARNESS-INFERRED GOAL. The harness solved level %d by its own search and looked at the board at the moment "
            "that level was completed. Compared with every other board seen on that level, these statements were true "
            "only there:" % level_done)
    body = "\n".join("  - %s" % s for s in stmts)
    if same_kind:
        tail = ("You are now past that level. Levels of one game usually share the KIND of goal and differ in the numbers "
                "(more objects, obstacles, distractors), so aim for the same kind of condition on this board and re-derive "
                "the exact numbers yourself. State the goal you settle on in `Goal model:` and verify it before long plans.")
    else:
        tail = ("Use this as the goal of the current level unless the board contradicts it; state your own goal in "
                "`Goal model:` and verify it.")
    return head + "\n" + body + "\n" + tail


if _gh_os.environ.get("GOALHINT", "1") != "0":
    _gh_orig_prompt = _gh_wta.ToolAgent._build_user_prompt

    def _gh_prompt(self, action_num, *args, **kwargs):
        st = getattr(self, "_gh_state", None)
        if st is None:
            st = {"block": None, "level": None}; self._gh_state = st
        try:
            cb = getattr(self, "_step_env_callback", None); sess = getattr(cb, "__self__", None)
            raw = getattr(sess, "_bf_goal_raw", None) if sess is not None else None
            if raw is not None and not st.get("done"):
                st["done"] = True
                stmts = _gh_infer(raw.get("goal"), raw.get("samples") or [])
                _gh_stats["games"] += 1
                if stmts:
                    _gh_stats["goals"] += 1; _gh_stats["statements"].append(stmts)
                    st["stmts"] = stmts; st["level_done"] = int(raw.get("level_done") or 1)
                    st["block"] = _gh_block(st["level_done"], stmts, False)
                    print("[[GOAL]] уровень %d взят перебором; цель: %s" % (st["level_done"], " | ".join(stmts)), flush=True)
                else:
                    print("[[GOAL]] уровень %d взят перебором, но ни одно утверждение не отделило цель" % int(raw.get("level_done") or 1), flush=True)
                _gh_dump()
            lv = getattr(kwargs.get("current_frame"), "level", None)
            if lv is not None and st.get("stmts"):
                lv = int(lv)
                if st["level"] is not None and lv > st["level"]:
                    st["block"] = _gh_block(st["level_done"], st["stmts"], True)   # тот же вид цели, другие числа
                st["level"] = lv if st["level"] is None else max(st["level"], lv)
        except Exception as _e:
            _gh_stats["errors"] += 1; print("[GOAL] сбой: %r" % (_e,), flush=True)
        text = _gh_orig_prompt(self, action_num, *args, **kwargs)
        if st.get("block"):
            _gh_stats["prompts"] += 1
            return st["block"] + "\n\n" + text
        return text

    _gh_wta.ToolAgent._build_user_prompt = _gh_prompt

    def _gh_dump():
        try:
            _gh_os.makedirs("/kaggle/working", exist_ok=True)
            _gh_json.dump(_gh_stats, open("/kaggle/working/goalhint_stats.json", "w"), ensure_ascii=False, indent=1)
        except Exception:
            pass

    _gh_atexit.register(_gh_dump)
    print("[[GOAL]] слой установлен: перебор берёт уровень 1, цель выводится по кадру взятия и идёт во вход модели "
          "(до %d утверждений); TRUE_SUBMISSION=%s" % (_GH_MAX, TRUE_SUBMISSION), flush=True)

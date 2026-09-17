
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
_BF_MODE, _BF_TAIL_S, _BF_STALL_S = "stall", 600.0, 1800.0   # tail: перебор в хвосте игры; pre: до первого вызова; stall: после застоя
_BF_MIN_LEFT = 900.0   # stall: перебор только если после него модели остаётся >= _BF_MIN_LEFT с
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
        sess.game.execute_action(_bf_arcengine.ActionInput(id=gid, data=data), generated_tokens=0, uncached_input_tokens=0)
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
    if found is None:
        _bf_replay(sess, [], cnt)   # модель начинает со старта
    else:
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
# ПЕРЕНОС МОДЕЛИ МИРА ЧЕРЕЗ ВЗЯТИЕ УРОВНЯ + ПОЛНЫЙ ПУТЬ УРОВНЯ (17.09). Выключатель CARRY_LAYER=0.
# =====================================================================
import os as _cr_os, json as _cr_json, atexit as _cr_atexit, re as _cr_re
import inference.agent.tool_agent as _cr_wta
_CR_KEEP = ("world_model", "goal_model", "action_model")
_CR_PATH_MAX = 150
_CR_TAG = _cr_re.compile(r"^\[carried from level \d+ -- re-check on this board\] ")
_cr_stats = {"transitions": 0, "carried_nonempty": 0, "carried_fields": 0, "paths": 0, "path_moves": [], "level_moves": [],
             "prompts_with_path": 0, "errors": 0}

def _cr_rle(acts):
    out = []
    for a in acts:
        if out and out[-1][0] == a:
            out[-1][1] += 1
        else:
            out.append([a, 1])
    return ", ".join(a if n == 1 else "%s x%d" % (a, n) for a, n in out)

def _cr_path(history_entries, new_level):
    """(ходов на уровне всего, ходы от последнего RESET до завершающего включительно) для уровня new_level-1."""
    ents = list(history_entries or [])
    idx = None
    for i, e in enumerate(ents):
        try:
            if int(getattr(e.frame, "level", 0) or 0) >= int(new_level):
                idx = i; break
        except Exception:
            continue
    if idx is None or idx == 0:
        return None
    # кадр записи -- ПОСЛЕ хода: первая запись с уровнем new_level-1 -- это завершающий ход предыдущего уровня
    # (или стартовый кадр без хода), поэтому путь начинается со следующей за ней.
    j = idx - 1
    while j - 1 >= 0 and int(getattr(ents[j - 1].frame, "level", 0) or 0) == int(new_level) - 1:
        j -= 1
    start = j + 1
    acts = [str(getattr(e, "action", "") or "").strip() for e in ents[start:idx + 1]]
    acts = [a for a in acts if a]
    total = len(acts)
    last_reset = max([k for k, a in enumerate(acts) if a.upper().startswith("RESET")], default=-1)
    eff = acts[last_reset + 1:]
    return total, eff

def _cr_block(level_done, total, eff):
    shown = eff[-_CR_PATH_MAX:]
    cut = len(eff) - len(shown)
    head = ("LEVEL %d SOLUTION PATH (exact, from the harness log): level %d was completed by this move sequence "
            "(%d moves since the last RESET of that level; %d moves spent on the level in total; the final move completed it):"
            % (level_done, level_done, len(eff), total))
    body = ("(first %d moves omitted) " % cut if cut > 0 else "") + _cr_rle(shown) + "."
    tail = ("Your world/goal/action model from level %d is carried into the world model below, marked [carried]. "
            "Check on the new board whether the same goal and mechanics still hold and whether an analogous sequence applies, "
            "then act; revise the carried model where the new board contradicts it." % level_done)
    return head + "\n" + body + "\n" + tail

def _cr_block_bfs(rec):
    level_done = int(rec.get("level_done") or 0); path = list(rec.get("path") or [])
    shown = path[-_CR_PATH_MAX:]; cut = len(path) - len(shown)
    head = ("LEVEL %d SOLVED BY HARNESS SEARCH (not by your own moves): while you were stuck, the harness searched the real game and "
            "completed level %d with this move sequence from the start of that level (right after RESET; %d moves in the path, "
            "%d search moves in total). You are now on level %d -- ignore any line saying you are still on the same level:"
            % (level_done, level_done, len(path), int(rec.get("moves") or 0), level_done + 1))
    body = ("(first %d moves omitted) " % cut if cut > 0 else "") + _cr_rle(shown) + "."
    tail = ("Study what this sequence did to infer the goal and mechanics of level %d, update the world model below (your model from level %d "
            "is carried, marked [carried]), then check whether the same goal holds on the new board and act." % (level_done, level_done))
    return head + "\n" + body + "\n" + tail

def _cr_carry_fields(self, prev_level, wipe_rest):
    """world/goal/action -- с пометкой уровня; при wipe_rest findings/questions/plan стираются (как в стоке при переходе)."""
    know = getattr(self, "_summarized_knowledge", None)
    if not isinstance(know, dict):
        return 0
    n = 0
    for k in _CR_KEEP:
        v = _CR_TAG.sub("", str(know.get(k, "") or ""))
        if v.strip():
            know[k] = "[carried from level %d -- re-check on this board] %s" % (prev_level, v); n += 1
    if wipe_rest:
        for k in ("recent_findings", "open_questions", "current_plan"):
            know[k] = ""
    return n

if _cr_os.environ.get("CARRY_LAYER", "1") != "0":
    _cr_orig_update = _cr_wta.ToolAgent._update_summarized_knowledge_from_step_summary
    def _cr_update(self):
        s = getattr(self, "_last_step_summary", None) or {}
        if not (s.get("level_transition") and not s.get("run_complete") and not s.get("game_over")):
            return _cr_orig_update(self)
        try:
            know = getattr(self, "_summarized_knowledge", None) or {}
            saved = {k: _CR_TAG.sub("", str(know.get(k, "") or "")) for k in _CR_KEEP}
        except Exception as _e:
            _cr_stats["errors"] += 1; print("[CARRY] сбой чтения модели мира: %r" % (_e,), flush=True)
            return _cr_orig_update(self)
        out = _cr_orig_update(self)
        try:
            try:
                new_level = int(s.get("level"))
            except (TypeError, ValueError):
                new_level = None
            prev = (new_level - 1) if new_level else 0
            n = 0
            for k, v in saved.items():
                if v.strip():
                    self._summarized_knowledge[k] = "[carried from level %d -- re-check on this board] %s" % (prev, v); n += 1
            _cr_stats["transitions"] += 1; _cr_stats["carried_fields"] += n
            if n:
                _cr_stats["carried_nonempty"] += 1
            print("[[CARRY]] переход на уровень %s: перенесено полей %d (%s)" % (new_level, n, ", ".join(k for k, v in saved.items() if v.strip()) or "-"), flush=True)
            _cr_dump()
        except Exception as _e:
            _cr_stats["errors"] += 1; print("[CARRY] сбой переноса: %r" % (_e,), flush=True)
        return out
    _cr_wta.ToolAgent._update_summarized_knowledge_from_step_summary = _cr_update

    _cr_orig_prompt = _cr_wta.ToolAgent._build_user_prompt
    def _cr_prompt(self, action_num, *args, **kwargs):
        st = getattr(self, "_cr_state", None)
        if st is None:
            st = {"level": None, "block": None}; self._cr_state = st
        try:
            # уровень, взятый перебором (слой BFS, если стоит): разбираем ДО сборки стокового промпта, чтобы перенесённая
            # модель мира и стёртый план попали в этот же промпт
            cb = getattr(self, "_step_env_callback", None); sess = getattr(cb, "__self__", None)
            bf = list(getattr(sess, "_bf_found", []) or []) if sess is not None else []
            if len(bf) > int(st.get("bf_seen", 0)):
                rec = bf[-1]; st["bf_seen"] = len(bf)
                st["block"] = _cr_block_bfs(rec)
                n = _cr_carry_fields(self, int(rec.get("level_done") or 0), True)
                st["level"] = max(int(st["level"] or 0), int(rec.get("level_done") or 0) + 1)
                _cr_stats["bfs_paths"] = int(_cr_stats.get("bfs_paths", 0)) + 1; _cr_stats["transitions"] += 1; _cr_stats["carried_fields"] += n
                if n:
                    _cr_stats["carried_nonempty"] += 1
                print("[[CARRY]] путь перебора уровня %d во входе: %d ходов, перенесено полей %d" % (int(rec.get("level_done") or 0), len(rec.get("path") or []), n), flush=True)
                _cr_dump()
        except Exception as _e:
            _cr_stats["errors"] += 1; print("[CARRY] сбой записи перебора: %r" % (_e,), flush=True)
        text = _cr_orig_prompt(self, action_num, *args, **kwargs)
        try:
            lv = getattr(kwargs.get("current_frame"), "level", None)
            if lv is not None:
                lv = int(lv)
                if st["level"] is not None and lv > st["level"]:
                    st["block"] = None
                    got = _cr_path(kwargs.get("history_entries"), lv)
                    if got is not None:
                        total, eff = got
                        st["block"] = _cr_block(lv - 1, total, eff)
                        _cr_stats["paths"] += 1; _cr_stats["path_moves"].append(len(eff)); _cr_stats["level_moves"].append(total)
                        print("[[CARRY]] путь уровня %d: %d ходов от последнего RESET, всего %d" % (lv - 1, len(eff), total), flush=True)
                        _cr_dump()
                st["level"] = lv if st["level"] is None else max(st["level"], lv)
            if st.get("block"):
                _cr_stats["prompts_with_path"] += 1
                return st["block"] + "\n\n" + text
            return text
        except Exception as _e:
            _cr_stats["errors"] += 1; print("[CARRY] сбой промпта: %r" % (_e,), flush=True)
            return text
    _cr_wta.ToolAgent._build_user_prompt = _cr_prompt

    def _cr_dump():
        try:
            _cr_os.makedirs("/kaggle/working", exist_ok=True)
            _cr_json.dump(_cr_stats, open("/kaggle/working/carry_stats.json", "w"), indent=1)
        except Exception:
            pass
    _cr_atexit.register(_cr_dump)
    print("[[CARRY]] слой установлен: модель мира (world/goal/action) переносится через взятие уровня, путь уровня во входе "
          "(до %d ходов); TRUE_SUBMISSION=%s" % (_CR_PATH_MAX, TRUE_SUBMISSION), flush=True)

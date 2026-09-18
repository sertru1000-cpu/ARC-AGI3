
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


def _gh_preds(f_goal, f_start=None):
    """(ранг, фраза, проверка). Ранг = ИЗМЕРЕННАЯ доля переноса вида цели на следующий уровень
    (18.09, 30 пар уровней, runs/goal_cross_replay_v5.json): «цвет исчез» 100%, «цвет убран по сравнению с началом» 100%,
    «весь цвет одной областью» 75%, «цвет -- прямоугольник» 50%, «областей стало меньше, чем в начале» 43%.
    НЕ показываем модели виды с низким переносом: «областей ровно m» 12%, «формы совпали» 11%, «ровно N клеток» --
    они верны для кадра, но целью следующего уровня почти никогда не остаются."""
    out = []; bg = f_goal["bg"]; cols = [c for c in f_goal["cnt"] if c != bg]
    start_cnt = (f_start or {}).get("cnt", {}); start_comps = (f_start or {}).get("comps", {})
    for c in range(16):
        if f_goal["cnt"].get(c, 0) == 0 and c != bg:
            if start_cnt.get(c, 0) > 0:
                out.append((0, "colour %d, which was on the board at the start of the level, is now completely gone" % c,
                            lambda f, c=c: f["cnt"].get(c, 0) == 0))
            else:
                out.append((1, "no cells of colour %d are left on the board" % c, lambda f, c=c: f["cnt"].get(c, 0) == 0))
    for c in cols:
        cs = f_goal["comps"].get(c, []); n = max([x["n"] for x in cs], default=0)
        if cs and n == f_goal["cnt"].get(c, 0):
            out.append((2, "all cells of colour %d are joined into a single group" % c,
                        lambda f, c=c: bool(f["comps"].get(c)) and max(x["n"] for x in f["comps"][c]) == f["cnt"].get(c, 0)))
        if f_goal["cnt"].get(c, 0) <= 2:
            k = f_goal["cnt"][c]
            out.append((3, "only %d cell(s) of colour %d remain" % (k, c), lambda f, c=c, k=k: f["cnt"].get(c, 0) == k))
        k0 = len(start_comps.get(c, []))
        if k0 > 1 and len(cs) < k0:
            out.append((4, "colour %d is split into fewer groups than at the start of the level (%d -> %d)" % (c, k0, len(cs)),
                        lambda f, c=c, k0=k0: 0 < len(f["comps"].get(c, [])) < k0))
    out.append((5, "exactly %d distinct colours are on the board" % f_goal["ncolors"],
                lambda f, k=f_goal["ncolors"]: f["ncolors"] == k))
    return sorted(out, key=lambda x: x[0])


def _gh_infer(goal, samples):
    """утверждения, истинные в кадре цели и ложные во всех образцах; первый образец уровня считается его СТАРТОМ."""
    if goal is None:
        return []
    f_goal = _gh_feat(goal)
    f_neg = [_gh_feat(g) for g in samples if getattr(g, "shape", None) == goal.shape]
    f_start = f_neg[0] if f_neg else None
    out = []
    for _rank, text, fn in _gh_preds(f_goal, f_start):
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



import inference.framework.solver as _ph_solver

_ph_stats = {"levels": 0, "goals": 0, "empty": 0, "prompts": 0, "errors": 0, "statements": []}
_ph_orig_exec = _ph_solver._HarnessGameSession._execute_action


def _ph_frames(state):
    """кадры анимации последнего хода. В бандле, который ставит стоковый ноутбук, функции _raw_frames НЕТ
    (ЛОВУШКА 18.09: она есть только в другом бандле) -- читаем raw.frame напрямую, с запасными путями."""
    raw = getattr(state, "raw", None)
    seq = getattr(raw, "frame", None)
    out = []
    for x in (seq or []):
        try:
            g = _gh_np.asarray(x, dtype=_gh_np.int16)
        except Exception:
            continue
        if g.ndim == 2 and g.size:
            out.append(g)
    if not out:
        try:
            out = [_gh_np.asarray(_ph_solver._grid_from_state(state), dtype=_gh_np.int16)]
        except Exception:
            out = []
    return out


def _ph_exec(self, action, *a, **k):
    """только ЧТЕНИЕ после хода модели: образцы состояний уровня и кадр взятия. Ходов не добавляет."""
    payload = _ph_orig_exec(self, action, *a, **k)
    try:
        st = getattr(self, "_ph_state", None)
        if st is None:
            st = {"samples": [], "n": 0}; self._ph_state = st
        cur = self.game.current_state
        if isinstance(payload, dict) and payload.get("level_completed"):
            frames = _ph_frames(cur)
            goal = (frames[:-1] or frames)[-1] if frames else None
            stmts = _gh_infer(goal, st["samples"])
            lvl = int(getattr(cur, "levels_completed", 0) or 0)
            _ph_stats["levels"] += 1
            if stmts:
                _ph_stats["goals"] += 1; _ph_stats["statements"].append(stmts)
                self._ph_goal = {"level_done": lvl, "stmts": stmts}
                print("[[PRED]] уровень %d взят моделью; цель: %s" % (lvl, " | ".join(stmts)), flush=True)
            else:
                _ph_stats["empty"] += 1
                print("[[PRED]] уровень %d взят моделью, ни одно утверждение цель не отделило (образцов %d)"
                      % (lvl, len(st["samples"])), flush=True)
            st["samples"] = []; st["n"] = 0
            _ph_dump()
        else:
            st["n"] += 1
            if len(st["samples"]) < 200 and st["n"] % 2 == 0:
                st["samples"].append(_gh_np.asarray(_ph_solver._grid_from_state(cur), dtype=_gh_np.int16))
    except Exception as _e:
        _ph_stats["errors"] += 1
        print("[PRED] сбой съёма: %r" % (_e,), flush=True)
    return payload


def _ph_block(level_done, stmts):
    return ("HARNESS-INFERRED GOAL OF THE PREVIOUS LEVEL. When level %d was completed, the harness compared that board "
            "with every other board seen on that level. These statements held only at completion:\n%s\n"
            "Levels of one game usually share the KIND of goal and differ in the numbers (more objects, obstacles, "
            "distractors). Aim for the same kind of condition here, re-derive the exact numbers yourself, write the goal "
            "you settle on in `Goal model:` and check it against the board before long plans."
            % (level_done, "\n".join("  - %s" % s for s in stmts)))


if _gh_os.environ.get("PREDHINT", "1") != "0":
    _ph_solver._HarnessGameSession._execute_action = _ph_exec
    _ph_orig_prompt = _gh_wta.ToolAgent._build_user_prompt

    def _ph_prompt(self, action_num, *args, **kwargs):
        text = _ph_orig_prompt(self, action_num, *args, **kwargs)
        try:
            cb = getattr(self, "_step_env_callback", None); sess = getattr(cb, "__self__", None)
            goal = getattr(sess, "_ph_goal", None) if sess is not None else None
            if goal and goal.get("stmts"):
                _ph_stats["prompts"] += 1
                return _ph_block(int(goal.get("level_done") or 0), list(goal["stmts"])) + "\n\n" + text
        except Exception as _e:
            _ph_stats["errors"] += 1; print("[PRED] сбой промпта: %r" % (_e,), flush=True)
        return text

    _gh_wta.ToolAgent._build_user_prompt = _ph_prompt

    def _ph_dump():
        try:
            _gh_os.makedirs("/kaggle/working", exist_ok=True)
            _gh_json.dump(_ph_stats, open("/kaggle/working/predhint_stats.json", "w"), ensure_ascii=False, indent=1)
        except Exception:
            pass

    _gh_atexit.register(_ph_dump)
    print("[[PRED]] слой установлен: ходов движка не тратит; цель выводится по кадру взятия уровня МОДЕЛЬЮ и идёт "
          "во вход на следующих уровнях (до %d утверждений); TRUE_SUBMISSION=%s" % (_GH_MAX, TRUE_SUBMISSION), flush=True)
else:
    def _ph_dump():
        pass

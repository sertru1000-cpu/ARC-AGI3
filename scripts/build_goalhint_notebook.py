"""Перебор берёт уровень 1 -> обвязка ВЫВОДИТ ЦЕЛЬ по кадру взятия -> цель во входе модели (18.09, слово владельца:
«Мы не хотим попробовать BFS на первом уровне, а после того, как он взят — передавать цель и уже решать обычной моделью?»,
затем «ставь на 80 минут»).

Чем отличается от bfscarry (17.09, балл 3.33): там модели передавался ПУТЬ (последовательность ходов), и ни в одной из
5 игр с найденным уровнем модель не взяла следующий (0 из 5). Здесь передаётся ЦЕЛЬ как утверждение о доске, выведенное
в сильном режиме: обвязка ВИДИТ кадр, в котором уровень был взят, поэтому параметры берутся из него, а не угадываются
(офлайн-замер: цель отделяется от виденных состояний в 9 играх из 10; при угадывании параметров было 36%).
Перенос на следующие уровни: вид цели сохраняется в 80% переходов между уровнями (измерено на 30 парах уровней),
поэтому подсказка держится и на уровне 2+, с явной оговоркой, что числа на новом уровне другие.

Состав ячейки 15:
  1. слой перебора из build_bfs_notebook.py в режиме pre (до первого вызова модели; кадр цели и образцы не-целевых
     состояний кладутся на сессию: sess._bf_goal_raw);
  2. индукция цели прямо в ядре (компактная копия словаря из scripts/goal_predicates.py: T1/T2/T3/T4/T6/T9/T11/T12)
     -- предикаты, истинные в кадре цели и ложные во ВСЕХ образцах, переводятся в короткие фразы на английском;
  3. подача во вход модели: блок HARNESS-INFERRED GOAL; после взятия уровня моделью блок сохраняется с пометкой,
     что вид цели тот же, а числа могут отличаться.

usage:  .venv/bin/python scripts/build_goalhint_notebook.py --probe [--cap 4800]
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_lvfact_reset_notebook import build  # noqa: E402
import build_bfs_notebook as _bfs  # noqa: E402

GOAL_HELPERS = r'''
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


'''

GOAL_WIRE_BFS = r'''if _gh_os.environ.get("GOALHINT", "1") != "0":
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
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true"); ap.add_argument("--cap", type=float, default=4800.0)
    ap.add_argument("--moves", type=int, default=12000); ap.add_argument("--seconds", type=float, default=600.0)
    a = ap.parse_args()
    cell = _bfs.cell(a.moves, a.seconds, "pre", 600.0, 600.0, 600.0) + "\n" + GOAL_HELPERS + GOAL_WIRE_BFS
    out = "kernels/notebooks_stockflash_goalhint"
    build(cell, out, "sergueimakarov/arc3-stock-flash-goalhint", "arc3 stock flash goalhint", "_gh_stats = ")
    if a.probe:
        p = os.path.join(out, "submission.ipynb")
        nb = json.load(open(p, encoding="utf-8"))
        c15 = "".join(nb["cells"][15]["source"])
        marker = "# Play the benchmark; watchdog stop and teardown run even if it raises."
        assert marker in c15
        c15 = c15.replace(marker, "if not TRUE_SUBMISSION:\n    bm.solver.max_runtime_s_per_game = %r    # проба вне боя\n\n" % a.cap + marker, 1)
        nb["cells"][15]["source"] = c15.splitlines(keepends=True)
        json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("ok   проба: потолок игры %s с" % a.cap)


if __name__ == "__main__":
    main()

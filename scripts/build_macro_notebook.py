"""Макроходы на стоковом Duck (14.09, слово владельца «макроходы -- пробуй»).

Замысел (план 13.09, пункт 5): вместо одного сырого хода на вызов -- три помощника в песочнице `python`,
каждый делает много настоящих ходов движка за ОДИН вызов модели и возвращает сводку:
  move_until_stuck(move, max_moves)      -- повторяет ход, пока доска меняется (дойти до стены/края);
  click_objects(...)                      -- клик по центру каждого подходящего объекта сегментации, отчёт что изменилось;
  sweep_moves()                           -- по одному разу каждый ход из valid_actions (кроме MOUSE), отчёт что изменилось.
Все останавливаются при взятии уровня / конце игры. Помощники приписываются спереди кода модели в
_run_python_tool (как у ворот цели); маркеры [[MACRO]] из stdout песочницы читает обвязка и вырезает.
Описание помощников -- во входе каждого хода (_build_user_prompt), выходных токенов слой не добавляет.

Пороги, записанные ДО пуска, против базы runs/flash_v1_phaseA (10.25, h115): польза -- дельта >= +4 и
парный критерий знаков p < 0.05; вред -- <= -4; иначе шум. Механизм: макрос вызван хотя бы раз в >= 15 играх,
иначе прогон измерил отсутствие слоя. Побочные: вызовов на игру, ходов на вызов, ходов внутри макросов,
уровней, взятых внутри макроса. Риск по RHAE: лишние ходы макроса дешевле невзятых уровней (макросы ограничены
max_moves/max_clicks). Только оффлайн (TRUE_SUBMISSION=False).
Проба: --probe ставит потолок игры 1800 с (сравнение с docs/base30_flash_v1_h115.json, 3.32).
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_lvfact_reset_notebook import build  # noqa: E402

MAX_MOVES, MAX_CLICKS = 40, 16
PROBE_CAP_S = 1800.0

HELPERS = '''
_MC_MAX_MOVES, _MC_MAX_CLICKS = __MAX_MOVES__, __MAX_CLICKS__
def _mc_stop(res):
    return (not res.get("executed")) or bool(res.get("level_completed") or res.get("done") or res.get("game_over") or res.get("run_complete"))
def _mc_report(out):
    print("[[MACRO]] " + repr({"macro": out.get("macro"), "moves": out.get("moves"), "level_completed": out.get("level_completed")}))
    return out
def _mc_center(node):
    pts = node.get("boundary") or []
    rs = [int(p[0]) for p in pts]; cs = [int(p[1]) for p in pts]
    if not rs or not cs:
        return None
    return (min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2
def move_until_stuck(move, max_moves=40):
    """Repeat ONE move (e.g. "RIGHT") while the board keeps changing. Stops at the first move that changes nothing
    (wall/edge), when a level completes, or after max_moves. Returns a summary dict."""
    move = str(move).strip().upper()
    if move == "MOUSE":
        raise ValueError("move_until_stuck takes a non-MOUSE move name")
    n = 0; changed = 0; last = {}; why = "max_moves"
    for _ in range(max(1, min(int(max_moves), _MC_MAX_MOVES))):
        last = action(move); n += 1
        if last.get("board_changed"):
            changed += 1
        if _mc_stop(last):
            why = "level_completed" if last.get("level_completed") else "game_end"; break
        if not last.get("board_changed"):
            why = "no_change"; break
    return _mc_report({"macro": "move_until_stuck", "move": move, "moves": n, "changed_moves": changed, "stopped_by": why,
                       "level_completed": bool(last.get("level_completed")), "level": last.get("level")})
def click_objects(color=None, min_pixels=None, max_pixels=None, ids=None, stop_on_change=False, max_clicks=16):
    """Click the center of every object of current_frame.segmentation that matches the filters (color letter,
    pixel-count range, or explicit node ids). Skips the background (largest object). Reports which clicks changed the
    board; stops when a level completes, or at the first change if stop_on_change=True. Returns a list of dicts."""
    nodes = list((current_frame.segmentation or {}).get("nodes") or [])
    if nodes:
        bg = max(nodes, key=lambda n: int(n.get("pixels") or 0))
        nodes = [n for n in nodes if n is not bg]
    if ids is not None:
        want = set(int(i) for i in ids); nodes = [n for n in nodes if int(n.get("id", -1)) in want]
    if color is not None:
        nodes = [n for n in nodes if str(n.get("color")) == str(color)]
    if min_pixels is not None:
        nodes = [n for n in nodes if int(n.get("pixels") or 0) >= int(min_pixels)]
    if max_pixels is not None:
        nodes = [n for n in nodes if int(n.get("pixels") or 0) <= int(max_pixels)]
    out = []; last = {}; lvl = False
    for n in nodes[:max(1, min(int(max_clicks), _MC_MAX_CLICKS))]:
        c = _mc_center(n)
        if c is None:
            continue
        last = action({"action": "MOUSE", "row": c[0], "col": c[1]})
        out.append({"id": n.get("id"), "color": n.get("color"), "pixels": n.get("pixels"), "row": c[0], "col": c[1],
                    "changed": bool(last.get("board_changed")), "level_completed": bool(last.get("level_completed"))})
        lvl = lvl or bool(last.get("level_completed"))
        if _mc_stop(last) or (stop_on_change and last.get("board_changed")):
            break
    _mc_report({"macro": "click_objects", "moves": len(out), "level_completed": lvl})
    return out
def sweep_moves():
    """Try each valid non-MOUSE move ONCE and report which of them changed the board. Returns {move: changed}."""
    res = {}; lvl = False
    for name in [str(a) for a in (valid_actions or [])]:
        if name.upper() in ("MOUSE", "RESET"):
            continue
        last = action(name)
        res[name] = bool(last.get("board_changed"))
        lvl = lvl or bool(last.get("level_completed"))
        if _mc_stop(last):
            break
    _mc_report({"macro": "sweep_moves", "moves": len(res), "level_completed": lvl})
    return res
'''.replace("__MAX_MOVES__", str(MAX_MOVES)).replace("__MAX_CLICKS__", str(MAX_CLICKS))

PROMPT_NOTE = (
    "MACRO HELPERS are defined inside the `python` tool (each runs several REAL environment moves in one call and "
    "returns a summary; all stop automatically when a level completes): "
    "`move_until_stuck(\"RIGHT\", max_moves=40)` repeats one move while the board keeps changing (walk to a wall/edge); "
    "`click_objects(color=None, min_pixels=None, max_pixels=None, ids=None, stop_on_change=False, max_clicks=16)` clicks "
    "the center of every matching object of `current_frame.segmentation` (background skipped) and returns which clicks "
    "changed the board; `sweep_moves()` tries every non-MOUSE valid action once and returns which changed the board. "
    "Prefer a macro over one raw action per call when exploring, testing objects, or traversing."
)

MACRO_CELL = '''
# =====================================================================
# МАКРОХОДЫ (14.09): три помощника поверх action() в песочнице python -- много ходов движка за один вызов
# модели. Помощники приписываются спереди кода модели; маркеры [[MACRO]] из stdout читает обвязка и вырезает.
# Только оффлайн.
# =====================================================================
import re as _mc_re, threading as _mc_thr, atexit as _mc_atexit
import inference.agent.tool_agent as _wta
_MC_HELPERS = __HELPERS__
_MC_NOTE = __NOTE__
_mc_stats = {"games": 0, "turns": 0, "macro_turns": 0, "macro_calls": 0, "macro_moves": 0, "by_name": {}, "levels_in_macro": 0, "games_used": 0}
_mc_tls = _mc_thr.local()

if not TRUE_SUBMISSION:
    _mc_orig_sandbox = _wta.run_sandboxed_python
    def _mc_sandbox(*a, **k):
        res = _mc_orig_sandbox(*a, **k)
        try:
            text = str(res.get("stdout", "") or "")
            ev = []
            for m in _mc_re.finditer(r"^\\[\\[MACRO\\]\\] (\\{.*\\})$", text, flags=_mc_re.M):
                try:
                    ev.append(eval(m.group(1), {"__builtins__": {}}, {}))
                except Exception:
                    pass
            res["stdout"] = _mc_re.sub(r"^\\[\\[MACRO\\]\\].*$\\n?", "", text, flags=_mc_re.M)
            _mc_tls.events = ev
        except Exception as _e:
            print("[MACRO] сбой разбора stdout: %r" % (_e,), flush=True)
        return res
    _wta.run_sandboxed_python = _mc_sandbox

    _mc_orig_run = _wta.ToolAgent._run_python_tool
    def _mc_run(self, state_path, arguments):
        st = getattr(self, "_mc_state", None)
        if st is None:
            st = {"used": 0}; self._mc_state = st; _mc_stats["games"] += 1
        _mc_stats["turns"] += 1
        arguments = dict(arguments or {})
        code = str(arguments.get("code", "") or "")
        if code.strip():
            arguments["code"] = _MC_HELPERS + "\\n" + code
        _mc_tls.events = None
        out = _mc_orig_run(self, state_path, arguments)
        try:
            ev = getattr(_mc_tls, "events", None) or []
            if ev:
                _mc_stats["macro_turns"] += 1
                if st["used"] == 0:
                    _mc_stats["games_used"] += 1
                st["used"] += len(ev)
                for e in ev:
                    nm = str(e.get("macro")); mv = int(e.get("moves") or 0)
                    _mc_stats["macro_calls"] += 1; _mc_stats["macro_moves"] += mv
                    _mc_stats["by_name"][nm] = _mc_stats["by_name"].get(nm, 0) + 1
                    if e.get("level_completed"):
                        _mc_stats["levels_in_macro"] += 1
                        print("[MACRO] уровень взят внутри макроса %s (%d ходов)" % (nm, mv), flush=True)
                if _mc_stats["macro_calls"] % 25 == 0:
                    print("[MACRO] статистика:", _mc_stats, flush=True)
        except Exception as _e:
            print("[MACRO] сбой учёта: %r" % (_e,), flush=True)
        return out
    _wta.ToolAgent._run_python_tool = _mc_run

    _mc_orig_prompt = _wta.ToolAgent._build_user_prompt
    def _mc_prompt(self, action_num, *args, **kwargs):
        text = _mc_orig_prompt(self, action_num, *args, **kwargs)
        return _MC_NOTE + "\\n\\n" + text
    _wta.ToolAgent._build_user_prompt = _mc_prompt
    _mc_atexit.register(lambda: print("[MACRO] ИТОГ:", _mc_stats, flush=True))
    print("MACRO: три макрохода в песочнице (move_until_stuck <= %d ходов, click_objects <= %d кликов, sweep_moves); описание во входе. "
          "ПОРОГИ против базы 10.25: польза -- дельта >= +4 и знаки p < 0.05; вред -- <= -4; механизм -- макрос вызван в >= 15 играх."
          % (__MM__, __MK__), flush=True)
'''.replace("__HELPERS__", repr(HELPERS)).replace("__NOTE__", repr(PROMPT_NOTE)).replace("__MM__", str(MAX_MOVES)).replace("__MK__", str(MAX_CLICKS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="потолок игры %s с (проба 30 мин)" % PROBE_CAP_S)
    a = ap.parse_args()
    out = "kernels/notebooks_stockflash_macro"
    build(MACRO_CELL, out, "sergueimakarov/arc3-stock-flash-macro", "arc3 stock flash macro", "_mc_stats = ")
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
    open(os.path.join(out, "helpers.py"), "w", encoding="utf-8").write(HELPERS)


if __name__ == "__main__":
    main()

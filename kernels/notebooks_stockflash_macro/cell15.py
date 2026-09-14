
# =====================================================================
# МАКРОХОДЫ (14.09): три помощника поверх action() в песочнице python -- много ходов движка за один вызов
# модели. Помощники приписываются спереди кода модели; маркеры [[MACRO]] из stdout читает обвязка и вырезает.
# Только оффлайн.
# =====================================================================
import re as _mc_re, threading as _mc_thr, atexit as _mc_atexit
import inference.agent.tool_agent as _wta
_MC_HELPERS = '\n_MC_MAX_MOVES, _MC_MAX_CLICKS = 40, 16\ndef _mc_stop(res):\n    return (not res.get("executed")) or bool(res.get("level_completed") or res.get("done") or res.get("game_over") or res.get("run_complete"))\ndef _mc_report(out):\n    print("[[MACRO]] " + repr({"macro": out.get("macro"), "moves": out.get("moves"), "level_completed": out.get("level_completed")}))\n    return out\ndef _mc_center(node):\n    pts = node.get("boundary") or []\n    rs = [int(p[0]) for p in pts]; cs = [int(p[1]) for p in pts]\n    if not rs or not cs:\n        return None\n    return (min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2\ndef move_until_stuck(move, max_moves=40):\n    """Repeat ONE move (e.g. "RIGHT") while the board keeps changing. Stops at the first move that changes nothing\n    (wall/edge), when a level completes, or after max_moves. Returns a summary dict."""\n    move = str(move).strip().upper()\n    if move == "MOUSE":\n        raise ValueError("move_until_stuck takes a non-MOUSE move name")\n    n = 0; changed = 0; last = {}; why = "max_moves"\n    for _ in range(max(1, min(int(max_moves), _MC_MAX_MOVES))):\n        last = action(move); n += 1\n        if last.get("board_changed"):\n            changed += 1\n        if _mc_stop(last):\n            why = "level_completed" if last.get("level_completed") else "game_end"; break\n        if not last.get("board_changed"):\n            why = "no_change"; break\n    return _mc_report({"macro": "move_until_stuck", "move": move, "moves": n, "changed_moves": changed, "stopped_by": why,\n                       "level_completed": bool(last.get("level_completed")), "level": last.get("level")})\ndef click_objects(color=None, min_pixels=None, max_pixels=None, ids=None, stop_on_change=False, max_clicks=16):\n    """Click the center of every object of current_frame.segmentation that matches the filters (color letter,\n    pixel-count range, or explicit node ids). Skips the background (largest object). Reports which clicks changed the\n    board; stops when a level completes, or at the first change if stop_on_change=True. Returns a list of dicts."""\n    nodes = list((current_frame.segmentation or {}).get("nodes") or [])\n    if nodes:\n        bg = max(nodes, key=lambda n: int(n.get("pixels") or 0))\n        nodes = [n for n in nodes if n is not bg]\n    if ids is not None:\n        want = set(int(i) for i in ids); nodes = [n for n in nodes if int(n.get("id", -1)) in want]\n    if color is not None:\n        nodes = [n for n in nodes if str(n.get("color")) == str(color)]\n    if min_pixels is not None:\n        nodes = [n for n in nodes if int(n.get("pixels") or 0) >= int(min_pixels)]\n    if max_pixels is not None:\n        nodes = [n for n in nodes if int(n.get("pixels") or 0) <= int(max_pixels)]\n    out = []; last = {}; lvl = False\n    for n in nodes[:max(1, min(int(max_clicks), _MC_MAX_CLICKS))]:\n        c = _mc_center(n)\n        if c is None:\n            continue\n        last = action({"action": "MOUSE", "row": c[0], "col": c[1]})\n        out.append({"id": n.get("id"), "color": n.get("color"), "pixels": n.get("pixels"), "row": c[0], "col": c[1],\n                    "changed": bool(last.get("board_changed")), "level_completed": bool(last.get("level_completed"))})\n        lvl = lvl or bool(last.get("level_completed"))\n        if _mc_stop(last) or (stop_on_change and last.get("board_changed")):\n            break\n    _mc_report({"macro": "click_objects", "moves": len(out), "level_completed": lvl})\n    return out\ndef sweep_moves():\n    """Try each valid non-MOUSE move ONCE and report which of them changed the board. Returns {move: changed}."""\n    res = {}; lvl = False\n    for name in [str(a) for a in (valid_actions or [])]:\n        if name.upper() in ("MOUSE", "RESET"):\n            continue\n        last = action(name)\n        res[name] = bool(last.get("board_changed"))\n        lvl = lvl or bool(last.get("level_completed"))\n        if _mc_stop(last):\n            break\n    _mc_report({"macro": "sweep_moves", "moves": len(res), "level_completed": lvl})\n    return res\n'
_MC_NOTE = 'MACRO HELPERS are defined inside the `python` tool (each runs several REAL environment moves in one call and returns a summary; all stop automatically when a level completes): `move_until_stuck("RIGHT", max_moves=40)` repeats one move while the board keeps changing (walk to a wall/edge); `click_objects(color=None, min_pixels=None, max_pixels=None, ids=None, stop_on_change=False, max_clicks=16)` clicks the center of every matching object of `current_frame.segmentation` (background skipped) and returns which clicks changed the board; `sweep_moves()` tries every non-MOUSE valid action once and returns which changed the board. Prefer a macro over one raw action per call when exploring, testing objects, or traversing.'
_mc_stats = {"games": 0, "turns": 0, "macro_turns": 0, "macro_calls": 0, "macro_moves": 0, "by_name": {}, "levels_in_macro": 0, "games_used": 0}
_mc_tls = _mc_thr.local()

if not TRUE_SUBMISSION:
    _mc_orig_sandbox = _wta.run_sandboxed_python
    def _mc_sandbox(*a, **k):
        res = _mc_orig_sandbox(*a, **k)
        try:
            text = str(res.get("stdout", "") or "")
            ev = []
            for m in _mc_re.finditer(r"^\[\[MACRO\]\] (\{.*\})$", text, flags=_mc_re.M):
                try:
                    ev.append(eval(m.group(1), {"__builtins__": {}}, {}))
                except Exception:
                    pass
            res["stdout"] = _mc_re.sub(r"^\[\[MACRO\]\].*$\n?", "", text, flags=_mc_re.M)
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
            arguments["code"] = _MC_HELPERS + "\n" + code
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
        return _MC_NOTE + "\n\n" + text
    _wta.ToolAgent._build_user_prompt = _mc_prompt
    _mc_atexit.register(lambda: print("[MACRO] ИТОГ:", _mc_stats, flush=True))
    print("MACRO: три макрохода в песочнице (move_until_stuck <= %d ходов, click_objects <= %d кликов, sweep_moves); описание во входе. "
          "ПОРОГИ против базы 10.25: польза -- дельта >= +4 и знаки p < 0.05; вред -- <= -4; механизм -- макрос вызван в >= 15 играх."
          % (40, 16), flush=True)

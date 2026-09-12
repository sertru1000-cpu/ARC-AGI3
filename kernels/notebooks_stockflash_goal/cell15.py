
# =====================================================================
# ВОРОТА ЦЕЛИ (12.09): «сформулировать и проверить цель до хода».
# До ходов модель регистрирует гипотезу цели уровня и измеритель прогресса
# set_goal(text, progress_code); обвязка считает измеритель после каждой пачки и после
# 3 пачек без роста объявляет гипотезу опровергнутой. Пробы <= 2 ходов без цели --
# не больше 4 на уровень. Состояние живёт на агенте, в песочницу подставляется литералом.
# Только оффлайн: боевая ветка не тронута.
# =====================================================================
import re as _gre, json as _gjson
import inference.agent.tool_agent as _wta

_GOAL_PATIENCE, _GOAL_PROBE_LEN, _GOAL_PROBE_BATCHES = 3, 2, 4
_GOAL_HELPERS = '\nimport json as _gj\n_GOAL = %(state)s\n_GOAL_PATIENCE, _GOAL_PROBE_LEN, _GOAL_PROBE_BATCHES = %(patience)d, %(probe_len)d, %(probe_batches)d\n\ndef _goal_eval(frame):\n    _ns = {}\n    exec(_GOAL["code"], _ns)\n    return float(_ns["progress"](frame))\n\ndef set_goal(text, progress_code):\n    """Register the level goal (checkable sentence) and a progress measure: Python source defining\n    `def progress(frame) -> number` (frame.grid = list of rows of ints, frame.ascii, frame.level) that RISES\n    as the board gets closer to the goal."""\n    text = str(text or "").strip()\n    if len(text) < 8:\n        raise ValueError("set_goal(text, progress_code): text must state what completes this level as a checkable condition")\n    if not isinstance(progress_code, str) or "def progress" not in progress_code:\n        raise ValueError("set_goal(text, progress_code): progress_code must be a Python SOURCE STRING defining "\n                         "`def progress(frame) -> number` computed from the board only; it must RISE as you approach the goal")\n    _ns = {}\n    try:\n        exec(progress_code, _ns); _v = float(_ns["progress"](current_frame))\n    except Exception as _e:\n        raise ValueError("progress(frame) failed on the current board: %%r" %% (_e,))\n    if text in (_GOAL.get("falsified_texts") or []):\n        raise ValueError("this exact goal was already FALSIFIED on this level (its measure did not rise in %%d batches); "\n                         "state a DIFFERENT hypothesis or a different measure" %% _GOAL_PATIENCE)\n    _GOAL.update({"text": text, "code": progress_code, "closed": None, "values": [_v], "no_progress": 0})\n    print("[[GOAL_SET]] " + _gj.dumps({"text": text, "code": progress_code, "value": _v}))\n    return {"ok": True, "progress_now": _v, "note": "action() is open; progress is re-measured after every batch and reported in the result"}\n\n_goal_orig_action = action\ndef action(actions):\n    _acts = actions if isinstance(actions, list) else [actions]\n    if _GOAL.get("closed") == "falsified":\n        raise RuntimeError("GOAL GATE: goal %%r is FALSIFIED -- its own progress measure did not rise in %%d batches (values %%s). "\n                           "Call set_goal() with a DIFFERENT hypothesis or measure before acting." %% (_GOAL.get("text"), _GOAL_PATIENCE, _GOAL.get("values")))\n    if not _GOAL.get("text"):\n        if len(_acts) <= _GOAL_PROBE_LEN and _GOAL.get("probes_used", 0) < _GOAL_PROBE_BATCHES:\n            _GOAL["probes_used"] = _GOAL.get("probes_used", 0) + 1\n            print("[[GOAL_PROBE]]")\n            return _goal_orig_action(actions)\n        raise RuntimeError("GOAL GATE: action() blocked -- no goal registered for this level (probes of <= %%d actions without a goal: "\n                           "%%d of %%d used). Formulate the level goal from what you have seen and call "\n                           "set_goal(text, progress_code) in THIS code block, then act." %% (_GOAL_PROBE_LEN, _GOAL.get("probes_used", 0), _GOAL_PROBE_BATCHES))\n    try:\n        _before = _goal_eval(current_frame)\n    except Exception:\n        _before = None\n    _res = _goal_orig_action(actions)\n    try:\n        _after = _goal_eval(current_frame)\n    except Exception as _e:\n        if isinstance(_res, dict):\n            _res["goal_progress"] = {"error": "progress(frame) failed after the batch: %%r" %% (_e,)}\n        return _res\n    _delta = None if _before is None else _after - _before\n    _GOAL["no_progress"] = 0 if (_delta is not None and _delta > 0) else _GOAL.get("no_progress", 0) + 1\n    _GOAL["values"] = (_GOAL.get("values") or []) + [_after]\n    _fals = _GOAL["no_progress"] >= _GOAL_PATIENCE\n    if _fals:\n        _GOAL["closed"] = "falsified"\n    if isinstance(_res, dict):\n        _res["goal_progress"] = {"before": _before, "after": _after, "delta": _delta,\n                                 "no_progress_batches": _GOAL["no_progress"], "falsified": _fals}\n    print("[[GOAL_BATCH]] " + _gj.dumps({"after": _after, "no_progress": _GOAL["no_progress"], "falsified": _fals}))\n    return _res\n'
_goal_stats = {"games": 0, "turns": 0, "acted": 0, "set": 0, "blocked": 0, "probes": 0,
               "batches": 0, "progress": 0, "falsified": 0, "confirmed": 0, "levels": 0}

_GOAL_PROTOCOL = (
    "GOAL GATE (mandatory): real moves are gated on a stated, checkable level goal. Before any batch longer than a short probe, "
    "call set_goal(\"<what completes this level>\", progress_code) where progress_code is a Python source string defining "
    "`def progress(frame) -> number` (frame.grid = rows of ints, frame.ascii, frame.level) that RISES as the board gets closer "
    "to the goal (e.g. count of matched targets, filled slots, negative distance to the exit). The harness re-measures it after every "
    "batch and reports 'goal_progress' in the action() result. If it does not rise for 3 batches in a row the goal is "
    "FALSIFIED and action() is blocked until set_goal() with a different hypothesis or measure. Probes of <= 2 actions without "
    "a goal: at most 4 per level. A level-up confirms the goal; the next level needs set_goal() again. "
    "Write the goal from EVIDENCE and say in your reasoning whether the last result SUPPORTS or CONTRADICTS it."
)

def _g_st(agent):
    st = getattr(agent, "_goal_state", None)
    if st is None:
        st = {"text": "", "code": "", "closed": None, "values": [], "no_progress": 0,
              "probes_used": 0, "falsified_texts": [], "level": None}
        agent._goal_state = st
        _goal_stats["games"] += 1
    return st

def _g_status(st):
    if not st.get("text"):
        return ("GOAL GATE STATUS: no goal registered for this level. Probes of <= %d actions without a goal: %d of %d used."
                % (_GOAL_PROBE_LEN, st.get("probes_used", 0), _GOAL_PROBE_BATCHES))
    vals = ", ".join("%g" % v for v in (st.get("values") or [])[-6:])
    if st.get("closed") == "falsified":
        return ("GOAL GATE STATUS: goal %r is FALSIFIED -- its progress measure did not rise in %d batches (values %s). "
                "action() is blocked until set_goal() with a different hypothesis or measure." % (st["text"][:120], _GOAL_PATIENCE, vals))
    return ("GOAL GATE STATUS: your goal: %r | progress values %s | batches without progress %d/%d"
            % (st["text"][:120], vals, st.get("no_progress", 0), _GOAL_PATIENCE))

if not TRUE_SUBMISSION:
    _g_orig_prompt = _wta.ToolAgent._build_user_prompt
    def _g_prompt(self, action_num, *args, **kwargs):
        text = _g_orig_prompt(self, action_num, *args, **kwargs)
        try:
            st = _g_st(self)
            lv = getattr(kwargs.get("current_frame"), "level", None)
            prev = st.get("level")
            if prev is not None and lv is not None and int(lv) > int(prev):
                _goal_stats["levels"] += 1
                if st.get("text") and st.get("closed") is None:
                    _goal_stats["confirmed"] += 1
                st.update({"text": "", "code": "", "closed": None, "values": [], "no_progress": 0,
                           "probes_used": 0, "falsified_texts": []})
            if lv is not None:
                st["level"] = lv
            return _GOAL_PROTOCOL + "\n" + _g_status(st) + "\n\n" + text
        except Exception:
            return text
    _wta.ToolAgent._build_user_prompt = _g_prompt

    _g_orig_run = _wta.ToolAgent._run_python_tool
    def _g_run(self, state_path, arguments):
        st = _g_st(self)
        code = str((arguments or {}).get("code", "") or "")
        _goal_stats["turns"] += 1
        if "action(" in code:
            _goal_stats["acted"] += 1
        lit = {k: st.get(k) for k in ("text", "code", "closed", "values", "no_progress", "probes_used", "falsified_texts")}
        arguments = dict(arguments or {})
        arguments["code"] = (_GOAL_HELPERS % {"state": repr(lit), "patience": _GOAL_PATIENCE,
                                              "probe_len": _GOAL_PROBE_LEN, "probe_batches": _GOAL_PROBE_BATCHES}
                             + "\n" + code)
        out = _g_orig_run(self, state_path, arguments)
        try:
            text = getattr(out, "content", "") or ""
            for m in _gre.finditer(r"\[\[GOAL_SET\]\] (\{.*\})", text):
                d = _gjson.loads(m.group(1))
                st.update({"text": d["text"], "code": d["code"], "closed": None, "values": [d["value"]], "no_progress": 0})
                _goal_stats["set"] += 1
            st["probes_used"] = st.get("probes_used", 0) + text.count("[[GOAL_PROBE]]")
            _goal_stats["probes"] += text.count("[[GOAL_PROBE]]")
            for m in _gre.finditer(r"\[\[GOAL_BATCH\]\] (\{.*\})", text):
                d = _gjson.loads(m.group(1))
                st["values"] = (st.get("values") or []) + [d["after"]]
                st["no_progress"] = int(d["no_progress"])
                _goal_stats["batches"] += 1
                if st["no_progress"] == 0:
                    _goal_stats["progress"] += 1
                if d.get("falsified"):
                    st["closed"] = "falsified"
                    st["falsified_texts"] = list(st.get("falsified_texts") or []) + [st.get("text", "")]
                    _goal_stats["falsified"] += 1
            _goal_stats["blocked"] += text.count("GOAL GATE: action() blocked") + text.count("is FALSIFIED -- its own")
            cleaned = _gre.sub(r"^\[\[GOAL_[A-Z]+\]\].*$\n?", "", text, flags=_gre.M)
            if cleaned != text:
                try:
                    out.content = cleaned
                except Exception:
                    out = _wta._ToolDispatchResult(content=cleaned, step_executed=getattr(out, "step_executed", True))
            if _goal_stats["turns"] % 50 == 0:
                print("[GOAL] " + _gjson.dumps(_goal_stats), flush=True)
        except Exception as _e:
            print("[GOAL] сбой учёта: %r" % (_e,), flush=True)
        return out
    _wta.ToolAgent._run_python_tool = _g_run
    print("GOAL GATE: set_goal(text, progress_code) до ходов; терпение 3 пачек, пробы <= 2 ходов x 4. "
          "ПОРОГИ против базы 10.25: польза -- победы-поражения >= +8 и медиана >= +8; вред -- <= -6. "
          "Механизм: set_goal хотя бы раз в >= 20 играх, доля вызовов с ходом >= 77%, генерация 1400-1600.", flush=True)

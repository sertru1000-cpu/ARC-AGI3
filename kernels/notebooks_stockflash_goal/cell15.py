
# =====================================================================
# ВОРОТА ЦЕЛИ (12.09): «сформулировать и проверить цель до хода».
# До ходов модель регистрирует гипотезу цели уровня и измеритель прогресса
# set_goal(text, progress_code); обвязка считает измеритель после каждой пачки и после
# 3 вызовов с ходами без роста объявляет гипотезу опровергнутой (измеритель исполняет ОБВЯЗКА: в песочнице нет exec). Пробы <= 2 ходов без цели --
# не больше 4 на уровень. Состояние живёт на агенте, в песочницу подставляется литералом.
# Только оффлайн: боевая ветка не тронута.
# =====================================================================
import re as _gre, json as _gjson
import inference.agent.tool_agent as _wta

_GOAL_PATIENCE, _GOAL_PROBE_LEN, _GOAL_PROBE_BATCHES = 3, 2, 4
_GOAL_REJECT_AFTER, _GOAL_RESET_AFTER = 2, 5
_GOAL_HELPERS = '\nimport json as _gj\n_GOAL = %(state)s\n_GOAL_PATIENCE, _GOAL_PROBE_LEN, _GOAL_PROBE_BATCHES = %(patience)d, %(probe_len)d, %(probe_batches)d\n\ndef set_goal(text, progress_code):\n    # Register the level goal (checkable sentence) and a progress measure: Python SOURCE defining\n    # `def progress(frame) -> number` (frame.grid = rows of ints, frame.ascii, frame.level) that RISES\n    # as the board gets closer to the goal. The harness validates and measures it after this call.\n    text = str(text or "").strip()\n    if len(text) < 8:\n        raise ValueError("set_goal(text, progress_code): text must state what completes this level as a checkable condition")\n    if not isinstance(progress_code, str) or "def progress" not in progress_code:\n        raise ValueError("set_goal(text, progress_code): progress_code must be a Python SOURCE STRING defining "\n                         "`def progress(frame) -> number` computed from the board only; it must RISE as you approach the goal")\n    if text in (_GOAL.get("falsified_texts") or []):\n        raise ValueError("this exact goal was already FALSIFIED on this level (its measure did not rise in %%d calls with moves); "\n                         "state a DIFFERENT hypothesis or a different measure" %% _GOAL_PATIENCE)\n    _GOAL.update({"text": text, "code": progress_code, "closed": None})\n    print("[[GOAL_SET]] " + _gj.dumps({"text": text, "code": progress_code}))\n    return {"ok": True, "note": "goal registered; action() is open. The harness validates progress(frame) and measures it after every call with moves; see GOAL GATE STATUS next turn"}\n\n_goal_orig_action = action\ndef action(actions):\n    _acts = actions if isinstance(actions, list) else [actions]\n    if _GOAL.get("closed") == "falsified":\n        raise RuntimeError("GOAL GATE: goal %%r is FALSIFIED -- its own progress measure did not rise in %%d calls with moves (values %%s). "\n                           "Call set_goal() with a DIFFERENT hypothesis or measure before acting." %% (_GOAL.get("text"), _GOAL_PATIENCE, _GOAL.get("values")))\n    if not _GOAL.get("text"):\n        if len(_acts) <= _GOAL_PROBE_LEN and _GOAL.get("probes_used", 0) < _GOAL_PROBE_BATCHES:\n            _GOAL["probes_used"] = _GOAL.get("probes_used", 0) + 1\n            print("[[GOAL_PROBE]]")\n            return _goal_orig_action(actions)\n        raise RuntimeError("GOAL GATE: action() blocked -- no goal registered for this level (probes of <= %%d actions without a goal: "\n                           "%%d of %%d used). Formulate the level goal from what you have seen and call "\n                           "set_goal(text, progress_code) in THIS code block, then act." %% (_GOAL_PROBE_LEN, _GOAL.get("probes_used", 0), _GOAL_PROBE_BATCHES))\n    _res = _goal_orig_action(actions)\n    print("[[GOAL_BATCH]] " + _gj.dumps({"n": len(_acts)}))\n    if isinstance(_res, dict):\n        _res["goal_note"] = "progress is measured by the harness after this call; see GOAL GATE STATUS next turn"\n    return _res\n'
_goal_stats = {"games": 0, "turns": 0, "acted": 0, "set": 0, "blocked": 0, "probes": 0,
               "batches": 0, "progress": 0, "falsified": 0, "confirmed": 0, "levels": 0,
               "rejected": 0, "same_measure": 0, "inspect_refused": 0, "gate_reset": 0}

_GOAL_PROTOCOL = (
    "GOAL GATE (mandatory): real moves are gated on a stated, checkable level goal. Before any batch longer than a short probe, "
    "call set_goal(\"<what completes this level>\", progress_code) where progress_code is a Python source string defining "
    "`def progress(frame) -> number` (frame.grid = rows of ints, frame.ascii, frame.level) that RISES as the board gets closer "
    "to the goal (e.g. count of matched targets, filled slots, negative distance to the exit). The harness re-measures it after every "
    "call with moves and shows the values in GOAL GATE STATUS. If it does not rise for 3 calls with moves in a row the goal is "
    "FALSIFIED and action() is blocked until set_goal() with a different hypothesis or measure. Probes of <= 2 actions without "
    "a goal: at most 4 per level. A level-up confirms the goal; the next level needs set_goal() again. "
    "Write the goal from EVIDENCE and say in your reasoning whether the last result SUPPORTS or CONTRADICTS it."
)

def _g_st(agent):
    st = getattr(agent, "_goal_state", None)
    if st is None:
        st = {"text": "", "code": "", "closed": None, "values": [], "no_progress": 0,
              "probes_used": 0, "falsified_texts": [], "level": None, "rejected": "", "since_fals": 0, "fals_codes": []}
        agent._goal_state = st
        _goal_stats["games"] += 1
    return st

def _g_status(st):
    if not st.get("text"):
        rej = (" LAST set_goal REJECTED: " + st["rejected"]) if st.get("rejected") else ""
        return ("GOAL GATE STATUS: no goal registered for this level. Probes of <= %d actions without a goal: %d of %d used.%s"
                % (_GOAL_PROBE_LEN, st.get("probes_used", 0), _GOAL_PROBE_BATCHES, rej))
    vals = ", ".join("%g" % v for v in (st.get("values") or [])[-6:])
    if st.get("closed") == "falsified":
        k = int(st.get("since_fals", 0))
        return ("GOAL GATE STATUS: goal %r is FALSIFIED -- its progress measure did not rise in %d calls with moves (values %s). "
                "You MUST call set_goal(text, progress_code) with a DIFFERENT MEASURE (one that reads a different number on the "
                "current board) IN THIS CODE BLOCK; code without set_goal is not executed after %d such turns (%d so far)."
                % (st["text"][:120], _GOAL_PATIENCE, vals, _GOAL_REJECT_AFTER, k))
    return ("GOAL GATE STATUS: your goal: %r | progress values %s | calls with moves and no progress %d/%d"
            % (st["text"][:120], vals, st.get("no_progress", 0), _GOAL_PATIENCE))

if not TRUE_SUBMISSION:
    # Маркеры разбираются из СЫРОГО stdout песочницы, до того как обвязка завернёт его в JSON
    # (в JSON они экранированы и регулярные выражения по строкам их не видят -- дефект v1, 12.09 20:25).
    # Перехват потокобезопасный: 28 игр идут в потоках, run_sandboxed_python и _run_python_tool -- один поток.
    import threading as _gthr
    _g_tls = _gthr.local()
    _g_orig_sandbox = _wta.run_sandboxed_python
    def _g_sandbox(*a, **k):
        res = _g_orig_sandbox(*a, **k)
        try:
            text = str(res.get("stdout", "") or "")
            ev = {"set": [], "probe": 0, "batch": 0}
            for m in _gre.finditer(r"\[\[GOAL_SET\]\] (\{.*\})", text):
                ev["set"].append(_gjson.loads(m.group(1)))
            ev["probe"] = text.count("[[GOAL_PROBE]]")
            ev["batch"] = text.count("[[GOAL_BATCH]]")
            res["stdout"] = _gre.sub(r"^\[\[GOAL_[A-Z]+\]\].*$\n?", "", text, flags=_gre.M)
            _g_tls.events = ev
        except Exception as _e:
            print("[GOAL] сбой разбора stdout: %r" % (_e,), flush=True)
        return res
    _wta.run_sandboxed_python = _g_sandbox

    def _g_frame(state_path):
        """Текущий кадр для измерителя: те же поля, что видит модель (grid, ascii, level, step, shape)."""
        frame, _hist = _wta.load_runtime_state(state_path)
        pl = _wta._ascii_frame_view_payload(frame) or {}
        import types as _gtypes
        return _gtypes.SimpleNamespace(grid=pl.get("grid"), ascii=pl.get("ascii", ""), level=pl.get("level"),
                                        step=pl.get("step"), shape=pl.get("shape"))

    def _g_measure(code, frame):
        ns = {}
        exec(code, ns)
        return float(ns["progress"](frame))

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
                           "probes_used": 0, "falsified_texts": [], "rejected": "", "since_fals": 0, "fals_codes": []})
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
        if st.get("closed") == "falsified":
            if "set_goal(" not in code:
                st["since_fals"] = int(st.get("since_fals", 0)) + 1
                if st["since_fals"] > _GOAL_RESET_AFTER:
                    # предохранитель: модель не формулирует -- ворота в режим проб, игра не умирает
                    st.update({"text": "", "code": "", "closed": None, "values": [], "no_progress": 0,
                               "probes_used": 0, "since_fals": 0, "rejected": ""})
                    _goal_stats["gate_reset"] += 1
                elif st["since_fals"] > _GOAL_REJECT_AFTER:
                    _goal_stats["inspect_refused"] += 1
                    return _wta._ToolDispatchResult(
                        content=_gjson.dumps({"error": "GOAL GATE: not executed. Your goal is FALSIFIED and this code block "
                                              "has no set_goal(). Register a new goal with a DIFFERENT measure in the same "
                                              "block as your next moves. (%d turns without a new goal; after %d the gate "
                                              "resets to probe mode.)" % (st["since_fals"], _GOAL_RESET_AFTER)}, indent=2),
                        step_executed=False)
            else:
                st["since_fals"] = 0
        lit = {k: st.get(k) for k in ("text", "code", "closed", "values", "no_progress", "probes_used", "falsified_texts")}
        arguments = dict(arguments or {})
        arguments["code"] = (_GOAL_HELPERS % {"state": repr(lit), "patience": _GOAL_PATIENCE,
                                              "probe_len": _GOAL_PROBE_LEN, "probe_batches": _GOAL_PROBE_BATCHES}
                             + "\n" + code)
        _g_tls.events = None
        out = _g_orig_run(self, state_path, arguments)
        try:
            ev = getattr(_g_tls, "events", None) or {"set": [], "probe": 0, "batch": 0}
            frame = None
            if ev["set"] or ev["batch"]:
                try:
                    frame = _g_frame(state_path)
                except Exception as _e:
                    print("[GOAL] кадр не прочитан: %r" % (_e,), flush=True)
            for d in ev["set"]:
                try:
                    v = _g_measure(d["code"], frame)
                    same = None
                    for fc in st.get("fals_codes") or []:
                        try:
                            if abs(_g_measure(fc, frame) - v) < 1e-9:
                                same = fc; break
                        except Exception:
                            continue
                    if same is not None:
                        st.update({"text": "", "code": "", "closed": None, "values": [], "no_progress": 0,
                                   "rejected": "your new measure reads the same value (%g) on the current board as a FALSIFIED one -- "
                                               "it is the same measure under a new name; define progress from a DIFFERENT observable "
                                               "(other objects, other relation, other count) and call set_goal again" % v})
                        _goal_stats["same_measure"] += 1
                        continue
                    st.update({"text": d["text"], "code": d["code"], "closed": None, "values": [v], "no_progress": 0, "rejected": "", "since_fals": 0})
                    _goal_stats["set"] += 1
                except Exception as _e:
                    st.update({"text": "", "code": "", "closed": None, "values": [], "no_progress": 0,
                               "rejected": "progress(frame) failed on the current board: %r -- fix it and call set_goal again" % (_e,)})
                    _goal_stats["rejected"] = _goal_stats.get("rejected", 0) + 1
            st["probes_used"] = st.get("probes_used", 0) + int(ev["probe"])
            _goal_stats["probes"] += int(ev["probe"])
            if ev["batch"] and st.get("text") and st.get("closed") is None and frame is not None:
                try:
                    v = _g_measure(st["code"], frame)
                    prev = (st.get("values") or [None])[-1]
                    st["no_progress"] = 0 if (prev is not None and v > prev) else st.get("no_progress", 0) + 1
                    st["values"] = (st.get("values") or []) + [v]
                    _goal_stats["batches"] += 1
                    if st["no_progress"] == 0:
                        _goal_stats["progress"] += 1
                    if st["no_progress"] >= _GOAL_PATIENCE:
                        st["closed"] = "falsified"
                        st["falsified_texts"] = list(st.get("falsified_texts") or []) + [st.get("text", "")]
                        st["fals_codes"] = list(st.get("fals_codes") or []) + [st.get("code", "")]
                        st["since_fals"] = 0
                        _goal_stats["falsified"] += 1
                except Exception as _e:
                    st["rejected"] = "progress(frame) failed after your moves: %r" % (_e,)
            text = getattr(out, "content", "") or ""
            _goal_stats["blocked"] += text.count("GOAL GATE: action() blocked") + text.count("is FALSIFIED -- its own")
            if _goal_stats["turns"] % 50 == 0:
                print("[GOAL] " + _gjson.dumps(_goal_stats), flush=True)
        except Exception as _e:
            print("[GOAL] сбой учёта: %r" % (_e,), flush=True)
        return out
    _wta.ToolAgent._run_python_tool = _g_run
    print("GOAL GATE: set_goal(text, progress_code) до ходов; терпение 3 пачек, пробы <= 2 ходов x 4. "
          "ПОРОГИ против базы 10.25: польза -- победы-поражения >= +8 и медиана >= +8; вред -- <= -6. "
          "Механизм: set_goal хотя бы раз в >= 20 играх, доля вызовов с ходом >= 77%, генерация 1400-1600.", flush=True)


# =====================================================================
# RESET ПРИ ЗАСТРЕВАНИИ С СОХРАНЕНИЕМ ПАМЯТИ (13.09): после 100 ходов на уровне без взятия обвязка
# сама сбрасывает уровень (взятые уровни сохраняются) и кладёт во вход сводку испробованного.
# Не больше 2 сбросов на уровень. Только оффлайн.
# =====================================================================
import hashlib as _rs_hashlib
import inference.agent.tool_agent as _wta
_RS_STALL, _RS_CALLS, _RS_MAX, _RS_NOTE_TURNS = 100, 40, 2, 3
_rs_stats = {"games": 0, "resets": 0, "turns": 0, "blocked_max": 0, "levels": 0}

def _rs_summary(history_entries, level):
    """Сводка по ходам текущего уровня: гистограмма ходов, различные доски, возвраты в виденное."""
    ents = [e for e in (history_entries or []) if int(getattr(e.frame, "level", 0) or 0) == int(level)]
    hist = {}
    seen = set(); returns = 0
    for e in ents:
        a = str(getattr(e, "action", "") or "?").split("(")[0]
        hist[a] = hist.get(a, 0) + 1
        h = _rs_hashlib.blake2b((getattr(e.frame, "ascii", "") or "").encode(), digest_size=8).hexdigest()
        if h in seen:
            returns += 1
        seen.add(h)
    top = ", ".join("%s x%d" % (k, v) for k, v in sorted(hist.items(), key=lambda kv: -kv[1])[:6])
    return len(ents), top, len(seen), returns

if not TRUE_SUBMISSION:
    _rs_orig_prompt = _wta.ToolAgent._build_user_prompt
    def _rs_prompt(self, action_num, *args, **kwargs):
        text = _rs_orig_prompt(self, action_num, *args, **kwargs)
        try:
            st = getattr(self, "_rs_state", None)
            if st is None:
                st = {"level": None, "level_start": int(action_num or 0), "resets": 0, "last_reset": None,
                      "note": None, "note_left": 0, "entries": None}; self._rs_state = st; _rs_stats["games"] += 1
            _rs_stats["turns"] += 1
            st["entries"] = kwargs.get("history_entries")
            lv = getattr(kwargs.get("current_frame"), "level", None)
            if lv is not None:
                lv = int(lv)
                if st["level"] is None or lv > st["level"]:
                    if st["level"] is not None:
                        _rs_stats["levels"] += 1
                    st.update({"level": lv, "level_start": int(action_num or 0), "resets": 0, "last_reset": None,
                               "calls": 0, "calls_at_reset": 0})
            st["action_num"] = int(action_num or 0)
            if st.get("note") and st.get("note_left", 0) > 0:
                st["note_left"] -= 1
                return st["note"] + "\n\n" + text
            return text
        except Exception as _e:
            print("[RESET] сбой промпта: %r" % (_e,), flush=True)
            return text
    _wta.ToolAgent._build_user_prompt = _rs_prompt

    _rs_orig_run = _wta.ToolAgent._run_python_tool
    def _rs_run(self, state_path, arguments):
        out = _rs_orig_run(self, state_path, arguments)
        try:
            st = getattr(self, "_rs_state", None)
            if st is not None:
                st["calls"] = int(st.get("calls", 0)) + 1   # один запуск кода = один вызов модели (промптов меньше: tool-loop)
            cb = getattr(self, "_step_env_callback", None)
            sess = getattr(cb, "__self__", None) if cb is not None else None
            if st is None or sess is None or st.get("level") is None:
                return out
            last = getattr(self, "_last_action_result", None) or {}
            cur_num = int(last.get("action_num") or st.get("action_num") or 0)
            if last.get("level_completed") or last.get("game_over") or last.get("done"):
                return out
            since_start = cur_num - int(st.get("level_start") or 0)
            since_reset = cur_num - int(st["last_reset"]) if st.get("last_reset") is not None else since_start
            calls_since = int(st.get("calls", 0)) - int(st.get("calls_at_reset", 0))
            by_moves = since_start >= _RS_STALL and since_reset >= _RS_STALL
            by_calls = calls_since >= _RS_CALLS
            if by_moves or by_calls:
                if st["resets"] >= _RS_MAX:
                    _rs_stats["blocked_max"] += 1
                    return out
                n, top, boards, returns = _rs_summary(st.get("entries"), st["level"])
                sess._execute_auto_reset()
                st["resets"] += 1; st["last_reset"] = cur_num; st["calls_at_reset"] = int(st.get("calls", 0)); _rs_stats["resets"] += 1
                _rs_stats["by_calls"] = _rs_stats.get("by_calls", 0) + int(by_calls and not by_moves)
                st["note"] = ("HARNESS RESET (attempt %d of %d): this level was RESET by the harness after %d moves and %d model calls without "
                              "completing it -- the board is back to the level's start; completed levels are kept and your history "
                              "is kept. What you tried on this level: %d moves (%s); %d distinct boards; returned to an already-seen "
                              "board %d times. Do NOT repeat the same sequences from the start: choose a different mechanic or "
                              "a different target first." % (st["resets"], _RS_MAX, since_start, calls_since, n, top or "-", boards, returns))
                st["note_left"] = _RS_NOTE_TURNS
                print("[RESET] уровень %d: сброс %d/%d после %d ходов / %d вызовов (досок %d, возвратов %d)%s"
                      % (st["level"], st["resets"], _RS_MAX, since_start, calls_since, boards, returns, " [по вызовам]" if (by_calls and not by_moves) else ""), flush=True)
        except Exception as _e:
            print("[RESET] сбой: %r" % (_e,), flush=True)
        return out
    _wta.ToolAgent._run_python_tool = _rs_run
    print("RESET: сброс уровня обвязкой после %d ходов ИЛИ %d вызовов без взятия, не больше %d на уровень, сводка испробованного во входе %d хода. "
          "ПОРОГИ против базы 10.25: польза -- победы-поражения >= +8 и медиана >= +8; вред -- <= -6; генерация 1400-1500."
          % (_RS_STALL, _RS_CALLS, _RS_MAX, _RS_NOTE_TURNS), flush=True)

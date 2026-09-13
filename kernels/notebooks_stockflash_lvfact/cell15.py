
# =====================================================================
# ФАКТ О ЗАВЕРШЕНИИ УРОВНЯ (13.09): при переходе на новый уровень обвязка сообщает, чем был взят
# предыдущий. Только входные токены. Только оффлайн.
# =====================================================================
import inference.agent.tool_agent as _wta
_lf_stats = {"games": 0, "levels": 0, "facts": 0, "turns": 0}

def _lf_fact(history_entries, new_level):
    """Завершивший ход, последние ходы перед ним, число ходов на уровне, доска перед завершением."""
    ents = list(history_entries or [])
    idx = None
    for i, e in enumerate(ents):
        try:
            if int(getattr(e.frame, "level", 0)) >= int(new_level):
                idx = i; break
        except Exception:
            continue
    if idx is None or idx == 0:
        return None, None
    start = 0
    for j in range(idx - 1, -1, -1):
        try:
            if int(getattr(ents[j].frame, "level", 0)) < int(new_level) - 1:
                start = j + 1; break
        except Exception:
            continue
    fin = str(getattr(ents[idx], "action", "") or "?")
    before = [str(getattr(e, "action", "") or "?") for e in ents[max(start, idx - 8):idx]]
    n_moves = idx - start + 1
    prev_ascii = getattr(ents[idx - 1].frame, "ascii", "") or ""
    short = ("LEVEL %d FACT (from the harness log): level %d was completed after %d moves; the completing move was %s; "
             "the moves right before it were: %s." % (new_level, new_level - 1, n_moves, fin, ", ".join(before) or "-"))
    full = short + ("\nThe board right BEFORE the completing move (level %d) looked like this -- use it to infer what "
                    "condition completes a level in this game:\n%s" % (new_level - 1, prev_ascii))
    return short, full

if not TRUE_SUBMISSION:
    _lf_orig_prompt = _wta.ToolAgent._build_user_prompt
    def _lf_prompt(self, action_num, *args, **kwargs):
        text = _lf_orig_prompt(self, action_num, *args, **kwargs)
        try:
            st = getattr(self, "_lf_state", None)
            if st is None:
                st = {"level": None, "short": None, "full_left": 0}; self._lf_state = st; _lf_stats["games"] += 1
            _lf_stats["turns"] += 1
            lv = getattr(kwargs.get("current_frame"), "level", None)
            if lv is not None:
                lv = int(lv)
                if st["level"] is not None and lv > st["level"]:
                    _lf_stats["levels"] += 1
                    short, full = _lf_fact(kwargs.get("history_entries"), lv)
                    if short:
                        st["short"], st["full"], st["full_left"] = short, full, 1
                        _lf_stats["facts"] += 1
                        print("[LVFACT] уровень %d: факт выдан (%s)" % (lv, short[:90]), flush=True)
                st["level"] = lv
            if st.get("short"):
                if st["full_left"] > 0:
                    st["full_left"] -= 1
                    return st["full"] + "\n\n" + text
                return st["short"] + "\n\n" + text
            return text
        except Exception as _e:
            print("[LVFACT] сбой: %r" % (_e,), flush=True)
            return text
    _wta.ToolAgent._build_user_prompt = _lf_prompt
    print("LVFACT: факт о завершении уровня во входе при переходе на новый уровень (полный один раз, дальше строка). "
          "ПОРОГИ против базы 10.25: польза -- победы-поражения >= +8 и медиана >= +8; вред -- <= -6; генерация 1400-1500.", flush=True)


# =====================================================================
# FRESH: сброс гипотезы при застревании (26.09). Ячейкой на датасет v3.
# =====================================================================
import sys as _fsys
from collections import Counter as _FCounter
import inference.agent.tool_agent as _fta
_FRESH_MOVES = 30        # ходов на уровне с последнего сброса (scripts/stall_signal.py)
_FRESH_MAX = 2           # сбросов на уровень
_f_orig = _fta.ToolAgent._build_user_prompt


def _f_build(self, action_num, **kw):
    block = ""
    try:
        fr = kw.get("current_frame"); hist = kw.get("history_entries") or []
        lvl = fr.level if fr is not None else 0
        key = (str(self.__dict__.get("_session_runtime_dir")), lvl)
        st = self.__dict__.setdefault("_f_state", {})
        if st.get("key") != key:
            st.clear(); st.update(key=key, base=0, resets=0)
        on_level = [e for e in hist if getattr(getattr(e, "frame", None), "level", None) == lvl]
        moves = len(on_level)
        if moves - st["base"] >= _FRESH_MOVES and st["resets"] < _FRESH_MAX:
            st["base"] = moves; st["resets"] += 1
            know = self._summarized_knowledge
            old_goal = (know.get("goal_model") or "").strip().replace("\n", " ")[:300]
            old_plan = (know.get("current_plan") or "").strip().replace("\n", " ")[:200]
            for k in ("goal_model", "current_plan", "open_questions"):
                know[k] = ""
            self._history_messages = []
            tried = _FCounter(str(e.action) for e in on_level).most_common(10)
            block = (
                "\nFRESH START on this level: %d moves on this level have not cleared it, so the previous goal "
                "hypothesis is DISCARDED and the conversation history was cleared (facts about mechanics are kept above).\n"
                "- Discarded goal hypothesis: %s\n- Discarded plan: %s\n- Actions already tried on this level (count): %s\n"
                "Do not resume the old plan. Propose a goal hypothesis of a DIFFERENT class (a different target object, a "
                "different win condition, or an interaction you have not tried yet), design one short probe that tests it, and act."
                % (moves, old_goal or "(none recorded)", old_plan or "(none recorded)",
                   ", ".join("%s x%d" % (a, n) for a, n in tried) or "(none)"))
            print("FRESH: сброс %d на уровне %s после %d ходов" % (st["resets"], lvl, moves), file=_fsys.__stderr__, flush=True)
    except Exception as exc:
        print("FRESH: сбой слоя: %r" % (exc,), file=_fsys.__stderr__, flush=True)
    return _f_orig(self, action_num, **kw) + block


_fta.ToolAgent._build_user_prompt = _f_build
print("FRESH: слой включён — сброс после %d ходов на уровне, не более %d раз, поверх датасета v3" % (_FRESH_MOVES, _FRESH_MAX), flush=True)

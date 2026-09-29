
# =====================================================================
# OPEN: разнообразие гипотез в начале каждого уровня (26.09). Ячейкой на датасет v3.
# =====================================================================
import sys as _osys
import inference.agent.tool_agent as _ota
_OPEN_BLOCK = (
    "\nOpening protocol for this level (first turns only): before committing to a plan, write 2-3 genuinely "
    "DIFFERENT hypotheses about this level's goal and key mechanic, labelled H1, H2, H3 -- different classes "
    "(different target object, different win condition, different interaction), not variations of one idea. "
    "Choose ONE short probe (1-3 actions) whose outcome distinguishes them, state what each hypothesis predicts "
    "for it, and execute only that probe. On the next turn, drop the hypotheses the result contradicts before "
    "acting further."
)
_o_orig = _ota.ToolAgent._build_user_prompt


def _o_build(self, action_num, **kw):
    text = _o_orig(self, action_num, **kw)
    try:
        fr = kw.get("current_frame")
        key = (str(self.__dict__.get("_session_runtime_dir")), fr.level if fr is not None else 0)
        if self.__dict__.get("_o_key") != key:
            self._o_key = key; self._o_turns = 0
        self._o_turns += 1
        if self._o_turns <= 2:
            text += _OPEN_BLOCK
            print("OPEN: блок гипотез, уровень %s, вызов %d" % (key[1], self._o_turns), file=_osys.__stderr__, flush=True)
    except Exception as exc:
        print("OPEN: сбой слоя: %r" % (exc,), file=_osys.__stderr__, flush=True)
    return text


_ota.ToolAgent._build_user_prompt = _o_build
print("OPEN: слой включён — первые 2 вызова каждого уровня, поверх датасета v3", flush=True)


# =====================================================================
# КРЮЧОК НА СМЕНЕ УРОВНЯ: один отдельный запрос-обзор на первом ходу нового уровня,
# ответ дописывается в промпт на всех ходах этого уровня. Перезаход того же шага запрос не повторяет.
# =====================================================================
import inference.agent.tool_agent as _hta

_HOOK_MAX_NOTE = 1500
_HOOK_STATS = {"reviews": 0, "fail": 0, "reuse": 0, "injected": 0}
_HOOK_SYSTEM = ("You are reviewing a grid puzzle game you are playing. Answer in plain text, no tool calls, "
                "no code, at most 8 short lines.")
_HOOK_ASK = ("LEVEL REVIEW (asked once by the harness right after you completed a level): in at most 8 short lines "
             "state (1) the rules or mechanics that solved the completed level, stated generally, not as coordinates; "
             "(2) which of them most likely carry over to the new level; (3) what to test first on the new level "
             "to detect what changed.")

_hook_orig_prompt = _hta.ToolAgent._build_user_prompt


def _hook_build_user_prompt(self, action_num, **kw):
    text = _hook_orig_prompt(self, action_num, **kw)
    try:
        frame = kw.get("current_frame")
        if frame is None:
            return text
        step = int(getattr(frame, "step", 0) or 0)
        level = int(getattr(frame, "level", 0) or 0)
        cache = getattr(self, "_hook_cache", None)
        if cache is not None and cache[0] == step:          # перезаход того же шага
            _HOOK_STATS["reuse"] += 1
            return text + cache[1]
        last_level = getattr(self, "_hook_level", None)
        self._hook_level = level
        if last_level is not None and level > last_level:
            note = ""
            try:
                res = self._chat_completion(
                    [{"role": "system", "content": _HOOK_SYSTEM},
                     {"role": "user", "content": text + "\n\n" + _HOOK_ASK}],
                    tools=None)
                try:
                    self._accumulate_usage_tokens(res.usage)
                except Exception:
                    pass
                msg = getattr(res, "message", None) or {}
                note = (msg.get("content") or "").strip()
                if not note:
                    note = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()[-_HOOK_MAX_NOTE:]
                note = note[:_HOOK_MAX_NOTE]
                _HOOK_STATS["reviews"] += 1
                print("[HOOK] уровень %d: обзор получен, %d знаков, всего обзоров %d"
                      % (level, len(note), _HOOK_STATS["reviews"]), flush=True)
            except Exception as _exc:
                _HOOK_STATS["fail"] += 1
                print("[HOOK] сбой запроса-обзора: %r" % (_exc,), flush=True)
            self._hook_note = (level, note)
        extra = ""
        hn = getattr(self, "_hook_note", None)
        if hn and hn[1] and hn[0] == level:
            extra = ("\n\nLEVEL REVIEW NOTES (your own review written right after completing level %d; "
                     "use them, do not restate them):\n%s" % (level - 1, hn[1]))
            _HOOK_STATS["injected"] += 1
        self._hook_cache = (step, extra)
        return text + extra
    except Exception as _exc:
        _HOOK_STATS["fail"] += 1
        print("[HOOK] сбой учёта: %r" % (_exc,), flush=True)
        return text


_hta.ToolAgent._build_user_prompt = _hook_build_user_prompt

print("HOOK: крючок на смене уровня включён (%s). Пороги: второй уровень >= 10 игр из 25 (база 7), "
      "действий на игру >= 165; вред — вызовов на игру < 50." % ("бой" if TRUE_SUBMISSION else "Фаза A"), flush=True)

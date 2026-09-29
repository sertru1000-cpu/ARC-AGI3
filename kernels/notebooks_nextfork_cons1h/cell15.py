
# =====================================================================
# СОГЛАСОВАНИЕ ДВУХ ОТВЕТОВ на первых 4 ходах игры (26.09) — наложено ячейкой на датасет v3.
# Основание: из 11 проверок колеи единственная разница — ход застрявшего сходит с известного
# пути на первых ходах втрое чаще (29% против 10%), обе ветки правдоподобны; единственный
# доступный сигнал — несогласие модели с собой в точке развилки. Цена: +1 вызов на каждый из
# первых 4 ходов, +1 при расхождении. Мерить по ОХВАТУ первых уровней, не по баллу.
# =====================================================================
import json as _cj, sys as _cs
import inference.agent.tool_agent as _cta

_CONS_STEPS = 4
_orig_analyze = _cta.ToolAgent.analyze
_orig_chat = _cta.ToolAgent._chat_completion


def _cons_analyze(self, state_path, action_num, *a, **kw):
    self._cons_action_num = action_num
    self._cons_state_path = state_path
    self._cons_done = False
    return _orig_analyze(self, state_path, action_num, *a, **kw)


def _cons_first_code(tool_calls):
    for tc in tool_calls or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        if str(fn.get("name", "")).strip() != "python":
            continue
        raw = fn.get("arguments", "{}")
        try:
            args = _cj.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except Exception:
            return None
        return str(args.get("code", "")).rstrip() or None
    return None


def _cons_parse(message):
    raw_r = _cta._extract_reasoning_text(message)
    raw_c = _cta._normalize_message_content(message.get("content", ""))
    tcs = _cj.loads(_cj.dumps(message.get("tool_calls") or []))
    if not tcs and _cta._contains_tool_call_markup(raw_r, raw_c):
        tcs = _cta._recover_tool_calls_from_markup(raw_r, raw_c)
    return tcs


def _cons_dry(self, code):
    captured = []

    def recorder(request):
        captured.append(list(request.get("actions") or []))
        raise RuntimeError("consensus dry-run: first action captured")

    real_cb, real_last = self._step_env_callback, self._last_action_result
    self._step_env_callback = recorder
    try:
        self._run_python_tool(self._cons_state_path, {"code": code})
    except Exception:
        pass
    finally:
        self._step_env_callback, self._last_action_result = real_cb, real_last
    return captured[0] if captured else None


def _cons_chat(self, messages, **kw):
    res = _orig_chat(self, messages, **kw)
    try:
        if getattr(self, "_cons_done", True) or int(getattr(self, "_cons_action_num", 99)) >= _CONS_STEPS:
            return res
        self._cons_done = True
        code_a = _cons_first_code(_cons_parse(res.message))
        if not code_a:
            return res
        a1 = _cons_dry(self, code_a)
        if a1 is None:
            return res
        res_b = _orig_chat(self, messages, **kw)
        self._accumulate_usage_tokens(res_b.usage)
        code_b = _cons_first_code(_cons_parse(res_b.message))
        a2 = _cons_dry(self, code_b) if code_b else None
        if a2 is None:
            return res
        same = _cj.dumps(a1, sort_keys=True) == _cj.dumps(a2, sort_keys=True)
        print("CONSENSUS: move %s, A=%s, B=%s -> %s" % (self._cons_action_num, a1, a2, "agree" if same else "DISAGREE"),
              file=_cs.__stderr__, flush=True)
        if same:
            return res
        note = {"role": "user", "content": (
            "Consistency check: two independent analyses of this exact same state proposed different first moves "
            "-- analysis A: %s; analysis B: %s. Both cannot be right. Re-derive from the board which one follows "
            "from the observed rules, state the reason in one line, then act." % (_cj.dumps(a1), _cj.dumps(a2)))}
        res_c = _orig_chat(self, messages + [note], **kw)
        self._accumulate_usage_tokens(res_c.usage)
        return res_c if _cons_parse(res_c.message) else res
    except Exception as exc:
        print("CONSENSUS: сбой слоя, отдаю ответ A: %r" % (exc,), file=_cs.__stderr__, flush=True)
        return res


_cta.ToolAgent.analyze = _cons_analyze
_cta.ToolAgent._chat_completion = _cons_chat
print("CONSENSUS: слой включён, первые %d хода игры, поверх датасета версии 3" % _CONS_STEPS, flush=True)

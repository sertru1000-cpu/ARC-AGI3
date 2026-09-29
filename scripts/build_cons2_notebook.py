"""Сборка `arc3-nextfork-cons2`: согласование С ПРЕДСКАЗАНИЯМИ и проверкой после хода — ячейкой на датасет v3 (26.09).

Отличия от cons (v1), по третьему критику и по тесту «память или суждение» (83% — отказ суждения):
  * на развилке каждый кандидат обязан напечатать PREDICT: что видимо изменится после хода;
  * арбитр при расхождении сравнивает не мнения, а ПРЕДСКАЗАНИЯ с доской;
  * после исполненного хода обвязка САМА проверяет предсказание (изменилась ли доска, куда сдвинулось)
    и кладёт итог в Recent findings следующего промпта — ноль выходных токенов; промах предсказания —
    первый сигнал колеи, который обвязка видит;
  * развилка — первые 4 хода КАЖДОГО уровня, а не только игры.
Предсказание берётся из текста кода регуляркой по print("PREDICT: ...") — не зависит от stdout песочницы.
usage: .venv/bin/python scripts/build_cons2_notebook.py
"""
import ast, json, os

CELL = r'''
# =====================================================================
# CONS2: согласование двух ответов С ПРЕДСКАЗАНИЯМИ + проверка предсказания после хода (26.09).
# Наложено ячейкой на датасет v3. Развилка — первые 4 хода каждого уровня.
# =====================================================================
import json as _cj, re as _cre, sys as _cs
import inference.agent.tool_agent as _cta
from inference.agent.runtime_state import load_runtime_state as _c2_load

_C2_STEPS = 4
_C2_PRED_RX = _cre.compile(r"PREDICT:\s*([^\"'\n]{3,240})")
_C2_ASK = ("\n\nBefore calling action(), print one line `PREDICT: <what will visibly change on the board after this move>` "
           "(e.g. 'PREDICT: blue 3x3 block moves 1 cell up; nothing else changes'). The harness will check it.")
_c2_orig_analyze = _cta.ToolAgent.analyze
_c2_orig_chat = _cta.ToolAgent._chat_completion


def _c2_first_code(tool_calls):
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


def _c2_parse(message):
    raw_r = _cta._extract_reasoning_text(message)
    raw_c = _cta._normalize_message_content(message.get("content", ""))
    tcs = _cj.loads(_cj.dumps(message.get("tool_calls") or []))
    if not tcs and _cta._contains_tool_call_markup(raw_r, raw_c):
        tcs = _cta._recover_tool_calls_from_markup(raw_r, raw_c)
    return tcs


def _c2_dry(self, code):
    captured = []

    def recorder(request):
        captured.append(list(request.get("actions") or []))
        raise RuntimeError("cons2 dry-run: first action captured")

    real_cb, real_last = self._step_env_callback, self._last_action_result
    self._step_env_callback = recorder
    try:
        self._run_python_tool(self._c2_state_path, {"code": code})
    except Exception:
        pass
    finally:
        self._step_env_callback, self._last_action_result = real_cb, real_last
    return captured[0] if captured else None


def _c2_pred(code):
    m = _C2_PRED_RX.search(code or "")
    return m.group(1).strip() if m else ""


def _c2_analyze(self, state_path, action_num, *a, **kw):
    self._c2_state_path = state_path
    self._c2_done = False
    frame, _ = _c2_load(state_path)
    level = getattr(frame, "level", None) if frame is not None else None
    if getattr(self, "_c2_level", None) != level:
        self._c2_level, self._c2_level_start = level, int(action_num)
    self._c2_fork = (int(action_num) - int(getattr(self, "_c2_level_start", action_num))) < _C2_STEPS
    self._c2_grid_before = list(frame.grid) if frame is not None and getattr(frame, "grid", None) is not None else None
    self._c2_pending = ""
    res = _c2_orig_analyze(self, state_path, action_num, *a, **kw)
    # --- проверка предсказания после исполненного хода: обвязка сама, ноль токенов ---
    try:
        pred = getattr(self, "_c2_pending", "")
        lar = self._last_action_result or {}
        if pred and lar.get("executed"):
            changed = bool(lar.get("board_changed"))
            pl = pred.lower()
            said_none = any(k in pl for k in ("nothing", "no change", "no visible", "unchanged", "stays the same"))
            after, _ = _c2_load(state_path)
            note = "changed" if changed else "NO change"
            dirn = ""
            gb, ga = self._c2_grid_before, (list(after.grid) if after is not None and getattr(after, "grid", None) is not None else None)
            if changed and gb and ga:
                gone = [(r, c) for r in range(min(len(gb), len(ga))) for c in range(min(len(gb[r]), len(ga[r]))) if gb[r][c] != ga[r][c] and ga[r][c] == gb[0][0]]
                came = [(r, c) for r in range(min(len(gb), len(ga))) for c in range(min(len(gb[r]), len(ga[r]))) if gb[r][c] != ga[r][c] and gb[r][c] == gb[0][0]]
                if gone and came:
                    dr = sum(p[0] for p in came) / len(came) - sum(p[0] for p in gone) / len(gone)
                    dc = sum(p[1] for p in came) / len(came) - sum(p[1] for p in gone) / len(gone)
                    dirn = ("down" if dr > 0.5 else "up" if dr < -0.5 else "") + (" right" if dc > 0.5 else " left" if dc < -0.5 else "")
                    dirn = dirn.strip()
            verdict = "MISS" if (said_none == changed) else "hit"
            if verdict == "hit" and dirn:
                for d in ("up", "down", "left", "right"):
                    if d in pl and d not in dirn and any(o in pl for o in ("move", "shift", "slide", "go")):
                        verdict = "MISS"
            fact = "Prediction check (harness, factual): you predicted '%s'; observed: board %s%s. %s" % (
                pred, note, (" — net shift " + dirn) if dirn else "", "Prediction failed: revise the rule before the next move." if verdict == "MISS" else "Prediction held.")
            self._summarized_knowledge["recent_findings"] = fact
            print("CONS2: verify %s | pred=%r | %s%s" % (verdict, pred[:80], note, (" " + dirn) if dirn else ""), file=_cs.__stderr__, flush=True)
    except Exception as exc:
        print("CONS2: сбой проверки: %r" % (exc,), file=_cs.__stderr__, flush=True)
    return res


def _c2_chat(self, messages, **kw):
    fork = bool(getattr(self, "_c2_fork", False)) and not getattr(self, "_c2_done", True)
    if fork:
        # просим предсказание: дописываем к последнему пользовательскому сообщению (только если это строка)
        try:
            last = messages[-1]
            if last.get("role") == "user" and isinstance(last.get("content"), str) and "PREDICT:" not in last["content"]:
                messages = messages[:-1] + [dict(last, content=last["content"] + _C2_ASK)]
        except Exception:
            pass
    res = _c2_orig_chat(self, messages, **kw)
    if not fork:
        return res
    try:
        self._c2_done = True
        code_a = _c2_first_code(_c2_parse(res.message))
        if not code_a:
            return res
        a1 = _c2_dry(self, code_a)
        if a1 is None:
            return res
        p1 = _c2_pred(code_a)
        res_b = _c2_orig_chat(self, messages, **kw)
        self._accumulate_usage_tokens(res_b.usage)
        code_b = _c2_first_code(_c2_parse(res_b.message))
        a2 = _c2_dry(self, code_b) if code_b else None
        p2 = _c2_pred(code_b) if code_b else ""
        if a2 is None:
            self._c2_pending = p1
            return res
        same = _cj.dumps(a1, sort_keys=True) == _cj.dumps(a2, sort_keys=True)
        print("CONS2: A=%s pred=%r | B=%s pred=%r -> %s" % (a1, p1[:60], a2, p2[:60], "agree" if same else "DISAGREE"),
              file=_cs.__stderr__, flush=True)
        if same:
            self._c2_pending = p1 or p2
            return res
        note = {"role": "user", "content": (
            "Consistency check: two independent analyses of this exact state proposed different moves.\n"
            "A: %s -- predicts: %s\nB: %s -- predicts: %s\n"
            "Do not pick by preference. Check each prediction against the observed objects and the rules you have "
            "actually seen work in this game; choose the move whose prediction is consistent with them, print its "
            "`PREDICT:` line, then act." % (_cj.dumps(a1), p1 or "(no prediction)", _cj.dumps(a2), p2 or "(no prediction)"))}
        res_c = _c2_orig_chat(self, messages + [note], **kw)
        self._accumulate_usage_tokens(res_c.usage)
        code_c = _c2_first_code(_c2_parse(res_c.message))
        if not code_c:
            self._c2_pending = p1
            return res
        self._c2_pending = _c2_pred(code_c) or p1
        return res_c
    except Exception as exc:
        print("CONS2: сбой слоя, отдаю ответ A: %r" % (exc,), file=_cs.__stderr__, flush=True)
        return res


_cta.ToolAgent.analyze = _c2_analyze
_cta.ToolAgent._chat_completion = _c2_chat
print("CONS2: слой включён — согласование с предсказаниями на первых %d ходах каждого уровня, проверка после хода" % _C2_STEPS, flush=True)
'''


def main() -> None:
    src = json.load(open("kernels/notebooks_nextfork/submission.ipynb", encoding="utf-8"))
    nb = json.loads(json.dumps(src)); cell = nb["cells"][15]; body = "".join(cell["source"])
    anchor = "    seconds=budget - 600.0\n)"; at = body.find(anchor)
    if at < 0 or body.find("await bm.run(") < 0:
        raise SystemExit("не нашёл стоковый soft_end или запуск прогона — сборка остановлена")
    at += len(anchor); code = body[:at] + "\n" + CELL + body[at:]; cell["source"] = code.splitlines(keepends=True)
    out = "kernels/notebooks_nextfork_cons2"; os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(CELL)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_nextfork/kernel-metadata.json")); meta["id"] = "sergueimakarov/arc3-nextfork-cons2"; meta["title"] = "arc3 nextfork cons2"
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    compile(code, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    print("ok   патч ДО запуска прогона:", code.find("_c2_chat") < code.find("await bm.run("))
    print("собрано:", out, "| датасет", meta["dataset_sources"][0])


if __name__ == "__main__":
    main()

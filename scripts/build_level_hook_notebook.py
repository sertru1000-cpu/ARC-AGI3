"""Сборка пункта 5 субботы: крючок на смене уровня — отдельный короткий запрос-обзор при переходе.

ВОПРОС. Мы берём 40 уровней из 183, второй уровень — в 7 играх из 25. Поможет ли модели на новом
уровне, если сразу после взятия прошлого она отдельно сформулирует его правила и что проверить первым?

ЧТО ДЕЛАЕТ ПАТЧ (одна ячейка поверх базы, после стокового soft_end).
  * При построении промпта харнесс сравнивает уровень текущего кадра с прошлым. На первом ходу нового
    уровня делается ОДИН отдельный запрос тем же `_chat_completion` агента, без инструментов:
    системная строка обзора + текст промпта этого хода (в нём мировая модель и сводка последних
    действий) + просьба в 8 коротких строк назвать правила пройденного уровня, что скорее переносится
    и что проверить первым. История сообщений в запрос не берётся — пары «вызов/результат
    инструмента» не рвутся.
  * Ответ (до 1500 знаков; пустой текст — хвост мышления) дописывается в конец промпта на всех ходах
    этого уровня как «LEVEL REVIEW NOTES». Входной токен почти бесплатен.
  * Перезаход солвера в тот же шаг получает тот же текст и повторного запроса не делает.
  * Сбой запроса не ломает ход: заметок нет, счётчик сбоев растёт.
Выход модели в самом ходе не трогается. Цена — один запрос на взятый уровень (~40 на прогон из ~1389).

Пушить ВЕРСИЕЙ в существующий `arc3-stock-flash-wm13` (источники как у базы). Полная Фаза A 2.2 ч.

usage:  .venv/bin/python scripts/build_level_hook_notebook.py
"""
import ast
import json
import os

CELL = r'''
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
'''


def main() -> None:
    src = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))
    nb = json.loads(json.dumps(src))
    body = "".join(nb["cells"][15]["source"])
    anchor = "    seconds=budget - 600.0\n)"
    at = body.find(anchor)
    if at < 0 or body.find("await bm.run(") < 0:
        raise SystemExit("не нашёл стоковый soft_end или запуск прогона")
    at += len(anchor)
    code = body[:at] + "\n" + CELL + body[at:]
    nb["cells"][15]["source"] = code.splitlines(keepends=True)
    out = "kernels/notebooks_stockflash_hook"
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(CELL)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash_wm13/kernel-metadata.json"))
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    compile(code, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    diff = [i for i in range(18) if "".join(nb["cells"][i]["source"]) != "".join(src["cells"][i]["source"])]
    print("ok   изменена только ячейка 15:", diff == [15])
    print("ok   выход хода не трогаем:", "_run_python_tool" not in CELL)
    print("ok   потолок не трогаем:", "max_runtime_s_per_game" not in CELL)
    print("ok   слаг для пуша:", meta["id"])
    print("ok   компилируется, %d символов" % len(code))


if __name__ == "__main__":
    main()

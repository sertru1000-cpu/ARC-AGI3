"""Сборка `arc3-stock-flash-loop`: харнесс сообщает модели, что доска уже была.

НАХОДКА, РАДИ КОТОРОЙ ЭТО ДЕЛАЕТСЯ (10.09, по событийным журналам базы, ноль квоты).
Застрявшая игра не «думает медленно» — она ходит по кругу:
  * доля троек подряд идущих действий, уже встречавшихся раньше, в последней трети игры:
    64% у игр с 0-1 уровнем против 21% у игр с 2+ уровнями;
  * когда доска ВОЗВРАЩАЕТСЯ в уже виденное состояние, застрявшая игра в 25% случаев делает
    оттуда ТОТ ЖЕ ход, что и в прошлый раз. У игр, берущих уровни, — 0%. Ровно ноль.
  * в худших играх возвраты составляют от четверти до двух третей всех ходов
    (sk48 64%, sp80 29%, tn36 25%, wa30 21%), и там же повтор хода доходит до 78%.

ЧТО ДЕЛАЕТ ПАТЧ. Считает хеш доски (в паре с номером уровня) на каждом ходу. Если состояние
уже встречалось, дописывает в КОНЕЦ пользовательского промпта короткий блок: на каком шаге
это состояние было, что из него делали и сколько раз оно повторялось. Ничего не запрещает
и не подменяет — только сообщает факт, который модель по своей истории не восстанавливает.

ПОЧЕМУ ВО ВХОД. Входной токен стоит в 2317 раз дешевле выходного (`scripts/concurrency_math.py`),
блок весит десятки токенов, а строится нашим кодом детерминированно, без обращений к модели.
Выход не трогаем вовсе: опыт `noreason` 07.09 показал, что вмешательство в мышление
модели стоит две трети балла (2.91 против 9.43).

ПОЧЕМУ БЕЗОПАСНО. У нас есть контрольная группа: игры, берущие уровни, из виденного состояния
тот же ход не повторяют НИКОГДА. Значит блок для них не несёт новой информации и ничего
не отнимает; он адресован ровно тем играм, которые иначе крутятся на месте.

usage:  .venv/bin/python scripts/build_loop_notebook.py
"""

import ast
import json
import os

CELL = r'''
# =====================================================================
# ПРОВЕРКА НА ПЕТЛЮ: харнесс сообщает, что эта доска уже была.
#
# Считаем хеш доски вместе с номером уровня. При возврате в виденное состояние дописываем
# в конец промпта: на каком шаге оно было, что из него делали, сколько раз повторялось.
# Ничего не запрещаем — только сообщаем факт, который модель по своей истории не восстановит.
# Выход модели не трогаем: см. noreason 07.09 (2.91 против 9.43).
# =====================================================================
import hashlib as _lh
import inference.agent.tool_agent as _lta

_LOOP_KEEP_ACTS = 6      # сколько действий прошлого захода показывать
_LOOP_STATS = {"hits": 0, "states": 0, "fail": 0, "reentry": 0}

_loop_orig_prompt = _lta.ToolAgent._build_user_prompt


def _loop_build_user_prompt(self, action_num, **kw):
    text = _loop_orig_prompt(self, action_num, **kw)
    try:
        frame = kw.get("current_frame")
        board = getattr(frame, "ascii", None) if frame is not None else None
        if not board:
            return text
        seen = getattr(self, "_loop_seen", None)
        if seen is None:
            seen = {}
            self._loop_seen = seen
            self._loop_last = None
        step = int(getattr(frame, "step", 0) or 0)
        key = (int(getattr(frame, "level", 0) or 0),
               _lh.md5(str(board).encode("utf-8", "ignore")).hexdigest())
        # ПЕРЕЗАХОД ТОГО ЖЕ ШАГА — НЕ ВОЗВРАТ. Солвер Duck повторно входит в тот же analysis_step
        # с той же доской при yielded_control (раз в 60 с) и при сбое запроса: это 36-38% всех
        # построений промпта. Версия 1 принимала их за петлю — 120 ложных тревог из 132 в пробе
        # 10.09. Пропускаем целиком и ДО записи «что делали», иначе туда попадут ходы, которые
        # в это состояние ПРИВЕЛИ.
        rec0 = seen.get(key)
        if rec0 is not None and rec0["step"] == step:
            _LOOP_STATS["reentry"] += 1
            return text
        # чем закончился прошлый заход из прошлого состояния
        summary = kw.get("previous_step_summary")
        last = getattr(self, "_loop_last", None)
        if last is not None and last != key and isinstance(summary, dict):
            acts = [str(a).strip() for a in (summary.get("executed_actions") or []) if str(a).strip()]
            prev = seen.get(last)
            if prev is not None and not prev["after"] and acts:
                prev["after"] = acts[:_LOOP_KEEP_ACTS]
        rec = seen.get(key)
        if rec is None:
            seen[key] = {"step": step, "after": [], "hits": 0}
            _LOOP_STATS["states"] += 1
        else:
            rec["hits"] += 1
            _LOOP_STATS["hits"] += 1
            note = ["", "LOOP CHECK (computed by the harness from the recorded frames, not by you):",
                    "This exact board (level %d) was already seen at step %d; you are back on it "
                    "for the %s time." % (key[0], rec["step"], "%d-th" % (rec["hits"] + 1))]
            if rec["after"]:
                note.append("From that state you then executed: %s." % ", ".join(rec["after"]))
                note.append("That path has already been tried and led back here. Repeating it "
                            "cannot produce new information -- choose an action you have NOT "
                            "tried from this board, or re-examine what the goal actually is.")
            else:
                note.append("Whatever you did from it led back here. Choose an action you have "
                            "NOT tried from this board.")
            text = text + "\n" + "\n".join(note)
            print("[LOOP] шаг %d: возврат в состояние шага %d (повтор %d), всего возвратов %d"
                  % (step, rec["step"], rec["hits"], _LOOP_STATS["hits"]), flush=True)
        self._loop_last = key
    except Exception as _exc:
        _LOOP_STATS["fail"] += 1
        print("[LOOP] сбой учёта: %r" % (_exc,), flush=True)
    return text


_lta.ToolAgent._build_user_prompt = _loop_build_user_prompt

if not TRUE_SUBMISSION:
    bm.solver.max_runtime_s_per_game = 1500.0    # дымовая проба 25 минут

print("LOOP: проверка на петлю включена (%s). Основание: у игр с 0-1 уровнем 25%% возвратов "
      "заканчиваются повтором того же хода, у игр с 2+ уровнями 0%%. Точка сравнения пробы — "
      "база-25: RHAE 2.47, первый уровень 10/25, действий на игру 23.6."
      % ("бой" if TRUE_SUBMISSION else "проба 25 мин"), flush=True)
'''


def main() -> None:
    src = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))
    nb = json.loads(json.dumps(src))
    cell = nb["cells"][15]
    body = "".join(cell["source"])
    anchor = "    seconds=budget - 600.0\n)"
    at = body.find(anchor)
    if at < 0 or body.find("await bm.run(") < 0:
        raise SystemExit("не нашёл стоковый soft_end или запуск прогона — сборка остановлена")
    at += len(anchor)
    code = body[:at] + "\n" + CELL + body[at:]
    cell["source"] = code.splitlines(keepends=True)
    out = "kernels/notebooks_stockflash_loop"
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(CELL)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    meta["id"] = "sergueimakarov/arc3-stock-flash-loop"
    meta["title"] = "arc3 stock flash loop"
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    compile(code, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    diff = [i for i in range(18) if "".join(nb["cells"][i]["source"]) != "".join(src["cells"][i]["source"])]
    new = "".join(nb["cells"][15]["source"])
    print("ok   изменена только ячейка 15:", diff == [15])
    print("ok   патч ПОСЛЕ стокового soft_end:", new.find("_LOOP_KEEP_ACTS") > new.find("budget - 600.0"))
    print("ok   патч ДО запуска прогона:", new.find("_LOOP_KEEP_ACTS") < new.find("await bm.run("))
    print("ok   выход модели не трогаем:", "_run_python_tool" not in CELL and "_chat_completion" not in CELL)
    print("ok   потолок 25 мин только вне боя:", "if not TRUE_SUBMISSION" in CELL)
    print("ok   компилируется, %d символов" % len(code))


if __name__ == "__main__":
    main()

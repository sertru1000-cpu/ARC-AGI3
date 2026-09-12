
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

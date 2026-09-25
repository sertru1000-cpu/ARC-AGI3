
# =====================================================================
# НАПОМИНАНИЕ О ПЕРЕПРОВЕРКЕ: доска изменилась — пора вернуться к брошенному действию.
#
# Измерено 25.09 на записях базы: после пустого хода модель в 61% случаев повторяет его один в один,
# а возврат к нему ПОСЛЕ изменения доски срабатывает в 84% случаев (117 из 140), но случается поздно
# (медиана 4 хода, четверть позже 16-го). Слой ничего не спрашивает у модели и не делает вызовов —
# только напоминает о том, что уже наблюдалось.
# =====================================================================
import inference.agent.tool_agent as _rtta

_RT_MIN_GAP = 3           # не напоминать раньше, чем через столько ходов (половина возвратов и так в 4)
_RT_MAX_ITEMS = 2         # не больше двух напоминаний за раз, чтобы не раздувать промпт
_RT_STATS = {"fired": 0}


def _rt_board(frame):
    return getattr(frame, "ascii", None) or getattr(frame, "grid", None)


def _rt_build_user_prompt(self, action_num, **kw):
    text = _rt_orig_prompt(self, action_num, **kw)
    try:
        hist = list(getattr(self, "history_entries", None) or [])
        if len(hist) < 2:
            return text
        cur = kw.get("current_frame") or getattr(hist[-1], "frame", None)
        cur_level = getattr(cur, "level", None)
        # собираем: действие -> (номер хода, доска при пустом исходе)
        dead = {}
        for i in range(1, len(hist)):
            prev, now = hist[i - 1], hist[i]
            act = (getattr(now, "action", "") or "").strip()
            if not act:
                continue
            if getattr(getattr(now, "frame", None), "level", None) != cur_level:
                dead.clear()                      # другой уровень — прежний опыт не про эту доску
                continue
            before, after = _rt_board(getattr(prev, "frame", None)), _rt_board(getattr(now, "frame", None))
            if before is None or after is None:
                continue
            if before == after:
                dead[act] = (i, before)           # пусто: запоминаем, при какой доске
            else:
                dead.pop(act, None)               # сработало — больше не кандидат
        if not dead:
            return text
        now_board = _rt_board(cur)
        ready = []
        for act, (i, board_then) in dead.items():
            gap = len(hist) - i
            if gap >= _RT_MIN_GAP and now_board is not None and board_then != now_board:
                ready.append((gap, act))
        if not ready:
            return text
        ready.sort(reverse=True)                  # сначала самые давно брошенные
        items = "; ".join("%s (%d actions ago)" % (a, g) for g, a in ready[:_RT_MAX_ITEMS])
        _RT_STATS["fired"] += 1
        if _RT_STATS["fired"] <= 3:
            print("RETEST: напоминание выдано (ход %s): %s" % (action_num, items), flush=True)
        return text + (
            "\n\n[HOST] These actions produced NO change earlier, but the board has changed since: %s. "
            "An action that did nothing before can work once its precondition is met. If one of them fits "
            "your current hypothesis, retest it now instead of repeating what already failed." % items)
    except Exception as exc:                      # слой не имеет права ронять прогон
        print("RETEST: сбой слоя, пропускаю: %r" % (exc,), flush=True)
    return text


_rt_orig_prompt = _rtta.ToolAgent._build_user_prompt
_rtta.ToolAgent._build_user_prompt = _rt_build_user_prompt

if not TRUE_SUBMISSION:
    bm.solver.max_runtime_s_per_game = 7920.0     # полный прогон 132 мин; сравнение flash_v1_phaseA = 9.43

print("RETEST: слой включён (%s). Никаких новых вызовов модели — только учёт в обвязке. "
      "Основание: возврат к пустому действию после изменения доски срабатывает в 84%% случаев "
      "(117 из 140), но модель в 61%% случаев вместо этого повторяет пустой ход."
      % ("бой" if TRUE_SUBMISSION else "полный прогон 132 мин"), flush=True)


# =====================================================================
# РАСПИСАНИЕ: 14 игр одновременно, без потолка на игру, снятие после 45 минут без уровня.
#
# Слот держит тот, кто движется. Игра, не взявшая нового уровня 45 минут (отсчёт от старта
# игры или от последнего уровня), снимается — её место занимает следующая из очереди.
# Модель об этом ничего не знает: снятие делает движок, счёт за него не штрафует, взятые
# уровни сохраняются.
#
# Только оффлайн: в боевой ветке ничего не меняется.
# =====================================================================
import time as _st, datetime as _sd
import inference.framework.solver as _ss

_SCHED_STALL_S = 2700.0     # 45 минут без нового уровня — снятие
_SCHED_CONC    = 18
_SCHED_WINDOW  = 5400.0     # полтора часа на весь прогон

if not TRUE_SUBMISSION:
    bm.solver.concurrency = _SCHED_CONC
    bm.solver.max_runtime_s_per_game = None      # потолка на игру нет вовсе
    soft_end = _sd.datetime.now() + _sd.timedelta(seconds=_SCHED_WINDOW)
    _sched_stats = {"dropped": 0, "started": 0}

    _s_orig_limit = _ss._HarnessGameSession.runtime_limit_reached
    def _s_limit(self):
        if _s_orig_limit(self):
            return True
        now = _st.monotonic()
        try:
            lv = int(self.game.current_state.levels_completed)
        except Exception:
            return False
        mark = getattr(self, "_s_mark", None)
        if mark is None or lv > mark[0]:
            self._s_mark = (lv, now)              # отсчёт заново от старта и от каждого уровня
            return False
        idle = now - mark[1]
        if idle < _SCHED_STALL_S:
            return False
        gid = getattr(getattr(self.game, "game_run", None), "game_id", "?")
        reason = "sched: снята — %.0f с без нового уровня (уровней %d)" % (idle, lv)
        try:
            run = self.game.game_run
            if run is not None and not run.solver_note:
                run.solver_note = reason
        except Exception:
            pass
        _sched_stats["dropped"] += 1
        print("[SCHED] %s: %s; снято всего %d" % (gid, reason, _sched_stats["dropped"]), flush=True)
        return True
    _ss._HarnessGameSession.runtime_limit_reached = _s_limit

    print("SCHED: одновременно %d игр, потолка на игру НЕТ, снятие после %.0f с без уровня, "
          "окно прогона %.0f с; точка сравнения — база, обрезанная на 90 мин: RHAE 7.78, "
          "первый уровень 21/25, действий на игру 111.2"
          % (_SCHED_CONC, _SCHED_STALL_S, _SCHED_WINDOW), flush=True)

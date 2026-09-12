
# =====================================================================
# ПАРКОВКА ЗАСТРЯВШЕЙ ИГРЫ: слот держим, к модели не ходим.
#
# Игра, не взявшая нового уровня за N ходов, перестаёт обращаться к модели, но её сессия
# живёт до штатного потолка. Движку нечего подсаживать, поэтому доля потока сервера
# достаётся соседям по волне — тем, кто ещё берёт уровни. Взятые уровни сохраняются.
#
# Если ВСЕ игры волны запарковались, парковка снимается сама: держать пустую волну незачем.
# =====================================================================
import threading as _pt, time as _ptime
import inference.framework.solver as _ps

_PARK_BATTLE = 100      # ходов без нового уровня -> парковка (бой: ~158 ходов на игру)
_PARK_PROBE  = 100      # Фаза A 4 ч: порог как в бою — он входит в метрику окна
_PARK_ACTS   = _PARK_BATTLE if TRUE_SUBMISSION else _PARK_PROBE
_PARK_MAX_S  = 4 * 3600.0   # страховка: если потолка на игру нет, дольше не паркуемся
# ЗАПРЕТ РАННЕЙ ПАРКОВКИ (правка после прогона 12.09, runs/flash_park_cap_v1).
# Там порог 100 ходов заморозил 18 игр из 25 РАНЬШЕ 132-й минуты, поле вымерло к 155-й,
# и окно замера 132..240 досталось трём играм вместо двадцати пяти. Вне боя парковка
# теперь запрещена до 132-й минуты: первые 132 минуты — точная копия базы, а весь эффект
# парковки измеряется внутри окна. Боевая ветка не меняется (запрет 0).
_PARK_NOT_BEFORE_S = 0.0 if TRUE_SUBMISSION else 7920.0

_park = {"playing": 0, "parked": 0, "lock": _pt.Lock(), "fired": 0}

_orig_should_stop = _ps._HarnessGameSession.should_stop


def _pk_leave(self):
    if getattr(self, "_pk_left", False):
        return
    self._pk_left = True
    with _park["lock"]:
        _park["playing"] -= 1


def _pk_should_stop(self):
    try:
        if not getattr(self, "_pk_reg", False):
            self._pk_reg = True
            with _park["lock"]:
                _park["playing"] += 1
        if _orig_should_stop(self):
            _pk_leave(self)
            return True
        lv = int(self.game.current_state.levels_completed)
        acts = int(self.action_count)
        mark = getattr(self, "_pk_mark", None)
        if mark is None or lv > mark[0]:
            self._pk_mark = (lv, acts)      # отсчёт заново от старта и от каждого уровня
            return False
        if acts - mark[1] < _PARK_ACTS:
            return False
        if _ptime.monotonic() - self.started_at < _PARK_NOT_BEFORE_S:
            return False      # окно замера должно начинаться на полном поле
    except Exception as _exc:
        print("[PARK] сбой учёта, играем как обычно: %r" % (_exc,), flush=True)
        try:
            return _orig_should_stop(self)
        except Exception:
            return False

    gid = getattr(getattr(self.game, "game_run", None), "game_id", "?")
    _pk_leave(self)
    with _park["lock"]:
        _park["parked"] += 1
        _park["fired"] += 1
        left = _park["playing"]
    print("[PARK] %s: %d ходов без нового уровня (уровней %d) -> парковка; "
          "ещё играют %d, запарковано %d" % (gid, acts - mark[1], lv, left, _park["parked"]),
          flush=True)
    try:
        run = self.game.game_run
        if run is not None and not getattr(run, "solver_note", None):
            run.solver_note = "park: %d ходов без нового уровня" % (acts - mark[1])
    except Exception:
        pass

    _t0 = _ptime.monotonic()
    while True:
        try:
            if self.stop_event.is_set():
                break
            if self.runtime_limit_reached():
                break
            if _ptime.monotonic() - _t0 > _PARK_MAX_S:
                break
            with _park["lock"]:
                if _park["playing"] <= 0:
                    break               # волна вымерла — держать слот незачем
        except Exception:
            break
        _ptime.sleep(5.0)
    with _park["lock"]:
        _park["parked"] -= 1
    print("[PARK] %s: расстыковка через %.0f с" % (gid, _ptime.monotonic() - _t0), flush=True)
    return True


_ps._HarnessGameSession.should_stop = _pk_should_stop

if not TRUE_SUBMISSION:
    bm.solver.max_runtime_s_per_game = 14400.0    # потолок 4 часа: живут ли уровни после 132-й минуты

print("PARK: парковка после %d ходов без уровня (%s); потолок на игру %s с. "
      "Фаза A 4 ч: парковка запрещена до 132-й минуты, порог 100 ходов как в бою; сравнение — та же игра на 132-й минуте этого прогона и база на 132-й минуте. "
      "ИМИТАЦИЯ БОЯ: 9.87 против 9.21 у нынешней схемы."
      % (_PARK_ACTS, "бой" if TRUE_SUBMISSION else "проба",
         bm.solver.max_runtime_s_per_game), flush=True)

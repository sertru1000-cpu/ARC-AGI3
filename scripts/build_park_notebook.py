"""Сборка `arc3-stock-flash-park`: застрявшая игра ПАРКУЕТСЯ, а не снимается.

ИДЕЯ ВЛАДЕЛЬЦА (10.09): «почему мы не можем отдать место играм из текущей волны?»
Ответ: можем — если снятая игра не отпускает слот. Тогда движок не подсаживает новую
из очереди, и освободившаяся доля сервера достаётся тем, кто в волне остался.

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ ПРЕЖНЕГО ПАТЧА РАСПИСАНИЯ. Тот возвращал «предел исчерпан», сессия
завершалась, движок немедленно сажал следующую игру — и доля потока уходила к тому, кто ещё
не начинал. Перенос живым не происходил. Здесь сессия остаётся жить до штатного потолка,
но больше не обращается к модели: слот занят, подсадки нет, поток достаётся соседям по волне.

ЧТО ПОСЧИТАНО ДО СБОРКИ (`scripts/battle_flow_sim.py`, поминутная раздача измеренного потока
1973 действия/ч по 110 играм на 28 мест, окно 530 мин, 30 испытаний):
    сейчас, подсадка из очереди .................. RHAE 9.21
    волнами без подсадки, порог 60 ходов ......... 8.78  (-5%)
    волнами без подсадки, порог 80 ходов ......... 9.73  (+6%)
    волнами без подсадки, ПОРОГ 100 ХОДОВ ........ 9.87  (+7%)
    волнами без подсадки, порог 120 ходов ........ 9.48  (+3%)
    все 110 сразу, порог 100 ходов ............... 9.99  (+8%)
Порог 100 выбран по разрывам между уровнями в базе: медиана 29 ходов, три четверти в 44,
девять десятых в 78; порог 100 режет один уровень из сорока.

ПОЧЕМУ ПОРОГ В ХОДАХ, А НЕ В МИНУТАХ. Доля потока на игру зависит от числа одновременных
игр: при 28 местах это 1.17 действия в минуту, при 110 — 0.30. Порог в минутах не переносится
между схемами: в первой имитации «45 минут без уровня» при 110 играх снял вообще всех.

ЧЕСТНО ПРО ПРОБУ. За 25 минут игра получает около 33 ходов (1973/ч ÷ 25 игр × 25 мин),
поэтому боевой порог 100 в пробе не сработает НИ РАЗУ. В пробе стоит отдельный порог 25:
первый уровень в базе берётся на 24-м ходу по медиане, значит правило сработает и механизм
будет виден. **Балл пробы вердиктом о пользе правила НЕ ЯВЛЯЕТСЯ** — проба проверяет, что
машинерия работает: парковка срабатывает, слоты не подсаживаются, ходы концентрируются.

usage:  .venv/bin/python scripts/build_park_notebook.py
"""

import ast
import json
import os

CELL = r'''
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
_PARK_PROBE  = 25       # проба 25 мин: на игру ~33 хода, первый уровень по медиане на 24-м
_PARK_ACTS   = _PARK_BATTLE if TRUE_SUBMISSION else _PARK_PROBE
_PARK_MAX_S  = 4 * 3600.0   # страховка: если потолка на игру нет, дольше не паркуемся

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
    bm.solver.max_runtime_s_per_game = 1500.0     # проба 25 минут на игру

print("PARK: парковка после %d ходов без уровня (%s); потолок на игру %s с. "
      "Точка сравнения пробы — база-25: RHAE 2.47, первый уровень 10/25, действий на игру 23.6. "
      "ИМИТАЦИЯ БОЯ: 9.87 против 9.21 у нынешней схемы."
      % (_PARK_ACTS, "бой" if TRUE_SUBMISSION else "проба",
         bm.solver.max_runtime_s_per_game), flush=True)
'''


def main() -> None:
    src = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))
    nb = json.loads(json.dumps(src))
    # ЯЧЕЙКА 15: стоковая ячейка пересчитывает soft_end от бюджета ноутбука и затирает всё,
    # что задано раньше (урок 09.09 — 2 ч 20 мин квоты). Патч ставится ПОСЛЕ этого
    # присваивания и ПЕРЕД запуском прогона.
    cell = nb["cells"][15]
    body = "".join(cell["source"])
    anchor = "    seconds=budget - 600.0\n)"
    at = body.find(anchor)
    run_at = body.find("await bm.run(")
    if at < 0 or run_at < 0:
        raise SystemExit("не нашёл стоковый soft_end или запуск прогона — сборка остановлена")
    at += len(anchor)
    code = body[:at] + "\n" + CELL + body[at:]
    cell["source"] = code.splitlines(keepends=True)

    out = "kernels/notebooks_stockflash_park"
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(CELL)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    meta["id"] = "sergueimakarov/arc3-stock-flash-park"
    meta["title"] = "arc3 stock flash park"
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)

    compile(code, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    diff = [i for i in range(18) if "".join(nb["cells"][i]["source"]) != "".join(src["cells"][i]["source"])]
    new = "".join(nb["cells"][15]["source"])
    print("ok   изменена только ячейка 15:", diff == [15])
    print("ok   патч ПОСЛЕ стокового soft_end:", new.find("_PARK_BATTLE") > new.find("budget - 600.0"))
    print("ok   патч ДО запуска прогона:", new.find("_PARK_BATTLE") < new.find("await bm.run("))
    print("ok   компилируется, %d символов" % len(code))
    print("ok   боевой порог отличается от пробного:", "_PARK_BATTLE if TRUE_SUBMISSION" in code)
    print("ok   потолок 25 мин только вне боя:", "if not TRUE_SUBMISSION" in CELL)
    print("ok   парковка ДЕРЖИТ слот (не возвращает True сразу):", "_ptime.sleep(5.0)" in CELL)


if __name__ == "__main__":
    main()

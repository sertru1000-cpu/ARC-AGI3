"""Сборка `arc3-stock-flash-sched`: расписание вместо потолка на игру.

ИДЕЯ ВЛАДЕЛЬЦА (09.09): играть 18 игр одновременно, потолка на игру НЕ ставить вовсе,
а игру, которая 45 минут не берёт уровень, снимать — её слот занимает следующая. Окно
прогона полтора часа. Одновременных игр было 14; поднято до 18 после того, как отменённый
прогон показал: при 14 слотах за 140 минут стартовали 19 игр из 25, шесть не сыграли вовсе,
а несыгранная игра — гарантированный ноль в среднем по всем двадцати пяти.

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ СЛОЯ exploit, который мы уже мерили и отвергли (8.09 при базе 9.43):
там все 25 игр уже шли, и освободившийся слот доставался тем же играм. Здесь слот достаётся
игре, которая иначе не сыграла бы совсем, а ноль от несыгранной гарантирован.

ЧТО ГОВОРЯТ НАШИ ЧИСЛА ДО ПУСКА:
  * первый уровень в базе приходит на медиане 26-й минуты; к 45-й минуте взяты 19 из 21,
    позже только tr87 (60) и cd82 (82) — цена порога по этим данным две игры;
  * НО время взятия гуляет между прогонами: отсечка exploit на 79-й минуте сняла семь игр,
    которые в базе брали уровень между 22-й и 77-й. Значит гильотина зарежет больше, чем
    обещает таблица, и это записано заранее;
  * снятие игр измеренно даёт живым больше вызовов: 06.09 одиннадцать живых получили 69
    вызовов вместо 54 (+28%);
  * ёмкость окна: 18 слотов × 90 мин = 1620 игро-минут на 25 игр, которым нужно примерно
    по 45 минут (~1125). Ждут очереди семеро, и место освобождается только когда кого-то
    снимают: при 14 слотах таких мест за 140 минут нашлось всего пять. Восемнадцать слотов
    закрывают риск почти целиком, но не полностью, поэтому число НЕСЫГРАВШИХ игр остаётся
    отдельной метрикой прогона;
  * плата за 18 вместо 14 — меньшая концентрация: вызовов на игру примерно на четверть
    меньше, чем было бы при 14, но всё ещё больше, чем у базы с её 25 одновременными.

ТОЧКА СРАВНЕНИЯ (посчитана бесплатно обрезкой базы на 90 минутах, docs/base90_flash_v1_h115.json):
RHAE 7.78, первый уровень в 21 игре из 25, второй в 7, действий на игру 111.2.

ПОРОГИ, ЗАПИСАННЫЕ ДО ПУСКА: польза — действий на игру ≥ 111 при первом уровне ≥ 19 игр;
вред — первый уровень < 17 игр или RHAE < 6.0; между — расписание не рычаг.

usage:  .venv/bin/python scripts/build_sched_notebook.py
"""

import ast
import json
import os

S = os.path.dirname(os.path.abspath(__file__))

CELL = r'''
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
'''


def main() -> None:
    src = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))
    nb = json.loads(json.dumps(src))
    # ЯЧЕЙКА 15, НЕ 13. Урок 09.09: стоковая ячейка 15 пересчитывает soft_end от бюджета
    # ноутбука (32400 − 600 = 8 ч 50 мин) и затирает любое окно, заданное раньше. Прогон
    # 14:59–17:20 честно собирался работать девять часов и съел 2 ч 20 мин квоты без единого
    # результата. Патч должен стоять ПОСЛЕ этого присваивания.
    cell = nb["cells"][15]
    body = "".join(cell["source"])
    # Патч вставляется ПОСЛЕ стокового soft_end и ПЕРЕД запуском прогона: в конце ячейки он
    # исполнился бы уже после игры, а в начале был бы затёрт (оба случая — пустой прогон).
    anchor = "    seconds=budget - 600.0\n)"          # конец стокового присваивания soft_end
    at = body.find(anchor)
    run_at = body.find("await bm.run(")
    if at < 0 or run_at < 0:
        raise SystemExit("не нашёл стоковый soft_end или запуск прогона — сборка остановлена")
    at += len(anchor)
    code = body[:at] + "\n" + CELL + body[at:]
    cell["source"] = code.splitlines(keepends=True)
    out = "kernels/notebooks_stockflash_sched"
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell13.py"), "w", encoding="utf-8").write(CELL)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    meta["id"] = "sergueimakarov/arc3-stock-flash-sched"
    meta["title"] = "arc3 stock flash sched"
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    compile(code, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)  # в ячейке есть await
    diff = [i for i in range(18) if "".join(nb["cells"][i]["source"]) != "".join(src["cells"][i]["source"])]
    print("ok   изменена только ячейка 15:", diff == [15])
    new = "".join(nb["cells"][15]["source"])
    print("ok   наше окно ПОСЛЕ стокового:", new.rfind("_SCHED_WINDOW)") > new.find("budget - 600.0"))
    print("ok   наш патч ДО запуска прогона:", new.find("_SCHED_WINDOW)") < new.find("await bm.run("))
    print("ok   компилируется, %d символов" % len(code))
    print("ok   правки только вне боя:", code.count("if not TRUE_SUBMISSION") == 2)
    print("ok   потолок на игру снят:", "max_runtime_s_per_game = None" in code)


if __name__ == "__main__":
    main()

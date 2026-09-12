"""Сборка пункта 3 субботы: нагрузка боя на знакомых играх — 25 публичных игр x 2 прохода.

ВОПРОС. Коэффициент переноса публичные -> бой 0.36 (9.43 -> 3.36): его создаёт боевое расписание
с очередью или приватные игры просто тяжелее? Здесь игры знакомые, а раскладка боевая — значит
разница с Фазой A будет только от расписания.

КАК УСТРОЕНО (проверено по первоисточнику `taaf/benchmark.py`): `Benchmark.run` делает `n_passes`
копий каждой игры, складывает ВСЕ в один список (сначала весь проход 0, потом проход 1) и один раз
отдаёт солверу `run_games`. Семафор солвера на 28 мест -> в первой волне 25 игр прохода 0 и 3 игры
прохода 1, ещё 22 ждут в очереди — ровно проверяемый механизм. Потолок на игру стоковый 7920 с.
Сокращено владельцем 11.09 «до 6 часов»: 2 прохода = 50 запусков, две полные волны, ~4.4 ч игры.

КАК СЧИТАТЬ. Итоговый балл харнесса при нескольких проходах берёт по игре ЛУЧШИЙ проход и завышен.
Мерить средний балл одного запуска: `scripts/passes_run_score.py` (записи `game_runs` лежат по схеме
`game_runs[проход * число_игр + игра]`).

Пушить ВЕРСИЕЙ в существующий `arc3-stock-flash-wm12` (источники как у базы).

usage:  .venv/bin/python scripts/build_passes_notebook.py
"""
import ast
import json
import os

CELL = r'''
# =====================================================================
# НАГРУЗКА БОЯ НА ЗНАКОМЫХ ИГРАХ: 2 прохода по 25 публичным играм = 50 запусков под боевой раскладкой
# (28 мест, потолок 132 мин). Очередь из 22 запусков воспроизводит боевое расписание на знакомых играх.
# Только вне боя. Средний балл одного запуска считать scripts/passes_run_score.py — итог завышен.
# =====================================================================
if not TRUE_SUBMISSION:
    bm.n_passes = 2
try:
    _pa_games = len(getattr(bm, "games", None) or [])
except Exception:
    _pa_games = -1
print("PASSES: проходов %d (%s); одновременно %s, потолок на игру %s с; игр %d, запусков %d. "
      "Пороги: средний балл запуска >= 7.5 — разрыв переноса от приватных игр, <= 5.0 — от расписания."
      % (bm.n_passes, "бой" if TRUE_SUBMISSION else "Фаза A", getattr(bm.solver, "concurrency", "?"),
         getattr(bm.solver, "max_runtime_s_per_game", "?"), _pa_games, bm.n_passes * max(_pa_games, 0)), flush=True)
'''


def main() -> None:
    src = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))
    nb = json.loads(json.dumps(src))
    body = "".join(nb["cells"][15]["source"])
    anchor = "    seconds=budget - 600.0\n)"
    at = body.find(anchor)
    if at < 0 or body.find("await bm.run(") < 0:
        raise SystemExit("не нашёл стоковый soft_end или запуск прогона")
    stock_passes = body.find("bm.n_passes = 1")
    if stock_passes < 0 or stock_passes > at:
        raise SystemExit("стоковое bm.n_passes = 1 не найдено ДО места вставки — наше значение затёрлось бы")
    at += len(anchor)
    code = body[:at] + "\n" + CELL + body[at:]
    nb["cells"][15]["source"] = code.splitlines(keepends=True)
    out = "kernels/notebooks_stockflash_passes"
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(CELL)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash_wm12/kernel-metadata.json"))
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    compile(code, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    diff = [i for i in range(18) if "".join(nb["cells"][i]["source"]) != "".join(src["cells"][i]["source"])]
    new = "".join(nb["cells"][15]["source"])
    print("ok   изменена только ячейка 15:", diff == [15])
    print("ok   стоковое n_passes = 1 стоит ДО нашей вставки:", new.find("bm.n_passes = 1") < new.find("bm.n_passes = 2"))
    print("ok   наша вставка ДО запуска прогона:", new.find("bm.n_passes = 2") < new.find("await bm.run("))
    print("ok   потолок и конкурентность не трогаем:", "max_runtime_s_per_game =" not in CELL and "concurrency =" not in CELL)
    print("ok   слаг для пуша:", meta["id"])
    print("ok   компилируется, %d символов" % len(code))


if __name__ == "__main__":
    main()

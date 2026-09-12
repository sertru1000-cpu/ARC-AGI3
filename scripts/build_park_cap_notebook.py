"""Сборка пункта 4 субботы: потолок 4 часа на игру + парковка застрявших (порог 100 ходов).

ЗАЧЕМ ВМЕСТЕ (решение владельца 11.09 после боя паркинга 3.01): пункт 4 проверяет допущение, на котором
держится вся линия расписания и парковки — берёт ли живая игра уровни после 132-й минуты. Паркинг
присоединён, чтобы в том же прогоне проверить схему целиком: застрявшие паркуются, живые получают их
поток и время до 240-й минуты. В Фазе A 25 игр на 28 мест — очереди нет, поэтому парковка передаёт
поток именно живым.

ЦЕНА ОБЪЕДИНЕНИЯ, НАЗВАННАЯ ДО СБОРКИ: если после 132-й минуты уровни появятся, прогон не разделит
вклад лишнего времени и вклад парковки. Чистое плечо «только потолок» можно поставить параллельно.

ОТЛИЧИЯ ОТ `build_park_notebook.py` (боевой сборки паркинга, прошедшей пробу 10.09 и бой 11.09):
  * порог в Фазе A 100 ходов (как в бою) вместо пробных 25 — за 4 часа игра получает ~300 ходов;
  * потолок на игру вне боя 14400 с вместо пробных 1500;
  * боевая ветка не меняется (эта сборка в бой не подаётся).

Пушить ВЕРСИЕЙ 2 в `arc3-stock-flash-park` (версия 1 уже отыграла сабмит 56156224, новая его не затрагивает).

usage:  .venv/bin/python scripts/build_park_cap_notebook.py
"""
import ast
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_park_notebook  # noqa: E402

PROBE_ACTS = 100   # как в боевой сборке: порог входит в метрику, выдумывать своё нельзя
CAP_S = 14400.0
NOT_BEFORE_S = 7920.0   # 132 минуты: до этого момента поле не режем


def make_cell() -> str:
    cell = build_park_notebook.CELL
    reps = [
        ("_PARK_PROBE  = 25       # проба 25 мин: на игру ~33 хода, первый уровень по медиане на 24-м",
         "_PARK_PROBE  = %d      # Фаза A 4 ч: порог как в бою — он входит в метрику окна" % PROBE_ACTS),
        ("_PARK_MAX_S  = 4 * 3600.0   # страховка: если потолка на игру нет, дольше не паркуемся",
         "_PARK_MAX_S  = 4 * 3600.0   # страховка: если потолка на игру нет, дольше не паркуемся\n"
         "# ЗАПРЕТ РАННЕЙ ПАРКОВКИ (правка после прогона 12.09, runs/flash_park_cap_v1).\n"
         "# Там порог 100 ходов заморозил 18 игр из 25 РАНЬШЕ 132-й минуты, поле вымерло к 155-й,\n"
         "# и окно замера 132..240 досталось трём играм вместо двадцати пяти. Вне боя парковка\n"
         "# теперь запрещена до 132-й минуты: первые 132 минуты — точная копия базы, а весь эффект\n"
         "# парковки измеряется внутри окна. Боевая ветка не меняется (запрет 0).\n"
         "_PARK_NOT_BEFORE_S = 0.0 if TRUE_SUBMISSION else %.1f" % NOT_BEFORE_S),
        ("        if acts - mark[1] < _PARK_ACTS:\n            return False",
         "        if acts - mark[1] < _PARK_ACTS:\n            return False\n"
         "        if _ptime.monotonic() - self.started_at < _PARK_NOT_BEFORE_S:\n"
         "            return False      # окно замера должно начинаться на полном поле"),
        ("    bm.solver.max_runtime_s_per_game = 1500.0     # проба 25 минут на игру",
         "    bm.solver.max_runtime_s_per_game = %.1f    # потолок 4 часа: живут ли уровни после 132-й минуты" % CAP_S),
        ("Точка сравнения пробы — база-25: RHAE 2.47, первый уровень 10/25, действий на игру 23.6. ",
         "Фаза A 4 ч: парковка запрещена до 132-й минуты, порог %d ходов как в бою; сравнение — "
         "та же игра на 132-й минуте этого прогона и база на 132-й минуте. " % PROBE_ACTS),
    ]
    for old, new in reps:
        assert old in cell, "не нашёл в ячейке паркинга: %r" % old[:60]
        cell = cell.replace(old, new)
    return cell


def main() -> None:
    cell = make_cell()
    src = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))
    nb = json.loads(json.dumps(src))
    body = "".join(nb["cells"][15]["source"])
    anchor = "    seconds=budget - 600.0\n)"
    at = body.find(anchor)
    if at < 0 or body.find("await bm.run(") < 0:
        raise SystemExit("не нашёл стоковый soft_end или запуск прогона")
    at += len(anchor)
    code = body[:at] + "\n" + cell + body[at:]
    nb["cells"][15]["source"] = code.splitlines(keepends=True)
    out = "kernels/notebooks_stockflash_park_cap"
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(cell)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash_park/kernel-metadata.json"))
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    compile(code, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    diff = [i for i in range(18) if "".join(nb["cells"][i]["source"]) != "".join(src["cells"][i]["source"])]
    print("ok   изменена только ячейка 15:", diff == [15])
    print("ok   порог вне боя %d ходов, потолок вне боя %.0f с" % (PROBE_ACTS, CAP_S))
    print("ok   9-часовой бюджет вмещает потолок:", CAP_S + 600 < 32400)
    print("ok   слаг для пуша:", meta["id"])
    print("ok   компилируется, %d символов" % len(code))


if __name__ == "__main__":
    main()

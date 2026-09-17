"""Перебор после застоя + перенос модели мира и пути уровня (17.09, слово владельца: «объединим с BFS версию 2» → «собирай и
проверяй локально»). Пуш -- только по отдельному слову.

Состав ячейки 15 (после стокового soft_end, до запуска прогона):
1. Слой перебора из scripts/build_bfs_notebook.py в режиме stall: перед вызовом модели, если на текущем уровне нет взятия
   >= --stall секунд и после перебора модели останется >= --min-left секунд, обвязка один раз на уровень ищет следующий
   уровень перебором по настоящей среде (RESET + повтор пути + ход), бюджет --moves ходов и --seconds секунд. Найденный путь
   (в записи модели: UP/DOWN/.../MOUSE(row, col)) кладётся на сессию: sess._bf_found.
2. Слой carry из scripts/build_carry_notebook.py: при взятии уровня моделью -- перенос world/goal/action и путь из истории;
   при взятии перебором (ходы перебора в историю модели не пишутся) -- блок LEVEL N SOLVED BY HARNESS SEARCH с путём
   перебора, тот же перенос модели мира и явное «ты уже на уровне N+1» против стоковой строки «still on the same level».

Арифметика до сборки (ИЗМЕРЕНО на трёх прогонах базы, обрезка 3600 с): уровни, взятые моделью после застоя >= S секунд:
S=1200 -- 39 из 77, S=1500 -- 29, S=1800 -- 18. Любой перебор на уровне обнуляет балл этого уровня (зачётные ходы перебора
в знаменателе), найден путь или нет. Выгода возможна только через следующий уровень, взятый моделью своими ходами.
Контрфактическая оценка до пуска: scripts/bfs_stall_counterfactual.py.

usage:  .venv/bin/python scripts/build_bfscarry_notebook.py --probe [--stall 1800 --min-left 900 --moves 12000 --seconds 600]
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_lvfact_reset_notebook import build  # noqa: E402
import build_bfs_notebook as _bfs  # noqa: E402
import build_carry_notebook as _carry  # noqa: E402

PROBE_CAP_S = 3600.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="потолок игры %s с (проба 1 ч)" % PROBE_CAP_S)
    ap.add_argument("--stall", type=float, default=1800.0); ap.add_argument("--min-left", type=float, default=900.0)
    ap.add_argument("--moves", type=int, default=12000); ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--path-max", type=int, default=150)
    a = ap.parse_args()
    cell = _bfs.cell(a.moves, a.seconds, "stall", 600.0, a.stall, a.min_left) + "\n" + _carry.cell(a.path_max)
    out = "kernels/notebooks_stockflash_bfscarry"
    build(cell, out, "sergueimakarov/arc3-stock-flash-bfscarry", "arc3 stock flash bfscarry", "_cr_stats = ")
    if a.probe:
        p = os.path.join(out, "submission.ipynb")
        nb = json.load(open(p, encoding="utf-8"))
        c15 = "".join(nb["cells"][15]["source"])
        marker = "# Play the benchmark; watchdog stop and teardown run even if it raises."
        assert marker in c15
        c15 = c15.replace(marker, "if not TRUE_SUBMISSION:\n    bm.solver.max_runtime_s_per_game = %r    # проба вне боя\n\n" % PROBE_CAP_S + marker, 1)
        nb["cells"][15]["source"] = c15.splitlines(keepends=True)
        json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("ok   проба: потолок игры %s с" % PROBE_CAP_S)
    print("ok   параметры: застой %.0f с, остаток модели %.0f с, перебор <= %d ходов / %.0f с" % (a.stall, a.min_left, a.moves, a.seconds))


if __name__ == "__main__":
    main()

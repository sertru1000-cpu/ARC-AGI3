"""Проба «4-часовой потолок игры» на текущем стоке (19.09; повтор снятой 14.09 пробы).

Вопрос — единственный неизмеренный факт под последним рычагом обвязки (перераспределение времени между играми):
продолжают ли «живые» к 132-й минуте игры брать уровни дальше. Замер 19.09 по трём прогонам базы: на 87–132 мин
уровни брали 7 / 9 / 14 игр из 25, и эти 45 минут дали 12–34% всего балла.

Правка одна: bm.solver.max_runtime_s_per_game = 14400 (вместо 7920), только под `if not TRUE_SUBMISSION`.
Всё остальное — сток: 25 игр, конкурентность 28, модель и сервер как в бою. Сравнение — с самим собой: усечение
этого же прогона до 7920 с (scripts/truncate_run.py) против полного, поэтому разброс между прогонами не мешает.

usage:  .venv/bin/python scripts/build_cap4h_notebook.py [--slug arc3-stock-flash-predhint]
"""
import argparse, json, os


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--slug", default="arc3-stock-flash-predhint")
    ap.add_argument("--cap", type=float, default=14400.0)
    a = ap.parse_args()
    nb = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))
    c15 = "".join(nb["cells"][15]["source"])
    marker = "# Play the benchmark; watchdog stop and teardown run even if it raises."
    assert marker in c15
    patch = ("if not TRUE_SUBMISSION:\n"
             "    bm.solver.max_runtime_s_per_game = %r    # [[CAP4H]] проба вне боя: 4 ч на игру вместо 7920 с\n"
             "    print('[[CAP4H]] потолок игры %%s с, игр %%d, конкурентность %%s' %% (bm.solver.max_runtime_s_per_game, "
             "len(bm.games), getattr(bm.solver, 'concurrency', '?')), flush=True)\n\n" % a.cap)
    c15 = c15.replace(marker, patch + marker, 1)
    nb["cells"][15]["source"] = c15.splitlines(keepends=True)
    out = "kernels/notebooks_cap4h"
    os.makedirs(out, exist_ok=True)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    meta["id"] = "sergueimakarov/" + a.slug; meta["title"] = a.slug.replace("-", " ")
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    compile("".join(nb["cells"][15]["source"]), "c15", "exec", 0x2000)
    print("ok   собрано: %s (кернел %s); потолок игры %.0f с, 25 игр, сток" % (out, meta["id"], a.cap))


if __name__ == "__main__":
    main()

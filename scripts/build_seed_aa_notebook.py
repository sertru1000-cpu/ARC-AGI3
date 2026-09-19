"""Проба «база против базы с одним зерном»: воспроизводима ли партия при фиксированном seed (19.09).

Зачем. Критик раунда 10 предложил мерить слои ПАРАМИ: одна игра, одно зерно, два варианта -- тогда разность
внутри пары свободна от случайности модели, и видны эффекты в 10-20%. Это держится на допущении, которое мы не
проверяли: даёт ли vLLM с одним seed одинаковые ответы, когда рядом в пакете идут другие запросы. Если нет --
пары не точнее одиночных прогонов, и протокол критика стоит в разы дороже.

Как: стоковая сборка без единой правки обвязки; LOCAL_ANALYZER_SEED задан (обвязка кладёт одно и то же зерно
в КАЖДЫЙ запрос, tool_agent.py: seed=_LOCAL_ANALYZER_SEED); 4 игры, каждая играется ДВАЖДЫ (n_passes = 2)
одновременно -- ровно боевые условия пакетной работы сервера. Сравниваем последовательности ходов двух проходов.

usage:  .venv/bin/python scripts/build_seed_aa_notebook.py --seed 17 --games lp85,r11l,tu93,lf52 --cap 1800
"""
import argparse, json, os

SEED_ENV = '''
# --- [[SEEDAA]] фиксированное зерно для проверки воспроизводимости (scripts/build_seed_aa_notebook.py) ---
import os as _sa_os
_sa_os.environ["LOCAL_ANALYZER_SEED"] = "%(seed)d"
print("[[SEEDAA]] LOCAL_ANALYZER_SEED = %%s" %% _sa_os.environ["LOCAL_ANALYZER_SEED"], flush=True)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=17); ap.add_argument("--games", default="lp85,r11l,tu93,lf52")
    ap.add_argument("--cap", type=float, default=1800.0); ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--slug", default="arc3-stock-flash-predhint")
    a = ap.parse_args()
    nb = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))

    # 1) зерно -- в конце ячейки 9 (после команд установки); модули обвязки импортируются в ячейке 11 при распаковке bm,
    #    а tool_agent читает переменную при импорте -- значит, зерно успевает
    c9 = "".join(nb["cells"][9]["source"]) + "\n" + SEED_ENV % {"seed": a.seed}
    nb["cells"][9]["source"] = c9.splitlines(keepends=True)

    # 2) проба: свои игры, потолок, два прохода; проверка «сыграны все 25 игр» -- только в бою
    c15 = "".join(nb["cells"][15]["source"])
    marker = "# Play the benchmark; watchdog stop and teardown run even if it raises."
    assert marker in c15
    games = ", ".join('"%s"' % g for g in a.games.split(","))
    patch = ("if not TRUE_SUBMISSION:\n"
             "    bm.solver.max_runtime_s_per_game = %r    # проба вне боя\n"
             "    _sa_want = [%s]\n"
             "    bm.games = [g for g in bm.games if str(getattr(g, 'env_name', getattr(g, 'game_id', '')))[:4] in _sa_want]\n"
             "    bm.n_passes = %d                          # каждая игра дважды, одновременно\n"
             "    print('[[SEEDAA]] игр %%d, проходов %%d' %% (len(bm.games), bm.n_passes), flush=True)\n\n"
             % (a.cap, games, a.passes))
    c15 = c15.replace(marker, patch + marker, 1)
    cov = "if len(public_runs) != 25 or public_run_ids != list(PUBLIC_GAME_IDS):"
    assert cov in c15
    c15 = c15.replace(cov, "if TRUE_SUBMISSION and (len(public_runs) != 25 or public_run_ids != list(PUBLIC_GAME_IDS)):")
    nb["cells"][15]["source"] = c15.splitlines(keepends=True)

    out = "kernels/notebooks_seed_aa"
    os.makedirs(out, exist_ok=True)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    meta["id"] = "sergueimakarov/" + a.slug; meta["title"] = a.slug.replace("-", " ")
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    for i in (9, 15):
        compile("".join(nb["cells"][i]["source"]), "c%d" % i, "exec", 0x2000)
    print("ok   собрано: %s (кернел %s); зерно %d, игры %s, потолок %.0f с, проходов %d"
          % (out, meta["id"], a.seed, a.games, a.cap, a.passes))


if __name__ == "__main__":
    main()

"""Ядро на НАШЕМ форке обвязки Duck вместо чужого бандла (19.09, см. scripts/duck_fork.py).

Меняется ровно одно: источник бандла. Стоковый ноутбук ищет бандл по файлу-маркеру `taaf-kaggle-bundle.json`
(первый найденный под /kaggle/input) и держит список входов в константе DATASET_SOURCES (ячейка 7), где бандл
стоит первым. Поэтому чужой бандл убирается ОТОВСЮДУ -- и из метаданных, и из константы: если подключить оба,
ноутбук возьмёт тот маркер, что найдётся первым, и какой именно -- не предсказать.

Всё остальное (модель Qwen3.8-Flash-Next, runtime vLLM, карта, соревнование) -- как у рабочего стока.
`serving_setup.py` упоминает эталонный датасет только в записи о происхождении (SOURCE_DATASET), для поиска
не используется -- его не трогаем, иначе упадёт сверка хеша с SOURCE_IDENTITY.json.

usage:
  .venv/bin/python scripts/build_fork_notebook.py                       # боевой вид: как сток, только бандл наш
  .venv/bin/python scripts/build_fork_notebook.py --probe --cap 1800    # проба Фазы A с потолком (только вне боя)
"""
import argparse, json, os

UPSTREAM = "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1"
FORK = "sergueimakarov/arc3-duck-fork"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true"); ap.add_argument("--cap", type=float, default=1800.0)
    ap.add_argument("--slug", default="arc3-duck-fork-run")
    a = ap.parse_args()
    nb = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))
    c7 = "".join(nb["cells"][7]["source"])
    assert c7.count(UPSTREAM) == 1, "ячейка 7: ожидал ровно одно упоминание чужого бандла"
    c7 = c7.replace(UPSTREAM, FORK)
    nb["cells"][7]["source"] = c7.splitlines(keepends=True)
    for i, c in enumerate(nb["cells"]):
        assert UPSTREAM not in "".join(c["source"]), "чужой бандл остался в ячейке %d" % i
    if a.probe:
        c15 = "".join(nb["cells"][15]["source"])
        marker = "# Play the benchmark; watchdog stop and teardown run even if it raises."
        assert marker in c15
        c15 = c15.replace(marker, "if not TRUE_SUBMISSION:\n    bm.solver.max_runtime_s_per_game = %r    # проба вне боя\n\n" % a.cap + marker, 1)
        nb["cells"][15]["source"] = c15.splitlines(keepends=True)
    out = "kernels/notebooks_duck_fork"
    os.makedirs(out, exist_ok=True)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    assert meta["dataset_sources"][0] == UPSTREAM
    meta["dataset_sources"] = [FORK] + meta["dataset_sources"][1:]
    meta["id"] = "sergueimakarov/" + a.slug; meta["title"] = a.slug.replace("-", " ")
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    for i in (7, 15):
        compile("".join(nb["cells"][i]["source"]), "c%d" % i, "exec", 0x2000)
    print("ok   собрано: %s (кернел %s); входы %s%s" % (out, meta["id"], meta["dataset_sources"],
                                                      "; проба, потолок %.0f с" % a.cap if a.probe else "; боевой вид"))


if __name__ == "__main__":
    main()

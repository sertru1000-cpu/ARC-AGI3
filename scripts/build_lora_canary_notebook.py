"""Проба ПУТИ ДОСТАВКИ обученной модели: пустой адаптер LoRA на нашем форке обвязки (20.09).

Зачем (см. docs/artifacts/training_cost_20_09.html): обучение боевой модели стоит $535-1425, и вся эта трата
обесценивается, если обученные веса нельзя довезти до ядра. Два известных препятствия: (1) ядро прибито к модели
тремя проверками в serving_setup.py, (2) vLLM умеет МОЛЧА пропустить модули адаптера -- так было со студентом 27B
в августе. Проба отвечает на оба вопроса бесплатно, квотой.

Что в кернеле: форк обвязки (sergueimakarov/arc3-duck-fork) + датасет пустого адаптера
(sergueimakarov/arc3-empty-lora, матрица B нулевая -- ответ модели не меняется). Правки форка включаются
переменной ARC3_LORA_PATH, которую ставит ячейка 7 ТОЛЬКО вне боя; без переменной форк ведёт себя как сток.

Как читать итог:
  * в логе есть [[LORA]] vLLM flags added, [[LORA]] /v1/models reports [... 'policy'] и балл на уровне базы-30
    (3.06 по docs/base30_flash_v1.json) -- путь доставки ЖИВ, обучение имеет смысл оплачивать;
  * vLLM падает при старте с адаптером -- путь мёртв в лоб, нужен другой способ (пересжатие целиком);
  * сервер поднялся, но в логе vLLM нет упоминания загруженных модулей адаптера -- тот самый молчаливый пропуск,
    это и есть главный риск, ради которого проба ставится.

usage:  .venv/bin/python scripts/build_lora_canary_notebook.py [--cap 1800] [--slug arc3-lora-canary]
"""
import argparse, json, os

UPSTREAM = "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1"
FORK = "sergueimakarov/arc3-duck-fork"
LORA = "sergueimakarov/arc3-empty-lora"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=1800.0)
    ap.add_argument("--slug", default="arc3-lora-canary")
    a = ap.parse_args()
    nb = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))

    # --- ячейка 7: бандл -> форк, подключаем датасет адаптера и ставим ARC3_LORA_PATH вне боя
    c7 = "".join(nb["cells"][7]["source"])
    assert c7.count(UPSTREAM) == 1, "ячейка 7: ожидал ровно одно упоминание чужого бандла"
    c7 = c7.replace(UPSTREAM, FORK)
    c7 = c7.replace('"%s"]\nKERNEL_SOURCES' % "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1",
                    '"%s", "%s"]\nKERNEL_SOURCES' % ("keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1", LORA))
    assert LORA in c7, "ячейка 7: не удалось дописать датасет адаптера в DATASET_SOURCES"
    c7 += '''

# [[LORA]] Проба пути доставки (только вне боя): показываем обвязке папку пустого адаптера.
if not TRUE_SUBMISSION:
    _lora_dir = _first_existing(_dataset_mount_candidates("%s")) or _dataset_mount_candidates("%s")[0]
    _lora_files = sorted(p.name for p in Path(_lora_dir).glob("*")) if Path(_lora_dir).exists() else []
    if "adapter_config.json" not in _lora_files:
        raise RuntimeError("[[LORA]] adapter not mounted at %%s (files=%%s)" %% (_lora_dir, _lora_files))
    setup_env["ARC3_LORA_PATH"] = str(_lora_dir)
    os.environ.update(setup_env)
    SETUP_ENV_PATH.write_text(json.dumps(setup_env, indent=2, sort_keys=True) + "\\n")
    print("[[LORA]] adapter dir = %%s, files = %%s" %% (_lora_dir, _lora_files), flush=True)
''' % (LORA, LORA)
    nb["cells"][7]["source"] = c7.splitlines(keepends=True)

    # --- ячейка 15: потолок игры на время пробы
    c15 = "".join(nb["cells"][15]["source"])
    marker = "# Play the benchmark; watchdog stop and teardown run even if it raises."
    assert marker in c15
    c15 = c15.replace(marker, "if not TRUE_SUBMISSION:\n    bm.solver.max_runtime_s_per_game = %r    # [[LORA]] проба вне боя\n\n" % a.cap + marker, 1)
    nb["cells"][15]["source"] = c15.splitlines(keepends=True)

    for i, c in enumerate(nb["cells"]):
        assert UPSTREAM not in "".join(c["source"]), "чужой бандл остался в ячейке %d" % i

    out = "kernels/notebooks_lora_canary"
    os.makedirs(out, exist_ok=True)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    assert meta["dataset_sources"][0] == UPSTREAM
    meta["dataset_sources"] = [FORK] + meta["dataset_sources"][1:] + [LORA]
    meta["id"] = "sergueimakarov/" + a.slug
    meta["title"] = a.slug.replace("-", " ")
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    for i in (7, 15):
        compile("".join(nb["cells"][i]["source"]), "c%d" % i, "exec", 0x2000)
    print("ok   собрано: %s (кернел %s)" % (out, meta["id"]))
    print("     входы: %s" % meta["dataset_sources"])
    print("     потолок игры %.0f с, адаптер включается только при TRUE_SUBMISSION=False" % a.cap)


if __name__ == "__main__":
    main()

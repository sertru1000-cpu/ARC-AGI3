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

PROBE_CELL = """
# [[LORA]] РЕШАЮЩАЯ ПРОВЕРКА (только вне боя): один и тот же запрос к базовой модели и к псевдониму адаптера
# при температуре 0. Адаптер "слышный" (матрица B шумная), поэтому ответы ОБЯЗАНЫ отличаться.
# Совпали -- значит vLLM модули адаптера молча пропустил, и путь доставки обученной модели мёртв.
if not TRUE_SUBMISSION and os.environ.get("ARC3_LORA_PATH"):
    import urllib.request as _u

    _base_url = os.environ.get("LOCAL_ANALYZER_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    _key = os.environ.get("LOCAL_ANALYZER_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    _msg = [{"role": "user", "content": "Count from 1 to 12, separated by commas. Answer with the list only."}]

    def _ask(model_id):
        body = json.dumps({"model": model_id, "messages": _msg, "temperature": 0.0,
                           "max_tokens": 48, "seed": 0}).encode()
        req = _u.Request(_base_url.rstrip("/") + "/chat/completions", data=body,
                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + _key})
        with _u.urlopen(req, timeout=180) as resp:
            out = json.loads(resp.read())
        return (out["choices"][0]["message"].get("content") or "").strip()

    _base_txt = _ask("Qwen/Qwen3.8-Flash-Next-NVFP4")
    _lora_txt = _ask("policy")
    print("[[LORA]] base  answer: %r" % _base_txt[:200], flush=True)
    print("[[LORA]] policy answer: %r" % _lora_txt[:200], flush=True)
    print("[[LORA]] VERDICT: %s" % ("МОДУЛИ ПРИМЕНЕНЫ (ответы различаются)" if _base_txt != _lora_txt
                                    else "МОЛЧАЛИВЫЙ ПРОПУСК (ответы совпали)"), flush=True)
"""

UPSTREAM = "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1"
FORK = "sergueimakarov/arc3-duck-fork"
LORA_ZERO = "sergueimakarov/arc3-empty-lora"
LORA_LOUD = "sergueimakarov/arc3-loud-lora"



def _runtime_patch_block() -> str:
    """Блок для ячейки 7: копирует чужой бандл в рабочую папку, кладёт НАШ serving_setup.py и чинит хеш.

    Почему не заплатки строками: наш файл кладётся целиком (gzip+base64, 41 КБ в ноутбуке), поэтому копия
    в кернеле побайтово равна проверенному локально файлу -- ошибиться заменой нельзя. SOURCE_IDENTITY.json
    пересчитывается на месте, иначе обвязка падает на сверке своего хеша.
    """
    import base64, gzip
    blob = base64.b64encode(gzip.compress(open("harness/duck/serving_setup.py", "rb").read(), 9)).decode()
    return (
        "\n# [[LORA]] Правка бандла на лету (только вне боя): копия чужого бандла + наш serving_setup.py.\n"
        "if not TRUE_SUBMISSION:\n"
        "    import base64 as _b64, gzip as _gz, hashlib as _hl, shutil as _sh\n"
        "    _patched = WORKING_DIR / 'bundle_patched'\n"
        "    if _patched.exists():\n"
        "        _sh.rmtree(_patched)\n"
        "    _sh.copytree(BUNDLE_DIR, _patched, symlinks=True)\n"
        "    _ss = _patched / 'serving_setup.py'\n"
        "    _old_sha = _hl.sha256(_ss.read_bytes()).hexdigest()\n"
        "    _ss.write_bytes(_gz.decompress(_b64.b64decode('" + blob + "')))\n"
        "    _new_sha = _hl.sha256(_ss.read_bytes()).hexdigest()\n"
        "    _si = _patched / 'SOURCE_IDENTITY.json'\n"
        "    _ident = json.loads(_si.read_text())\n"
        "    if _ident.get('serving_setup_sha256') != _old_sha:\n"
        "        raise RuntimeError('[[LORA]] upstream serving_setup hash mismatch: %s' % _old_sha)\n"
        "    _ident['serving_setup_sha256'] = _new_sha\n"
        "    _si.write_text(json.dumps(_ident, indent=1))\n"
        "    BUNDLE_DIR = _patched\n"
        "    print('[[LORA]] patched bundle at %s (serving_setup %s -> %s)' % (BUNDLE_DIR, _old_sha[:12], _new_sha[:12]), flush=True)\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=300.0, help="потолок игры: проба короткая, квота дорога")
    ap.add_argument("--adapter", choices=["loud", "zero"], default="loud")
    ap.add_argument("--bundle", choices=["fork", "runtime"], default="runtime",
                    help="fork -- наш датасет-форк (ждёт обработки Kaggle); "
                         "runtime -- взять чужой бандл и починить копию прямо в кернеле")
    ap.add_argument("--slug", default="arc3-lora-canary")
    a = ap.parse_args()
    nb = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))

    # --- ячейка 7: бандл -> форк, подключаем датасет адаптера и ставим ARC3_LORA_PATH вне боя
    c7 = "".join(nb["cells"][7]["source"])
    assert c7.count(UPSTREAM) == 1, "ячейка 7: ожидал ровно одно упоминание чужого бандла"
    if a.bundle == "fork":
        c7 = c7.replace(UPSTREAM, FORK)
    else:
        # Бандл остаётся чужой; копию правим в кернеле сразу после того, как она найдена по маркеру.
        anchor = 'print(f"taaf.kaggle: source bundle = {BUNDLE_DIR}")'
        assert anchor in c7, "ячейка 7: не найдено место для правки бандла на лету"
        c7 = c7.replace(anchor, anchor + "\n" + _runtime_patch_block(), 1)
    lora = LORA_LOUD if a.adapter == "loud" else LORA_ZERO
    c7 = c7.replace('"%s"]\nKERNEL_SOURCES' % "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1",
                    '"%s", "%s"]\nKERNEL_SOURCES' % ("keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1", lora))
    assert lora in c7, "ячейка 7: не удалось дописать датасет адаптера в DATASET_SOURCES"
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
''' % (lora, lora)
    nb["cells"][7]["source"] = c7.splitlines(keepends=True)

    # --- ячейка 13: решающее сравнение базы и псевдонима
    c13 = "".join(nb["cells"][13]["source"]) + PROBE_CELL
    nb["cells"][13]["source"] = c13.splitlines(keepends=True)

    # --- ячейка 15: потолок игры на время пробы
    c15 = "".join(nb["cells"][15]["source"])
    marker = "# Play the benchmark; watchdog stop and teardown run even if it raises."
    assert marker in c15
    c15 = c15.replace(marker, "if not TRUE_SUBMISSION:\n    bm.solver.max_runtime_s_per_game = %r    # [[LORA]] проба вне боя\n\n" % a.cap + marker, 1)
    nb["cells"][15]["source"] = c15.splitlines(keepends=True)

    if a.bundle == "fork":
        for i, c in enumerate(nb["cells"]):
            assert UPSTREAM not in "".join(c["source"]), "чужой бандл остался в ячейке %d" % i

    out = "kernels/notebooks_lora_canary"
    os.makedirs(out, exist_ok=True)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    assert meta["dataset_sources"][0] == UPSTREAM
    meta["dataset_sources"] = ([FORK] if a.bundle == "fork" else [UPSTREAM]) + meta["dataset_sources"][1:] + [lora]
    meta["id"] = "sergueimakarov/" + a.slug
    meta["title"] = a.slug.replace("-", " ")
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    for i in (7, 13, 15):
        compile("".join(nb["cells"][i]["source"]), "c%d" % i, "exec", 0x2000)
    print("ok   собрано: %s (кернел %s)" % (out, meta["id"]))
    print("     входы: %s" % meta["dataset_sources"])
    print("     потолок игры %.0f с, адаптер включается только при TRUE_SUBMISSION=False" % a.cap)


if __name__ == "__main__":
    main()

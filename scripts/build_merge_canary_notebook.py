"""Канарейка ВТОРОГО пути доставки: обученные веса вливаются в bf16-файлы модели (22.09).

Зачем. Первый путь (адаптер на лету) проверен прогоном arc3-lora-canary-j: работает, но требует четырёх
уступок в настройке сервера -- кэш внимания 5 -> 3 ГиБ, пакет 8192 -> 2048 токенов, выключенные графы CUDA,
один адаптер в пакете. Все они бьют по пропускной способности, то есть в бою такая доставка не бесплатна.

Второй путь этих уступок не требует вовсе: в чекпойнте 419 файлов, из них 119 ГБ -- эксперты в 4 битах,
а обучаемая часть лежит в четырёх файлах bf16 (16 ГБ). Обученные веса можно влить прямо в них, эксперты
оставить байт в байт, и ядро стартует СТОКОВО.

Что делает этот кернел: строит рядом папку-оверлей из ссылок на оригинальные файлы, подменяя ОДИН файл --
`model-bf16-00010.safetensors` (0.32 ГБ). В нём лежат и матрицы линейного внимания слоя 10, и q_proj/o_proj
слоя 11 (полное внимание), то есть одним файлом затрагиваются оба типа слоёв. В копии ОБНУЛЯЮТСЯ два тензора
(правка побайтовая, размер файла и заголовок сохраняются), манифест пересобирается с новым хешем.

Как читать итог:
  * модель отвечает мусором или молчит и игра не делает ходов -- оверлей ЗАГРУЗИЛСЯ, путь доставки работает
    (для сверки: на стоке тот же запрос дал '1, 2, 3, 4,' в прогоне arc3-lora-canary-j);
  * модель отвечает нормально и играет как обычно -- vLLM читает оригинальные файлы, оверлей мимо;
  * падение на сверке модели -- обвязка не приняла наш манифест, читать причину в логе.

usage:  .venv/bin/python scripts/build_merge_canary_notebook.py [--slug arc3-merge-canary]
"""
import argparse, json, os

UPSTREAM = "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1"
SHARD = "model-bf16-00010.safetensors"
TENSORS = [
    "model.language_model.layers.11.self_attn.q_proj.weight",      # полное внимание
    "model.language_model.layers.10.linear_attn.out_proj.weight",  # линейное внимание
]


def _overlay_block() -> str:
    return (
        "\n# [[MERGE]] Оверлей модели: ссылки на оригинал + один подменённый bf16-файл (только вне боя).\n"
        "if not TRUE_SUBMISSION:\n"
        "    import hashlib as _hl, shutil as _sh, struct as _st\n"
        "    _src = Path('/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1')\n"
        "    _dst = WORKING_DIR / 'model_overlay'\n"
        "    if _dst.exists():\n"
        "        _sh.rmtree(_dst)\n"
        "    _dst.mkdir(parents=True)\n"
        "    _shard = %r\n"
        "    _n = 0\n"
        "    for _f in sorted(_src.rglob('*')):\n"
        "        if not _f.is_file():\n"
        "            continue\n"
        "        _rel = _f.relative_to(_src)\n"
        "        _out = _dst / _rel\n"
        "        _out.parent.mkdir(parents=True, exist_ok=True)\n"
        "        if _rel.name in (_shard, 'MODEL_MANIFEST.json'):\n"
        "            _sh.copy2(_f, _out)\n"
        "        else:\n"
        "            _out.symlink_to(_f)\n"
        "        _n += 1\n"
        "    print('[[MERGE]] оверлей собран: %%d файлов, копий 2, остальное ссылки' %% _n, flush=True)\n"
        "\n"
        "    # побайтовая правка: обнуляем два тензора, заголовок и размер файла не трогаем\n"
        "    import torch as _t\n"
        "    _path = _dst / _shard\n"
        "    with open(_path, 'rb') as _fh:\n"
        "        _hn = _st.unpack('<Q', _fh.read(8))[0]\n"
        "        _hdr = json.loads(_fh.read(_hn).decode())\n"
        "    _base = 8 + _hn\n"
        "    for _name in %r:\n"
        "        _row = _hdr[_name]\n"
        "        _a, _b = _row['data_offsets']\n"
        "        with open(_path, 'r+b') as _fh:\n"
        "            _fh.seek(_base + _a)\n"
        "            _fh.write(bytes(_b - _a))\n"
        "        print('[[MERGE]] обнулён %%s %%s (%%d байт)' %% (_name.split('language_model.')[-1], _row['shape'], _b - _a), flush=True)\n"
        "\n"
        "    # манифест пересобираем: тот же список файлов, новый хеш у подменённого\n"
        "    _mf = json.loads((_dst / 'MODEL_MANIFEST.json').read_text())\n"
        "    _sha = _hl.sha256(_path.read_bytes()).hexdigest()\n"
        "    _hit = 0\n"
        "    for _row in _mf['files']:\n"
        "        if str(_row['path']).endswith(_shard):\n"
        "            if int(_row['size']) != _path.stat().st_size:\n"
        "                raise RuntimeError('[[MERGE]] размер файла поехал: %%d против %%d' %% (_path.stat().st_size, int(_row['size'])))\n"
        "            _row['sha256'] = _sha\n"
        "            _hit += 1\n"
        "    if _hit != 1:\n"
        "        raise RuntimeError('[[MERGE]] в манифесте не найден %%s' %% _shard)\n"
        "    (_dst / 'MODEL_MANIFEST.json').write_text(json.dumps(_mf))\n"
        "    print('[[MERGE]] манифест обновлён, новый хеш файла %%s' %% _sha[:12], flush=True)\n"
        "    setup_env['ARC3_MODEL_OVERLAY'] = str(_dst)\n"
        "    os.environ.update(setup_env)\n"
        "    SETUP_ENV_PATH.write_text(json.dumps(setup_env, indent=2, sort_keys=True) + '\\n')\n"
    ) % (SHARD, TENSORS)


def _runtime_patch_block() -> str:
    """тот же приём, что и у канарейки адаптера: наш serving_setup кладётся в копию чужого бандла"""
    import base64, gzip
    blob = base64.b64encode(gzip.compress(open("harness/duck/serving_setup.py", "rb").read(), 9)).decode()
    return (
        "\n# [[MERGE]] Копия чужого бандла + наш serving_setup.py (он умеет оверлей модели).\n"
        "if not TRUE_SUBMISSION:\n"
        "    import base64 as _b64, gzip as _gz, hashlib as _hl2, shutil as _sh2\n"
        "    _patched = WORKING_DIR / 'bundle_patched'\n"
        "    if _patched.exists():\n"
        "        _sh2.rmtree(_patched)\n"
        "    _sh2.copytree(BUNDLE_DIR, _patched, symlinks=True)\n"
        "    _ss = _patched / 'serving_setup.py'\n"
        "    _old_sha = _hl2.sha256(_ss.read_bytes()).hexdigest()\n"
        "    _ss.write_bytes(_gz.decompress(_b64.b64decode('" + blob + "')))\n"
        "    _new_sha = _hl2.sha256(_ss.read_bytes()).hexdigest()\n"
        "    _si = _patched / 'SOURCE_IDENTITY.json'\n"
        "    _ident = json.loads(_si.read_text())\n"
        "    if _ident.get('serving_setup_sha256') != _old_sha:\n"
        "        raise RuntimeError('[[MERGE]] upstream serving_setup hash mismatch: %s' % _old_sha)\n"
        "    _ident['serving_setup_sha256'] = _new_sha\n"
        "    _si.write_text(json.dumps(_ident, indent=1))\n"
        "    BUNDLE_DIR = _patched\n"
        "    print('[[MERGE]] patched bundle at %s (serving_setup %s -> %s)' % (BUNDLE_DIR, _old_sha[:12], _new_sha[:12]), flush=True)\n"
    )


PROBE = '''

# [[MERGE]] Проверка (только вне боя): тот же запрос, что давал на стоке '1, 2, 3, 4,'.
if not TRUE_SUBMISSION and os.environ.get("ARC3_MODEL_OVERLAY"):
    import urllib.request as _u

    _base_url = os.environ.get("LOCAL_ANALYZER_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    _body = json.dumps({"model": "Qwen/Qwen3.8-Flash-Next-NVFP4",
                        "messages": [{"role": "user", "content": "Count from 1 to 12, separated by commas. Answer with the list only."}],
                        "temperature": 0.0, "max_tokens": 48, "seed": 0}).encode()
    _req = _u.Request(_base_url.rstrip("/") + "/chat/completions", data=_body,
                      headers={"Content-Type": "application/json"})
    with _u.urlopen(_req, timeout=180) as _resp:
        _out = json.loads(_resp.read())
    _txt = (_out["choices"][0]["message"].get("content") or "").strip()
    print("[[MERGE]] answer: %r" % _txt[:200], flush=True)
    print("[[MERGE]] VERDICT: %s" % ("ОВЕРЛЕЙ ЗАГРУЗИЛСЯ (ответ не как на стоке)" if _txt != "1, 2, 3, 4,"
                                     else "ОВЕРЛЕЙ МИМО (ответ как на стоке)"), flush=True)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="arc3-merge-canary")
    ap.add_argument("--cap", type=float, default=300.0)
    a = ap.parse_args()
    nb = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))

    c7 = "".join(nb["cells"][7]["source"])
    anchor = 'print(f"taaf.kaggle: source bundle = {BUNDLE_DIR}")'
    assert anchor in c7
    c7 = c7.replace(anchor, anchor + "\n" + _runtime_patch_block(), 1)
    c7 += _overlay_block()
    nb["cells"][7]["source"] = c7.splitlines(keepends=True)

    c13 = "".join(nb["cells"][13]["source"]) + PROBE
    nb["cells"][13]["source"] = c13.splitlines(keepends=True)

    c15 = "".join(nb["cells"][15]["source"])
    marker = "# Play the benchmark; watchdog stop and teardown run even if it raises."
    assert marker in c15
    c15 = c15.replace(marker, "if not TRUE_SUBMISSION:\n    bm.solver.max_runtime_s_per_game = %r    # [[MERGE]] проба вне боя\n\n" % a.cap + marker, 1)
    nb["cells"][15]["source"] = c15.splitlines(keepends=True)

    out = "kernels/notebooks_merge_canary"
    os.makedirs(out, exist_ok=True)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    meta["id"] = "sergueimakarov/" + a.slug
    meta["title"] = a.slug.replace("-", " ")
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    for i in (7, 13, 15):
        compile("".join(nb["cells"][i]["source"]), "c%d" % i, "exec", 0x2000)
    print("ok   собрано: %s (кернел %s)" % (out, meta["id"]))
    print("     подменяется %s, обнуляются %d тензора, потолок игры %.0f с" % (SHARD, len(TENSORS), a.cap))


if __name__ == "__main__":
    main()

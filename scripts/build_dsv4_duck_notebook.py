"""Обвязка Duck на DeepSeek-V4-Flash через llama.cpp вместо vLLM (18.09, шаг 2 после пробы arc3-dsv4-smoke).

Что меняется в стоковом ноутбуке (kernels/notebooks_stockflash):
  * ячейка 9 -- команды установки из бандла (они поднимают vLLM с Qwen3.8-Flash-Next) НЕ выполняются: карта занята
    другой моделью, второй сервер туда не влезет;
  * добавляется запуск `llama-server` из сборки под SM120 на кванте DeepSeek-V4-Flash (GGUF), с ожиданием /health;
  * `LOCAL_ANALYZER_BASE_URL` / `LOCAL_ANALYZER_MODEL_ID` перенаправляются на него (обвязка Duck говорит по
    совместимому протоколу, поэтому больше ничего менять не нужно);
  * ячейка 15 -- сторож vLLM не запускается (его сервера нет);
  * список игр и потолок урезаются под пробу (--games, --cap).

Пороги пробы записываются в план ДО пуска. Смысл шага: увидеть, играет ли более сильная модель вообще, и с какой
скоростью; сравнивать балл с базой на 3-5 играх бессмысленно, поэтому смотрим на число вызовов, длину ответов,
взятые уровни и ошибки протокола.

usage:  .venv/bin/python scripts/build_dsv4_duck_notebook.py [--games tu93,ft09,lp85] [--cap 1800]
"""
import argparse, json, os

START_SERVER = '''
# --- DeepSeek-V4-Flash через llama.cpp вместо vLLM (см. scripts/build_dsv4_duck_notebook.py) ---
import glob as _ds_glob, os as _ds_os, subprocess as _ds_sub, time as _ds_time, urllib.request as _ds_url

_DS_CTX, _DS_PARALLEL, _DS_MAX_GB = %(ctx)d, %(parallel)d, %(max_gb).1f
_DS_PICK, _DS_EXTRA = %(pick)r, %(extra)r
# журнал сервера: без --log-disable llama-server пишет тайминги каждого запроса (чтение промпта / генерация) в
# /kaggle/working/llama-server.log -- без них непонятно, на что уходит время вызова (19.09)
_DS_LOGFLAG = %(logflag)r
import re as _ds_re
_ds_gguf = []
for _root, _dirs, _files in _ds_os.walk("/kaggle/input"):
    for _f in _files:
        if _f.endswith(".gguf") and _DS_PICK in _f:
            _p = _ds_os.path.join(_root, _f)
            try:
                _ds_gguf.append((_ds_os.path.getsize(_p) / 1e9, _p))
            except OSError:
                pass
# Квант, разбитый на части (…-00001-of-00004.gguf), лежит в РАЗНЫХ датасетах, то есть в разных папках, а llama.cpp
# ищет остальные части рядом с первой. Поэтому части ссылками собираются в одну папку, и модель задаётся первой частью.
_ds_shards = sorted(p for s, p in _ds_gguf if _ds_re.search(r"-\\d{5}-of-\\d{5}\\.gguf$", p))
if _ds_shards:
    _ds_dir = "/kaggle/working/dsv4_shards"; _ds_os.makedirs(_ds_dir, exist_ok=True)
    for _p in _ds_shards:
        _l = _ds_os.path.join(_ds_dir, _ds_os.path.basename(_p))
        if not _ds_os.path.exists(_l):
            _ds_os.symlink(_p, _l)
    _ds_total = sum(s for s, p in _ds_gguf if p in _ds_shards)
    _ds_first = sorted(_ds_os.listdir(_ds_dir))[0]
    print("[[DSV4]] квант из %%d частей, всего %%.1f ГБ: %%s" %% (len(_ds_shards), _ds_total, sorted(_ds_os.listdir(_ds_dir))), flush=True)
    _ds_fit = [(_ds_total, _ds_os.path.join(_ds_dir, _ds_first))]
else:
    _ds_gguf.sort(reverse=True)
    _ds_fit = [(s, p) for s, p in _ds_gguf if s <= _DS_MAX_GB]
_ds_server = None
for _root, _dirs, _files in _ds_os.walk("/kaggle/input"):
    if "llama-server" in _files:
        _ds_server = _ds_os.path.join(_root, "llama-server")
print("[[DSV4]] квантов найдено %%d, подходящих %%d; сервер %%s" %% (len(_ds_gguf), len(_ds_fit), _ds_server), flush=True)
if not _ds_fit or _ds_server is None:
    raise RuntimeError("[[DSV4]] нет подходящего кванта или llama-server")
_ds_size, _ds_model = _ds_fit[0]
import shutil as _ds_sh
_ds_local = "/kaggle/working/llama-server"; _ds_sh.copy2(_ds_server, _ds_local); _ds_os.chmod(_ds_local, 0o755)
_ds_server = _ds_local
_ds_cmd = ("%%s -m %%s --host 127.0.0.1 --port 8080 -ngl 999 -c %%d --parallel %%d --flash-attn on %%s--jinja %%s"
           %% (_ds_server, _ds_model, _DS_CTX, _DS_PARALLEL, _DS_LOGFLAG, _DS_EXTRA)).strip()
print("[[DSV4]] запуск: %%s (квант %%.1f ГБ)" %% (_ds_cmd, _ds_size), flush=True)
_ds_log = open("/kaggle/working/llama-server.log", "w")
_ds_proc = _ds_sub.Popen(_ds_cmd, shell=True, stdout=_ds_log, stderr=_ds_sub.STDOUT)
_ds_t0 = _ds_time.time(); _ds_ok = False
while _ds_time.time() - _ds_t0 < 2400:
    _ds_time.sleep(10)
    if _ds_proc.poll() is not None:
        raise RuntimeError("[[DSV4]] сервер упал через %%.0f с" %% (_ds_time.time() - _ds_t0))
    try:
        with _ds_url.urlopen("http://127.0.0.1:8080/health", timeout=5) as _r:
            if _r.status == 200:
                _ds_ok = True; break
    except Exception:
        pass
print("[[DSV4]] сервер готов: %%s за %%.0f с" %% (_ds_ok, _ds_time.time() - _ds_t0), flush=True)
try:   # запас памяти -- главное число для выбора, сколько экспертов выносить в память хоста
    _ds_mem = _ds_sub.run("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader", shell=True,
                          capture_output=True, text=True, timeout=20).stdout.strip()
    _ds_ram = _ds_sub.run("free -g | head -2", shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    print("[[DSV4]] видеопамять после загрузки: %%s | память хоста:\\n%%s" %% (_ds_mem, _ds_ram), flush=True)
except Exception as _e:
    print("[[DSV4]] память не прочитана: %%r" %% (_e,), flush=True)
if not _ds_ok:
    raise RuntimeError("[[DSV4]] сервер не поднялся за 40 минут")
_ds_os.environ["LOCAL_ANALYZER_BASE_URL"] = "http://127.0.0.1:8080/v1"
_ds_os.environ["LOCAL_ANALYZER_MODEL_ID"] = "dsv4"
_ds_os.environ["INFERENCE_ANALYZER_MODEL"] = "dsv4"
_ds_os.environ["LOCAL_ANALYZER_API_KEY"] = "local"
_ds_os.environ["LOCAL_ANALYZER_PROVIDER"] = "vllm"
# ЛОВУШКА 19.09: llama.cpp делит общий контекст на места, обвязке надо сообщать контекст ОДНОГО места,
# иначе она строит промпт длиннее места и получает 400 (запрос 20573 токенов при доступных 20480)
_ds_os.environ["LOCAL_ANALYZER_CONTEXT_WINDOW"] = str(_DS_CTX // _DS_PARALLEL)
# ИЗМЕРЕНО 19.09 (версия 3): модель выдала по 64-69 тысяч токенов на игру и сделала 4-51 ход -- всё время ушло
# в рассуждение. Ограничиваем длину ответа и просим шаблон без размышления (как делали для Qwen).
_ds_os.environ["LOCAL_ANALYZER_MAX_OUTPUT"] = "%(maxout)d"
# ИСПРАВЛЕНО 19.09: раньше здесь стояло LLM_DISABLE_THINKING=1 -- обвязка Duck эту переменную НЕ ЧИТАЕТ
# (проверено grep по harness/duck), её выключатель -- LOCAL_ANALYZER_ENABLE_THINKING (по умолчанию включено).
# Поэтому v4-v7 и 3-битная v1 шли С рассуждением. Теперь режим задаётся явно ключом сборщика --thinking.
_ds_os.environ["LOCAL_ANALYZER_ENABLE_THINKING"] = "%(thinking)s"
print("[[DSV4]] анализатор перенаправлен на llama.cpp; потолок ответа %(maxout)d токенов; рассуждение %(thinking)s", flush=True)
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="tu93,ft09,lp85,sp80"); ap.add_argument("--cap", type=float, default=1800.0)
    ap.add_argument("--ctx", type=int, default=131072); ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--max-gb", type=float, default=88.0); ap.add_argument("--max-output", type=int, default=1500)
    # --quant q3: публичный UD-IQ3_XXS от Unsloth (4 части в трёх датасетах, 104.2 ГБ -- больше всей карты),
    # поэтому часть экспертов выносится в память хоста ключом --n-cpu-moe (слои экспертов ~2.3 ГБ каждый)
    ap.add_argument("--quant", choices=["q2", "q3"], default="q2")
    ap.add_argument("--server-log", choices=["on", "off"], default="off", help="on -- журнал сервера с таймингами запросов")
    ap.add_argument("--thinking", choices=["on", "off"], default="on",
                    help="рассуждение модели; on -- как фактически шли все пробы до 19.09 и как работает база Qwen")
    ap.add_argument("--n-cpu-moe", type=int, default=3)
    ap.add_argument("--slug", default="")
    a = ap.parse_args()
    src = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))
    nb = json.loads(json.dumps(src))

    # 1) не выполнять команды установки из бандла (они поднимают vLLM и занимают карту)
    c9 = "".join(nb["cells"][9]["source"])
    marker = 'for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):'
    assert marker in c9, "ячейка 9: не нашёл цикл команд установки"
    c9 = c9.replace(marker, 'for command in []:   # [[DSV4]] команды бандла отключены: vLLM не поднимаем')
    pick = "IQ2XXS" if a.quant == "q2" else "UD-IQ3_XXS"
    extra = "" if a.quant == "q2" else "--n-cpu-moe %d" % a.n_cpu_moe
    c9 += "\n" + (START_SERVER % {"ctx": a.ctx, "parallel": a.parallel, "max_gb": a.max_gb, "maxout": a.max_output,
                                  "pick": pick, "extra": extra, "thinking": "1" if a.thinking == "on" else "0",
                                  "logflag": "" if a.server_log == "on" else "--log-disable "})
    nb["cells"][9]["source"] = c9.splitlines(keepends=True)

    # 2) сторож vLLM не запускать
    c15 = "".join(nb["cells"][15]["source"])
    assert "vllm_watchdog.start_background(" in c15
    # сторож vLLM не просто не запускается -- его модуль даже не ЗАГРУЖАЕТСЯ: serving_setup.py требует переменных
    # окружения от команд бандла, которые мы отключили (падение 19.09: KeyError TAAF_KAGGLE_BUNDLE_DIR)
    c15 = c15.replace("vllm_watchdog_setup = vllm_watchdog.load_setup(BUNDLE_DIR / 'serving_setup.py')",
                      "vllm_watchdog_setup = None   # [[DSV4]] сторожа vLLM нет")
    c15 = c15.replace("vllm_watchdog.start_background(", "_DSV4_SKIP_WATCHDOG = (")
    c15 = c15.replace("vllm_watchdog.stop_background(timeout_seconds=15.0)", "pass   # [[DSV4]] сторожа нет")
    # 3) проба: свои игры и потолок
    games = ", ".join('"%s"' % g for g in a.games.split(",")) if a.games != "all" else ""
    marker2 = "# Play the benchmark; watchdog stop and teardown run even if it raises."
    assert marker2 in c15
    # --games all: играем все 25, отбор не ставим (тогда и стоковая проверка покрытия проходит сама)
    sel = ("" if a.games == "all" else
           "    _ds_want = [%s]\n"
           "    bm.games = [g for g in bm.games if str(getattr(g, 'env_name', getattr(g, 'game_id', '')))[:4] in _ds_want]\n" % games)
    patch = ("if not TRUE_SUBMISSION:\n"
             "    bm.solver.max_runtime_s_per_game = %r    # проба вне боя\n"
             "    bm.solver.concurrency = %d              # ровно столько мест у llama-server\n"
             "%s"
             "    print('[[DSV4]] игр в пробе: %%d' %% len(bm.games), flush=True)\n\n" % (a.cap, a.parallel, sel))
    c15 = c15.replace(marker2, patch + marker2, 1)
    cov = "if len(public_runs) != 25 or public_run_ids != list(PUBLIC_GAME_IDS):"
    if cov in c15:
        c15 = c15.replace(cov, "if TRUE_SUBMISSION and (len(public_runs) != 25 or public_run_ids != list(PUBLIC_GAME_IDS)):")
    nb["cells"][15]["source"] = c15.splitlines(keepends=True)

    out = "kernels/notebooks_dsv4_duck" if a.quant == "q2" else "kernels/notebooks_dsv4q3_duck"
    os.makedirs(out, exist_ok=True)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    slug = a.slug or ("arc3-dsv4-duck" if a.quant == "q2" else "arc3-dsv4q3-duck")
    meta["id"] = "sergueimakarov/" + slug; meta["title"] = slug.replace("-", " ")
    weights = (["johnannis/test-ok-3-q2-new"] if a.quant == "q2" else
               ["benitomallamaci/dsv4flash-q3-part1", "benitomallamaci/dsv4flash-q3-part2", "benitomallamaci/dsv4flash-q3-part3"])
    meta["dataset_sources"] = [d for d in meta["dataset_sources"] if "runtime" not in d] + [
        "bachhg/dsv4-llamacpp-runtime-sm120-ba360efe-r1"] + weights
    meta["model_sources"] = []
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    for i in (9, 15):
        compile("".join(nb["cells"][i]["source"]), "c%d" % i, "exec", 0x2000 if i == 15 else 0)
    print("ok   собрано: %s (кернел %s); игры %s, потолок %.0f с, контекст %d, параллельно %d"
          % (out, meta["id"], a.games, a.cap, a.ctx, a.parallel))
    print("     входы:", meta["dataset_sources"], "| модель vLLM отцеплена:", all("qwen" not in d for d in meta["dataset_sources"]))


if __name__ == "__main__":
    main()

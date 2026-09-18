"""Пробный ноутбук: поднимается ли DeepSeek-V4-Flash в сжатом виде на нашей карте и с какой скоростью (18.09,
слово владельца «попытка поднять более сильную модель в ядре — выполняй»).

Зачем именно так. Правило проекта: минутная проба пригодности до многочасового прогона. Этот ноутбук НЕ играет в игры
и не трогает обвязку Duck. Он делает три вещи: показывает, что реально лежит во входах (какие файлы квантов и их
размеры), пробует поднять llama-server на самом крупном кванте, который влезает в 96 ГБ, и меряет скорость выдачи
на двух запросах. Этого достаточно, чтобы решить, стоит ли вкладываться в интеграцию.

Что монтируется (всё уже есть на Kaggle, интернет в ядре не нужен):
  * bachhg/dsv4-llamacpp-runtime-sm120-ba360efe-r1 -- сборка llama.cpp ПОД SM120 (наша RTX Pro 6000 Blackwell);
    именно на SM120 у vLLM известен сбой ядер DeepGEMM для моделей класса V4, поэтому берём llama.cpp;
  * johnannis/test-ok-3-q2-new -- один файл DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix, 86.7 ГБ:
    самый качественный из квантов, влезающих в 96 ГБ (кванты 90.9 ГБ и полная модель 155 ГиБ не влезают с KV-кэшем).
Порог выбора кванта: файл <= MAX_GB, иначе не влезет в 96 ГБ вместе с KV-кэшем.

Что печатает (это и есть результат пробы):
  [[DSV4]] список файлов с размерами; выбранный квант; время загрузки; токенов/с на генерации; ответ модели.
Пороги (записаны до пуска): сервер поднялся и ответил осмысленным текстом; скорость >= 30 токенов/с суммарно (v3: 63.5 на одном запросе, 102.8 на двух -- порог пройден); в v4
проверяются пустой content (модель пишет в reasoning_content), реальная занятость видеопамяти и скорость на 8 запросах -- иначе для 25 игр за 132 минуты модель бесполезна (нужно ~250 токенов/с, но 30 -- порог,
ниже которого дальше идти точно нет смысла).

usage:  .venv/bin/python scripts/build_dsv4_smoke_notebook.py
"""
import json, os

CELL = r'''
import glob, json, os, subprocess, sys, time, urllib.request

MAX_GB = 88.0          # больше не влезет в 96 ГБ вместе с KV-кэшем
CTX = 81920      # 4 места по 20к токенов: столько занимает промпт нашей обвязки
PARALLEL = 4

def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw).stdout

print("[[DSV4]] GPU:", sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").strip(), flush=True)
print("[[DSV4]] диск:", sh("df -h /kaggle/working | tail -1").strip(), flush=True)
print("[[DSV4]] входы:", sorted(os.listdir("/kaggle/input")), flush=True)

# --- 1. что реально лежит во входах ---
gguf = []
for root, _dirs, files in os.walk("/kaggle/input"):
    for f in files:
        p = os.path.join(root, f)
        try:
            sz = os.path.getsize(p) / 1e9
        except OSError:
            continue
        if f.endswith(".gguf"):
            gguf.append((sz, p))
gguf.sort(reverse=True)
print("[[DSV4]] найдено файлов .gguf: %d" % len(gguf), flush=True)
for sz, p in gguf[:25]:
    print("   %7.1f ГБ  %s" % (sz, p), flush=True)

server_bin = None
for root, _dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "llama-server":
            server_bin = os.path.join(root, f)
print("[[DSV4]] llama-server:", server_bin, flush=True)

fit = [(sz, p) for sz, p in gguf if sz <= MAX_GB]
if not fit or server_bin is None:
    print("[[DSV4]] ИТОГ: нечего запускать (подходящий квант или сервер не найдены)", flush=True)
else:
    sz, model = fit[0]          # самый крупный из влезающих
    print("[[DSV4]] выбран квант: %.1f ГБ %s" % (sz, model), flush=True)
    # ЛОВУШКА 19.09: /kaggle/input только для ЧТЕНИЯ -- права на запуск там не поставить, копируем в рабочую папку
    import shutil
    local_bin = "/kaggle/working/llama-server"
    shutil.copy2(server_bin, local_bin)
    os.chmod(local_bin, 0o755)
    runtime_root = os.path.dirname(os.path.dirname(server_bin))
    libs = [d for d in glob.glob(runtime_root + "/lib*") if os.path.isdir(d)]
    if libs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(libs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    print("[[DSV4]] сервер скопирован в %s; библиотеки: %s" % (local_bin, libs), flush=True)
    server_bin = local_bin
    cmd = ("%s -m %s --host 127.0.0.1 --port 8080 -ngl 999 -c %d --parallel %d "
           "--flash-attn on --log-disable" % (server_bin, model, CTX, PARALLEL))
    print("[[DSV4]] запуск:", cmd, flush=True)
    t0 = time.time()
    log = open("/kaggle/working/llama-server.log", "w")
    proc = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
    ok = False
    while time.time() - t0 < 1800:
        time.sleep(10)
        if proc.poll() is not None:
            print("[[DSV4]] сервер упал через %.0f с, код %s" % (time.time() - t0, proc.returncode), flush=True)
            break
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=5) as r:
                if r.status == 200:
                    ok = True; break
        except Exception:
            pass
        if int(time.time() - t0) % 120 < 10:
            print("[[DSV4]] жду сервер %.0f с; %s" % (time.time() - t0, sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader").strip()), flush=True)
    load_s = time.time() - t0
    print("[[DSV4]] сервер поднялся: %s, загрузка %.0f с" % (ok, load_s), flush=True)
    if ok:
        body = json.dumps({"model": "dsv4", "messages": [
            {"role": "user", "content": "Reply with exactly one short sentence: what information would you need to "
                                        "infer the goal of an unknown grid puzzle game?"}],
            "max_tokens": 300, "temperature": 0.6, "chat_template_kwargs": {"enable_thinking": False}}).encode()
        t1 = time.time(); out = None
        try:
            req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.loads(r.read())
        except Exception as exc:
            print("[[DSV4]] запрос не прошёл: %r" % (exc,), flush=True)
        dt = time.time() - t1
        print("[[DSV4]] видеопамять после загрузки: %s" % sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader").strip(), flush=True)
        if out:
            print("[[DSV4]] ответ целиком (первые 1200 знаков): %s" % json.dumps(out, ensure_ascii=False)[:1200], flush=True)
            _m = out["choices"][0].get("message", {})
            txt = _m.get("content") or _m.get("reasoning_content") or ""
            n = out.get("usage", {}).get("completion_tokens", 0)
            print("[[DSV4]] ответ за %.1f с, токенов %s, скорость %.1f ток/с" % (dt, n, n / max(dt, 1e-9)), flush=True)
            print("[[DSV4]] текст ответа: %s" % txt[:400].replace("\n", " "), flush=True)
        # ДЛИННЫЙ ПРОМПТ: главный расход в бою -- чтение 20к токенов на каждый вызов
        _board = "\n".join("".join("ABCDEFGHIJKLMNOP"[(i * 7 + j * 3) % 16] for j in range(64)) for i in range(64))
        _prompt = ("You are playing an unknown grid game. Board history follows.\n" + (_board + "\n\n") * 4 +
                   "Reply with one short sentence: which single action would you try next and why?")
        _b = json.dumps({"model": "dsv4", "messages": [{"role": "user", "content": _prompt}],
                         "max_tokens": 120, "temperature": 0.6,
                         "chat_template_kwargs": {"enable_thinking": False}}).encode()
        _t3 = time.time(); _outl = None
        try:
            _rq = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=_b,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(_rq, timeout=1200) as r:
                _outl = json.loads(r.read())
        except Exception as exc:
            _body = ""
            try:
                _body = exc.read().decode()[:300]
            except Exception:
                pass
            print("[[DSV4]] длинный промпт не прошёл: %r %s" % (exc, _body), flush=True)
        _dt3 = time.time() - _t3
        if _outl:
            _u = _outl.get("usage", {})
            print("[[DSV4]] ДЛИННЫЙ ПРОМПТ: %d токенов прочитано, %d сгенерировано, за %.1f с -> чтение %.0f ток/с"
                  % (_u.get("prompt_tokens", 0), _u.get("completion_tokens", 0), _dt3,
                     _u.get("prompt_tokens", 0) / max(_dt3, 1e-9)), flush=True)
            print("[[DSV4]] ответ на длинный: %s" % (_outl["choices"][0]["message"].get("content") or "")[:200], flush=True)
        # параллельная нагрузка: PARALLEL запросов разом
        import threading
        res = []
        def one(i):
            b = json.dumps({"model": "dsv4", "messages": [{"role": "user", "content": "Count from 1 to 40, comma separated."}],
                            "max_tokens": 200, "temperature": 0.6}).encode()
            try:
                rq = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=b,
                                            headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(rq, timeout=900) as r:
                    res.append(json.loads(r.read()).get("usage", {}).get("completion_tokens", 0))
            except Exception as exc:
                res.append(0)
        t2 = time.time()
        th = [threading.Thread(target=one, args=(i,)) for i in range(PARALLEL)]
        [x.start() for x in th]; [x.join() for x in th]
        dt2 = time.time() - t2
        print("[[DSV4]] %d параллельных запросов: токенов %d за %.1f с = %.1f ток/с суммарно"
              % (PARALLEL, sum(res), dt2, sum(res) / max(dt2, 1e-9)), flush=True)
    print("[[DSV4]] хвост лога сервера:", flush=True)
    print(sh("tail -40 /kaggle/working/llama-server.log"), flush=True)
    try:
        proc.terminate()
    except Exception:
        pass
print("[[DSV4]] проба завершена", flush=True)
'''


def main() -> None:
    out = "kernels/notebooks_dsv4_smoke"
    os.makedirs(out, exist_ok=True)
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "source": CELL.splitlines(keepends=True),
                     "execution_count": None, "outputs": []}],
          "metadata": {"kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = {"id": "sergueimakarov/arc3-dsv4-smoke", "title": "arc3 dsv4 smoke", "code_file": "submission.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "machine_shape": "NvidiaRtxPro6000", "enable_tpu": False, "enable_internet": False, "keywords": [],
            "dataset_sources": ["bachhg/dsv4-llamacpp-runtime-sm120-ba360efe-r1", "johnannis/test-ok-3-q2-new"],
            # ЛОВУШКА 19.09: без подключённого соревнования Kaggle выдаёт T4 x2 (16 ГБ), даже если machine_shape
            # просит NvidiaRtxPro6000. Мощная карта доступна только ядрам соревнования.
            "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-3"],
            "model_sources": []}
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    compile(CELL, "dsv4_smoke", "exec")
    print("ok   собрана проба: %s (кернел %s)" % (out, meta["id"]))
    print("     входы: рантайм llama.cpp SM120 + GGUF-кванты DeepSeek-V4-Flash; порог кванта 86 ГБ")


if __name__ == "__main__":
    main()

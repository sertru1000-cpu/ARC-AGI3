"""Сборка пункта 6 субботы: пакетный синтез программ модели мира на Kaggle, без игры.

ВОПРОС (ради ответа, не ради балла): программы state_of/predict не принимались из-за плохой идеи или
из-за того, что в игре у синтезатора было 8 попыток и буфер в 4 перехода? Здесь — 20 попыток на игру
по готовым сводкам наблюдений `docs/wm_digests.json` (21 игра с буфером >= 10 переходов, 420 запросов).

ЧТО ВНУТРИ. Ячейки 0-9 базового ноутбука без изменений — они поднимают vLLM на 127.0.0.1:1234 с профилем
базы (та же обвязка, что у пробы промпта 11.09, отработала 100 запросов без сбоев). Вместо игры —
ячейка синтеза: системный промпт, инструмент `python` и разбор ответа по четырём каналам взяты ДОСЛОВНО
из `scripts/wm_synth_generate.py`; температура 0.9 (нужен разброс программ), мышление включено как у
агента, потолок 8192 токена на ответ; 16 запросов разом. Результат — /kaggle/working/programs.json
в формате генератора, затем teardown сервера бандла.

ОЦЕНКА ЛОКАЛЬНО: `.venv/bin/python scripts/wm_eval_programs.py programs.json`.
ПОРОГ (записан 09.09): принимается по горизонту меньше трети программ ИЛИ путь по принятым находится
реже чем в половине случаев — ветка закрыта окончательно.

Пушить ВЕРСИЕЙ в существующий `arc3-stock-flash-exploit` (источники как у базы).

usage:  .venv/bin/python scripts/build_synth_batch_notebook.py
"""
import ast
import base64
import json
import lzma
import os

GEN = open("scripts/wm_synth_generate.py", encoding="utf-8").read()


def segment(name):
    tree = ast.parse(GEN)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == name for t in node.targets):
            return ast.get_source_segment(GEN, node)
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(GEN, node)
    raise SystemExit("в генераторе нет %s" % name)


SYNTH = r'''
# =====================================================================
# ПАКЕТНЫЙ СИНТЕЗ ПРОГРАММ МОДЕЛИ МИРА: 20 попыток на игру по готовым сводкам, без игры.
# Промпт, инструмент и разбор ответа — дословно из scripts/wm_synth_generate.py.
# =====================================================================
import base64 as _b64, json as _json, lzma as _lz, re, time as _t, threading as _th
import json  # разбор ответа из генератора зовёт json.loads; без этого программы из вызова инструмента
             # терялись молча (исключение внутри try) — поймано сухим тестом 11.09
import urllib.request as _ur
from concurrent.futures import ThreadPoolExecutor as _TPE

_BASE = "http://127.0.0.1:1234/v1"
_MODEL = "Qwen/Qwen3.8-Flash-Next-NVFP4"
_DIG = _json.loads(_lz.decompress(_b64.b64decode(DIGESTS_B64)).decode("utf-8"))
_ATTEMPTS = 20
_MIN_PAIRS = 10
_CONC = 16
_SAMPLING = {"temperature": 0.9, "top_p": 0.95, "top_k": 20, "max_tokens": 8192,
             "chat_template_kwargs": {"enable_thinking": True}}
_OUT = WORKING_DIR / "programs.json"

__GEN_PARTS__

def _get(url, timeout=10):
    with _ur.urlopen(url, timeout=timeout) as r:
        return r.read()

_t0 = _t.time()
while True:
    try:
        _get(_BASE + "/models"); break
    except Exception as _e:
        if _t.time() - _t0 > 1800:
            raise RuntimeError("сервер не поднялся за 30 минут: %r" % (_e,))
        _t.sleep(10)
_games = [g for g in _DIG if g["pairs"] >= _MIN_PAIRS]
_jobs = [(g, i) for g in _games for i in range(_ATTEMPTS)]
print("SYNTH: сервер готов через %.0f с; игр %d, попыток на игру %d, запросов %d, разом %d"
      % (_t.time() - _t0, len(_games), _ATTEMPTS, len(_jobs), _CONC), flush=True)

_res, _lock = [], _th.Lock()

def _one(job):
    g, i = job
    rec = {"game": g["game"], "attempt": i, "level": g["level"], "pairs": g["pairs"]}
    t = _t.time()
    body = {"model": _MODEL, "messages": [{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": "Наблюдения:\n" + g["digest"]}],
            "tools": TOOLS, "tool_choice": "auto", "stream": False, **_SAMPLING}
    try:
        req = _ur.Request(_BASE + "/chat/completions", data=_json.dumps(body).encode("utf-8"),
                          headers={"Content-Type": "application/json"})
        with _ur.urlopen(req, timeout=1800) as r:
            resp = _json.loads(r.read())
        ch = (resp.get("choices") or [{}])[0]
        code, channel = pick_code(ch.get("message") or {})
        rec.update({"code": code, "channel": channel, "finish": ch.get("finish_reason"),
                    "usage": (resp.get("usage") or {}).get("completion_tokens", 0)})
    except Exception as e:
        rec.update({"code": "", "channel": "сбой", "error": repr(e)[:200]})
    rec["seconds"] = round(_t.time() - t, 1)
    with _lock:
        _res.append(rec)
        if len(_res) % 10 == 0 or len(_res) == len(_jobs):
            _OUT.write_text(_json.dumps(sorted(_res, key=lambda r: (r["game"], r["attempt"])), ensure_ascii=False))
            got = sum(1 for x in _res if "def predict" in (x.get("code") or ""))
            print("SYNTH [%d/%d] программ %d, прошло %.0f с" % (len(_res), len(_jobs), got, _t.time() - _t0), flush=True)
    return rec

with _TPE(max_workers=_CONC) as _pool:
    list(_pool.map(_one, _jobs))
_OUT.write_text(_json.dumps(sorted(_res, key=lambda r: (r["game"], r["attempt"])), ensure_ascii=False))
_chan = {}
for _r in _res:
    _chan[_r.get("channel", "?")] = _chan.get(_r.get("channel", "?"), 0) + 1
print("SYNTH ИТОГ: программ %d из %d попыток; каналы %s; обрезано по длине %d"
      % (sum(1 for x in _res if "def predict" in (x.get("code") or "")), len(_res), _chan,
         sum(1 for x in _res if x.get("finish") == "length")), flush=True)

for _cmd in _json.loads((BUNDLE_DIR / "teardown_commands.json").read_text()):
    print("SYNTH teardown: %s" % _cmd, flush=True)
    subprocess.run(_cmd, shell=True, check=False, cwd=WORKING_DIR, env=_command_env())
'''


def main():
    parts = "\n\n".join([segment("SYSTEM"), segment("TOOLS"), segment("pick_code")])
    cell_body = SYNTH.replace("__GEN_PARTS__", parts)
    dig = lzma.compress(open("docs/wm_digests.json", "rb").read())
    cell_src = "DIGESTS_B64 = %r\n" % base64.b64encode(dig).decode("ascii") + cell_body
    src = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))
    nb = json.loads(json.dumps(src))
    nb["cells"] = nb["cells"][:10] + [{"cell_type": "code", "execution_count": None, "metadata": {},
                                       "outputs": [], "source": cell_src.splitlines(keepends=True)}]
    out = "kernels/notebooks_synth_batch"
    os.makedirs(out, exist_ok=True)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash_exploit/kernel-metadata.json"))
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    compile(cell_src, "synth", "exec")
    kept = ["".join(c["source"]) for c in nb["cells"][:10]]
    base = ["".join(c["source"]) for c in src["cells"][:10]]
    size = os.path.getsize(os.path.join(out, "submission.ipynb"))
    print("ok   ячейки 0-9 совпадают с базой побайтно:", kept == base)
    print("ok   игры нет (нет bm.run):", "bm.run" not in cell_src)
    print("ok   промпт, инструмент и разбор ответа — дословно из генератора:", all(p in cell_src for p in (segment("SYSTEM"), segment("TOOLS"), segment("pick_code"))))
    print("ok   teardown в конце:", "teardown_commands.json" in cell_src)
    import re as _re
    names_needed = {m for m in ("json", "re") if _re.search(r"\b%s\." % m, segment("pick_code"))}
    print("ok   модули, нужные разбору ответа, импортированы под своими именами:",
          all(_re.search(r"^import %s\b|, %s,|, %s\b" % (n, n, n), cell_src, _re.M) for n in names_needed), sorted(names_needed))
    print("ok   слаг для пуша:", meta["id"])
    print("ok   размер ноутбука %.0f КБ, меньше предела 1 МБ: %s" % (size / 1024, size < 1_000_000))


if __name__ == "__main__":
    main()

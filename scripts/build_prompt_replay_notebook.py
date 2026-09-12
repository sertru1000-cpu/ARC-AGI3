"""Сборка `arc3-prompt-replay`: повтор 25 записанных запросов, вариант A против Б, без игры.

ЧТО ВНУТРИ. Ячейки 0-9 базового ноутбука `arc3-stock-flash` без изменений (окружение, колёса,
бандл, setup-команды бандла — они и поднимают vLLM на 127.0.0.1:1234 с профилем базы). Ячейки
10-17 (загрузка бенчмарка и сама игра) НЕ включены. Вместо них — ячейка повтора: ждёт здоровья
сервера, шлёт каждый запрос дважды в варианте A и дважды в варианте Б (100 запросов, 4 разом),
пишет ответы в /kaggle/working/replay_results.json и гасит сервер командой teardown бандла.

ЗАЧЕМ ПОВТОРЫ. Семплирование при температуре 0.6 шумное: два прогона одного и того же запроса
расходятся сами по себе. Пара A1/A2 даёт естественный разброс длины и выбранного действия,
на её фоне и читается разница A против Б.

usage:  .venv/bin/python scripts/build_prompt_replay_notebook.py
"""
import ast
import base64
import json
import os

REPLAY = r'''
# =====================================================================
# ПОВТОР ЗАПИСАННЫХ ЗАПРОСОВ: вариант A (как было) против Б (с разбором перехода).
# Игры нет. Сервер поднят setup-командами бандла в ячейке 9.
# =====================================================================
import base64 as _b64, gzip as _gz, json as _json, time as _t, threading as _th, re as _re
import urllib.request as _ur, urllib.error as _ue
from concurrent.futures import ThreadPoolExecutor as _TPE

import lzma as _lz
_BASE = "http://127.0.0.1:1234/v1"
_PAYLOAD = _json.loads(_lz.decompress(_b64.b64decode(PAYLOAD_B64)).decode("utf-8"))

# Картинки рисует СОБСТВЕННАЯ функция бандла — ровно так, как их рисует харнесс в бою.
# upscale задан явно: без бенчмарка переменная MULTIMODAL_UPSCALE не выставлена,
# а по умолчанию функция берёт 16, тогда как базовый прогон шёл с 4.
from inference.agent.runtime_state import Frame as _Frame
from inference.agent.vision_context import frame_to_png_data_url as _png

def _user_msg(m, extra=None):
    grid = tuple(tuple(int(ch, 16) for ch in row) for row in m["board"].split("/"))
    frame = _Frame(grid=grid, step=int(m["step"]), level=int(m["level"]))
    text = m["text"] if extra is None else m["text"] + "\n\n" + extra
    return {"role": "user", "content": [
        {"type": "text", "text": text + "\n\nCurrent grid image:"},
        {"type": "image_url", "image_url": {"url": _png(frame, upscale=int(_PAYLOAD["upscale"]))}}]}

for _it in _PAYLOAD["items"]:
    _msgs = _it["messages"]
    _it["messages_A"] = [_user_msg(m) if m["role"] == "user" else m for m in _msgs]
    # Б совпадает с A во всём, кроме текста последнего сообщения пользователя
    _it["messages_B"] = _it["messages_A"][:-1] + [_user_msg(_msgs[-1], extra=_it["diff"])]
print("REPLAY: сообщения восстановлены, картинок %d"
      % sum(1 for it in _PAYLOAD["items"] for m in it["messages_A"] if m["role"] == "user"), flush=True)
_OUT = WORKING_DIR / "replay_results.json"
_REPS = 2
_CONC = 4

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
print("REPLAY: сервер готов через %.0f с; снимков %d, повторов %d, одновременно %d"
      % (_t.time() - _t0, len(_PAYLOAD["items"]), _REPS, _CONC), flush=True)

def _post(item, variant, rep):
    body = {"model": _PAYLOAD["model"], "messages": item["messages_" + variant], "tools": item["tools"],
            "tool_choice": "auto", "stream": False, **_PAYLOAD["sampling"]}
    data = _json.dumps(body).encode("utf-8")
    req = _ur.Request(_BASE + "/chat/completions", data=data, headers={"Content-Type": "application/json"})
    t = _t.time()
    try:
        with _ur.urlopen(req, timeout=1200) as r:
            resp = _json.loads(r.read())
        ch = resp["choices"][0]; msg = ch.get("message") or {}
        calls = msg.get("tool_calls") or []
        code = ""
        for c in calls:
            try: code = _json.loads(c["function"]["arguments"]).get("code", ""); break
            except Exception: pass
        rec = {"game": item["game"], "variant": variant, "rep": rep, "ok": True,
               "seconds": round(_t.time() - t, 1), "finish": ch.get("finish_reason"),
               "usage": resp.get("usage") or {},
               "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or "",
               "content": msg.get("content") or "", "code": code, "n_calls": len(calls)}
    except Exception as e:
        rec = {"game": item["game"], "variant": variant, "rep": rep, "ok": False,
               "seconds": round(_t.time() - t, 1), "error": repr(e)[:300]}
    print("REPLAY %s %s%d %s %s ток за %.0f с" % (item["game"], variant, rep, "ок" if rec["ok"] else "СБОЙ",
          (rec.get("usage") or {}).get("completion_tokens", "-"), rec["seconds"]), flush=True)
    return rec

_jobs = [(it, v, r) for r in range(_REPS) for it in _PAYLOAD["items"] for v in ("A", "B")]
_res = []
with _TPE(max_workers=_CONC) as _pool:
    for _rec in _pool.map(lambda j: _post(*j), _jobs):
        _res.append(_rec)
        if len(_res) % 10 == 0:
            _OUT.write_text(_json.dumps(_res, ensure_ascii=False))
_OUT.write_text(_json.dumps(_res, ensure_ascii=False))
_ok = [r for r in _res if r["ok"]]
for _v in ("A", "B"):
    _ct = [r["usage"].get("completion_tokens", 0) for r in _ok if r["variant"] == _v]
    print("REPLAY ИТОГ %s: ответов %d, средняя генерация %.0f токенов" % (_v, len(_ct), sum(_ct) / max(1, len(_ct))), flush=True)
print("REPLAY: сбоев %d из %d, записано %s" % (len(_res) - len(_ok), len(_res), _OUT), flush=True)

for _cmd in _json.loads((BUNDLE_DIR / "teardown_commands.json").read_text()):
    print("REPLAY teardown: %s" % _cmd, flush=True)
    subprocess.run(_cmd, shell=True, check=False, cwd=WORKING_DIR, env=_command_env())
'''


def main():
    import sys
    use_dataset = "--dataset" in sys.argv
    src = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))
    if use_dataset:
        # запасной путь: нагрузка приватным датасетом, ноутбук остаётся маленьким
        cell_src = ("import glob as _glob\n"
                    "_pf = _glob.glob('/kaggle/input/**/payload.json.xz', recursive=True)\n"
                    "assert _pf, 'нет payload.json.gz во входных данных'\n"
                    "PAYLOAD_B64 = __import__('base64').b64encode(open(_pf[0], 'rb').read()).decode('ascii')\n") + REPLAY
    else:
        payload = open("kernels/notebooks_prompt_replay/payload.json.xz", "rb").read()
        b64 = base64.b64encode(payload).decode("ascii")
        cell_src = "PAYLOAD_B64 = %r\n" % b64 + REPLAY
    nb = json.loads(json.dumps(src))
    nb["cells"] = nb["cells"][:10] + [{
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": cell_src.splitlines(keepends=True)}]
    out = "kernels/notebooks_prompt_replay"
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    meta["id"] = "sergueimakarov/arc3-prompt-replay"
    if use_dataset:
        meta["dataset_sources"] = list(meta.get("dataset_sources", [])) + ["sergueimakarov/arc3-prompt-replay-payload"]
    meta["title"] = "arc3 prompt replay"
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    compile(REPLAY, "replay", "exec")
    kept = ["".join(c["source"]) for c in nb["cells"][:10]]
    base = ["".join(c["source"]) for c in src["cells"][:10]]
    print("ok   ячейки 0-9 совпадают с базой побайтно:", kept == base)
    print("ok   игры нет (нет bm.run):", "bm.run" not in "".join("".join(c["source"]) for c in nb["cells"]))
    print("ok   сервер поднимают setup-команды ячейки 9:", "setup_commands.json" in kept[9])
    print("ok   teardown в конце:", "teardown_commands.json" in REPLAY)
    size = os.path.getsize(os.path.join(out, "submission.ipynb"))
    print("ok   компилируется; размер ноутбука %.0f КБ" % (size / 1024))
    print("ok   меньше предела Kaggle 1 МБ:", size < 1_000_000)


if __name__ == "__main__":
    main()

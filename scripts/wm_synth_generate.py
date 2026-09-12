"""Пакетный синтез программ модели мира: много попыток на записанных наблюдениях.

ГДЕ РАБОТАЕТ. На поде рядом с сервером модели (или локально против любого совместимого
адреса). Игрового движка не требует вовсе: берёт готовые сводки наблюдений
(`scripts/wm_make_digests.py`) и просит модель написать программу столько раз, сколько велено.

ЗАЧЕМ ТАК. В живом прогоне синтезатор получает считанные попытки: v13 за полчаса дал восемь,
из них четыре умерли на транспорте. Чтобы проверить правило приёма, нужны десятки программ,
а игра для этого только помеха. Здесь на одну игру приходится столько попыток, сколько задано.

ЧТО ПРОСИМ. Промпт согласован с тем, как мы теперь принимаем: программа должна вести состояние
МНОГО ШАГОВ подряд, а не только воспроизводить отдельные переходы. В версиях до 13 мы просили
одно (точное воспроизведение), а судили по другому — это исправлено здесь.

Оценка (горизонт, старое правило, поиск пути) делается ОТДЕЛЬНО и локально:
`scripts/wm_eval_programs.py`. Так на поде тратится только генерация.

ПОЧЕМУ В НЕСКОЛЬКО ПОТОКОВ. Запросов пятьсот (25 игр x 20 попыток), каждый — целая программа
с рассуждением, то есть десятки секунд. Подряд это ночь работы пода; сервер же держит десятки
запросов разом, и упирается он в карту, а не в нас. Поэтому попытки раздаются пулу потоков,
а порядок в файле восстанавливается по номерам игры и попытки.

usage:
    python3 scripts/wm_synth_generate.py --digests wm_digests.json --out programs.json \\
        --attempts 20 --workers 16 --base-url http://127.0.0.1:8000/v1 \\
        --model Qwen/Qwen3.8-Flash-Next-NVFP4
"""

from __future__ import annotations

import argparse
import concurrent.futures as _cf
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request

SYSTEM = (
    "Ты — агент-синтезатор модели мира. Ты НЕ играешь: твоя работа — написать на Python "
    "программу, которая ВЕДЁТ состояние игры много шагов подряд.\n"
    "ОТВЕЧАЙ ВЫЗОВОМ ИНСТРУМЕНТА `python`, положив программу в аргумент `code`.\n"
    "В программе ровно две функции:\n"
    "  def state_of(frame): ...   # структурное состояние кадра; ОБЯЗАТЕЛЬНО включи wm_level(frame)\n"
    "  def predict(state, action): ...   # следующее состояние или None, если случай не покрыт\n"
    "Доступны помощники: wm_level(frame), wm_objects(frame) — объекты с полем 't' (подпись цвета и\n"
    "формы без привязки к позиции), 'x', 'y', 'w', 'h', 'pixels'; frame.ascii — сетка символов.\n"
    "КАК ТЕБЯ ОЦЕНИВАЮТ: программу раскатывают от разных состояний и смотрят, сколько шагов подряд "
    "она предсказывает верно. Принимается та, что держится восемь шагов и дольше. Поэтому пиши "
    "правило на ТИП объекта, а не на координаты, и не подгоняй ответ под отдельные записанные "
    "переходы — подгонка ломается на втором шаге.\n"
    "Не больше 60 строк, без объяснений и примеров."
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "python",
        "description": "Верни программу модели мира: две функции state_of(frame) и predict(state, action).",
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string", "description": "Программа из двух функций."}},
                       "required": ["code"]},
    },
}]


def pick_code(msg: dict) -> tuple[str, str]:
    """Четыре канала по убыванию надёжности — та же цепочка, что в боевой сборке v11+."""
    def from_calls(calls):
        for c in calls or []:
            fn = (c or {}).get("function", {}) or {}
            args = fn.get("arguments", "")
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except Exception:
                    args = {}
            if isinstance(args, dict) and args.get("code"):
                return str(args["code"])
        return ""

    code = from_calls(msg.get("tool_calls"))
    if code:
        return code, "вызов инструмента"
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    for text, name in ((content, "content"), (reasoning, "канал рассуждений")):
        if not text:
            continue
        blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
        if blocks:
            return blocks[-1].strip(), name
        if "def predict" in text and "def state_of" in text:
            return text.strip(), name + " (без ограды)"
    return "", "кода нет"


def ask(base_url: str, model: str, digest: str, timeout: float, temperature: float) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": "Наблюдения:\n" + digest}],
        "tools": TOOLS,
        "temperature": temperature,
        "max_tokens": 4096,
    }).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--digests", default="wm_digests.json")
    ap.add_argument("--out", default="programs.json")
    ap.add_argument("--attempts", type=int, default=20, help="попыток на игру")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--temperature", type=float, default=0.9,
                    help="выше стоковых 0.6: нужен разброс программ, а не одна и та же")
    ap.add_argument("--min-pairs", type=int, default=10, help="игры с меньшим буфером пропускаем")
    ap.add_argument("--workers", type=int, default=16,
                    help="сколько попыток идёт к серверу одновременно")
    a = ap.parse_args()

    games = [g for g in json.load(open(a.digests, encoding="utf-8")) if g["pairs"] >= a.min_pairs]
    jobs = [(g, i) for g in games for i in range(a.attempts)]
    print(f"игр: {len(games)}, попыток на игру: {a.attempts}, всего запросов: {len(jobs)}, "
          f"одновременно: {a.workers}", flush=True)

    out, lock, t0 = [], threading.Lock(), time.time()

    def one(job):
        g, i = job
        rec = {"game": g["game"], "attempt": i, "level": g["level"], "pairs": g["pairs"]}
        try:
            resp = ask(a.base_url, a.model, g["digest"], a.timeout, a.temperature)
            msg = (resp.get("choices") or [{}])[0].get("message", {}) or {}
            code, channel = pick_code(msg)
            rec.update({"code": code, "channel": channel,
                        "usage": (resp.get("usage") or {}).get("completion_tokens", 0)})
        except Exception as exc:
            rec.update({"code": "", "channel": "сбой", "error": repr(exc)[:200]})
        with lock:
            out.append(rec)
            n = len(out)
            # сохраняемся по ходу: обрыв пода не должен стоить всей работы
            if n % 10 == 0 or n == len(jobs):
                ordered = sorted(out, key=lambda r: (r["game"], r["attempt"]))
                json.dump(ordered, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
                got = sum(1 for r in out if "def predict" in (r.get("code") or ""))
                print(f"[{n}/{len(jobs)}] программ {got}, прошло {time.time() - t0:.0f} с", flush=True)
        return rec

    with _cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(one, jobs))

    ordered = sorted(out, key=lambda r: (r["game"], r["attempt"]))
    json.dump(ordered, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    total = sum(1 for r in out if "def predict" in (r.get("code") or ""))
    by_ch = {}
    for r in out:
        by_ch[r.get("channel", "?")] = by_ch.get(r.get("channel", "?"), 0) + 1
    print(f"\nготово: программ {total} из {len(out)} попыток, {time.time() - t0:.0f} с -> {a.out}")
    print("каналы ответа:", by_ch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

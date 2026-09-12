"""Проигрывание ЗАПИСАННЫХ боевых ходов через облачный эндпойнт (Alibaba Model Studio).

ЗАЧЕМ. Слово владельца: «а почему бы нам к нашей модели не подключиться через API и не
тестировать промпты в реальном времени? API оплачу». Стенд для промптов, который работает
с мака за секунды и не тратит квоту Kaggle.

ЧТО ШЛЁТСЯ. Настоящие промпты боевого прогона из `kernels/notebooks_prompt_replay/payload.json.xz`:
25 ходов, в каждом по 32 сообщения (система, история пользователь/ассистент/инструмент и
текущий ход), инструмент `python` с его описанием и картинка доски — она рисуется здесь же
тем же рисовальщиком и тем же увеличением 4, что и в бою.

ЧЕГО ЭТОТ СТЕНД НЕ ДАЁТ, и это надо помнить при каждом выводе. В облаке живёт `qwen3.8-flash` —
управляемая версия, сделанная НА ОСНОВЕ наших весов Qwen3.8-Flash-Next, но не они сами:
другое квантование, другое окно контекста (миллион против наших 32 тысяч) и свои умолчания
сэмплирования. Выводы отсюда переносимы качественно («перестала ли пересказывать доску»,
«воспользовалась ли выданным правилом»), числа — нет. Всё, что должно перенестись в бой,
проверяется бесплатным точным стендом внутри Kaggle.

КЛЮЧ берётся из окружения (`DASHSCOPE_API_KEY`) и никогда не печатается.

usage:
    export DASHSCOPE_API_KEY=...            # ключ из консоли Model Studio, в репозиторий не кладём
    .venv/bin/python scripts/dashscope_replay.py --limit 3
    .venv/bin/python scripts/dashscope_replay.py --limit 10 --variant-file docs/variant_rule.txt
"""
import argparse
import json
import lzma
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_replay_payload import png  # noqa: E402  — тот же рисовальщик, что и в бою

DEFAULT_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.8-flash"
PRICE_IN, PRICE_OUT, CNY_USD = 0.8, 2.7, 0.128   # юаней за млн токенов, пекинский прайс


def unpack_board(packed):
    return [[int(ch, 16) for ch in row] for row in packed.split("/")]


def message_for(m, extra=None):
    """Сообщение пользователя = текст (плюс необязательная вставка) и картинка доски."""
    if m["role"] != "user" or "board" not in m:
        return {k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "tool_call_id", "name")}
    text = m["text"] if extra is None else m["text"] + "\n\n" + extra
    return {"role": "user", "content": [
        {"type": "text", "text": text + "\n\nCurrent grid image:"},
        {"type": "image_url", "image_url": {"url": png(unpack_board(m["board"]))}}]}


def call(url, key, model, item, extra, sampling, max_tokens, timeout):
    msgs = [message_for(m) for m in item["messages"][:-1]]
    msgs.append(message_for(item["messages"][-1], extra))
    body = {"model": model, "messages": msgs, "tools": item["tools"],
            "tool_choice": "auto", "stream": False, "max_tokens": max_tokens, **sampling}
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read()[:400].decode("utf-8", "replace")
        return {"game": item["game"], "ok": False, "error": "HTTP %s: %s" % (e.code, detail),
                "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"game": item["game"], "ok": False, "error": repr(e)[:300], "seconds": round(time.time() - t0, 1)}
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    calls = msg.get("tool_calls") or []
    code = ""
    for c in calls:
        try:
            code = json.loads(c["function"]["arguments"]).get("code", "")
            break
        except Exception:  # noqa: BLE001
            pass
    usage = resp.get("usage") or {}
    # ВАЖНО: «вернула код» и «сделала ход» — разные вещи. Первая проба это смешивала, и выдача
    # показывала «ход» у ответов, которые лишь печатали доску. Ход = вызов action() внутри кода.
    acts = bool(re.search(r"(?<![A-Za-z_])action\s*\(", code))
    return {"game": item["game"], "ok": True, "seconds": round(time.time() - t0, 1),
            "finish": ch.get("finish_reason"), "in": usage.get("prompt_tokens"),
            "out": usage.get("completion_tokens"), "returned_code": bool(calls), "acts": acts,
            "code": code,
            "code_head": code.strip().splitlines()[0][:90] if code.strip() else "",
            "text_head": (msg.get("content") or "").strip()[:90]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", default="kernels/notebooks_prompt_replay/payload.json.xz")
    ap.add_argument("--base-url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=3, help="сколько записанных ходов послать")
    ap.add_argument("--repeats", type=int, default=1, help="повторов каждого хода (сэмплирование шумное)")
    ap.add_argument("--variant-file", help="файл с текстом, который дописывается к последнему сообщению")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--send-top-k", action="store_true",
                    help="слать top_k и chat_template_kwargs (наши боевые значения); по умолчанию НЕТ — "
                         "облачный эндпойнт может их не принять")
    ap.add_argument("--out", default="docs/dashscope_replay_results.json")
    a = ap.parse_args()

    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise SystemExit("нет ключа: export DASHSCOPE_API_KEY=... (в репозиторий и в переписку он не попадает)")

    d = json.loads(lzma.open(a.payload).read().decode("utf-8"))
    items = d["items"][:a.limit]
    extra = Path(a.variant_file).read_text(encoding="utf-8") if a.variant_file else None
    sampling = {"temperature": a.temperature, "top_p": a.top_p}
    if a.send_top_k:
        sampling.update({"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}})

    print("шлём %d ходов x %d повторов на %s, модель %s%s"
          % (len(items), a.repeats, a.base_url, a.model, ", со вставкой" if extra else ""))
    jobs = [(it,) for _ in range(a.repeats) for it in items]
    res = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for r in pool.map(lambda j: call(a.base_url, key, a.model, j[0], extra, sampling, a.max_tokens, a.timeout), jobs):
            res.append(r)
            if r["ok"]:
                print("  %-6s %5s вх / %5s вых | %-9s | %.0f с | %s"
                      % (r["game"], r["in"], r["out"],
                         "ДЕЙСТВИЕ" if r["acts"] else ("осмотр" if r["returned_code"] else "без кода"),
                         r["seconds"], r["code_head"] or r["text_head"]))
            else:
                print("  %-6s СБОЙ: %s" % (r["game"], r["error"]))

    ok = [r for r in res if r["ok"]]
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    if not ok:
        raise SystemExit("ни одного успешного ответа — смотрите текст ошибки выше")
    tin = sum(r["in"] or 0 for r in ok)
    tout = sum(r["out"] or 0 for r in ok)
    cost = (tin * PRICE_IN + tout * PRICE_OUT) / 1e6
    print()
    print("ответов %d из %d; генерация в среднем %.0f токенов" % (len(ok), len(res), tout / len(ok)))
    print("кодом ответили %.0f%%, из них ДЕЙСТВИЕМ (вызов action) — %.0f%% от всех ответов"
          % (100 * sum(r["returned_code"] for r in ok) / len(ok),
             100 * sum(r["acts"] for r in ok) / len(ok)))
    print("токенов: %d входных, %d выходных" % (tin, tout))
    print("стоимость: %.3f юаня (~$%.3f); результаты в %s" % (cost, cost * CNY_USD, a.out))


if __name__ == "__main__":
    main()

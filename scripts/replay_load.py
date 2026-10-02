"""Проигрывание записанной нагрузки обвязки на сервер модели — стенд серверов без игр (для пода).

Берёт записанные запросы прогона (runs/<run>/*_p0_requests.jsonl: полные промпты с картинками, tools, kwargs шаблона)
и шлёт их на OpenAI-совместимый сервер. Каждая игра — отдельный последовательный поток (следующий запрос после
ответа на предыдущий), одновременно идут `--streams` игр. Длина ответа фиксирована записанной
(max_tokens = completion_tokens, ignore_eos), поэтому нагрузка одинакова для любого сервера и настройки.
Итог: время, скорость генерации, задержки, доля промпта из кэша — в stdout и JSON.

usage: .venv/bin/python scripts/replay_load.py --run runs/kaggle_graft_p10_v1 --url http://127.0.0.1:8001/v1 \
           --model flashnext --streams 10 [--games 10] [--max-requests-per-game 0] [--out result.json] [--dry]
"""
import argparse, glob, json, statistics as st, sys, threading, time, urllib.request
from pathlib import Path


def load_streams(run, games, max_per_game):
    streams = []
    for f in sorted(glob.glob(str(Path(run) / "*_p0_requests.jsonl")))[:games or None]:
        reqs = []
        for line in open(f):
            r = json.loads(line)
            if not r.get("messages") or not isinstance(r.get("usage"), dict):
                continue
            ct = int(r["usage"].get("completion_tokens") or 0)
            if ct <= 0:
                continue
            reqs.append({"messages": r["messages"], "tools": r.get("tools"), "kw": r.get("chat_template_kwargs") or {},
                         "tool_choice": r.get("tool_choice") or "auto", "max_tokens": ct,
                         "rec_prompt": int(r["usage"].get("prompt_tokens") or 0)})
        if max_per_game:
            reqs = reqs[:max_per_game]
        if reqs:
            streams.append((Path(f).name[:4], reqs))
    return streams


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/kaggle_graft_p10_v1")
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="flashnext")
    ap.add_argument("--streams", type=int, default=10)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--max-requests-per-game", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--dry", action="store_true", help="только посчитать нагрузку, без запросов")
    a = ap.parse_args()

    streams = load_streams(a.run, a.games, a.max_requests_per_game)
    n_req = sum(len(r) for _, r in streams)
    gen = sum(q["max_tokens"] for _, r in streams for q in r)
    print(f"игр {len(streams)}, запросов {n_req}, токенов ответа {gen:,}, промптов (записано) "
          f"{sum(q['rec_prompt'] for _, r in streams for q in r):,}")
    if a.dry:
        return

    results, lock = [], threading.Lock()
    queue = list(streams)

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                game, reqs = queue.pop(0)
            for i, q in enumerate(reqs):
                body = {"model": a.model, "messages": q["messages"], "max_tokens": q["max_tokens"],
                        "temperature": 0.7, "top_p": 0.95, "top_k": 20, "ignore_eos": True,
                        "chat_template_kwargs": q["kw"], "stream": False}
                if q["tools"]:
                    body["tools"], body["tool_choice"] = q["tools"], q["tool_choice"]
                t0 = time.time()
                try:
                    out = post(a.url + "/chat/completions", body)
                    u = out.get("usage") or {}
                    rec = {"game": game, "i": i, "s": time.time() - t0, "prompt": u.get("prompt_tokens", 0),
                           "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
                           "completion": u.get("completion_tokens", 0), "err": None}
                except Exception as e:  # noqa: BLE001
                    rec = {"game": game, "i": i, "s": time.time() - t0, "err": str(e)[:200]}
                with lock:
                    results.append(rec)

    t_start = time.time()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.streams)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t_start

    ok = [r for r in results if not r["err"]]
    errs = [r for r in results if r["err"]]
    comp = sum(r["completion"] for r in ok)
    prompt = sum(r["prompt"] for r in ok)
    cached = sum(r["cached"] for r in ok)
    summary = {"streams": a.streams, "requests": len(results), "errors": len(errs), "wall_s": round(wall, 1),
               "gen_tok_per_s": round(comp / wall, 1), "completion_tokens": comp,
               "prompt_cached_pct": round(100 * cached / max(1, prompt), 2),
               "latency_median_s": round(st.median(r["s"] for r in ok), 2) if ok else None,
               "latency_p90_s": round(sorted(r["s"] for r in ok)[int(0.9 * (len(ok) - 1))], 2) if ok else None,
               "low_cache_requests": sum(1 for r in ok if r["prompt"] and r["cached"] < 0.5 * r["prompt"])}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if errs:
        print("первая ошибка:", errs[0]["err"])
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "requests": results}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())

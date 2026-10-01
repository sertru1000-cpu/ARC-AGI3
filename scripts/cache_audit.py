"""Аудит кэша и истории по прогону на сервере SGLang (копия решения Франзена).

Считает по serve.log (ReqTimeStats, Prefill/Decode batch, server_args) и по *_p0_requests.jsonl:
долю токенов промпта из кэша, длину промпта, очередь, одновременность, скорость генерации, обрезки истории.
usage: .venv/bin/python scripts/cache_audit.py runs/kaggle_graft_dfranzen_v1
"""
import glob, json, re, statistics as st, sys
from pathlib import Path

run = Path(sys.argv[1])
log = (run / "serve.log").read_text(errors="replace").splitlines()

# --- фактические аргументы сервера
want = ["kv_cache_dtype", "context_length", "max_running_requests", "mem_fraction_static", "chunked_prefill_size",
        "max_prefill_tokens", "speculative_algorithm", "speculative_num_steps", "quantization", "mamba_ssm_dtype",
        "max_mamba_cache_size", "mamba_radix_cache_strategy", "schedule_policy", "page_size"]
args_line = next((l for l in log if "server_args=" in l), "")
print("== аргументы запущенного сервера ==")
for k in want:
    m = re.search(rf"'{k}': ([^,}}]+)", args_line)
    print(f"  {k:28s} {m.group(1) if m else 'нет в логе'}")

# --- запросы по логу сервера
rq = []
for l in log:
    m = re.search(r"ReqTimeStats\(rid=\w+, input_len=(\d+), cached_input_len=(\d+), output_len=(\d+).*?queue_duration=([\d.]+)ms, "
                  r"initial_prefill_elapsed=([\d.]+)ms, post_prefill_elapsed=([\d.]+)ms, forward_duration=([\d.]+)ms", l)
    if m:
        rq.append(tuple(float(x) for x in m.groups()))
rq = [r for r in rq if r[0] > 500]          # без прогревочных запросов
inp = sum(r[0] for r in rq); cached = sum(r[1] for r in rq); out = sum(r[2] for r in rq)
print(f"\n== запросы (лог сервера), n={len(rq)} ==")
print(f"  токенов промпта {inp:,.0f}, из кэша {cached:,.0f} = {100*cached/inp:.2f}%")
print(f"  ответ: всего {out:,.0f}, медиана {st.median(r[2] for r in rq):.0f}")
print(f"  длина промпта: медиана {st.median(r[0] for r in rq):,.0f}, макс {max(r[0] for r in rq):,.0f}")
miss = [r for r in rq if r[1] < 0.5 * r[0]]
print(f"  запросов с кэшем < 50% промпта: {len(miss)} из {len(rq)}")
print(f"  очередь: медиана {st.median(r[3] for r in rq):.1f} мс, макс {max(r[3] for r in rq):.0f} мс")
print(f"  префилл: медиана {st.median(r[4] for r in rq):.0f} мс, макс {max(r[4] for r in rq):.0f} мс; "
      f"доля времени запроса {100*sum(r[4] for r in rq)/sum(r[6] for r in rq):.1f}%")
new = inp - cached
print(f"  новых токенов промпта на запрос: {new/len(rq):,.0f}; на один токен ответа {new/out:.2f}")

# --- одновременность и скорость
dec = []
for l in log:
    m = re.search(r"Decode batch, #running-req: (\d+), #full token: (\d+), full token usage: ([\d.]+).*?accept len: ([\d.]+).*?"
                  r"gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)", l)
    if m:
        dec.append(tuple(float(x) for x in m.groups()))
if dec:
    print(f"\n== декодирование (n={len(dec)} строк) ==")
    print(f"  одновременно запросов: медиана {st.median(d[0] for d in dec):.0f}, макс {max(d[0] for d in dec):.0f}")
    print(f"  в очереди сервера: макс {max(d[5] for d in dec):.0f}")
    print(f"  заполнение кэша KV: макс {100*max(d[2] for d in dec):.0f}% ({max(d[1] for d in dec):,.0f} токенов)")
    print(f"  генерация: медиана {st.median(d[4] for d in dec):.0f} т/с, макс {max(d[4] for d in dec):.0f} т/с")
    print(f"  черновик: принято за шаг {st.median(d[3] for d in dec):.2f} токена (медиана)")

# --- по играм: рост промпта и обрезки
print("\n== по играм (логи обвязки) ==")
print(f"  {'игра':6s} {'запр.':>5s} {'промпт макс':>11s} {'из кэша %':>9s} {'ответ':>7s} {'рассужд. %':>10s} {'обрезок':>7s}")
for f in sorted(glob.glob(str(run / "*_p0_requests.jsonl"))):
    rows = [json.loads(l) for l in open(f)]
    us = [r["usage"] for r in rows if isinstance(r.get("usage"), dict) and r["usage"].get("prompt_tokens")]
    if not us:
        continue
    p = [u["prompt_tokens"] for u in us]
    c = [(u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) for u in us]
    o = sum(u.get("completion_tokens", 0) for u in us); rs = sum(u.get("reasoning_tokens", 0) for u in us)
    trims = sum(1 for a, b in zip(p, p[1:]) if b < a - 2000)
    print(f"  {Path(f).name[:4]:6s} {len(us):5d} {max(p):11,d} {100*sum(c)/sum(p):9.1f} {o:7,d} {100*rs/max(o,1):10.0f} {trims:7d}")

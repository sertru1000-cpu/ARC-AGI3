"""Профиль декодирования сервера на поде (ответ критика 04.10, пункт «сначала диагноз»).

Пока идёт записанная нагрузка, включает встроенный профилировщик SGLang (/start_profile, CPU+GPU, N шагов),
затем разбирает трассу: доля времени карты по группам ядер (MoE, внимание QSA/индексатор, линейное внимание GDN/Mamba,
нормы/эмбеддинги, копирования H2D/D2H, прочее) и простои карты между ядрами.
usage: python3 profile_trace.py --port 8101 --out /workspace/run/prof [--steps 60] [--delay 120]
"""
import argparse, glob, gzip, json, os, re, time, urllib.request
from collections import defaultdict

GROUPS = [
    ("moe / gemm int4", r"marlin|moe|expert|gptq|fused_moe|topk_gating|w4a16|awq"),
    ("qsa / indexer / sparse attn", r"qsa|sparse|indexer|topk(?!_gating)|compact_kv|fa2|flash_attn|fmha|xqa|trtllm"),
    ("gdn / mamba / linear attn", r"gdn|mamba|chunk_|fused_recurrent|delta|conv1d|causal_conv|ssm|short_conv"),
    ("gemm (dense)", r"gemm|cutlass|cublas|sm90|sm100|sm120|matmul|nvjet"),
    ("norm / act / elementwise", r"norm|rms|silu|gelu|act_and_mul|elementwise|vectorized|reduce|softmax|rotary|rope"),
    ("sampling / spec", r"sampl|argmax|accept|verify|eagle|nextn|draft"),
    ("memcpy H2D/D2H", r"memcpy|Memcpy|copy_kernel|CatArrayBatched"),
]


def post(port, path, obj):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read().decode()[:300]


def analyse(path, steps):
    op = gzip.open if path.endswith(".gz") else open
    ev = json.load(op(path, "rt")).get("traceEvents", [])
    k = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not k:
        return {"error": "нет ядер в трассе"}
    durs = defaultdict(list); top = defaultdict(list); h2d_bytes = 0
    for e in k:
        n = e.get("name", ""); d = float(e.get("dur", 0))
        top[n[:90]].append(d)
        if e.get("cat") == "gpu_memcpy":
            h2d_bytes += int((e.get("args") or {}).get("bytes", 0) or 0)
        for g, rx in GROUPS:
            if re.search(rx, n, re.I):
                durs[g].append(d); break
        else:
            durs["прочее"].append(d)
    busy = sum(sum(v) for v in durs.values())
    k.sort(key=lambda e: e["ts"])
    span = k[-1]["ts"] + k[-1].get("dur", 0) - k[0]["ts"]
    gaps, end = [], k[0]["ts"]
    for e in k:
        if e["ts"] > end:
            gaps.append(e["ts"] - end)
        end = max(end, e["ts"] + e.get("dur", 0))
    idle = sum(gaps)
    def row(v):
        v = sorted(v)
        return {"calls_per_step": round(len(v) / steps, 1), "gpu_ms_per_step": round(sum(v) / steps / 1000, 3),
                "pct_gpu": round(100 * sum(v) / busy, 1), "avg_us": round(sum(v) / len(v), 1),
                "p95_us": round(v[int(0.95 * (len(v) - 1))], 1)}
    big_gaps = sorted(gaps)[-5:]
    return {"steps_assumed": steps, "span_ms": round(span / 1000, 1), "span_ms_per_step": round(span / steps / 1000, 2),
            "gpu_idle_pct": round(100 * idle / span, 1), "gaps_gt_100us": sum(g > 100 for g in gaps),
            "largest_gaps_us": [round(g) for g in big_gaps], "memcpy_bytes_total": h2d_bytes,
            "groups": {g: row(v) for g, v in sorted(durs.items(), key=lambda x: -sum(x[1]))},
            "top_kernels": [(n, row(v)) for n, v in sorted(top.items(), key=lambda x: -sum(x[1]))[:20]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--out", default="/workspace/run/prof")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--delay", type=int, default=120, help="сколько секунд нагрузки до включения профиля")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    time.sleep(a.delay)
    print("start_profile:", post(a.port, "/start_profile", {"output_dir": a.out, "num_steps": a.steps,
                                                           "activities": ["CPU", "GPU"]}), flush=True)
    for _ in range(120):
        files = glob.glob(os.path.join(a.out, "**", "*.json*"), recursive=True)
        if files:
            time.sleep(20); break
        time.sleep(5)
    files = sorted(glob.glob(os.path.join(a.out, "**", "*.json*"), recursive=True), key=os.path.getsize)
    if not files:
        print("ТРАССЫ НЕТ"); return
    res = analyse(files[-1], a.steps)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

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


def analyse(path):
    op = gzip.open if path.endswith(".gz") else open
    ev = json.load(op(path, "rt")).get("traceEvents", [])
    k = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not k:
        return {"error": "нет ядер в трассе"}
    by = defaultdict(float); top = defaultdict(float)
    for e in k:
        n = e.get("name", ""); d = float(e.get("dur", 0))
        top[n[:90]] += d
        for g, rx in GROUPS:
            if re.search(rx, n, re.I):
                by[g] += d; break
        else:
            by["прочее"] += d
    busy = sum(by.values())
    k.sort(key=lambda e: e["ts"])
    span = k[-1]["ts"] + k[-1].get("dur", 0) - k[0]["ts"]
    return {"gpu_busy_pct": round(100 * busy / span, 1), "span_ms": round(span / 1000, 1),
            "groups_pct": {g: round(100 * v / busy, 1) for g, v in sorted(by.items(), key=lambda x: -x[1])},
            "top_kernels_pct": [(n, round(100 * v / busy, 1)) for n, v in sorted(top.items(), key=lambda x: -x[1])[:15]]}


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
    res = analyse(files[-1])
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

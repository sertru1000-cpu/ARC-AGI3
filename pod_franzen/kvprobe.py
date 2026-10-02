"""Проба качества кэша KV на работающем сервере (под): NLL записанных ответов модели при длинной и короткой истории.
Выигрыш от памяти = NLL_short - NLL_long. Сравнивать между запусками сервера с разным POD_KVDTYPE.
usage: python3 kvprobe.py --tag fp8 [--probe /workspace/probe.json] [--port 8001] [--out /workspace/run/kvprobe_fp8.json]
"""
import argparse, json, time, urllib.request
from pathlib import Path


def generate(port, ids, start):
    payload = {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
               "return_logprob": True, "logprob_start_len": start}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    return [x[0] for x in out["meta_info"]["input_token_logprobs"] if x[0] is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--probe", default="/workspace/probe.json")
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    items = json.load(open(a.probe))
    rows = []
    for it in items:
        rec = {"game": it["game"], "pair": it["pair"], "long_prompt": it["long_prompt"], "resp_len": it["resp_len"]}
        for kind in ("long", "short"):
            ids = it[kind + "_ids"]
            t0 = time.time()
            lp = generate(a.port, ids, len(ids) - it["resp_len"])
            rec[kind + "_nll"] = -sum(lp) / max(1, len(lp))
            rec[kind + "_n"] = len(lp)
            rec[kind + "_s"] = round(time.time() - t0, 1)
        rec["gain"] = rec["short_nll"] - rec["long_nll"]
        rows.append(rec)
        print(f"[{a.tag}] {rec['game']} {rec['long_prompt']:6d}: long {rec['long_nll']:.4f} short {rec['short_nll']:.4f} "
              f"gain {rec['gain']:+.4f} (n={rec['long_n']}/{it['resp_len']}, {rec['long_s']}s)", flush=True)
    n = len(rows)
    ml = sum(r["long_nll"] for r in rows) / n
    ms = sum(r["short_nll"] for r in rows) / n
    print(f"==== {a.tag}: NLL long {ml:.4f} | short {ms:.4f} | выигрыш от памяти {ms - ml:+.4f}")
    out = a.out or f"/workspace/run/kvprobe_{a.tag}.json"
    Path(out).write_text(json.dumps({"tag": a.tag, "rows": rows, "long": ml, "short": ms}, indent=1))


if __name__ == "__main__":
    main()

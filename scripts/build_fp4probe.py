"""Сборка короткого кернела kernels/graft_fp4probe: поднимается ли сервер Франзена на кэше nvfp4 с патчем
nextfork/fp4qsa, и чего это стоит в качестве и скорости против fp8 на той же машине.

Основа — kernels/graft_kvprobe (сервер fp8 из ячейки 12 + проба NLL на записанных длинных контекстах).
Ячейка пробы заменяется:
  1) fp8 (бой): NLL long/short + жадное продолжение 64 токенов по каждому контексту + скорость на 10 параллельных;
  2) стоп сервера, патч fp4qsa в установленный sglang, сервер nvfp4 (NEXTFORK_FP4_QSA=1, NEXTFORK_FP4_NOWS=1,
     кэш черновика fp8) — то же самое; плюс совпадение жадных продолжений с fp8 и размер пула из лога.
usage: .venv/bin/python scripts/build_fp4probe.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels/graft_kvprobe"
DST = ROOT / "kernels/graft_fp4probe"
NF = (ROOT / "nextfork/fp4qsa/nf_fp4.py").read_text()
APPLY = (ROOT / "nextfork/fp4qsa/apply_fp4qsa.py").read_text()

TAIL = r'''
RESULTS = {}
GEN = {}
import glob as _glob
SGL = None  # путь к установленному sglang ищется после установки (версия python образа Kaggle может отличаться)

def _post(path, payload, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{SERVED_MODEL_PORT}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def _generate(ids, start):
    out = _post("/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                              "return_logprob": True, "logprob_start_len": start})
    return [x[0] for x in out["meta_info"]["input_token_logprobs"] if x[0] is not None]

def score_all(tag):
    rows = []
    for it in items:
        rec = {"game": it["game"], "pair": it["pair"], "long_prompt": it["long_prompt"], "resp_len": it["resp_len"]}
        resp = min(it["resp_len"], 384)   # логиты по словарю 248k на длинном ответе не влезают в память
        for kind in ("long", "short"):
            full = it[kind + "_ids"]; ids = full[: len(full) - it["resp_len"] + resp]
            t0 = time.time(); lp = _generate(ids, len(ids) - resp)
            rec[kind + "_nll"] = -sum(lp) / max(1, len(lp)); rec[kind + "_n"] = len(lp); rec[kind + "_s"] = round(time.time() - t0, 1)
        rec["gain"] = rec["short_nll"] - rec["long_nll"]
        rows.append(rec)
        print(f"[{tag}] {rec['game']} {rec['long_prompt']:6d}: long {rec['long_nll']:.4f} short {rec['short_nll']:.4f} gain {rec['gain']:+.4f}", flush=True)
    RESULTS[tag] = {"rows": rows}
    n = len(rows); ml = sum(r['long_nll'] for r in rows) / n; ms = sum(r['short_nll'] for r in rows) / n
    RESULTS[tag].update(long=ml, short=ms)
    print(f"==== {tag}: NLL long {ml:.4f} | short {ms:.4f} | выигрыш от памяти {ms-ml:+.4f}", flush=True)

def greedy_and_speed(tag):
    """жадные 64 токена по каждому длинному контексту (сверка fp4 с fp8) и скорость на 10 параллельных по 512 токенов"""
    from concurrent.futures import ThreadPoolExecutor
    outs = []
    for it in items:
        ids = it["long_ids"][: len(it["long_ids"]) - it["resp_len"]]
        o = _post("/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 64, "temperature": 0.0}})
        outs.append(o.get("output_ids") or o["meta_info"].get("output_ids") or o.get("text"))
    GEN[tag] = outs
    def one(it):
        ids = it["long_ids"][: len(it["long_ids"]) - it["resp_len"]]
        o = _post("/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 512, "temperature": 0.7, "ignore_eos": True}})
        return o["meta_info"]["completion_tokens"]
    batch = (items * 4)[:10]
    t0 = time.time()
    with ThreadPoolExecutor(10) as ex:
        toks = sum(ex.map(one, batch))
    RESULTS[tag]["speed10_tok_s"] = round(toks / (time.time() - t0), 1)
    print(f"==== {tag}: 10 параллельных x 512 токенов: {RESULTS[tag]['speed10_tok_s']} т/с", flush=True)

def pool_tokens():
    import re
    txt = Path(LOG).read_text(errors="ignore")
    m = re.findall(r"KV Cache is allocated\. dtype: ([\w.]+), #tokens: (\d+)", txt)
    return m[:1]

def save():
    Path(WORKING_DIR, 'fp4probe_results.json').write_text(json.dumps({"results": RESULTS, "gen": GEN}, indent=1, default=str))

def stop_server():
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception as e:
        print('kill:', e)
    for _ in range(120):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", int(SERVED_MODEL_PORT))) != 0:
                break
        time.sleep(2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass
    time.sleep(15)

def start_server(kvdtype):
    global NOTEBOOK_START_TIME, SERVER_STARTUP_TIMEOUT
    NOTEBOOK_START_TIME = time.time(); SERVER_STARTUP_TIMEOUT = 30 * 60
    src = (C12_SRC.replace('KVDTYPE="fp8_e4m3"', 'KVDTYPE=%r' % kvdtype)
           .replace('"--speculative-draft-kv-cache-dtype", CFG["KVDTYPE"]', '"--speculative-draft-kv-cache-dtype", "fp8_e4m3"')
           .replace('precache_model_thread.start()', 'pass'))
    exec(compile(src, 'serve_' + kvdtype, 'exec'), globals())
    for _ in range(900):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        if proc.poll() is not None:
            print('СЕРВЕР УПАЛ', kvdtype); show_log_tail(); return False
        time.sleep(3)
    return False

# --- 1) fp8: сервер уже поднят ячейкой 12 ---
for _ in range(600):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", timeout=5) as r:
            if r.status == 200: break
    except Exception: time.sleep(3)
score_all('fp8'); RESULTS['fp8']['pool'] = pool_tokens(); greedy_and_speed('fp8'); save()

# --- 2) nvfp4 с патчем QSA ---
stop_server()
Path('/tmp/fp4qsa').mkdir(exist_ok=True)
Path('/tmp/fp4qsa/nf_fp4.py').write_text(NF_FP4_SRC)
Path('/tmp/fp4qsa/apply_fp4qsa.py').write_text(APPLY_SRC)
SGL = sorted(_glob.glob('/tmp/sgl-intel/venv/lib/python3*/site-packages/sglang'))[0]
_r = subprocess.run([sys.executable, '/tmp/fp4qsa/apply_fp4qsa.py', SGL], capture_output=True, text=True)
print(_r.stdout, _r.stderr[-3000:]); assert _r.returncode == 0, 'патч fp4qsa не применился'
os.environ.update(NEXTFORK_FP4_QSA="1", NEXTFORK_FP4_NOWS="1")
if start_server('nvfp4'):
    score_all('nvfp4'); RESULTS['nvfp4']['pool'] = pool_tokens(); greedy_and_speed('nvfp4')
    agree = []
    for a, b in zip(GEN.get('fp8', []), GEN.get('nvfp4', [])):
        if isinstance(a, list) and isinstance(b, list):
            k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            agree.append(k)
    RESULTS['nvfp4']['greedy_same_prefix'] = agree
    print('жадные продолжения: совпадающий префикс с fp8 (из 64):', agree, flush=True)
else:
    RESULTS['nvfp4'] = 'server failed'
save()
print('ГОТОВО', json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'rows'} if isinstance(v, dict) else v
                            for k, v in RESULTS.items()}, ensure_ascii=False, default=str))
import pandas as pd
pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(WORKING_DIR / "submission.parquet", index=False)
'''


def main():
    nb = json.loads((SRC / "submission.ipynb").read_text())
    cell = nb["cells"][14]
    s = "".join(cell["source"])
    i = s.find('C12_SRC = """')
    j = s.find('"""', i + 14) + 3
    assert i > 0 and j > i
    head = s[:j]
    for need in ('KVDTYPE="fp8_e4m3"', '"--speculative-draft-kv-cache-dtype", CFG["KVDTYPE"]', 'precache_model_thread.start()'):
        assert need in head, f"нет в C12_SRC: {need}"
    head = head.replace("# ==== Проба качества кэша KV (02.10): fp8 (бой) против nvfp4 и bf16",
                        "# ==== Проба fp4 (03.10): сервер Франзена на nvfp4 с патчем QSA против fp8 (бой)")
    emb = f"\nNF_FP4_SRC = {NF!r}\nAPPLY_SRC = {APPLY!r}\nimport subprocess, sys\n"
    cell["source"] = (head + emb + TAIL).splitlines(keepends=True)
    nb["cells"][13]["source"] = ["## Проба fp4: nvfp4 + патч QSA против fp8\n"]
    DST.mkdir(exist_ok=True)
    (DST / "submission.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    meta.update(id="sergueimakarov/arc3-graft-fp4probe", title="arc3 graft fp4probe")
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    compile("".join(cell["source"]), "cell14", "exec")
    print("собрано:", DST, "ячейка пробы", len("".join(cell["source"])), "симв.")


if __name__ == "__main__":
    main()

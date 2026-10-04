
# ==== Проба «дообучение во время игры», вариант A (04.10) ====
# 1) сервер (черновик выключен, патч выгрузки включён) считает логарифмы вероятностей записанных окон игр и
#    сохраняет скрытые состояния перед lm_head; 2) сервер останавливается; 3) на карте для каждой игры обучается
#    низкоранговая поправка h' = h + B(A h) на наблюдениях первой половины игры и проверяется на наблюдениях второй.
#    Контроль: поправка, обученная на другой игре. Решает разница «своя игра» против «чужая игра».
import json, os, signal, socket, time, urllib.request, glob
from pathlib import Path
import torch

GAMES = json.loads(next(Path(c) for c in (PROBE_DIR, '/kaggle/input/arc3-ttt-probe') if Path(c, 'probe.json').is_file()).joinpath('probe.json').read_text())
DUMP = os.environ["NF_DUMP_DIR"]
print('игр:', len(GAMES), 'выгрузка в', DUMP, flush=True)

for _ in range(600):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", timeout=5) as r:
            if r.status == 200: break
    except Exception: time.sleep(3)

def _post(payload):
    req = urllib.request.Request(f"http://127.0.0.1:{SERVED_MODEL_PORT}/generate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())

def _ndump():
    return len(glob.glob(os.path.join(DUMP, "*.pt")))

ITEMS = []   # (игра, split, путь, ids, mask, server_lp)
t0 = time.time()
for gi, g in enumerate(GAMES):
    for it in g["items"]:
        before = _ndump()
        out = _post({"input_ids": it["ids"], "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                     "return_logprob": True, "logprob_start_len": 0})
        after = _ndump()
        assert after == before + 1, f"ожидался один новый файл, было {before}, стало {after}"
        lp = [x[0] for x in out["meta_info"]["input_token_logprobs"]]
        ITEMS.append((g["game"], it["split"], os.path.join(DUMP, f"{after - 1:05d}.pt"), it["ids"], it["mask"], lp))
    print(f"[{gi+1}/{len(GAMES)}] {g['game']} готово, {time.time()-t0:.0f} с", flush=True)

# сервер больше не нужен: освобождаем карту
try:
    os.killpg(proc.pid, signal.SIGTERM)
except Exception as e:
    print('kill:', e)
for _ in range(120):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", int(SERVED_MODEL_PORT))) != 0: break
    time.sleep(2)
try: os.killpg(proc.pid, signal.SIGKILL)
except Exception: pass
time.sleep(20)
torch.cuda.empty_cache()
print('сервер остановлен; свободно на карте, ГБ:', round(torch.cuda.mem_get_info()[0] / 1e9, 1), flush=True)

from safetensors import safe_open
_lmf = next(Path(MODEL_DIR).glob("model-*17-of-*.safetensors")) if not Path(MODEL_DIR, "model-00017-of-00017.safetensors").exists() else Path(MODEL_DIR, "model-00017-of-00017.safetensors")
with safe_open(str(_lmf), framework="pt", device="cuda") as f:
    W = f.get_tensor("lm_head.weight").to(torch.bfloat16)   # [V, H]
print('lm_head', tuple(W.shape), flush=True)

def gather(game, split):
    H, Y, check = [], [], []
    for (g, sp, path, ids, mask, lp) in ITEMS:
        if g != game or sp != split: continue
        h = torch.load(path).to("cuda")
        n = min(h.shape[0], len(ids) - 1)
        idx = [i for i in range(n) if mask[i + 1]]
        if not idx: continue
        H.append(h[idx]); Y.append(torch.tensor([ids[i + 1] for i in idx], device="cuda"))
        # сверка: логарифм вероятности из выгрузки против серверного (lp[i+1] — токен на позиции i+1)
        j = idx[len(idx) // 2]
        if j + 1 < len(lp) and lp[j + 1] is not None:
            mine = torch.log_softmax((h[j:j+1].to(torch.bfloat16) @ W.T).float(), -1)[0, ids[j + 1]].item()
            check.append(abs(mine - lp[j + 1]))
    if not H: return None, None, check
    return torch.cat(H), torch.cat(Y), check

def nll(H, Y, A=None, B=None, bs=2048):
    tot = 0.0
    with torch.no_grad():
        for i in range(0, H.shape[0], bs):
            h = H[i:i+bs].to(torch.bfloat16)
            if A is not None: h = h + (h.float() @ A @ B).to(torch.bfloat16)
            tot += torch.nn.functional.cross_entropy((h @ W.T).float(), Y[i:i+bs], reduction="sum").item()
    return tot / H.shape[0]

def train(H, Y, rank=32, steps=300, lr=2e-3, bs=1024, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    A = (torch.randn(H.shape[1], rank, device="cuda", generator=g) / H.shape[1] ** 0.5).requires_grad_()
    B = torch.zeros(rank, H.shape[1], device="cuda", requires_grad=True)
    opt = torch.optim.AdamW([A, B], lr=lr, weight_decay=0.0)
    for s in range(steps):
        k = torch.randint(0, H.shape[0], (min(bs, H.shape[0]),), device="cuda", generator=g)
        h = H[k].float()
        h2 = (h + h @ A @ B).to(torch.bfloat16)
        loss = torch.nn.functional.cross_entropy((h2 @ W.T).float(), Y[k])
        opt.zero_grad(); loss.backward(); opt.step()
    return A.detach(), B.detach()

DATA = {}
checks = []
for g in GAMES:
    tr = gather(g["game"], "train"); ev = gather(g["game"], "eval")
    checks += tr[2] + ev[2]
    if tr[0] is not None and ev[0] is not None:
        DATA[g["game"]] = (tr[0], tr[1], ev[0], ev[1])
print(f"сверка с сервером: медиана |Δlogp| {sorted(checks)[len(checks)//2]:.4f}, макс {max(checks):.4f}, n={len(checks)}", flush=True)

names = list(DATA)
ADAPT = {n: train(DATA[n][0], DATA[n][1]) for n in names}
rows = []
for i, n in enumerate(names):
    _, _, He, Ye = DATA[n]
    base = nll(He, Ye)
    own = nll(He, Ye, *ADAPT[n])
    other = names[(i + 1) % len(names)]
    cross = nll(He, Ye, *ADAPT[other])
    tr_base = nll(DATA[n][0], DATA[n][1]); tr_own = nll(DATA[n][0], DATA[n][1], *ADAPT[n])
    rows.append({"game": n, "n_train": int(DATA[n][0].shape[0]), "n_eval": int(He.shape[0]), "base": base, "own": own,
                 "cross": cross, "cross_from": other, "train_base": tr_base, "train_own": tr_own})
    print(f"{n}: проверка base {base:.4f} | своя {own:.4f} ({(own-base)/base*100:+.1f}%) | чужая ({other}) {cross:.4f} "
          f"({(cross-base)/base*100:+.1f}%) | обучение {tr_base:.4f}→{tr_own:.4f}", flush=True)

import statistics as st
rel_own = [(r["own"] - r["base"]) / r["base"] for r in rows]
rel_cross = [(r["cross"] - r["base"]) / r["base"] for r in rows]
diff = [a - b for a, b in zip(rel_own, rel_cross)]
print(f"==== ИТОГ: своя игра {st.mean(rel_own)*100:+.2f}% | чужая игра {st.mean(rel_cross)*100:+.2f}% | "
      f"своя лучше чужой в {sum(d < 0 for d in diff)} из {len(diff)} игр, средняя разница {st.mean(diff)*100:+.2f} п.п.", flush=True)
Path(WORKING_DIR, 'ttt_probe_results.json').write_text(json.dumps({"rows": rows, "checks": checks}, indent=1))
import pandas as pd
pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(Path(WORKING_DIR) / "submission.parquet", index=False)

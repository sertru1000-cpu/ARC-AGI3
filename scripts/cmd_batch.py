# Проверка: счёт ПАЧКОЙ. Разжатие экспертов (111 с постоянных) делится на все примеры пачки.
import sys, time
sys.path.insert(0, "/workspace/code")
import torch

LOSS_CHUNK, L = 1024, 8192

def render(messages):
    ids, labels = [], []
    for m in messages:
        txt = m.get("content") or ""
        if m.get("tool_calls"):
            txt = (txt + "\n" + json.dumps(m["tool_calls"], ensure_ascii=False)).strip()
        if m["role"] == "assistant":
            r = (m.get("reasoning_content") or "").strip()
            if r:
                txt = "<think>\n%s\n</think>\n\n%s" % (r, txt)
        role = m["role"] if m["role"] != "tool" else "user"
        t = tok("<|im_start|>%s\n%s<|im_end|>\n" % (role, txt), add_special_tokens=False)["input_ids"]
        ids += t
        labels += t if m["role"] == "assistant" else [-100] * len(t)
    return ids, labels

prepared = []
for r in rows:
    i2, l2 = render(r["messages"])
    if not any(v != -100 for v in l2):
        continue
    i2, l2 = i2[-L:], l2[-L:]
    if len(i2) < L:                      # дополняем слева, дополнение не учится
        pad = L - len(i2)
        i2 = [tok.pad_token_id or 0] * pad + i2
        l2 = [-100] * pad + l2
    prepared.append((i2, l2))
    if len(prepared) >= 8:
        break

def run(batch):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    x = torch.tensor([b[0] for b in batch]).cuda()
    y = torch.tensor([b[1] for b in batch]).cuda()
    hs = dec(input_ids=x, use_cache=False).last_hidden_state
    head = model.get_output_embeddings()
    s = torch.zeros((), device=hs.device, dtype=torch.float32); n = 0
    for c0 in range(0, hs.shape[1] - 1, LOSS_CHUNK):
        c1 = min(c0 + LOSS_CHUNK, hs.shape[1] - 1)
        lg = head(hs[:, c0:c1]).float(); tg = y[:, c0+1:c1+1]; mk = tg != -100
        if mk.any():
            s = s + torch.nn.functional.cross_entropy(lg[mk], tg[mk], reduction="sum"); n += int(mk.sum())
        del lg
    loss = s / max(1, n)
    loss.backward(); torch.cuda.synchronize()
    dt = time.time() - t0
    opt.zero_grad(set_to_none=True)
    tokens = x.numel()
    return dt, tokens / dt, torch.cuda.max_memory_allocated() / 1e9, float(loss.detach())

model.train()
for bs in (1, 2, 4):
    try:
        dt, v, mem, loss = run(prepared[:bs])
        ep_tokens = 495 * L
        print("  пачка %d x %d | %.1f с | %.0f ток/с | память %.0f ГБ | потеря %.3f"
              " | эпоха %.2f млн ток = %.1f ч = $%.0f"
              % (bs, L, dt, v, mem, loss, ep_tokens / 1e6, ep_tokens / v / 3600,
                 ep_tokens / v / 3600 * 7.94), flush=True)
    except Exception as e:
        print("  пачка %d: не вышло -- %s" % (bs, str(e)[:100]), flush=True)
        break
print("ГОТОВО: пачки проверены", flush=True)

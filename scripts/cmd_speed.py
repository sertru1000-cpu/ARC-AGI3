# Команда живому сеансу: измерить СКОРОСТЬ ОБУЧЕНИЯ на нескольких примерах и напечатать после каждого.
# Исполняется внутри процесса, где модель уже в памяти (scripts/live_session.py).
import time

import torch

MAX_LEN = 32768
LOSS_CHUNK = 1024
N = 4                      # столько примеров хватит: нужна скорость, а не обучение


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
        piece = "<|im_start|>%s\n%s<|im_end|>\n" % (role, txt)
        t = tok(piece, add_special_tokens=False)["input_ids"]
        ids += t
        labels += t if m["role"] == "assistant" else [-100] * len(t)
    return ids, labels


model.train()
print("замер скорости обучения на %d примерах, окно %d" % (N, MAX_LEN), flush=True)
done = 0
for r in rows:
    if done >= N:
        break
    ids, labels = render(r["messages"])
    if len(ids) > MAX_LEN:
        ids, labels = ids[-MAX_LEN:], labels[-MAX_LEN:]
    if not any(x != -100 for x in labels):
        continue
    t0 = time.time()
    x = torch.tensor([ids]).cuda(); y = torch.tensor([labels]).cuda()
    hs = dec(input_ids=x, use_cache=False).last_hidden_state
    head = model.get_output_embeddings()
    loss_sum = torch.zeros((), device=hs.device, dtype=torch.float32); n_tok = 0
    for c0 in range(0, hs.shape[1] - 1, LOSS_CHUNK):
        c1 = min(c0 + LOSS_CHUNK, hs.shape[1] - 1)
        lg = head(hs[:, c0:c1]).float()
        tgt = y[:, c0 + 1:c1 + 1]
        m = tgt != -100
        if m.any():
            loss_sum = loss_sum + torch.nn.functional.cross_entropy(lg[m], tgt[m], reduction="sum")
            n_tok += int(m.sum())
        del lg
    t_fwd = time.time() - t0
    loss = loss_sum / max(1, n_tok)
    loss.backward()
    torch.cuda.synchronize()
    dt = time.time() - t0
    done += 1
    print("  пример %d | %d токенов (учим %d) | потеря %.3f | прямой %.1f с | всего %.1f с"
          " | %.0f ток/с | память %.0f ГБ"
          % (done, len(ids), n_tok, float(loss), t_fwd, dt, len(ids) / dt,
             torch.cuda.max_memory_allocated() / 1e9), flush=True)
    opt.zero_grad(set_to_none=True)

print("ГОТОВО: по этим числам считается цена эпохи = 495 примеров x токены / скорость", flush=True)

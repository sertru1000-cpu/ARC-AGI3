# Только групповой счёт: включить и сразу замерить скорость обучения.
import sys, time
sys.path.insert(0, "/workspace/code")
import torch
from fp8_experts_grouped import use_grouped

MAX_LEN, LOSS_CHUNK, N = 32768, 1024, 3
use_grouped(model, verbose=True)

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

model.train()
done, speeds = 0, []
for r in rows:
    if done >= N: break
    ids, labels = render(r["messages"])
    if len(ids) > MAX_LEN: ids, labels = ids[-MAX_LEN:], labels[-MAX_LEN:]
    if not any(v != -100 for v in labels): continue
    t0 = time.time()
    x = torch.tensor([ids]).cuda(); y = torch.tensor([labels]).cuda()
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
    done += 1; speeds.append(len(ids) / dt)
    print("  пример %d | %d токенов | потеря %.3f | %.1f с | %.0f ток/с | память %.0f ГБ"
          % (done, len(ids), float(loss.detach()), dt, len(ids)/dt,
             torch.cuda.max_memory_allocated()/1e9), flush=True)

v = sum(speeds) / max(len(speeds), 1)
print("СКОРОСТЬ %.0f ток/с (было 60). Эпоха 14.11 млн токенов = %.1f ч = $%.0f"
      % (v, 14.11e6 / v / 3600, 14.11e6 / v / 3600 * 7.94), flush=True)
print("ГОТОВО: замер закончен", flush=True)

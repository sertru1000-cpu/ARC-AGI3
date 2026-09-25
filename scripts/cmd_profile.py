# Профилировщик: на что уходит время в одном шаге обучения. Даёт ответ, а не гипотезу.
import sys, time
sys.path.insert(0, "/workspace/code")
import torch
from torch.profiler import profile, ProfilerActivity

MAX_LEN, LOSS_CHUNK = 32768, 1024

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

ids = labels = None
for r in rows:
    i2, l2 = render(r["messages"])
    if len(i2) > MAX_LEN: i2, l2 = i2[-MAX_LEN:], l2[-MAX_LEN:]
    if any(v != -100 for v in l2):
        ids, labels = i2, l2; break

model.train()
x = torch.tensor([ids]).cuda(); y = torch.tensor([labels]).cuda()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    hs = dec(input_ids=x, use_cache=False).last_hidden_state
    head = model.get_output_embeddings()
    s = torch.zeros((), device=hs.device, dtype=torch.float32); n = 0
    for c0 in range(0, hs.shape[1] - 1, LOSS_CHUNK):
        c1 = min(c0 + LOSS_CHUNK, hs.shape[1] - 1)
        lg = head(hs[:, c0:c1]).float(); tg = y[:, c0+1:c1+1]; mk = tg != -100
        if mk.any():
            s = s + torch.nn.functional.cross_entropy(lg[mk], tg[mk], reduction="sum"); n += int(mk.sum())
        del lg
    (s / max(1, n)).backward()
    torch.cuda.synchronize()
opt.zero_grad(set_to_none=True)

print("длина примера %d токенов" % len(ids), flush=True)
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=18), flush=True)
print("ГОТОВО: профиль снят", flush=True)

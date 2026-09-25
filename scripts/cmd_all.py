# Проверка всех дешёвых гипотез разом, в одном живом сеансе.
import sys, time
sys.path.insert(0, "/workspace/code")
import torch
from fp8_experts_grouped import use_grouped

LOSS_CHUNK = 1024

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

full = None
for r in rows:
    i2, l2 = render(r["messages"])
    if any(v != -100 for v in l2) and len(i2) > 20000:
        full = (i2, l2); break
if full is None:
    for r in rows:
        i2, l2 = render(r["messages"])
        if any(v != -100 for v in l2):
            full = (i2, l2); break

def step(ids, labels):
    torch.cuda.reset_peak_memory_stats()
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
    return dt, len(ids) / dt, torch.cuda.max_memory_allocated() / 1e9

def cut(n):
    i2, l2 = full
    return i2[-n:], l2[-n:]

model.train()
res = []

print("1) ПЕРЕБОР экспертов, окно 32768 (как вчера)", flush=True)
ids, labels = cut(32768)
dt, v, mem = step(ids, labels); res.append(("перебор 32768", v, dt, mem))
print("   %.1f с | %.0f ток/с | память %.0f ГБ" % (dt, v, mem), flush=True)

use_grouped(model, verbose=True)
print("2) ГРУППОВОЙ счёт, окно 32768", flush=True)
dt, v, mem = step(ids, labels); res.append(("групповой 32768", v, dt, mem))
print("   %.1f с | %.0f ток/с | память %.0f ГБ" % (dt, v, mem), flush=True)

print("3) ГРУППОВОЙ + БЕЗ пересчёта активаций", flush=True)
try:
    model.gradient_checkpointing_disable()
    dt, v, mem = step(ids, labels); res.append(("без пересчёта 32768", v, dt, mem))
    print("   %.1f с | %.0f ток/с | память %.0f ГБ" % (dt, v, mem), flush=True)
except Exception as e:
    print("   не вышло: %s" % str(e)[:120], flush=True)
finally:
    model.gradient_checkpointing_enable()

print("4) ГРУППОВОЙ, окно 8192 (короче -- дешевле внимание)", flush=True)
i8, l8 = cut(8192)
dt, v, mem = step(i8, l8); res.append(("групповой 8192", v, dt, mem))
print("   %.1f с | %.0f ток/с | память %.0f ГБ" % (dt, v, mem), flush=True)

print("\n=== СВОДКА ===", flush=True)
for name, v, dt, mem in res:
    print("  %-22s %6.0f ток/с | %6.1f с | %5.0f ГБ | эпоха 14.11 млн = %5.1f ч = $%4.0f"
          % (name, v, dt, mem, 14.11e6 / v / 3600, 14.11e6 / v / 3600 * 7.94), flush=True)
print("ГОТОВО: все гипотезы проверены", flush=True)

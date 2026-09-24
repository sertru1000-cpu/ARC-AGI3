"""Обучение адаптера на ИСХОДНОЙ модели bf16 со сжатием ТОЛЬКО экспертов (24.09).

Почему не боевой чекпойнт. Боевой файл сжат инструментом NVIDIA в формат `modelopt`; transformers его не
читает вовсе, а встроенный `nvfp4` умеет только сжимать сам («Loading pre-quantized NVFP4 checkpoints is not
supported yet»). Поэтому берём исходные веса Qwen/Qwen3.8-Flash-Next (360 ГБ, bf16) и сжимаем экспертов сами,
библиотекой bitsandbytes, которую transformers поддерживает.

Ключевая деталь, ради которой всё сходится: обучаемые модули (внимание, линейное внимание, маршрутизаторы,
общий эксперт) мы НЕ сжимаем -- они остаются bf16. В боевом чекпойнте эти же модули тоже лежат в bf16 и
НИКОГДА не сжимались, то есть это побитово те же веса. Значит адаптер, обученный здесь, ляжет в боевую
модель точно, без приближения, через уже проверенную подмену bf16-файлов.

Память на карте: эксперты в четырёх битах ~60 ГБ + bf16-часть ~16 ГБ + активации. Влезает в H200 (141 ГБ).

usage:
  python3 train_lora_bnb.py --smoke            # загрузка + 2 шага, проверка памяти
  python3 train_lora_bnb.py --epochs 1 --out /workspace/out/lora_v1
"""
import argparse, json, math, os, time
from pathlib import Path

# Модули, которые НЕ сжимаем: их мы обучаем, и они же лежат в bf16 в боевом чекпойнте.
KEEP_BF16 = ["self_attn", "linear_attn", "mlp.gate", "shared_expert", "hyper_connection",
             "embed_tokens", "lm_head", "mtp", "visual", "ple"]
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/base")
    ap.add_argument("--data", default="/workspace/data/train.jsonl")
    ap.add_argument("--valid", default="/workspace/data/valid.jsonl")
    ap.add_argument("--out", default="/workspace/out/lora_v1")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=26000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true", help="загрузка + 2 шага: проверка памяти и скорости")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    t0 = time.time()
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                            llm_int8_skip_modules=KEEP_BF16)
    print("загружаю модель (эксперты -> 4 бита, обучаемое остаётся bf16)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=qc,
                                                 dtype=torch.bfloat16, device_map={"": 0})
    print("ЗАГРУЖЕНА за %.1f мин | на карте %.1f ГБ" % ((time.time()-t0)/60, torch.cuda.memory_allocated()/1e9), flush=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2*a.rank, lora_dropout=0.0,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("обучаемых параметров %.1f млн" % (trainable/1e6), flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    print("примеров %d" % len(rows), flush=True)

    def render(messages):
        ids, labels = [], []
        for m in messages:
            txt = m.get("content") or ""
            if m.get("tool_calls"):
                txt = (txt + "\n" + json.dumps(m["tool_calls"], ensure_ascii=False)).strip()
            if m["role"] == "assistant":
                r = (m.get("reasoning_content") or "").strip()
                txt = "<think>\n%s\n</think>\n\n%s" % (r, txt) if r else "<think>\n\n</think>\n\n" + txt
            role = m["role"] if m["role"] != "tool" else "user"
            piece = "<|im_start|>%s\n%s<|im_end|>\n" % (role, txt)
            t = tok(piece, add_special_tokens=False)["input_ids"]
            ids += t
            labels += t if m["role"] == "assistant" else [-100]*len(t)
        return ids, labels

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    model.train()
    step = seen = 0; losses = []; t1 = time.time()
    max_steps = 2 if a.smoke else 0
    for epoch in range(math.ceil(a.epochs)):
        for i, r in enumerate(rows):
            ids, labels = render(r["messages"])
            if len(ids) > a.max_len:
                ids, labels = ids[-a.max_len:], labels[-a.max_len:]
            if not any(x != -100 for x in labels):
                continue
            out = model(input_ids=torch.tensor([ids]).cuda(), labels=torch.tensor([labels]).cuda())
            (out.loss / a.accum).backward()
            losses.append(float(out.loss.detach())); seen += len(ids)
            if len(losses) % a.accum == 0:
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                dt = time.time()-t1
                print("шаг %d | потеря %.3f | %.0f ток/с | память %.0f ГБ | прошло %.1f мин"
                      % (step, sum(losses[-a.accum:])/a.accum, seen/dt, torch.cuda.max_memory_allocated()/1e9, dt/60), flush=True)
                if max_steps and step >= max_steps:
                    print("ДЫМОВОЙ ПРОГОН ПРОЙДЕН"); return
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    print("ГОТОВО: шагов %d, средняя потеря %.3f, %.0f ток/с, адаптер в %s"
          % (step, sum(losses)/max(1,len(losses)), seen/(time.time()-t1), a.out), flush=True)


if __name__ == "__main__":
    main()

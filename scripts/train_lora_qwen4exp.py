"""Обучение адаптера LoRA к боевой модели (Qwen3.8-Flash-Next, архитектура qwen4_exp) — 21.09.

Замысел «C-lite» (см. docs/artifacts/training_cost_20_09.html). В боевой модели в NVFP4 ужаты ТОЛЬКО
маршрутизируемые эксперты (120.8 млрд параметров, 99.4% весов); внимание, линейное внимание, гейты, общий
эксперт и эмбеддинги остались в bf16. Поэтому адаптер вешается на неквантованные модули и обучается в обычном
темпе, а не в 12 раз медленнее, как по слитым тензорам экспертов (замер августа: 34 с на микропакет против 2-3.6).

ПРАВИЛО ПРОЕКТА: на платной карте не отлаживаются. Поэтому у скрипта есть режим `--smoke-tiny`: он собирает
КРОШЕЧНУЮ модель той же архитектуры (те же типы слоёв, случайные веса, процессор) и прогоняет через весь путь
-- разметку, маскирование потерь, шаг оптимизатора, сохранение адаптера. Всё, что ловится локально, должно быть
поймано до аренды.

Маскирование: потеря считается только на ответах модели (по замеру это 28.7% токенов примера), контекст в цель
не идёт. Формат примеров -- выход scripts/build_sft_from_probe.py (диалог с вызовами инструмента и историей,
медиана 26.8 тыс. токенов).

usage:
  .venv/bin/python scripts/train_lora_qwen4exp.py --smoke-tiny                 # локальная проверка пути
  python scripts/train_lora_qwen4exp.py --model <путь> --data data/sft_v3/train.jsonl \
      --targets attn+linear --max-len 26000 --epochs 2 --out out/lora_v1       # на арендованной карте
"""
import argparse, json, math, os, time
from pathlib import Path

TARGET_SETS = {
    # только полное внимание: 12 слоёв из 48, самый осторожный вариант (именно его проверяет канарейка на Kaggle)
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    # плюс линейное внимание: достаёт до всех 48 слоёв
    "attn+linear": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj"],
    # плюс общий эксперт и маршрутизатор -- всё неквантованное
    "unquant": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                "gate_proj", "up_proj", "down_proj"],
}


def tiny_config(tok_vocab: int = 512):
    """Крошечная копия боевой архитектуры: те же типы слоёв, все размеры урезаны."""
    from transformers import AutoConfig
    big = json.load(open(Path(__file__).resolve().parents[1] / "data/model_config_qwen38.json"))
    tc = dict(big["text_config"]); tc.pop("model_type", None); tc.pop("quantization_config", None)
    tc.update(hidden_size=128, head_dim=32, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=4,
              num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, intermediate_size=64,
              vocab_size=tok_vocab, layer_types=["linear_attention"] * 3 + ["full_attention"],
              shared_expert_intermediate_size=32, ngram_vocab_size_base=2000, split_ngram_parts=4,
              make_ngram_vocab_size_divisible_by=8, ple_embed_dim=128, indexer_budget=64, indexer_head_dim=16,
              linear_key_head_dim=32, linear_value_head_dim=32, hc_lowrank=16, max_position_embeddings=512,
              bos_token_id=1, eos_token_id=1)
    return AutoConfig.for_model("qwen4_exp_text", **tc)


def render(tok, messages: list) -> list:
    """текст примера по шаблону модели + границы ответов (для маски потерь)"""
    # Рассуждение вставляем ровно так, как его рисует шаблон чата модели:
    #   '<|im_start|>assistant\n<think>\n' + reasoning_content + '\n</think>\n\n' + content
    # Без этого студент учится выдавать ход БЕЗ мысли -- измеренная катастрофа (2.91 против 9.43).
    flat = []
    for m in messages:
        txt = m.get("content") or ""
        if m.get("tool_calls"):
            txt = (txt + "\n" + json.dumps(m["tool_calls"], ensure_ascii=False)).strip()
        if m["role"] == "assistant":
            reasoning = (m.get("reasoning_content") or "").strip()
            txt = "<think>\n%s\n</think>\n\n%s" % (reasoning, txt) if reasoning else "<think>\n\n</think>\n\n" + txt
        flat.append({"role": m["role"] if m["role"] != "tool" else "user", "content": txt})
    ids, labels = [], []
    for m in flat:
        try:
            piece = tok.apply_chat_template([m], tokenize=False, add_generation_prompt=False)
        except Exception:
            piece = "<|im_start|>%s\n%s<|im_end|>\n" % (m["role"], m["content"])
        t = tok(piece, add_special_tokens=False)["input_ids"]
        ids += t
        labels += t if m["role"] == "assistant" else [-100] * len(t)
    return ids, labels


def batches(rows, tok, max_len, bs=1):
    import torch
    buf = []
    for r in rows:
        ids, labels = render(tok, r["messages"])
        if len(ids) > max_len:                      # режем СЛЕВА: сохраняем текущий ход и ближайшую историю
            ids, labels = ids[-max_len:], labels[-max_len:]
        if not any(x != -100 for x in labels):
            continue
        buf.append((ids, labels))
        if len(buf) == bs:
            n = max(len(x[0]) for x in buf)
            yield (torch.tensor([x[0] + [0] * (n - len(x[0])) for x in buf]),
                   torch.tensor([x[1] + [-100] * (n - len(x[1])) for x in buf]))
            buf = []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--tokenizer", default="data/tokenizer_qwen38")
    ap.add_argument("--data", default="data/sft_v3/train.jsonl")
    ap.add_argument("--targets", choices=sorted(TARGET_SETS), default="attn+linear")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=26000)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--out", default="out/lora_v1")
    ap.add_argument("--smoke-tiny", action="store_true", help="крошечная модель на процессоре: проверка пути, не обучение")
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    if a.smoke_tiny:
        cfg = tiny_config(len(tok))
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
        a.max_len, a.max_steps, a.accum = 512, a.max_steps or 2, 1
        dev = "cpu"
    else:
        if not a.model:
            raise SystemExit("нужен --model (папка с весами боевой модели)")
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")
        model.gradient_checkpointing_enable()
        dev = "cuda"
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                                             target_modules=TARGET_SETS[a.targets], task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("цели: %s | обучаемых параметров %.1f млн | примеров %d" % (a.targets, trainable / 1e6, len(rows)), flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    model.train()
    step = seen_tokens = 0; t0 = time.time(); losses = []
    total_steps = a.max_steps or max(1, int(len(rows) * a.epochs / max(1, a.accum)))
    for epoch in range(math.ceil(a.epochs)):
        for ids, labels in batches(rows, tok, a.max_len):
            out = model(input_ids=ids.to(dev), labels=labels.to(dev))
            (out.loss / a.accum).backward()
            seen_tokens += ids.numel(); losses.append(float(out.loss.detach()))
            if (len(losses)) % a.accum == 0:
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 5 == 0 or step == 1:
                    dt = time.time() - t0
                    print("шаг %d/%d | потеря %.3f | %.0f токенов/с | прошло %.1f мин"
                          % (step, total_steps, sum(losses[-a.accum:]) / a.accum, seen_tokens / max(1e-9, dt), dt / 60), flush=True)
                if a.max_steps and step >= a.max_steps:
                    break
        if a.max_steps and step >= a.max_steps:
            break
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    print("готово: шагов %d, средняя потеря %.3f, %.0f токенов/с, адаптер в %s"
          % (step, sum(losses) / max(1, len(losses)), seen_tokens / max(1e-9, time.time() - t0), a.out))
    print("ВАЖНО: число токенов/с из этого прогона -- прямой вход в смету аренды, записать его в план")


if __name__ == "__main__":
    main()

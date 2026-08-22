"""LoRA SFT of the VISION student: Qwen3.6 (VL, MoE) sees the board PNG.

Mirrors the runtime exactly: `VisionLLM` attaches the current board image to
the LAST user message (multipart [image, caption + text]); here every example
carries `image_b64` = the PNG the teacher looked at (rebuilt from the trace's
frame_seen by build_sft_dataset.py --vision), injected into the last user
message the same way before the chat template is applied.

Why not TRL's SFTTrainer: its VLM path makes assumptions about column names
and collation that vary across versions; a plain transformers Trainer with an
explicit collator keeps the prompt/label construction under our control and
identical to what the model sees at inference:
  prompt  = processor.apply_chat_template(messages[:-1], add_generation_prompt=True)
  full    = processor.apply_chat_template(messages,      add_generation_prompt=False)
  labels  = full ids with the prompt span masked (-100)  -> loss on the reply only

Trainable: LoRA on the LANGUAGE model only (attention/linear-attn/shared-expert
linears by full module name + fused expert nn.Parameters via peft
target_parameters, lora_dropout=0 as ParamWrapper requires). The vision tower
and the projector stay frozen -- the base already sees; we teach it to play.

Env knobs (same names as train.py): BASE_MODEL, DATA_DIR, OUT_DIR, EPOCHS,
MAX_STEPS (smoke), MAX_LENGTH (default 16384), LR (1e-4), LORA_R (16),
EXPERT_LORA (1), IMAGE_SCALE (8, must match how image_b64 was rendered).
"""
import base64
import io
import json
import os
import re
import time

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForImageTextToText, AutoProcessor,
                          BitsAndBytesConfig, Trainer, TrainingArguments)

BASE = os.getenv("BASE_MODEL", "Qwen/Qwen3.6-27B")  # dense VL: the only base our stack can both TRAIN and DEPLOY
DATA = os.getenv("DATA_DIR", "./data")
OUT = os.getenv("OUT_DIR", "/workspace/out_vl")
MAX_STEPS = int(os.getenv("MAX_STEPS", "0"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "12288"))
# 16384 OOMs on a 96 GB card: the loss materialises a seq x vocab(152k)
# logits tensor -- 13.5 GB in one allocation on top of 54 GB of weights
# (RTX PRO 6000, 22.08). 12288 keeps the median example (11450 tokens)
# untruncated and the tensor near 10 GB. Truncation drops OLDEST context,
# never the target reply.
LR = float(os.getenv("LR", "1e-4"))
LORA_R = int(os.getenv("LORA_R", "32"))  # v6 used 16 and under-fit (replies 41% of target length)
EXPERT_LORA = os.getenv("EXPERT_LORA", "1") == "1"
IMAGE_SCALE = int(os.getenv("IMAGE_SCALE", "8"))
QUANT_4BIT = os.getenv("QUANT_4BIT", "1") == "1"
SMOKE = MAX_STEPS > 0
if SMOKE:
    print(f"SMOKE MODE (VL): max_steps={MAX_STEPS}, max_length={MAX_LENGTH}, base={BASE}")

processor = AutoProcessor.from_pretrained(BASE)
tok = processor.tokenizer
tok.padding_side = "right"

_cfg = AutoConfig.from_pretrained(BASE)
_txt = getattr(_cfg, "text_config", None) or _cfg
N_EXPERTS = getattr(_txt, "num_experts", None)
IS_MOE = N_EXPERTS is not None
print(f"base arch {_cfg.architectures} | MoE={IS_MOE} (experts={N_EXPERTS})")


def image_caption(h: int, w: int, scale: int) -> str:
    """Byte-identical to agent/harness/vision.py::image_caption (runtime)."""
    return (
        f"[image] The PNG above is the CURRENT board ({h}x{w} cells, each cell "
        f"drawn as a {scale}x{scale} pixel block, same 16-color palette as the "
        "ascii symbols 0-F in current_frame.ascii; pixel (px,py) -> cell "
        f"(x=px//{scale}, y=py//{scale})). Use it to see shapes, zones and "
        "layout at a glance; use the grid/ascii/segmentation in code for exact "
        "coordinates."
    )


def with_image(messages: list[dict], img: Image.Image) -> list[dict]:
    """Attach the image to the LAST user message, runtime-style."""
    msgs = [dict(m) for m in messages]
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] == "user":
            h = img.height // IMAGE_SCALE
            w = img.width // IMAGE_SCALE
            msgs[i]["content"] = [
                {"type": "image"},
                {"type": "text", "text": image_caption(h, w, IMAGE_SCALE) + "\n\n" + msgs[i]["content"]},
            ]
            break
    return msgs


class VLDataset(Dataset):
    def __init__(self, path: str):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        self.rows = [r for r in self.rows if r.get("image_b64")]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(io.BytesIO(base64.b64decode(r["image_b64"]))).convert("RGB")
        msgs = with_image(r["messages"], img)
        prompt_txt = processor.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full_txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        enc = processor(text=[full_txt], images=[img], return_tensors="pt")
        prompt_enc = processor(text=[prompt_txt], images=[img], return_tensors="pt")
        input_ids = enc["input_ids"][0]
        labels = input_ids.clone()
        labels[: prompt_enc["input_ids"].shape[1]] = -100
        if input_ids.shape[0] > MAX_LENGTH:
            # Should not happen (dataset is pre-trimmed), but never train on a
            # reply cut in half: truncate from the LEFT of the prompt instead.
            cut = input_ids.shape[0] - MAX_LENGTH
            input_ids, labels = input_ids[cut:], labels[cut:]
            enc["attention_mask"] = enc["attention_mask"][:, cut:]
            if "mm_token_type_ids" in enc:
                enc["mm_token_type_ids"] = enc["mm_token_type_ids"][:, cut:]
        item = {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"][0],
            "labels": labels,
            "pixel_values": enc["pixel_values"],
            "image_grid_thw": enc["image_grid_thw"],
        }
        if "mm_token_type_ids" in enc:
            item["mm_token_type_ids"] = enc["mm_token_type_ids"][0]
        return item


def collate(batch):
    # batch size is 1 (long multimodal sequences) -- keep it simple and exact.
    assert len(batch) == 1, "this collator expects per_device_train_batch_size=1"
    b = batch[0]
    out = {
        "input_ids": b["input_ids"][None],
        "attention_mask": b["attention_mask"][None],
        "labels": b["labels"][None],
        "pixel_values": b["pixel_values"],
        "image_grid_thw": b["image_grid_thw"],
    }
    if "mm_token_type_ids" in b:
        out["mm_token_type_ids"] = b["mm_token_type_ids"][None]
    return out


train_ds = VLDataset(f"{DATA}/train.jsonl")
eval_ds = VLDataset(f"{DATA}/valid.jsonl")
print(f"dataset: train={len(train_ds)} valid={len(eval_ds)} (with images)")

# 4-bit weights (QLoRA). bf16 weights of the dense 27B are 54 GB, leaving under
# 40 GB for activations and the vocab-sized logits -- OOM on a 96 GB card at
# both 16k and 12k context (22.08). NF4 puts the weights near 16 GB. The vision
# tower stays unquantised: it is frozen anyway and quantising it risks the image
# path. MoE is excluded -- bitsandbytes cannot quantise fused expert tensors
# (bitsandbytes#1849).
if QUANT_4BIT and not IS_MOE:
    print("loading in 4-bit (NF4); vision tower kept in bf16", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["visual", "lm_head"]),
        dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
    )
else:
    print(f"loading in bf16 (QUANT_4BIT={QUANT_4BIT}, MoE={IS_MOE})", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
    )
model.config.use_cache = False

# Freeze everything; LoRA decides what trains. Vision tower + merger stay frozen.
for p_ in model.parameters():
    p_.requires_grad_(False)

# Language-model linears by FULL name (regex would also catch the vision tower).
lang_linears = sorted({
    name for name, mod in model.named_modules()
    if isinstance(mod, torch.nn.Linear)
    and "language_model" in name
    and not name.endswith("lm_head")
    and "visual" not in name
})
print(f"LoRA target linears (language model): {len(lang_linears)}; sample: {lang_linears[:3]}")

lora_kwargs = dict(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.0,
                   target_modules=lang_linears, task_type="CAUSAL_LM")
if IS_MOE and EXPERT_LORA:
    expert_r = max(1, LORA_R // N_EXPERTS)
    lora_kwargs.update(
        target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        rank_pattern={"experts.gate_up_proj": expert_r, "experts.down_proj": expert_r},
    )
model = get_peft_model(model, LoraConfig(**lora_kwargs))
model.enable_input_require_grads()

trainable = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
total = sum(p_.numel() for p_ in model.parameters())
on_experts = sum(p_.numel() for n, p_ in model.named_parameters() if p_.requires_grad and "experts" in n)
on_visual = sum(p_.numel() for n, p_ in model.named_parameters() if p_.requires_grad and "visual" in n)
print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")
print(f"  of which on experts: {on_experts:,} ({'OK' if on_experts or not (IS_MOE and EXPERT_LORA) else 'ZERO -- target_parameters did NOT attach!'})")
print(f"  on vision tower: {on_visual:,} ({'OK frozen' if on_visual == 0 else 'UNEXPECTED'})")

args = TrainingArguments(
    output_dir=OUT,
    num_train_epochs=float(os.getenv("EPOCHS", "2")),
    max_steps=MAX_STEPS if SMOKE else -1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM", "8")),
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=5,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=1 if SMOKE else int(os.getenv("LOG_STEPS", "20")),
    eval_strategy="no" if SMOKE else "steps",  # smoke: no 112-example eval pass
    eval_steps=100,
    save_steps=MAX_STEPS if SMOKE else 100,
    save_total_limit=3,
    remove_unused_columns=False,
    dataloader_num_workers=2,
    report_to=[],
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                  data_collator=collate)

last_ckpt = None
if os.path.isdir(OUT) and not SMOKE:
    ckpts = sorted((d for d in os.listdir(OUT) if d.startswith("checkpoint-")),
                   key=lambda d: int(d.split("-")[1]))
    last_ckpt = os.path.join(OUT, ckpts[-1]) if ckpts else None

torch.cuda.reset_peak_memory_stats()
_t0 = time.time()
train_out = trainer.train(resume_from_checkpoint=last_ckpt)
trainer.save_model(f"{OUT}/adapter_final")
processor.save_pretrained(f"{OUT}/adapter_final")
report = {"done": True, "adapter": f"{OUT}/adapter_final", "vision": True,
          "train_loss": getattr(train_out, "training_loss", None),
          "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
          "sec_per_step": round((time.time() - _t0) / max(1, trainer.state.global_step), 1),
          "loss_curve": [round(h["loss"], 4) for h in trainer.state.log_history if "loss" in h][:40],
          "eval_loss": next((h["eval_loss"] for h in reversed(trainer.state.log_history)
                             if "eval_loss" in h), None)}
if SMOKE:
    report.update({"smoke": True, "steps": MAX_STEPS})
print(json.dumps(report))

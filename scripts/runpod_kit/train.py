"""QLoRA SFT: Qwen3.6 student on teacher traces. RunPod GPU (96GB+ VRAM).

Recipe mirrors the validated Mac rehearsal (mlx_lm.lora), scaled up:
full 8-context-pair examples, 16k max seq, checkpoints every 100 steps
to /workspace (survives pod death).

MoE bases (e.g. Qwen3.6-35B-A3B): bitsandbytes 4-bit CANNOT quantize the
fused nn.Parameter expert weights that transformers 5.x uses for MoE layers
(confirmed bug: bitsandbytes-foundation/bitsandbytes#1849 -- experts land in
bf16 regardless of load_in_4bit, ballooning memory instead of shrinking it).
So for MoE bases we skip bnb entirely and load bf16 directly -- needs a
bigger card (~70GB just for a 35B model's weights) but avoids a broken
quantization path silently doing nothing. LoRA on MoE experts also needs
peft's target_parameters (nn.Parameter has no forward to patch, unlike
nn.Linear) -- see https://huggingface.co/docs/peft/en/package_reference/lora.
Dense bases keep the original bnb 4-bit path (this is how student v1 was
trained on Qwen3.6-27B, 19.08, unaffected by the MoE-specific bug).

Env knobs:
  BASE_MODEL   (default Qwen/Qwen3.6-27B; for MoE use the bf16 repo, e.g.
                Qwen/Qwen3.6-35B-A3B -- NOT the -FP8 inference checkpoint,
                QLoRA needs a higher-precision source to work from)
  DATA_DIR     (default ./data)   -- expects train.jsonl / valid.jsonl (chat "messages")
  OUT_DIR      (default /workspace/out)
  EPOCHS       (default 2)
  MAX_STEPS    (default 0 = full epochs) -- SMOKE MODE: stop after N optimizer
               steps, eval once, save once, print peak VRAM + step time. Use
               before any full run on a new base/card (MoE recipe untested).
  MAX_LENGTH   (default 16384) -- our examples run to ~15k tokens; 4096 would
               truncate most of the context (grids tokenize ~2.5 chars/token)
  LR           (default 1e-4, student v1 recipe)
  LORA_R       (default 16)
  EXPERT_LORA  (default 1) -- 0 = MoE fallback: adapt only attention/shared
               linears, leave expert nn.Parameters frozen (use if peft's
               target_parameters path fails or OOMs on the smoke)
"""
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE = os.getenv("BASE_MODEL", "Qwen/Qwen3.6-27B")
DATA = os.getenv("DATA_DIR", "./data")
OUT = os.getenv("OUT_DIR", "/workspace/out")
MAX_STEPS = int(os.getenv("MAX_STEPS", "0"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "16384"))
LR = float(os.getenv("LR", "1e-4"))
LORA_R = int(os.getenv("LORA_R", "16"))
EXPERT_LORA = os.getenv("EXPERT_LORA", "1") == "1"
SMOKE = MAX_STEPS > 0
if SMOKE:
    print(f"SMOKE MODE: max_steps={MAX_STEPS}, max_length={MAX_LENGTH}, base={BASE}")

ds = load_dataset("json", data_files={
    "train": f"{DATA}/train.jsonl", "eval": f"{DATA}/valid.jsonl"})

tok = AutoTokenizer.from_pretrained(BASE)

# Detect MoE (fused-expert) architectures before loading, so we can skip the
# broken bnb path for them rather than discover the memory blowup at OOM time.
_cfg = AutoConfig.from_pretrained(BASE)
IS_MOE = getattr(_cfg, "num_experts", None) is not None

if IS_MOE:
    print(f"MoE base detected (num_experts={_cfg.num_experts}) -> "
          "loading bf16 directly, no bnb quantization (see module docstring).")
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
    )
else:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
    )
model.config.use_cache = False

if IS_MOE:
    # Fused expert weights are nn.Parameter, not nn.Linear -- target_modules
    # can't reach them. r is divided across experts (256 here) so the total
    # adapter size stays sane; router is intentionally left untouched
    # (tuning it destabilizes token routing -- Unsloth/peft guidance).
    n_experts = _cfg.num_experts
    expert_r = max(1, LORA_R // n_experts)
    if EXPERT_LORA:
        lora = LoraConfig(
            r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05,
            target_modules="all-linear",  # attention + shared-expert FFN (real nn.Linear)
            target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
            rank_pattern={
                "experts.gate_up_proj": expert_r,
                "experts.down_proj": expert_r,
            },
            task_type="CAUSAL_LM",
        )
    else:
        print("EXPERT_LORA=0 -> experts frozen, adapting attention/shared linears only")
        lora = LoraConfig(
            r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05,
            target_modules="all-linear", task_type="CAUSAL_LM",
        )
else:
    lora = LoraConfig(
        r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )

cfg = SFTConfig(
    output_dir=OUT,
    num_train_epochs=float(os.getenv("EPOCHS", "2")),
    max_steps=MAX_STEPS if SMOKE else -1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=5,  # transformers 5.x dropped warmup_ratio
    bf16=True,
    gradient_checkpointing=True,
    max_length=MAX_LENGTH,
    packing=False,
    logging_steps=1 if SMOKE else 10,
    eval_strategy="steps",
    eval_steps=MAX_STEPS if SMOKE else 100,
    save_steps=MAX_STEPS if SMOKE else 100,
    save_total_limit=3,
    report_to=[],
)

trainer = SFTTrainer(
    model=model, args=cfg, processing_class=tok,
    train_dataset=ds["train"], eval_dataset=ds["eval"],
    peft_config=lora,
)

if SMOKE:
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")
    if IS_MOE and EXPERT_LORA:
        n_expert_lora = sum(p.numel() for n, p in trainer.model.named_parameters()
                            if p.requires_grad and "experts" in n)
        print(f"  of which on experts: {n_expert_lora:,} "
              f"({'OK' if n_expert_lora else 'ZERO -- target_parameters did NOT attach!'})")
    torch.cuda.reset_peak_memory_stats()
    import time as _time
    _t0 = _time.time()

# Resume automatically if a checkpoint survives a pod restart.
last_ckpt = None
if os.path.isdir(OUT) and not SMOKE:
    ckpts = sorted(
        (d for d in os.listdir(OUT) if d.startswith("checkpoint-")),
        key=lambda d: int(d.split("-")[1]))
    last_ckpt = os.path.join(OUT, ckpts[-1]) if ckpts else None

train_out = trainer.train(resume_from_checkpoint=last_ckpt)
trainer.save_model(f"{OUT}/adapter_final")
report = {"done": True, "adapter": f"{OUT}/adapter_final",
          "train_loss": getattr(train_out, "training_loss", None)}
if SMOKE:
    elapsed = _time.time() - _t0
    report.update({
        "smoke": True, "steps": MAX_STEPS,
        "sec_per_step": round(elapsed / MAX_STEPS, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "loss_curve": [round(h["loss"], 4) for h in trainer.state.log_history if "loss" in h],
        "eval_loss": next((h["eval_loss"] for h in reversed(trainer.state.log_history)
                           if "eval_loss" in h), None),
    })
print(json.dumps(report))

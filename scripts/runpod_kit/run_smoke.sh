#!/usr/bin/env bash
# SMOKE TEST of the MoE (Qwen3.6-35B-A3B) LoRA recipe -- a few optimizer steps,
# no merge, no upload. Answers, for ~$2-3, the questions a 3-4h run must not be
# the first to ask: does the model load in bf16 on this card, does peft attach
# LoRA to the fused expert parameters, does the loss move, what is peak VRAM
# and sec/step (=> cost of the real run).
#
# Usage (on the pod, after tar xzf runpod-kit.tgz && cd runpod-kit):
#   export HF_TOKEN=hf_...            # read token is enough (gated/private base)
#   bash run_smoke.sh                 # defaults: 35B-A3B, 12 steps, 16k ctx
#   MAX_STEPS=30 MAX_LENGTH=8192 bash run_smoke.sh
#   EXPERT_LORA=0 bash run_smoke.sh   # fallback: experts frozen
# Output: /workspace/smoke.log, final JSON line = the report to paste back.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1   # PEP 668 images (RunPod 2.8 template, 21.08)

export HF_HOME=/workspace/hf
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.6-35B-A3B}"
export MAX_STEPS="${MAX_STEPS:-12}"
export MAX_LENGTH="${MAX_LENGTH:-16384}"
export OUT_DIR="${OUT_DIR:-/workspace/smoke_out}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "== [0/3] deps (same pins as run_all.sh) =="
pip install -q -U "torch==2.8.0" "transformers==5.15.0" "trl>=0.17" peft \
    bitsandbytes accelerate datasets ninja
pip install -q -U flash-linear-attention
pip install -q causal-conv1d --no-build-isolation

echo "== [1/3] sanity =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'EOF'
import json, os
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(os.environ["BASE_MODEL"])
print("base:", os.environ["BASE_MODEL"], "| num_experts:", getattr(cfg, "num_experts", None),
      "| layers:", getattr(cfg, "num_hidden_layers", None))
n = sum(1 for _ in open("data/train.jsonl")); v = sum(1 for _ in open("data/valid.jsonl"))
print(f"dataset ok: train={n} valid={v}")
EOF

echo "== [2/3] smoke train ($MAX_STEPS steps, ctx $MAX_LENGTH) =="
rm -rf "$OUT_DIR"
python train.py 2>&1 | tee /workspace/smoke.log

echo "== [3/3] report =="
grep '"smoke": true' /workspace/smoke.log | tail -1 || echo "NO REPORT LINE -- see /workspace/smoke.log"
echo "Decision guide: loss_curve must trend down over the steps; 'of which on experts'"
echo "must be non-zero; peak_vram_gb < card total - 8; sec_per_step * (examples*epochs/8)"
echo "= real-run duration. Then: Stop (not Terminate) the pod if proceeding with the same setup."

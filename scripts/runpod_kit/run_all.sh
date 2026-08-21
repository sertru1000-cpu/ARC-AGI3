#!/usr/bin/env bash
# One-shot student training on a RunPod RTX Pro 6000 pod.
# Usage:  export HF_TOKEN=hf_...; export HF_REPO=you/arc3-student-v1; bash run_all.sh
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1   # PEP 668 images (RunPod 2.8 template, 21.08)

export HF_HOME=/workspace/hf          # cache on the persistent volume
export TOKENIZERS_PARALLELISM=false

echo "== [0/4] deps =="
# Version pins earned the hard way (19.08, A100 pod):
#  - torch stays at the image's CUDA-toolkit version (12.8) or JIT builds break;
#  - transformers 5.15+ needed for the qwen3_5 architecture;
#  - llmcompressor is BANNED here: it drags torch>=2.10 and transformers<5
#    (install it separately at convert time, after training);
#  - fast kernels: fla is pure triton; causal-conv1d must build against the
#    INSTALLED torch (--no-build-isolation), with ninja for speed.
pip install -q -U "torch==2.8.0" "transformers==5.15.0" "trl>=0.17" peft \
    bitsandbytes accelerate datasets ninja
pip install -q -U flash-linear-attention tilelang   # tilelang: fla backward on Hopper (triton 3.4 bug #640)
pip install -q causal-conv1d --no-build-isolation
# hf_transfer hung on a shard mid-download; plain downloader is slower but robust.
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "== [1/4] sanity =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'EOF'
import json
n = sum(1 for _ in open("data/train.jsonl")); v = sum(1 for _ in open("data/valid.jsonl"))
assert n > 100, f"train.jsonl suspiciously small: {n}"
print(f"dataset ok: train={n} valid={v}")
EOF

echo "== [2/4] train (QLoRA) =="
python train.py 2>&1 | tee /workspace/train.log

echo "== [3/4] merge + FP8 + upload =="
python convert.py 2>&1 | tee /workspace/convert.log

echo "== [4/4] done =="
echo "Artifacts: /workspace/student-fp8 (serve this), /workspace/out/adapter_final"
echo "If HF_REPO was set, both are uploaded - the pod can be TERMINATED now."

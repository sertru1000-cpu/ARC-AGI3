#!/usr/bin/env bash
# SMOKE TEST of the VISION student recipe (train_vl.py): a few steps on the
# image-carrying dataset. Same questions as run_smoke.sh plus: does the VL
# model load + accept our multimodal batches, is the vision tower frozen, how
# much VRAM/time does the image add. Assumes deps from run_smoke.sh are
# already installed on this pod (same container) -- re-runs them otherwise.
#
# Usage:  export HF_TOKEN=...; bash run_smoke_vl.sh
#         MAX_STEPS=20 bash run_smoke_vl.sh
# Output: /workspace/smoke_vl.log, final JSON line = report.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1

export HF_HOME=/workspace/hf
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.6-35B-A3B}"
export MAX_STEPS="${MAX_STEPS:-12}"
export MAX_LENGTH="${MAX_LENGTH:-16384}"
export OUT_DIR="${OUT_DIR:-/workspace/smoke_vl_out}"
export DATA_DIR="${DATA_DIR:-./data_vision}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "== [0/3] deps =="
python -c "import transformers, peft, trl; print('deps present:', transformers.__version__, peft.__version__)" 2>/dev/null || {
  pip install -q -U "torch==2.8.0" "transformers==5.15.0" "trl>=0.17" peft \
      bitsandbytes accelerate datasets ninja pillow
  pip install -q -U flash-linear-attention
  pip install -q causal-conv1d --no-build-isolation
}
pip install -q pillow 2>/dev/null || true

echo "== [1/3] sanity =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'EOF'
import json, os, base64, io
from PIL import Image
n = 0; v = 0
for l in open(f"{os.environ['DATA_DIR']}/train.jsonl", encoding="utf-8"):
    n += 1
    if n == 1:
        ex = json.loads(l); img = Image.open(io.BytesIO(base64.b64decode(ex["image_b64"])))
        print("first image:", img.size, img.mode)
for l in open(f"{os.environ['DATA_DIR']}/valid.jsonl", encoding="utf-8"):
    v += 1
print(f"dataset ok: train={n} valid={v}")
EOF

echo "== [2/3] smoke train VL ($MAX_STEPS steps, ctx $MAX_LENGTH) =="
rm -rf "$OUT_DIR"
python train_vl.py 2>&1 | tee /workspace/smoke_vl.log

echo "== [3/3] report =="
grep '"smoke": true' /workspace/smoke_vl.log | tail -1 || echo "NO REPORT LINE -- see /workspace/smoke_vl.log"

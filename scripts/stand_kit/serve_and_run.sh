#!/usr/bin/env bash
# Stand session: serve base+LoRA (Kaggle-like stack) and run the exam matrix
# CONCURRENTLY. One command after exports.
#
#   export HF_TOKEN=hf_...            # read access to the private adapter repo
#   bash serve_and_run.sh
set -euo pipefail
cd "$(dirname "$0")"

export HF_HOME=/workspace/hf
# Keep pip's cache off the network volume (only big model weights belong on
# /workspace) — a stalled network mount otherwise wedges pip in D-state I/O
# wait that can't even be killed.
export PIP_CACHE_DIR=/root/.cache/pip
BASE_REPO="${BASE_REPO:-vrfai/Qwen3.6-27B-FP8}"
ADAPTER_REPO="${ADAPTER_REPO:-sertru1000gpu/arc3-student-v1}"
# WITH_ADAPTER=0 -> serve the bare base model, no LoRA (capability-ceiling
# check: is a bigger/different base worth distilling before we pay for it).
WITH_ADAPTER="${WITH_ADAPTER:-1}"

echo "== [1/5] deps (vllm serving stack + game engine) =="
# Two steps, earned the hard way (20.08 A100 pod, cf. runpod_kit lesson 19.08):
#  1. --ignore-installed ONLY for blinker (apt-installed, no RECORD file, pip
#     refuses to uninstall it).
#  2. Everything else installed NORMALLY. --ignore-installed on the whole list
#     layered torch 2.10 over the image's torch without removing old files ->
#     двойная регистрация Triton-шаблонов -> "duplicate template name" crash
#     in torch._inductor at vLLM startup.
pip install -q --ignore-installed blinker
pip install -q "vllm==0.19.0" "huggingface_hub" "hf_transfer" arc-agi arcengine numpy requests python-dotenv

echo "== [2/5] models =="
python - <<EOF
import os
from huggingface_hub import snapshot_download
base = snapshot_download(os.environ.get("BASE_REPO", "vrfai/Qwen3.6-27B-FP8"))
print("base ->", base)
lines = [f"export BASE={base}\n"]
if "$WITH_ADAPTER" == "1":
    ad = snapshot_download(os.environ.get("ADAPTER_REPO", "sertru1000gpu/arc3-student-v1"),
                           allow_patterns=["adapter/*"])
    print("adapter ->", ad + "/adapter")
    lines.append(f"export ADAPTER={ad}/adapter\n")
open("/workspace/paths.env", "w").writelines(lines)
EOF
source /workspace/paths.env

if [ "$WITH_ADAPTER" = "1" ]; then
  echo "== [2b/5] adapter keys -> composite layout (same rename as Kaggle dataset) =="
  python - <<'EOF'
import struct, json, os, shutil
src = os.environ["ADAPTER"]
dst = "/workspace/adapter-composite"
os.makedirs(dst, exist_ok=True)
raw = open(os.path.join(src, "adapter_model.safetensors"), "rb").read()
n = struct.unpack("<Q", raw[:8])[0]
hdr = json.loads(raw[8:8+n]); data = raw[8+n:]
OLD, NEW = "base_model.model.model.layers.", "base_model.model.model.language_model.layers."
renamed = {(NEW + k[len(OLD):] if k.startswith(OLD) else k): v for k, v in hdr.items()}
hb = json.dumps(renamed, separators=(",", ":")).encode()
with open(os.path.join(dst, "adapter_model.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(hb))); f.write(hb); f.write(data)
shutil.copy(os.path.join(src, "adapter_config.json"), dst)
print("composite adapter ->", dst)
EOF
else
  echo "== [2b/5] skipped (WITH_ADAPTER=0, bare base capability check) =="
fi

echo "== [3/5] vLLM serve (background; THE RUNTIME FIXES UNDER TEST) =="
LORA_FLAGS=()
SERVED_NAME="base"
if [ "$WITH_ADAPTER" = "1" ]; then
  LORA_FLAGS=(--enable-lora --max-loras 1 --max-lora-rank 16 --lora-modules student=/workspace/adapter-composite)
  SERVED_NAME="student"
fi
nohup python -m vllm.entrypoints.openai.api_server \
    --model "$BASE" --served-model-name base \
    --host 127.0.0.1 --port 1234 \
    "${LORA_FLAGS[@]}" \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --enable-prefix-caching --max-num-seqs 24 \
    --max-model-len 65536 \
    > /workspace/vllm.log 2>&1 &

echo "waiting for server..."
for i in $(seq 1 240); do
  curl -s http://127.0.0.1:1234/v1/models >/dev/null 2>&1 && break
  sleep 5
done
curl -s http://127.0.0.1:1234/v1/models | head -c 300; echo

echo "== [4/5] exam matrix: 8 games x 3 reps, CONCURRENT =="
export LLM_BASE_URL=http://127.0.0.1:1234/v1 LLM_MODEL="$SERVED_NAME" LLM_API_KEY=x
export LLM_TIMEOUT_S=600 LLM_MAX_TOKENS=8192 PYTHONUNBUFFERED=1
LABEL="student-nothink"; [ "$WITH_ADAPTER" = "0" ] && LABEL="base-nodistill"
GAMES="${GAMES:-lp85,sb26,wa30,vc33,ft09,m0r0,tn36,ls20}"
REPS="${REPS:-3}"
CONCURRENCY="${CONCURRENCY:-12}"
MAX_TURNS="${MAX_TURNS:-80}"
GAME_SECONDS="${GAME_SECONDS:-2400}"
MAX_ACTIONS="${MAX_ACTIONS:-400}"
python scripts/run_stand.py \
    --games "$GAMES" \
    --reps "$REPS" --concurrency "$CONCURRENCY" --label "$LABEL" \
    --max-turns "$MAX_TURNS" --game-seconds "$GAME_SECONDS" --max-actions "$MAX_ACTIONS" \
    2>&1 | tee /workspace/stand.log

echo "== [5/5] done — summary above; full logs /workspace/stand.log =="

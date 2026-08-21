#!/usr/bin/env bash
# FULL training of the vision student (train_vl.py), unattended:
#   deps -> train (EXPERT_LORA=0, 16k ctx, EPOCHS epochs, checkpoints every
#   100 steps on /workspace) -> tarball of the adapter -> DONE marker ->
#   wait up to FETCH_WAIT_MIN for /workspace/FETCHED (the VM scp's the tarball
#   and touches it) -> stop this pod via the RunPod API (RUNPOD_API_KEY).
# Usage: RUNPOD_API_KEY=rpa_... POD_ID=... bash run_full_vl.sh
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
export HF_HOME=/workspace/hf
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"      # base is cached on the volume
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
export PIP_CACHE_DIR=/root/.cache/pip
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.6-35B-A3B}"
export DATA_DIR="${DATA_DIR:-./data_vision}"
export OUT_DIR="${OUT_DIR:-/workspace/out_vl}"
export EPOCHS="${EPOCHS:-2}"
export MAX_LENGTH="${MAX_LENGTH:-16384}"
export EXPERT_LORA="${EXPERT_LORA:-0}"
export MAX_STEPS=0
FETCH_WAIT_MIN="${FETCH_WAIT_MIN:-30}"

echo "=== FULL VL START $(date -u +%H:%M:%S) base=$BASE_MODEL epochs=$EPOCHS ctx=$MAX_LENGTH experts=$EXPERT_LORA"
echo "== [0/4] deps =="
python -c "import transformers, peft, tilelang, fla" 2>/dev/null && echo "deps present" || {
  pip install -q -U "torch==2.8.0" "transformers==5.15.0" "trl>=0.17" peft \
      bitsandbytes accelerate datasets ninja pillow
  pip install -q -U flash-linear-attention tilelang
  pip install -q causal-conv1d --no-build-isolation
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== [1/4] train =="
python train_vl.py 2>&1 | tee /workspace/full_vl_train.log
echo "=== TRAIN EXIT ${PIPESTATUS[0]} $(date -u +%H:%M:%S)"

echo "== [2/4] pack adapter =="
cd /workspace && tar -czf /workspace/adapter_vl_final.tgz -C "$OUT_DIR" adapter_final && ls -la /workspace/adapter_vl_final.tgz
cp /workspace/full_vl_train.log "$OUT_DIR"/ 2>/dev/null || true
echo "=== DONE $(date -u +%H:%M:%S)"

echo "== [3/4] wait for fetch (max ${FETCH_WAIT_MIN} min) =="
for i in $(seq 1 $((FETCH_WAIT_MIN * 6))); do
  [ -f /workspace/FETCHED ] && { echo "fetched by VM"; break; }
  sleep 10
done

echo "== [4/4] stop pod =="
if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${POD_ID:-}" ]; then
  curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation { podStop(input: {podId: \\\"${POD_ID}\\\"}) { id desiredStatus } }\"}"
  echo " === POD STOP REQUESTED $(date -u +%H:%M:%S)"
else
  echo "no RUNPOD_API_KEY/POD_ID -- pod left running"
fi

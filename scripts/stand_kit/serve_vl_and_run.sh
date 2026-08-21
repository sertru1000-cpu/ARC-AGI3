#!/usr/bin/env bash
# Stand for the VISION student (Qwen3.6-35B-A3B VL + our LoRA):
#   [1] merge the LoRA into the bf16 base (vLLM's runtime-LoRA support for a
#       MoE + VL + gated-delta-net model is unverified and silently drops
#       unsupported modules -- Duck's lora_guard lesson; a merged checkpoint
#       sidesteps that entirely)
#   [2] serve the merged model with vLLM, images enabled, thinking off
#   [3] run the exam matrix through run_stand.py with MY_AGENT_VISION=1
#
#   ADAPTER=/workspace/out_vl/adapter_final bash serve_vl_and_run.sh
# Env knobs: BASE_MODEL, ADAPTER, MERGED (output dir), GAMES, REPS,
#            CONCURRENCY, MAX_TURNS, MAX_ACTIONS, GAME_SECONDS, LABEL
set -euo pipefail
cd "$(dirname "$0")"
export PIP_BREAK_SYSTEM_PACKAGES=1
export HF_HOME=/workspace/hf
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PIP_CACHE_DIR=/root/.cache/pip
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.6-35B-A3B}"
ADAPTER="${ADAPTER:-/workspace/out_vl/adapter_final}"
MERGED="${MERGED:-/workspace/student_vl_merged}"

echo "== [1/4] merge LoRA -> bf16 checkpoint ($MERGED) =="
if [ -f "$MERGED/config.json" ]; then
  echo "merged checkpoint already present, skipping"
else
  python - <<EOF
import torch, os
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
base = AutoModelForImageTextToText.from_pretrained("$BASE_MODEL", dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(base, "$ADAPTER")
model = model.merge_and_unload()
model.save_pretrained("$MERGED", safe_serialization=True, max_shard_size="5GB")
AutoProcessor.from_pretrained("$BASE_MODEL").save_pretrained("$MERGED")
print("merged ->", "$MERGED")
EOF
fi
du -sh "$MERGED"

echo "== [2/4] vLLM stack (own venv: must not disturb the training env's torch) =="
VENV=/workspace/venv-vllm
if [ ! -x "$VENV/bin/python" ]; then
  python -m venv "$VENV"
  "$VENV/bin/pip" install -q -U pip
  "$VENV/bin/pip" install -q "vllm==0.19.0"
fi
"$VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
pip install -q arc-agi arcengine numpy requests python-dotenv pillow   # game engine for run_stand (system python)

echo "== [3/4] serve (bf16 merged, images on, thinking off) =="
nohup "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MERGED" --served-model-name student \
    --host 127.0.0.1 --port 1234 \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --limit-mm-per-prompt '{"image": 1}' \
    --enable-prefix-caching --max-num-seqs "${MAX_NUM_SEQS:-8}" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" \
    --gpu-memory-utilization 0.92 \
    > /workspace/vllm_vl.log 2>&1 &
echo "waiting for server..."
for i in $(seq 1 360); do
  curl -s http://127.0.0.1:1234/v1/models >/dev/null 2>&1 && break
  sleep 5
done
curl -s http://127.0.0.1:1234/v1/models | head -c 200; echo
# Smoke: one multimodal request must succeed before we spend a matrix on it.
python - <<'EOF'
import base64, io, json, urllib.request
from PIL import Image
img = Image.new("RGB", (512, 512), (255, 255, 255))
for x in range(32, 96):
    for y in range(32, 96):
        img.putpixel((x, y), (249, 60, 49))
buf = io.BytesIO(); img.save(buf, format="PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
payload = {"model": "student", "max_tokens": 64, "temperature": 0,
           "messages": [{"role": "user", "content": [
               {"type": "image_url", "image_url": {"url": url}},
               {"type": "text", "text": "One sentence: what do you see and where?"}]}]}
req = urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions",
                             data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
print("VISION SMOKE:", json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"])
EOF

echo "== [4/4] exam matrix (vision on, teacher-like budgets) =="
export LLM_BASE_URL=http://127.0.0.1:1234/v1 LLM_MODEL=student LLM_API_KEY=x
export LLM_TIMEOUT_S=600 LLM_MAX_TOKENS=4096 PYTHONUNBUFFERED=1
export MY_AGENT_VISION=1 MY_AGENT_VISION_SCALE=8
export MY_AGENT_TURN_ACTION_CAP="${MY_AGENT_TURN_ACTION_CAP:-40}"
export MY_AGENT_VERIFY_GATE_ATTEMPTS="${MY_AGENT_VERIFY_GATE_ATTEMPTS:-999}"
export MY_AGENT_MAX_NO_CODE_STRIKES="${MY_AGENT_MAX_NO_CODE_STRIKES:-1000}"
export MY_AGENT_MAX_TURN_FAILURES="${MY_AGENT_MAX_TURN_FAILURES:-50}"
LABEL="${LABEL:-student-vl-35b}"
GAMES="${GAMES:-sc25,su15,m0r0,sb26,ft09,lp85,tn36,sp80}"
REPS="${REPS:-2}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_TURNS="${MAX_TURNS:-120}"
GAME_SECONDS="${GAME_SECONDS:-5400}"
MAX_ACTIONS="${MAX_ACTIONS:-1600}"
python scripts/run_stand.py \
    --games "$GAMES" --reps "$REPS" --concurrency "$CONCURRENCY" --label "$LABEL" \
    --max-turns "$MAX_TURNS" --game-seconds "$GAME_SECONDS" --max-actions "$MAX_ACTIONS" \
    2>&1 | tee /workspace/stand_vl.log
echo "=== STAND DONE $(date -u +%H:%M:%S)"

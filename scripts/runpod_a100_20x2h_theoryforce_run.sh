#!/usr/bin/env bash
# One-shot setup + run: concurrency=20, 2h clean gameplay budget, on the SAME
# rented RunPod A100 80GB pod as runpod_a100_20x3h_run.sh (resumed, not
# recreated -- template runpod-torch-v280, PUBLIC_KEY already set, network
# volume already has the venv/model cache from the earlier successful run).
#
# Direct derivative of scripts/runpod_a100_20x3h_run.sh, same deployment
# fixes, two differences: (1) 2h budget instead of 3h -- user asked to test
# TODAY's newest feature (theory-force-override) specifically, not to
# reproduce the 26.08 3h/1.43 baseline for a score comparison, so a shorter
# clean run is fine here; (2) the sanity-check grep below looks for
# ATLAS_THEORY_FORCE_OVERRIDE (27.08, not yet tested outside Kaggle Phase A)
# instead of the 26.08 features it already superseded in coverage.
#
# Usage on the pod (as root, fresh or resumed container):
#   bash runpod_a100_20x2h_theoryforce_run.sh
#
# Assumes:
#   - ~/.kaggle/access_token has been scp'd/curl'd in already.
#   - An A100 80GB pod (Marlin W8A16 FP8 fallback, not native FP8 like the
#     real Kaggle RTX Pro 6000/Blackwell submission target).
set -euo pipefail

REPO_URL="https://github.com/sertru1000-cpu/ARC-AGI3.git"
BRANCH="main"
REPO_DIR="/workspace/ARC-AGI-3"
VENV_DIR="/workspace/venv312"
MODEL_HANDLE="foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1"
MODEL_DIR="/workspace/qwen3.8-27b-fp8"
SERVED_MODEL_NAME="Qwen/Qwen3.8-27B-FP8"
VLLM_HOST="127.0.0.1"
VLLM_PORT="1234"
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
VLLM_LOG="/workspace/vllm-openai-server.log"
RUN_NAME="runpod-a100-theoryforce-20x2h-$(date -u +%Y%m%d-%H%M%S)"
EXPERIMENT_DIR="/workspace/atlas_runpod_runs/${RUN_NAME}"

echo "=== [1/7] GPU check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "${GPU_MEM_MB}" -lt 70000 ]; then
  echo "FATAL: expected an 80GB+-class GPU, got ${GPU_MEM_MB} MiB. Aborting before spending more time." >&2
  exit 1
fi
case "${GPU_NAME}" in
  *A100*) : ;;
  *"RTX PRO 6000"*|*"RTX 6000"*) echo "NOTE: got RTX Pro 6000, not A100 -- native FP8 available, timing will differ from the A100 baseline. Continuing." >&2 ;;
  *) echo "WARNING: expected an A100, got '${GPU_NAME}'. Has enough memory, continuing anyway." >&2 ;;
esac

echo "=== [2/7] Clone repo ==="
rm -rf "${REPO_DIR}"
git clone --branch "${BRANCH}" --single-branch --depth 1 "${REPO_URL}" "${REPO_DIR}"
ATLAS_SRC="${REPO_DIR}/atlas_src/src"
test -f "${ATLAS_SRC}/tufa-arc-agi-framework/pyproject.toml" || { echo "FATAL: clone incomplete" >&2; exit 1; }
test -f "${ATLAS_SRC}/ARC3-Inference/pyproject.toml" || { echo "FATAL: clone incomplete" >&2; exit 1; }
grep -q "ATLAS_THEORY_FORCE_OVERRIDE" "${ATLAS_SRC}/ARC3-Inference/inference/agent/prompts.py" \
  || { echo "FATAL: cloned code is missing ATLAS_THEORY_FORCE_OVERRIDE (27.08) -- this run's whole point is to test it outside Kaggle. Wrong branch/commit, or the push didn't land?" >&2; exit 1; }
grep -q "_ATLAS_THEORY_FORCE_AFTER_CALLS" "${ATLAS_SRC}/ARC3-Inference/inference/agent/tool_agent.py" \
  || { echo "FATAL: cloned code is missing the theory-force trigger logic in tool_agent.py" >&2; exit 1; }
echo "Cloned code includes ATLAS_THEORY_FORCE_OVERRIDE -- confirmed."

# Bug #1 (26.08): BOTH ARC3-Inference AND tufa-arc-agi-framework pin
# requires-python=="3.12.12" exactly -- relax both in this cloned copy only.
sed -i 's/requires-python = "==3.12.12"/requires-python = ">=3.12,<3.13"/' \
  "${ATLAS_SRC}/ARC3-Inference/pyproject.toml" \
  "${ATLAS_SRC}/tufa-arc-agi-framework/pyproject.toml"
grep -n "requires-python" "${ATLAS_SRC}/ARC3-Inference/pyproject.toml" "${ATLAS_SRC}/tufa-arc-agi-framework/pyproject.toml"

echo "=== [3/7] Python 3.12 venv via uv (base-image-agnostic) ==="
if ! command -v uv >/dev/null 2>&1; then
  pip install -q uv || pip install -q --break-system-packages uv
fi
uv venv --python 3.12 --clear "${VENV_DIR}"
VENV_PYTHON="${VENV_DIR}/bin/python"
"${VENV_PYTHON}" --version

echo "=== [4/7] Install deps (torch/vllm pinned, flashinfer best-effort, harness editable) ==="
export PIP_CACHE_DIR=/root/.cache/pip   # never on a network volume -- can wedge in D-state
rm -rf /root/.cache/uv
export UV_LINK_MODE=symlink   # cache (local disk) and venv (network volume) are different filesystems
uv pip install --python "${VENV_PYTHON}" "torch==2.10.0" "vllm==0.19.0"
uv pip install --python "${VENV_PYTHON}" flashinfer || echo "flashinfer install failed -- continuing without it (Marlin FP8 path still works on Ampere)"
uv pip install --python "${VENV_PYTHON}" -e "${ATLAS_SRC}/tufa-arc-agi-framework" -e "${ATLAS_SRC}/ARC3-Inference"
"${VENV_PYTHON}" -c "
import importlib.util as u
for m in ('vllm','torch','arc_agi','taaf','inference'):
    print(m, '->', 'OK' if u.find_spec(m) else 'MISSING')
"

echo "=== [5/7] Download the model via kaggle CLI (access_token must already be at ~/.kaggle/access_token) ==="
uv pip install --python "${VENV_PYTHON}" kaggle
mkdir -p "${MODEL_DIR}"
"${VENV_DIR}/bin/kaggle" models instances versions download "${MODEL_HANDLE}" -p "${MODEL_DIR}" --untar
CONFIG_PATH=$(find "${MODEL_DIR}" -maxdepth 3 -iname "config.json" | head -1)
if [ -z "${CONFIG_PATH}" ]; then
  echo "FATAL: model download looks incomplete (no config.json found under ${MODEL_DIR})" >&2
  find "${MODEL_DIR}" -maxdepth 3 >&2
  exit 1
fi
MODEL_DIR=$(dirname "${CONFIG_PATH}")
echo "Resolved model dir: ${MODEL_DIR}"
find "${MODEL_DIR}" -maxdepth 1 -iname "*.safetensors" | wc -l

echo "=== [6/7] Start vLLM (real settings, no Colab-style compromises -- 80GB has headroom) ==="
cat > /tmp/vllm_cmd.txt <<CMD
${VENV_PYTHON} -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_DIR} \
  --served-model-name ${SERVED_MODEL_NAME} \
  --host ${VLLM_HOST} --port ${VLLM_PORT} \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --max-num-seqs 24 \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code \
  --generation-config vllm \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"preserve_thinking": true}'
CMD
nohup bash -c "$(cat /tmp/vllm_cmd.txt)" > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID} (log: ${VLLM_LOG})"

echo "Waiting for vLLM to become ready (up to 15 min -- CUDA-graph capture included)..."
deadline=$(($(date +%s) + 900))
while [ "$(date +%s)" -lt "${deadline}" ]; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "FATAL: vLLM process exited early. Last log lines:" >&2
    tail -n 100 "${VLLM_LOG}" >&2
    exit 1
  fi
  if curl -sf "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
    echo "vLLM ready."
    break
  fi
  sleep 5
done
curl -sf "${VLLM_BASE_URL}/models" >/dev/null 2>&1 || { echo "FATAL: vLLM never became ready" >&2; tail -n 100 "${VLLM_LOG}" >&2; exit 1; }

echo "=== [7/7] Run all 25 official games, concurrency=20, 2h clean gameplay budget ==="
# 25 games / concurrency 20 = 2 waves. 2h total / 2 waves = 60 min/wave.
mkdir -p "${EXPERIMENT_DIR}"
export MPLBACKEND=Agg
export LOCAL_ANALYZER_BASE_URL="${VLLM_BASE_URL}"
export OPENAI_BASE_URL="${VLLM_BASE_URL}"
export LOCAL_ANALYZER_PROVIDER=vllm
export OPENAI_PROVIDER=vllm
export LOCAL_ANALYZER_API_KEY=not-needed
export LOCAL_ANALYZER_MODEL_ID="${SERVED_MODEL_NAME}"
export INFERENCE_ANALYZER_MODEL="${SERVED_MODEL_NAME}"
export LOCAL_ANALYZER_APP_NAME="ARC3 Agent Harness"
export LOCAL_ANALYZER_CONTEXT_WINDOW=32768
export LOCAL_ANALYZER_MAX_OUTPUT=8000
export LOCAL_ANALYZER_TIMEOUT=480
export ANALYZER_TIMEOUT=480
export LOCAL_ANALYZER_TEMPERATURE=0.6
export LOCAL_ANALYZER_TOP_P=0.95
export LOCAL_ANALYZER_TOP_K=20
export LOCAL_ANALYZER_TOOL_STEPS=0
export LOCAL_ANALYZER_TOOL_TIMEOUT=30
export LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS=1024
export LOCAL_ANALYZER_YIELD_SECONDS=60
export LOCAL_ANALYZER_ENABLE_THINKING=true
export MULTIMODAL_CONTEXT=current_grid
export MULTIMODAL_UPSCALE=4
# ARC_API_KEY intentionally unset -> anonymous ONLINE key.

cd "${ATLAS_SRC}/ARC3-Inference"
set +e
"${VENV_DIR}/bin/inference-taaf-run" \
  --include-tags official \
  --agent inference \
  --model "${SERVED_MODEL_NAME}" \
  --analyzer-timeout 480 \
  --deployment-target inline \
  --concurrent-jobs 20 \
  --n-passes 1 \
  --max-runtime-minutes 60 \
  --max-experiment-runtime-hours 2 \
  --run-name "${RUN_NAME}" \
  --experiment-dir "${EXPERIMENT_DIR}" \
  2>&1 | tee "/workspace/${RUN_NAME}.log"
RUN_EXIT=${PIPESTATUS[0]}
set -e
echo "=== DONE (inference-taaf-run exit code: ${RUN_EXIT}). Results in ${EXPERIMENT_DIR} and /workspace/${RUN_NAME}.log ==="
echo "scp these off NOW, before stopping/terminating the pod:"
echo "  ${EXPERIMENT_DIR}"
echo "  /workspace/${RUN_NAME}.log"
echo "  ${VLLM_LOG}"
echo ""
echo "Reminder: this pod likely has a SEPARATE Network Volume resource that"
echo "survives Terminate and keeps billing standalone -- delete it manually"
echo "via the RunPod web UI (Storage -> Network Volumes) after scp'ing results."

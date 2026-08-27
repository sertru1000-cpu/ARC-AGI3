#!/usr/bin/env bash
# Concurrency=20, 3h clean gameplay budget, on a rented RunPod A100 80GB pod
# using the OFFICIAL vllm/vllm-openai:v0.19.0 Docker image as the base.
#
# Switched to this after TWO failed attempts (26-27.08) with a generic
# PyTorch-template pod + pip-installing torch/vllm/flashinfer into a fresh
# venv on the network volume: even with UV_LINK_MODE=symlink (fixes the
# hardlink-vs-copy problem), the sheer NUMBER of small-file operations
# against the network volume (torch ships thousands of tiny header files
# under include/ATen/ops/) crawled at ~7KB/s regardless -- the bottleneck is
# per-file network round-trip latency, which no client-side link-mode trick
# fully solves. The official image ships torch/vllm/flashinfer/CUDA
# PRE-INSTALLED and pre-matched inside the image layers (no network-volume
# I/O at all for them) -- this whole class of problem disappears. This was
# already a documented recommendation from 19-20.08
# (arc-agi-3-runpod-lessons.md: "Serving kits should use the official vLLM
# docker image... instead of PyTorch template + pip") that simply wasn't
# applied to this specific script until now.
#
# REQUIRES the pod to be created with:
#   Image: vllm/vllm-openai:v0.19.0
#   Container start command override: sleep infinity
#     (the image's own default ENTRYPOINT launches vllm's API server
#      immediately with no --model -- it would crash/exit before SSH is
#      even usable otherwise. Overriding the start command keeps the
#      container alive so this script can start vllm itself, later, with
#      the right model/args.)
#
# Usage on the pod (as root, once SSH is up):
#   bash runpod_a100_vllmimg_20x3h_run.sh
#
# Assumes:
#   - ~/.kaggle/access_token has been scp'd in already.
#   - An A100 80GB pod.
#
# Confirmed via reading both harness pyproject.toml files (27.08) before
# writing this that neither declares torch or vllm as a base dependency --
# vllm is only an OPTIONAL [server] extra on ARC3-Inference, never pulled in
# by a plain `-e path` install -- so installing the harness on top of this
# image cannot trigger a duplicate-torch/vllm-registration crash (the exact
# failure mode `--ignore-installed` on the whole dep list caused on 20.08,
# see arc-agi-3-runpod-lessons.md).
set -euo pipefail

REPO_URL="https://github.com/sertru1000-cpu/ARC-AGI3.git"
BRANCH="main"
REPO_DIR="/workspace/ARC-AGI-3"
MODEL_HANDLE="foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1"
MODEL_DIR="/workspace/qwen3.8-27b-fp8"
SERVED_MODEL_NAME="Qwen/Qwen3.8-27B-FP8"
VLLM_HOST="127.0.0.1"
VLLM_PORT="1234"
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
VLLM_LOG="/workspace/vllm-openai-server.log"
RUN_NAME="runpod-a100-vllmimg-20x3h-$(date -u +%Y%m%d-%H%M%S)"
EXPERIMENT_DIR="/workspace/atlas_runpod_runs/${RUN_NAME}"

echo "=== [1/6] GPU + image sanity check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "${GPU_MEM_MB}" -lt 70000 ]; then
  echo "FATAL: expected an 80GB+-class GPU, got ${GPU_MEM_MB} MiB. Aborting before spending more time." >&2
  exit 1
fi
case "${GPU_NAME}" in
  *A100*) : ;;
  *) echo "WARNING: expected an A100, got '${GPU_NAME}'. Has enough memory, continuing anyway." >&2 ;;
esac
# Confirms the pod was actually created with the vllm-openai image, cheaply,
# before spending any more time -- torch/vllm should already be importable
# with zero installs if the image is right.
python3 -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)" \
  || { echo "FATAL: torch/vllm not importable -- was this pod created with image vllm/vllm-openai:v0.19.0?" >&2; exit 1; }
command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git; }
command -v curl >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq curl; }

echo "=== [2/6] Clone repo ==="
rm -rf "${REPO_DIR}"
git clone --branch "${BRANCH}" --single-branch --depth 1 "${REPO_URL}" "${REPO_DIR}"
ATLAS_SRC="${REPO_DIR}/atlas_src/src"
test -f "${ATLAS_SRC}/tufa-arc-agi-framework/pyproject.toml" || { echo "FATAL: clone incomplete" >&2; exit 1; }
test -f "${ATLAS_SRC}/ARC3-Inference/pyproject.toml" || { echo "FATAL: clone incomplete" >&2; exit 1; }
grep -q "ATLAS_FORCE_ROLLBACK_CHECKPOINT" "${ATLAS_SRC}/ARC3-Inference/inference/agent/prompts.py" \
  || { echo "FATAL: cloned code is missing the rollback/extract-suggestion/context-sanitizer features (commits f77d17a/d9ab882) -- wrong branch/commit, or the push didn't land?" >&2; exit 1; }
echo "Cloned code includes the rollback/extract-suggestion/context-sanitizer features -- confirmed."

echo "=== [3/6] Install ONLY the harness (torch/vllm/CUDA already in the image -- nothing heavy to install) ==="
# --break-system-packages defensively covers PEP 668 "externally-managed"
# images (seen on other RunPod templates, 21.08); harmless no-op otherwise.
pip install --break-system-packages -e "${ATLAS_SRC}/tufa-arc-agi-framework" -e "${ATLAS_SRC}/ARC3-Inference" \
  || pip install -e "${ATLAS_SRC}/tufa-arc-agi-framework" -e "${ATLAS_SRC}/ARC3-Inference"
python3 -c "
import importlib.util as u
for m in ('vllm','torch','arc_agi','taaf','inference'):
    print(m, '->', 'OK' if u.find_spec(m) else 'MISSING')
"

echo "=== [4/6] Download the model via kaggle CLI (access_token must already be at ~/.kaggle/access_token) ==="
pip install --break-system-packages kaggle 2>/dev/null || pip install kaggle
mkdir -p "${MODEL_DIR}"
kaggle models instances versions download "${MODEL_HANDLE}" -p "${MODEL_DIR}" --untar
CONFIG_PATH=$(find "${MODEL_DIR}" -maxdepth 3 -iname "config.json" | head -1)
if [ -z "${CONFIG_PATH}" ]; then
  echo "FATAL: model download looks incomplete (no config.json found under ${MODEL_DIR})" >&2
  find "${MODEL_DIR}" -maxdepth 3 >&2
  exit 1
fi
MODEL_DIR=$(dirname "${CONFIG_PATH}")
echo "Resolved model dir: ${MODEL_DIR}"
find "${MODEL_DIR}" -maxdepth 1 -iname "*.safetensors" | wc -l

echo "=== [5/6] Start vLLM (the image's own pre-installed vllm, no venv, same flags as the earlier working run) ==="
cat > /tmp/vllm_cmd.txt <<CMD
python3 -m vllm.entrypoints.openai.api_server \
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

echo "=== [6/6] Run all 25 official games, concurrency=20, 3h clean gameplay budget ==="
# 25 games / concurrency 20 = 2 waves. 3h total / 2 waves = 90 min/wave --
# same per-game budget as 26.08's baseline run, so this run's mean score is
# directly comparable to that 1.43.
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
# Do NOT let set -e/pipefail abort the script on a nonzero exit here -- the
# results-retrieval reminder below must print regardless of how the run
# itself ended.
set +e
inference-taaf-run \
  --include-tags official \
  --agent inference \
  --model "${SERVED_MODEL_NAME}" \
  --analyzer-timeout 480 \
  --deployment-target inline \
  --concurrent-jobs 20 \
  --n-passes 1 \
  --max-runtime-minutes 90 \
  --max-experiment-runtime-hours 3 \
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

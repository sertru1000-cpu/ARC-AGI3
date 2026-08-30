#!/usr/bin/env bash
# A*-TRANSFER A/B (30.08): does the learned h(s) heuristic help plan_real
# on the REAL 25 public games? Two matched 1h arms on one pod session:
#   arm A: ATLAS_ASTAR_MODEL unset  (novelty ordering -- battle default)
#   arm B: ATLAS_ASTAR_MODEL=h_model.pkl, weight 2.0
# conc 28 (Duck stock = ladder point 2 regime), cap 3600s/game, one wave.
# Measures: plan_real found/executed counts + nodes, official levels/RHAE.
#
# Usage on the pod (as root, fresh container):
#   bash runpod_astar_ab_25pub.sh
# Assumes ~/.kaggle/access_token AND /root/h_model.pkl scp'd in already.
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
STAMP=$(date -u +%Y%m%d-%H%M%S)
RUN_A="runpod-astar-A-off-${STAMP}"
RUN_B="runpod-astar-B-on-${STAMP}"
EXP_A="/workspace/atlas_runpod_runs/${RUN_A}"
EXP_B="/workspace/atlas_runpod_runs/${RUN_B}"
H_MODEL="/root/h_model.pkl"

echo "=== [1/7] GPU check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "${GPU_MEM_MB}" -ge 70000 ] || { echo "FATAL: need an 80GB-class GPU" >&2; exit 1; }
[ -f "${H_MODEL}" ] || { echo "FATAL: ${H_MODEL} missing -- scp it first" >&2; exit 1; }

echo "=== [2/7] Clone repo ==="
rm -rf "${REPO_DIR}"
git clone --branch "${BRANCH}" --single-branch --depth 1 "${REPO_URL}" "${REPO_DIR}"
ATLAS_SRC="${REPO_DIR}/atlas_src/src"
grep -q "_atlas_astar_model" "${ATLAS_SRC}/ARC3-Inference/inference/agent/tool_agent.py" \
  || { echo "FATAL: cloned code lacks the A* wiring (17a0648) -- wrong branch?" >&2; exit 1; }
sed -i 's/requires-python = "==3.12.12"/requires-python = ">=3.12,<3.13"/' \
  "${ATLAS_SRC}/ARC3-Inference/pyproject.toml" \
  "${ATLAS_SRC}/tufa-arc-agi-framework/pyproject.toml"

echo "=== [3/7] venv via uv ==="
command -v uv >/dev/null 2>&1 || pip install -q uv || pip install -q --break-system-packages uv
VENV_PYTHON="${VENV_DIR}/bin/python"
if [ -x "${VENV_DIR}/bin/inference-taaf-run" ] && "${VENV_PYTHON}" -c "import vllm, taaf, inference, sklearn" 2>/dev/null; then
  echo "venv reused"
  SKIP_DEPS=1
else
  SKIP_DEPS=0
  uv venv --python 3.12 --clear "${VENV_DIR}"
fi
export PIP_CACHE_DIR=/root/.cache/pip
export UV_LINK_MODE=symlink
if [ "${SKIP_DEPS}" = "0" ]; then
  rm -rf /root/.cache/uv
  uv pip install --python "${VENV_PYTHON}" "torch==2.10.0" "vllm==0.19.0" scikit-learn
  uv pip install --python "${VENV_PYTHON}" flashinfer || echo "flashinfer failed -- continuing"
  uv pip install --python "${VENV_PYTHON}" -e "${ATLAS_SRC}/tufa-arc-agi-framework" -e "${ATLAS_SRC}/ARC3-Inference"
fi
"${VENV_PYTHON}" -c "import vllm, arc_agi, taaf, inference, sklearn; print('imports OK')"

echo "=== [4/7] Model download ==="
if [ -z "$(find "${MODEL_DIR}" -maxdepth 3 -iname 'config.json' 2>/dev/null | head -1)" ]; then
  uv pip install --python "${VENV_PYTHON}" kaggle
  mkdir -p "${MODEL_DIR}"
  "${VENV_DIR}/bin/kaggle" models instances versions download "${MODEL_HANDLE}" -p "${MODEL_DIR}" --untar
fi
MODEL_DIR=$(dirname "$(find "${MODEL_DIR}" -maxdepth 3 -iname config.json | head -1)")
echo "model: ${MODEL_DIR}"

echo "=== [5/7] Competition data (OFFLINE mandatory) ==="
COMP_DIR="/workspace/comp"
ENV_FILES_DIR="${COMP_DIR}/environment_files"
if [ ! -d "${ENV_FILES_DIR}" ] || [ -z "$(ls "${ENV_FILES_DIR}" 2>/dev/null)" ]; then
  uv pip install --python "${VENV_PYTHON}" kaggle
  mkdir -p "${COMP_DIR}"
  "${VENV_DIR}/bin/kaggle" competitions download -c arc-prize-2026-arc-agi-3 -p "${COMP_DIR}"
  ( cd "${COMP_DIR}" && { unzip -oq arc-prize-2026-arc-agi-3.zip \
      || "${VENV_PYTHON}" -m zipfile -e arc-prize-2026-arc-agi-3.zip .; } )
fi
[ "$(find "${ENV_FILES_DIR}" -mindepth 1 -maxdepth 1 | wc -l)" -ge 25 ] \
  || { echo "FATAL: expected 25 game dirs" >&2; exit 1; }

echo "=== [6/7] vLLM (kernel-mirrored) ==="
nohup "${VENV_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_DIR}" --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${VLLM_HOST}" --port "${VLLM_PORT}" --tensor-parallel-size 1 \
  --max-model-len 65536 --trust-remote-code --generation-config vllm \
  --enable-prefix-caching --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"preserve_thinking": true}' \
  > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
deadline=$(($(date +%s) + 900))
while [ "$(date +%s)" -lt "${deadline}" ]; do
  kill -0 "${VLLM_PID}" 2>/dev/null || { tail -50 "${VLLM_LOG}" >&2; exit 1; }
  curl -sf "${VLLM_BASE_URL}/models" >/dev/null 2>&1 && { echo "vLLM ready"; break; }
  sleep 5
done
curl -sf "${VLLM_BASE_URL}/models" >/dev/null 2>&1 || { echo "FATAL: vLLM not ready" >&2; exit 1; }

export MPLBACKEND=Agg
export LOCAL_ANALYZER_BASE_URL="${VLLM_BASE_URL}"
export OPENAI_BASE_URL="${VLLM_BASE_URL}"
export LOCAL_ANALYZER_PROVIDER=vllm OPENAI_PROVIDER=vllm LOCAL_ANALYZER_API_KEY=not-needed
export LOCAL_ANALYZER_MODEL_ID="${SERVED_MODEL_NAME}" INFERENCE_ANALYZER_MODEL="${SERVED_MODEL_NAME}"
export LOCAL_ANALYZER_APP_NAME="ARC3 Agent Harness"
export LOCAL_ANALYZER_CONTEXT_WINDOW=32768 LOCAL_ANALYZER_MAX_OUTPUT=4000
export ATLAS_LLM_MAX_CONCURRENT_REQUESTS=25 ATLAS_TIME_BANK_DRAWS=1
export LOCAL_ANALYZER_TIMEOUT=480 ANALYZER_TIMEOUT=480
export LOCAL_ANALYZER_TEMPERATURE=0.6 LOCAL_ANALYZER_TOP_P=0.95 LOCAL_ANALYZER_TOP_K=20
export LOCAL_ANALYZER_TOOL_STEPS=0 LOCAL_ANALYZER_TOOL_TIMEOUT=30 LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS=1024
export LOCAL_ANALYZER_YIELD_SECONDS=60 LOCAL_ANALYZER_ENABLE_THINKING=true
export MULTIMODAL_CONTEXT=current_grid MULTIMODAL_UPSCALE=4

run_arm() {  # $1=run name, $2=experiment dir
  mkdir -p "$2"
  cd "${ATLAS_SRC}/ARC3-Inference"
  set +e
  "${VENV_DIR}/bin/inference-taaf-run" \
    --agent inference --model "${SERVED_MODEL_NAME}" \
    --analyzer-timeout 480 --deployment-target inline \
    --concurrent-jobs 28 --n-passes 1 \
    --max-runtime-minutes 60 --max-experiment-runtime-minutes 80 \
    --environments-dir "${ENV_FILES_DIR}" \
    --run-name "$1" --experiment-dir "$2" \
    2>&1 | tee "/workspace/$1.log"
  set -e
  echo "=== arm $1 done ==="
}

echo "=== [7/7] arm A (A* OFF) then arm B (A* ON) ==="
unset ATLAS_ASTAR_MODEL || true
run_arm "${RUN_A}" "${EXP_A}"
export ATLAS_ASTAR_MODEL="${H_MODEL}"
export ATLAS_ASTAR_WEIGHT=2.0
run_arm "${RUN_B}" "${EXP_B}"

echo "=== DONE. scp off: ${EXP_A} ${EXP_B} /workspace/${RUN_A}.log /workspace/${RUN_B}.log ==="
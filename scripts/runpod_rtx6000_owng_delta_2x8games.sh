#!/usr/bin/env bash
# OWN-GAMES DELTA RUN (29.08): 8 original testbed games (our_games/, built
# under the Gemini round-7 designer-of-record protocol), TWO builds back to
# back on one pod session, one vLLM server for both:
#   build A = 60f76d5  (v25 code: Gemini rounds 3-5, sub110-tested)
#   build B = main     (v26 code: + round 6 depth levers: handoff, proactive
#                       budgets, hail mary, zombie gate, auto-replay)
# Each build: 8 games x 3 passes = 24 concurrent instances, 60 min/game.
# Purpose per the round-7 protocol: DELTAS between builds + harness-failure
# census per mechanic. Absolute numbers are meaningless outside this pair.
#
# Usage on the pod (as root, fresh container):
#   bash runpod_rtx6000_owng_delta_2x8games.sh
# Assumes ~/.kaggle/access_token scp'd in already (model download only --
# the games themselves ship IN the repo, no competition data needed).
set -euo pipefail

REPO_URL="https://github.com/sertru1000-cpu/ARC-AGI3.git"
BRANCH="main"
COMMIT_A="60f76d5"          # v25 code (rounds 3-5)
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
RUN_A="runpod-owng-delta-A-60f76d5-${STAMP}"
RUN_B="runpod-owng-delta-B-main-${STAMP}"
EXP_A="/workspace/atlas_runpod_runs/${RUN_A}"
EXP_B="/workspace/atlas_runpod_runs/${RUN_B}"
GAME_IDS="kq01-a1b2c3d4,bx01-b2c3d4e5,ic01-c3d4e5f6,rg01-d4e5f6a7,mn01-e5f6a7b8,mr01-f6a7b8c9,ph01-a7b8c9d0,fl01-b8c9d0e1"

echo "=== [1/7] GPU check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "${GPU_MEM_MB}" -lt 70000 ]; then
  echo "FATAL: expected an 80GB+-class GPU, got ${GPU_MEM_MB} MiB." >&2
  exit 1
fi

echo "=== [2/7] Clone repo (full history -- build A needs ${COMMIT_A}) ==="
rm -rf "${REPO_DIR}"
git clone --branch "${BRANCH}" "${REPO_URL}" "${REPO_DIR}"
ATLAS_SRC="${REPO_DIR}/atlas_src/src"
OUR_GAMES_DIR="${REPO_DIR}/our_games"
test -f "${ATLAS_SRC}/ARC3-Inference/pyproject.toml" || { echo "FATAL: clone incomplete" >&2; exit 1; }
GAMES_FOUND=$(find "${OUR_GAMES_DIR}" -name metadata.json | wc -l)
if [ "${GAMES_FOUND}" -lt 8 ]; then
  echo "FATAL: expected 8 own games in ${OUR_GAMES_DIR}, found ${GAMES_FOUND}" >&2
  exit 1
fi
( cd "${REPO_DIR}" && git rev-parse --verify "${COMMIT_A}^{commit}" >/dev/null ) \
  || { echo "FATAL: commit ${COMMIT_A} not in cloned history" >&2; exit 1; }

# requires-python pin relax (26.08 bugs #1/#2) -- must be re-applied after
# EVERY atlas_src checkout switch below, since checkout overwrites pyproject.
relax_python_pin() {
  sed -i 's/requires-python = "==3.12.12"/requires-python = ">=3.12,<3.13"/' \
    "${ATLAS_SRC}/ARC3-Inference/pyproject.toml" \
    "${ATLAS_SRC}/tufa-arc-agi-framework/pyproject.toml"
}
relax_python_pin

echo "=== [3/7] venv via uv (volume-reuse guard) ==="
if ! command -v uv >/dev/null 2>&1; then
  pip install -q uv || pip install -q --break-system-packages uv
fi
VENV_PYTHON="${VENV_DIR}/bin/python"
if [ -x "${VENV_DIR}/bin/inference-taaf-run" ] && "${VENV_PYTHON}" -c "import vllm, taaf, inference" 2>/dev/null; then
  echo "Existing venv passes the import check -- reusing."
  SKIP_DEPS=1
else
  SKIP_DEPS=0
  uv venv --python 3.12 --clear "${VENV_DIR}"
fi
export PIP_CACHE_DIR=/root/.cache/pip
export UV_LINK_MODE=symlink   # cache local, venv on volume: symlinks cross filesystems (27.08 bug #4)
if [ "${SKIP_DEPS}" = "0" ]; then
  rm -rf /root/.cache/uv      # ONLY when rebuilding (28.08 symlink-rot lesson)
  uv pip install --python "${VENV_PYTHON}" "torch==2.10.0" "vllm==0.19.0"
  uv pip install --python "${VENV_PYTHON}" flashinfer || echo "flashinfer failed -- continuing"
  uv pip install --python "${VENV_PYTHON}" -e "${ATLAS_SRC}/tufa-arc-agi-framework" -e "${ATLAS_SRC}/ARC3-Inference"
fi
"${VENV_PYTHON}" -c "
import importlib.util as u
for m in ('vllm','torch','arc_agi','taaf','inference'):
    print(m, '->', 'OK' if u.find_spec(m) else 'MISSING')
"

echo "=== [4/7] Model download (skip if on volume) ==="
if [ -n "$(find "${MODEL_DIR}" -maxdepth 3 -iname 'config.json' 2>/dev/null | head -1)" ]; then
  echo "Model already on the volume -- skipping download."
else
  uv pip install --python "${VENV_PYTHON}" kaggle
  mkdir -p "${MODEL_DIR}"
  "${VENV_DIR}/bin/kaggle" models instances versions download "${MODEL_HANDLE}" -p "${MODEL_DIR}" --untar
fi
CONFIG_PATH=$(find "${MODEL_DIR}" -maxdepth 3 -iname "config.json" | head -1)
[ -n "${CONFIG_PATH}" ] || { echo "FATAL: no config.json under ${MODEL_DIR}" >&2; exit 1; }
MODEL_DIR=$(dirname "${CONFIG_PATH}")
echo "Resolved model dir: ${MODEL_DIR}"

echo "=== [5/7] Start vLLM (kernel-mirrored settings; ONE server for both builds) ==="
nohup "${VENV_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${VLLM_HOST}" --port "${VLLM_PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --trust-remote-code \
  --generation-config vllm \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"preserve_thinking": true}' \
  > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"
deadline=$(($(date +%s) + 900))
while [ "$(date +%s)" -lt "${deadline}" ]; do
  kill -0 "${VLLM_PID}" 2>/dev/null || { echo "FATAL: vLLM exited early" >&2; tail -n 100 "${VLLM_LOG}" >&2; exit 1; }
  curl -sf "${VLLM_BASE_URL}/models" >/dev/null 2>&1 && { echo "vLLM ready."; break; }
  sleep 5
done
curl -sf "${VLLM_BASE_URL}/models" >/dev/null 2>&1 || { echo "FATAL: vLLM never became ready" >&2; exit 1; }

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
export LOCAL_ANALYZER_MAX_OUTPUT=4000
export ATLAS_LLM_MAX_CONCURRENT_REQUESTS=25
export ATLAS_TIME_BANK_DRAWS=0
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

run_build() {  # $1=run name, $2=experiment dir
  mkdir -p "$2"
  cd "${ATLAS_SRC}/ARC3-Inference"
  set +e
  "${VENV_DIR}/bin/inference-taaf-run" \
    --game "${GAME_IDS}" \
    --agent inference \
    --model "${SERVED_MODEL_NAME}" \
    --analyzer-timeout 480 \
    --deployment-target inline \
    --concurrent-jobs 24 \
    --n-passes 3 \
    --max-runtime-minutes 60 \
    --max-experiment-runtime-hours 2 \
    --environments-dir "${OUR_GAMES_DIR}" \
    --run-name "$1" \
    --experiment-dir "$2" \
    2>&1 | tee "/workspace/$1.log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "=== build run $1 finished (exit ${rc}) ==="
}

echo "=== [6/7] BUILD A: ${COMMIT_A} (v25 code, rounds 3-5) ==="
( cd "${REPO_DIR}" && git checkout "${COMMIT_A}" -- atlas_src )
relax_python_pin
# sanity: A must have rounds 3-5 but NOT round 6
grep -q "Probe budget exhausted" "${ATLAS_SRC}/ARC3-Inference/inference/agent/tool_agent.py" \
  || { echo "FATAL: build A missing round-4 hard gate -- wrong commit?" >&2; exit 1; }
if grep -q "_atlas_run_mechanic_handoff" "${ATLAS_SRC}/ARC3-Inference/inference/agent/tool_agent.py"; then
  echo "FATAL: build A contains round-6 handoff -- checkout did not switch?" >&2
  exit 1
fi
run_build "${RUN_A}" "${EXP_A}"

echo "=== [7/7] BUILD B: main (v26 code, + round 6) ==="
( cd "${REPO_DIR}" && git checkout "${BRANCH}" -- atlas_src )
relax_python_pin
grep -q "_atlas_run_mechanic_handoff" "${ATLAS_SRC}/ARC3-Inference/inference/agent/tool_agent.py" \
  || { echo "FATAL: build B missing round-6 handoff -- checkout did not restore main?" >&2; exit 1; }
run_build "${RUN_B}" "${EXP_B}"

echo "=== DONE. scp these off NOW, before terminating the pod: ==="
echo "  ${EXP_A}"
echo "  ${EXP_B}"
echo "  /workspace/${RUN_A}.log"
echo "  /workspace/${RUN_B}.log"
echo "  ${VLLM_LOG}"
echo "Then locally: python scripts/owng_delta_table.py --a <EXP_A>/benchmark.json --b <EXP_B>/benchmark.json --log-a <A.log> --log-b <B.log>"
echo "Reminder: delete any separate Network Volume via the RunPod UI after scp."

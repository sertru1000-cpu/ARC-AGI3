#!/usr/bin/env bash
# LEAN v2 on a RunPod pod: STOCK Duck harness (upstream bundle, not our fork)
# + the lean wrappers, 25 public games, concurrency 28, 2.2 h per game --
# the Kaggle Phase-A regime (solver pickle: max_runtime_s_per_game=7920,
# concurrency=28, analyzer_timeout=900; env from taaf_setup_env.json).
#
# Budget: the pod session is 3 h. Setup (venv, model, data, vLLM) is expected
# to take 20-25 min; the run is capped at 132 min per game and 145 min per
# experiment, so everything ends by ~2h50 and the tarball is ready to scp.
# Every stage prints its wall-clock so a slow stage is visible immediately.
#
# Usage on the pod (root, fresh "RunPod Pytorch 2.8.0" container, 80+ GB GPU;
# for tempo comparable with Kaggle prefer RTX Pro 6000 / H100, not A100):
#   scp -P <port> -i ~/.ssh/id_ed25519 scripts/runpod_lean_25pub_2h2.sh scripts/pod_lean_driver.py root@<ip>:/root/
#   scp -P <port> -i ~/.ssh/id_ed25519 ~/.kaggle/access_token root@<ip>:/root/.kaggle/access_token
#   nohup bash /root/runpod_lean_25pub_2h2.sh > /root/deploy.log 2>&1 &
# Then poll /root/deploy.log; at the end scp /workspace/lean_pod_<stamp>.tar.gz back.
set -euo pipefail

T0=$(date +%s)
stage() { echo "=== [$(( ($(date +%s) - T0) / 60 )) min] $* ==="; }

BUNDLE_REF="jakobbrggen/taaf-kaggle-source-anim-20260807-anim"
BUNDLE_DIR="/workspace/taaf_bundle"
VENV_DIR="/workspace/venv312"
MODEL_HANDLE="foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1"
MODEL_DIR="/workspace/qwen3.8-27b-fp8"
SERVED_MODEL_NAME="Qwen/Qwen3.8-27B-FP8"
VLLM_HOST="127.0.0.1"
VLLM_PORT="1234"
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
VLLM_LOG="/workspace/vllm-openai-server.log"
STAMP=$(date -u +%Y%m%d-%H%M%S)
RUN_NAME="pod-lean-25pub-${STAMP}"
EXP_DIR="/workspace/lean_runs/${RUN_NAME}"
DRIVER="/root/pod_lean_driver.py"
CAP_MIN="${CAP_MIN:-132}"          # 7920 s = the stock kernel's own per-game cap
EXP_MIN="${EXP_MIN:-145}"          # hard stop for the whole experiment
CONC="${CONC:-28}"

stage "GPU check"
nvidia-smi --query-gpu=name,memory.total --format=csv
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "${GPU_MEM_MB}" -ge 70000 ] || { echo "FATAL: need an 80GB-class GPU" >&2; exit 1; }
[ -f "${DRIVER}" ] || { echo "FATAL: ${DRIVER} missing -- scp scripts/pod_lean_driver.py first" >&2; exit 1; }
[ -f /root/.kaggle/access_token ] || { echo "FATAL: /root/.kaggle/access_token missing" >&2; exit 1; }
df -h / /workspace | tail -2

stage "venv via uv"
command -v uv >/dev/null 2>&1 || pip install -q uv || pip install -q --break-system-packages uv
VENV_PYTHON="${VENV_DIR}/bin/python"
export PIP_CACHE_DIR=/root/.cache/pip
export UV_LINK_MODE=symlink        # cache on local disk, venv on the volume: symlink, never copy (27.08 lesson)
if [ -x "${VENV_PYTHON}" ] && "${VENV_PYTHON}" -c "import vllm, kaggle" 2>/dev/null; then
  echo "venv reused"
else
  rm -rf /root/.cache/uv
  uv venv --python 3.12 --clear "${VENV_DIR}"
  uv pip install --python "${VENV_PYTHON}" "torch==2.10.0" "vllm==0.19.0" kaggle
  uv pip install --python "${VENV_PYTHON}" "flashinfer==0.6.6" || uv pip install --python "${VENV_PYTHON}" flashinfer || echo "flashinfer failed -- continuing"
fi
grep -q "falling back to full copy" /root/deploy.log 2>/dev/null && echo "WARNING: uv fell back to copy -- UV_LINK_MODE not honoured" || true

stage "upstream bundle (STOCK source, byte-identical to the Kaggle dataset)"
if [ ! -f "${BUNDLE_DIR}/taaf-kaggle-bundle.json" ]; then
  mkdir -p "${BUNDLE_DIR}"
  "${VENV_DIR}/bin/kaggle" datasets download -d "${BUNDLE_REF}" -p "${BUNDLE_DIR}" --unzip
fi
[ -f "${BUNDLE_DIR}/taaf-kaggle-bundle.json" ] || { echo "FATAL: bundle marker missing under ${BUNDLE_DIR}" >&2; exit 1; }
SRC="${BUNDLE_DIR}/src"
[ -d "${SRC}/ARC3-Inference" ] && [ -d "${SRC}/tufa-arc-agi-framework" ] || { echo "FATAL: bundle src layout unexpected: $(ls "${SRC}")" >&2; exit 1; }
sed -i 's/requires-python = "==3.12.12"/requires-python = ">=3.12,<3.13"/' \
  "${SRC}/ARC3-Inference/pyproject.toml" "${SRC}/tufa-arc-agi-framework/pyproject.toml" || true
"${VENV_PYTHON}" -c "import inference" 2>/dev/null \
  || uv pip install --python "${VENV_PYTHON}" -e "${SRC}/tufa-arc-agi-framework" -e "${SRC}/ARC3-Inference"
"${VENV_PYTHON}" -c "import vllm, arc_agi, taaf, inference; print('imports OK')"
grep -q "def _run_python_tool" "${SRC}/ARC3-Inference/inference/agent/tool_agent.py" \
  || { echo "FATAL: bundle tool_agent lacks _run_python_tool -- lean wrappers would not attach" >&2; exit 1; }

stage "model download (Kaggle model, same as the kernel)"
if [ -z "$(find "${MODEL_DIR}" -maxdepth 3 -iname 'config.json' 2>/dev/null | head -1)" ]; then
  mkdir -p "${MODEL_DIR}"
  "${VENV_DIR}/bin/kaggle" models instances versions download "${MODEL_HANDLE}" -p "${MODEL_DIR}" --untar
fi
MODEL_DIR=$(dirname "$(find "${MODEL_DIR}" -maxdepth 3 -iname config.json | head -1)")
echo "model: ${MODEL_DIR} ($(du -sh "${MODEL_DIR}" | cut -f1))"

stage "competition data (OFFLINE engine, mandatory)"
COMP_DIR="/workspace/comp"
ENV_FILES_DIR="${COMP_DIR}/environment_files"
if [ ! -d "${ENV_FILES_DIR}" ] || [ -z "$(ls "${ENV_FILES_DIR}" 2>/dev/null)" ]; then
  mkdir -p "${COMP_DIR}"
  "${VENV_DIR}/bin/kaggle" competitions download -c arc-prize-2026-arc-agi-3 -p "${COMP_DIR}"
  ( cd "${COMP_DIR}" && { unzip -oq arc-prize-2026-arc-agi-3.zip \
      || "${VENV_PYTHON}" -m zipfile -e arc-prize-2026-arc-agi-3.zip .; } )
fi
[ "$(find "${ENV_FILES_DIR}" -mindepth 1 -maxdepth 1 | wc -l)" -ge 25 ] \
  || { echo "FATAL: expected 25 game dirs under ${ENV_FILES_DIR}" >&2; exit 1; }

stage "vLLM (flags mirrored from the bundle's setup_commands)"
export VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 TRANSFORMERS_NO_TF=1 TRANSFORMERS_NO_TORCHVISION=1
nohup "${VENV_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_DIR}" --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${VLLM_HOST}" --port "${VLLM_PORT}" --tensor-parallel-size 1 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --generation-config vllm --enable-prefix-caching \
  --default-chat-template-kwargs '{"preserve_thinking": true}' \
  --reasoning-parser qwen3 --max-model-len 65536 \
  --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}' \
  > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
deadline=$(($(date +%s) + 900))
while [ "$(date +%s)" -lt "${deadline}" ]; do
  kill -0 "${VLLM_PID}" 2>/dev/null || { tail -50 "${VLLM_LOG}" >&2; exit 1; }
  curl -sf "${VLLM_BASE_URL}/models" >/dev/null 2>&1 && { echo "vLLM ready"; break; }
  sleep 5
done
curl -sf "${VLLM_BASE_URL}/models" >/dev/null 2>&1 || { echo "FATAL: vLLM not ready" >&2; exit 1; }

# Analyzer environment: taaf_setup_env.json of the Kaggle Phase-A run, verbatim.
export MPLBACKEND=Agg
export LOCAL_ANALYZER_BASE_URL="${VLLM_BASE_URL}" OPENAI_BASE_URL="${VLLM_BASE_URL}"
export LOCAL_ANALYZER_PROVIDER=vllm OPENAI_PROVIDER=vllm
export LOCAL_ANALYZER_MODEL_ID="${SERVED_MODEL_NAME}" INFERENCE_ANALYZER_MODEL="${SERVED_MODEL_NAME}"
export LOCAL_ANALYZER_APP_NAME="ARC3 Agent Harness"
export LOCAL_ANALYZER_CONTEXT_WINDOW=32768 LOCAL_ANALYZER_MAX_OUTPUT=0
export LOCAL_ANALYZER_TOOL_STEPS=0 LOCAL_ANALYZER_TOOL_TIMEOUT=30 LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS=1024
export LOCAL_ANALYZER_YIELD_SECONDS=60 LOCAL_ANALYZER_TEMPERATURE=0.6 LOCAL_ANALYZER_TOP_P=0.95 LOCAL_ANALYZER_TOP_K=20
export LOCAL_ANALYZER_ENABLE_THINKING=true
export MULTIMODAL_CONTEXT=current_grid MULTIMODAL_UPSCALE=4
# Kernel cell 7 / solver pickle: level-only resets, non-submission diagnostics, analyzer timeout 900 s.
export ONLY_RESET_LEVELS=true TAAF_RUN_AS_SUBMISSION=0 TAAF_MINIMAL_DIAGNOSTICS=0
export LOCAL_ANALYZER_TIMEOUT=900

stage "RUN: lean v2, 25 public games, conc ${CONC}, cap ${CAP_MIN} min/game, experiment stop ${EXP_MIN} min"
mkdir -p "${EXP_DIR}"
cd "${SRC}/ARC3-Inference"
set +e
"${VENV_PYTHON}" "${DRIVER}" \
  --include-tags official --agent inference --model "${SERVED_MODEL_NAME}" \
  --analyzer-timeout 900 --deployment-target inline \
  --concurrent-jobs "${CONC}" --n-passes 1 \
  --max-runtime-minutes "${CAP_MIN}" --max-experiment-runtime-minutes "${EXP_MIN}" \
  --environments-dir "${ENV_FILES_DIR}" \
  --run-name "${RUN_NAME}" --experiment-dir "${EXP_DIR}" \
  2>&1 | tee "/workspace/${RUN_NAME}.log"
RC=${PIPESTATUS[0]}
set -e
echo "runner exit code: ${RC}"
grep -m1 "pod-lean: wrappers verified" "/workspace/${RUN_NAME}.log" >/dev/null \
  || echo "WARNING: wrappers-verified line not found in the run log -- check the driver output"
echo "games with events: $(find "${EXP_DIR}" -name '*_events.jsonl' | wc -l)"

stage "pack results"
cp "${VLLM_LOG}" "${EXP_DIR}/" 2>/dev/null || true
cp "/workspace/${RUN_NAME}.log" "${EXP_DIR}/" 2>/dev/null || true
tar -czf "/workspace/lean_pod_${STAMP}.tar.gz" -C "/workspace/lean_runs" "${RUN_NAME}"
ls -la "/workspace/lean_pod_${STAMP}.tar.gz"
stage "DONE -- scp /workspace/lean_pod_${STAMP}.tar.gz, then TERMINATE the pod (not Stop)"

#!/usr/bin/env bash
# V23 CONTROL RUN (28.08): the checkpoint-only submission candidate at its
# EXACT commit (7265cbd, planforce+rollbackfix -- no probes, no plan_real,
# no round-3/4/5 code), concurrency=25, 4h per game, ONE wave (4h total),
# 25 public games, OFFLINE. The matched-duration baseline that was missing
# from every probes-vs-checkpoint comparison: same pod class (RTX PRO
# 6000), same vLLM settings, same wave shape as the decisive probe run's
# first 4 hours. Run AFTER the 55x55 rehearsal (user's order 28.08:
# "после 55x55 запускаем v23 с concurrency 25, время прогона 4 часа,
# время на игру - 4 часа").
#
# OFFLINE is MANDATORY (step [6/8]): without --environments-dir v23's
# rollback/anchor stack is silently dead (27.08 lesson). Verify via the
# sys_start-anchor grep printed at the end.
#
# Usage on the pod (as root, fresh container):
#   bash runpod_rtx6000_v23ctl_25x4h_run.sh
#
# Assumes:
#   - ~/.kaggle/access_token has been scp'd in already.
#   - An 80GB+-class GPU pod (RTX PRO 6000 preferred -- parity with the
#     probe-branch decisive run).
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
RUN_NAME="runpod-a100-v23ctl-off-25x4h-$(date -u +%Y%m%d-%H%M%S)"
EXPERIMENT_DIR="/workspace/atlas_runpod_runs/${RUN_NAME}"

echo "=== [1/8] GPU check ==="
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

echo "=== [2/8] Clone repo at the EXACT v23 commit (7265cbd) ==="
# THE CONTROL RUN: this is the checkpoint-only submission candidate v23,
# byte-identical to the Kaggle dataset it was built from (7265cbd,
# planforce+rollbackfix). NO probes, NO plan_real, NO round-3/4/5 code --
# the point is a matched-duration baseline against the probe-branch runs
# (user, 28.08: after the 55x55 rehearsal, v23 at concurrency 25, 4h/game,
# 4h total -- the exact wave shape of the decisive probe run's first 4h).
rm -rf "${REPO_DIR}"
git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${REPO_DIR}"
git -C "${REPO_DIR}" checkout --quiet 7265cbd \
  || { echo "FATAL: commit 7265cbd not found in the clone" >&2; exit 1; }
ATLAS_SRC="${REPO_DIR}/atlas_src/src"
test -f "${ATLAS_SRC}/tufa-arc-agi-framework/pyproject.toml" || { echo "FATAL: clone incomplete" >&2; exit 1; }
test -f "${ATLAS_SRC}/ARC3-Inference/pyproject.toml" || { echo "FATAL: clone incomplete" >&2; exit 1; }
# v23 MUST have the plan-force override and MUST NOT have any probe code --
# the inverse of the probe-run checks, same stale-code paranoia.
grep -q "ATLAS_PLAN_FORCE_OVERRIDE" "${ATLAS_SRC}/ARC3-Inference/inference/agent/prompts.py" \
  || { echo "FATAL: checked-out code is missing v23's plan-force override -- wrong commit?" >&2; exit 1; }
if grep -q "_atlas_search_real_plan" "${ATLAS_SRC}/ARC3-Inference/inference/agent/tool_agent.py"; then
  echo "FATAL: checked-out code CONTAINS plan_real -- this is not v23 (7265cbd), aborting the control." >&2
  exit 1
fi
echo "Checked-out code is v23 (plan-force present, probe stack absent) -- confirmed."

# Bug #1 (26.08): BOTH ARC3-Inference AND tufa-arc-agi-framework pin
# requires-python=="3.12.12" exactly (missed the second one on the first
# attempt -- the uv error only ever names whichever package it evaluates
# first). uv's own standalone-interpreter catalog doesn't carry that exact
# patch (confirmed live: "No interpreter found for Python 3.12.12"). The pin
# isn't actually load-bearing for correctness (whichever 3.12.x uv provides
# works fine, per the earlier Colab run on 3.12.14) -- relax both in this
# cloned copy only, never touching the committed files.
sed -i 's/requires-python = "==3.12.12"/requires-python = ">=3.12,<3.13"/' \
  "${ATLAS_SRC}/ARC3-Inference/pyproject.toml" \
  "${ATLAS_SRC}/tufa-arc-agi-framework/pyproject.toml"
grep -n "requires-python" "${ATLAS_SRC}/ARC3-Inference/pyproject.toml" "${ATLAS_SRC}/tufa-arc-agi-framework/pyproject.toml"

echo "=== [3/8] Python 3.12 venv via uv (base-image-agnostic) ==="
if ! command -v uv >/dev/null 2>&1; then
  pip install -q uv || pip install -q --break-system-packages uv
fi
# Volume-reuse guard (28.08): when the attached network volume already
# carries a working venv (today's volume lhzzwir24g does), skip the
# venv+deps rebuild entirely -- the editable installs point at the same
# /workspace/ARC-AGI-3 path the fresh clone just landed on, so the newly
# cloned code is what actually runs. This turns a warm start into ~2 min.
VENV_PYTHON="${VENV_DIR}/bin/python"
if [ -x "${VENV_DIR}/bin/inference-taaf-run" ] && "${VENV_PYTHON}" -c "import vllm, taaf, inference" 2>/dev/null; then
  echo "Existing venv on the volume passes the import check -- skipping venv+deps rebuild."
  SKIP_DEPS=1
else
  SKIP_DEPS=0
  # Bug #2 (26.08): `uv venv --python 3.12.12` (the exact patch) fails outright
  # -- use the generic `3.12` selector instead and let uv resolve whichever
  # patch its catalog has for this host.
  uv venv --python 3.12 --clear "${VENV_DIR}"
fi
"${VENV_PYTHON}" --version

echo "=== [4/8] Install deps (torch/vllm pinned, flashinfer best-effort, harness editable) ==="
export PIP_CACHE_DIR=/root/.cache/pip   # never on a network volume -- can wedge in D-state
# Bug #3 (26.08): uv caches resolved metadata for local/editable path deps
# across script re-runs on the SAME pod -- confirmed live: after
# sed-relaxing ARC3-Inference's requires-python above, uv still resolved the
# OLD ==3.12.12 pin from a previous attempt's cache and failed the same way.
# Clear it every run so a locally-patched pyproject.toml is always honored.
# 28.08 FIX (caught live): the venv is built with UV_LINK_MODE=symlink,
# so its packages POINT INTO /root/.cache/uv. Wiping the cache while
# REUSING the venv (SKIP_DEPS=1) deletes the symlink targets and the
# reused venv rots (vllm import passed the step-3 check, then the wipe
# broke it). Only wipe the cache when actually rebuilding.
if [ "${SKIP_DEPS}" = "0" ]; then rm -rf /root/.cache/uv; fi
# Bug #4 (27.08): uv's cache (/root/.cache/uv, local disk) and VENV_DIR
# (/workspace, network volume) are on DIFFERENT filesystems, so uv's default
# hardlink install mode silently falls back to a full byte-for-byte copy for
# every file -- confirmed live: torch alone ships thousands of tiny header
# files under include/ATen/ops/, and flashinfer's cubins/ has many small
# binaries too, so the fallback copy crawled at ~7KB/s on this network mount
# (each small file costs a full round-trip). The fix is NOT to move the
# cache onto /workspace to enable hardlinking -- that would put the cache on
# the same network mount PIP_CACHE_DIR above is deliberately kept OFF of
# (see the comment there: a stalled network mount wedges an open cache
# handle in unkillable D-state I/O wait). Symlinks are the actual fix: they
# work ACROSS filesystems (unlike hardlinks) and are a metadata-only
# operation (unlike copy) -- cache stays safely on local disk, venv gets
# instant symlinks into it.
export UV_LINK_MODE=symlink
if [ "${SKIP_DEPS}" = "0" ]; then
  uv pip install --python "${VENV_PYTHON}" "torch==2.10.0" "vllm==0.19.0"
  uv pip install --python "${VENV_PYTHON}" flashinfer || echo "flashinfer install failed -- continuing without it (Marlin FP8 path still works on Ampere)"
  uv pip install --python "${VENV_PYTHON}" -e "${ATLAS_SRC}/tufa-arc-agi-framework" -e "${ATLAS_SRC}/ARC3-Inference"
else
  echo "(deps skipped -- reusing the volume's venv)"
fi
"${VENV_PYTHON}" -c "
import importlib.util as u
for m in ('vllm','torch','arc_agi','taaf','inference'):
    print(m, '->', 'OK' if u.find_spec(m) else 'MISSING')
"

echo "=== [5/8] Download the model via kaggle CLI (access_token must already be at ~/.kaggle/access_token) ==="
# Volume-reuse guard (28.08, the 27.08 redundant-23.7GB-download mistake):
# skip the download when the volume already has the model.
if [ -n "$(find "${MODEL_DIR}" -maxdepth 3 -iname 'config.json' 2>/dev/null | head -1)" ]; then
  echo "Model already on the volume -- skipping download."
else
  uv pip install --python "${VENV_PYTHON}" kaggle
  mkdir -p "${MODEL_DIR}"
  "${VENV_DIR}/bin/kaggle" models instances versions download "${MODEL_HANDLE}" -p "${MODEL_DIR}" --untar
fi
# --untar's extraction layout (flat vs one nested dir) isn't hardcoded here --
# find the real model dir by locating config.json, wherever it landed.
CONFIG_PATH=$(find "${MODEL_DIR}" -maxdepth 3 -iname "config.json" | head -1)
if [ -z "${CONFIG_PATH}" ]; then
  echo "FATAL: model download looks incomplete (no config.json found under ${MODEL_DIR})" >&2
  find "${MODEL_DIR}" -maxdepth 3 >&2
  exit 1
fi
MODEL_DIR=$(dirname "${CONFIG_PATH}")
echo "Resolved model dir: ${MODEL_DIR}"
find "${MODEL_DIR}" -maxdepth 1 -iname "*.safetensors" | wc -l

echo "=== [6/8] Download competition data -> OFFLINE engine (MANDATORY since 27.08) ==="
# Without --environments-dir the runner silently falls back to ONLINE
# (three.arcprize.org): no local engine object, so checkpoint_env returns
# None and the ENTIRE snapshot stack (rollback, sys anchors, try_actions/
# plan_real, plan_real principle-force) is silently dead -- this invalidated
# a whole day of 27.08 experiments before it was caught. Offline is also
# faster and immune to dead-session retry storms.
COMP_DIR="/workspace/comp"
ENV_FILES_DIR="${COMP_DIR}/environment_files"
if [ ! -d "${ENV_FILES_DIR}" ] || [ -z "$(ls "${ENV_FILES_DIR}" 2>/dev/null)" ]; then
  mkdir -p "${COMP_DIR}"
  "${VENV_DIR}/bin/kaggle" competitions download -c arc-prize-2026-arc-agi-3 -p "${COMP_DIR}"
  ( cd "${COMP_DIR}" && { unzip -oq arc-prize-2026-arc-agi-3.zip \
      || "${VENV_PYTHON}" -m zipfile -e arc-prize-2026-arc-agi-3.zip .; } )
fi
# NOTE: environment_files/ holds one SUBDIRECTORY per game (ar25/, bp35/,
# ...), not flat .py files. The first version of this check globbed *.py --
# under set -e/pipefail the failed glob killed the whole script SILENTLY
# right here (28.08, caught live). Count entries with find, which exits 0
# either way.
ENV_FILE_COUNT=$(find "${ENV_FILES_DIR}" -mindepth 1 -maxdepth 1 | wc -l)
echo "environment_files: ${ENV_FILE_COUNT} game entry(ies)"
if [ "${ENV_FILE_COUNT}" -lt 25 ]; then
  echo "FATAL: expected 25 game entries in ${ENV_FILES_DIR}, found ${ENV_FILE_COUNT} -- offline mode would silently degrade to ONLINE. Aborting." >&2
  exit 1
fi

echo "=== [7/8] Start vLLM (real settings, no Colab-style compromises -- 80GB has headroom) ==="
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

echo "Waiting for vLLM to become ready (up to 15 min -- CUDA-graph capture included this time)..."
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

echo "=== [8/8] V23 CONTROL: 25 official games OFFLINE, concurrency=25, 4h/game, one wave ==="
# 25 games / concurrency 25 = ONE wave: every game gets the full 4h --
# matched to the probe-branch decisive run for a fair baseline.


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
# ARC_API_KEY intentionally unset -- irrelevant in OFFLINE mode (the local
# ArcadeSpec engine never talks to three.arcprize.org).

cd "${ATLAS_SRC}/ARC3-Inference"
# Do NOT let set -e/pipefail abort the script on a nonzero exit here -- the
# results-retrieval reminder below must print regardless of how the run
# itself ended (a game-level failure is not the same as "nothing to save").
set +e
"${VENV_DIR}/bin/inference-taaf-run" \
  --include-tags official \
  --agent inference \
  --model "${SERVED_MODEL_NAME}" \
  --analyzer-timeout 480 \
  --deployment-target inline \
  --concurrent-jobs 25 \
  --n-passes 1 \
  --max-runtime-minutes 240 \
  --max-experiment-runtime-hours 4 \
  --environments-dir "${ENV_FILES_DIR}" \
  --run-name "${RUN_NAME}" \
  --experiment-dir "${EXPERIMENT_DIR}" \
  2>&1 | tee "/workspace/${RUN_NAME}.log"
RUN_EXIT=${PIPESTATUS[0]}
set -e
ANCHOR_COUNT=$(grep -c 'auto-anchor created (sys_start)' "/workspace/${RUN_NAME}.log" 2>/dev/null || true)
echo "OFFLINE check: ${ANCHOR_COUNT:-0} sys_start auto-anchor line(s) in the run log (0 would mean the snapshot stack was DEAD -- treat the run as invalid)."
echo "=== DONE (inference-taaf-run exit code: ${RUN_EXIT}). Results in ${EXPERIMENT_DIR} and /workspace/${RUN_NAME}.log ==="
echo "scp these off NOW, before stopping/terminating the pod:"
echo "  ${EXPERIMENT_DIR}"
echo "  /workspace/${RUN_NAME}.log"
echo "  ${VLLM_LOG}"
echo ""
echo "Reminder: this pod likely has a SEPARATE Network Volume resource that"
echo "survives Terminate and keeps billing standalone -- delete it manually"
echo "via the RunPod web UI (Storage -> Network Volumes) after scp'ing results."

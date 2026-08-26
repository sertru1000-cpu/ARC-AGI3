#!/usr/bin/env bash
# Reproduce the real v1 atlas_src code + settings (the 22.08 run that scored
# 4.99) for 2h, on the SAME already-running pod/vLLM instance as the v20
# concurrency=20 test -- to check whether v1's high score reproduces under a
# fresh random sample, or was mostly run-to-run RHAE variance.
#
# v1 = git commit 19f5116 (the first atlas_src fork commit, before
# nudge/A4/memo/invariants were added in 8c83fcf) + its real operational
# settings: concurrency=14 (not our current 20), analyzer_timeout=180s (the
# actual historical value -- known to cause heavy retry-thrash, that IS part
# of what "v1" means, not a mistake to fix here), LOCAL_ANALYZER_MAX_OUTPUT=0
# (unbounded -- the real root cause of those timeouts, also intentionally
# reproduced).
#
# Waits for the CURRENT inference-taaf-run (v20 test) to finish on its own
# before starting -- do not kill it early. Reuses the already-loaded vLLM
# server (same model, same port) -- no re-download, no re-warmup.
set -uo pipefail

V1_COMMIT="19f5116"
REPO_DIR="/workspace/ARC-AGI-3"
VENV_DIR="/workspace/venv312"
VENV_PYTHON="${VENV_DIR}/bin/python"
RUN_NAME="runpod-v1-repro-$(date -u +%Y%m%d-%H%M%S)"
EXPERIMENT_DIR="/workspace/atlas_runpod_runs/${RUN_NAME}"

echo "=== [1/5] Waiting for the current v20 test (inference-taaf-run) to finish on its own ==="
while pgrep -f 'bin/inference-taaf-run' >/dev/null 2>&1; do
  sleep 30
done
echo "v20 test finished. Proceeding."

echo "=== [2/5] Reset atlas_src to the real v1 commit (${V1_COMMIT}) ==="
cd "${REPO_DIR}"
# The v20 script's clone used --depth 1 -- only the latest commit exists
# locally, so 19f5116 (much earlier) is unreachable without full history.
git fetch --unshallow origin 2>&1 || git fetch origin 2>&1
git rev-parse --verify "${V1_COMMIT}^{commit}" >/dev/null 2>&1 || { echo "FATAL: commit ${V1_COMMIT} not found even after unshallow fetch" >&2; exit 1; }
git checkout "${V1_COMMIT}" -- atlas_src
ATLAS_SRC="${REPO_DIR}/atlas_src/src"
# Content-based verification -- `git log` on the current (unchanged) HEAD
# would misleadingly show the latest main-branch commit regardless of what
# `checkout -- atlas_src` just staged, since we never moved HEAD.
if grep -q "ATLAS_MEMO_CHECKPOINT\|_ATLAS_THEORY_NAG_AFTER_CALLS" "${ATLAS_SRC}/ARC3-Inference/inference/agent/tool_agent.py" 2>/dev/null; then
  echo "FATAL: reverted tool_agent.py still contains post-v1 nudge code -- checkout did not take effect" >&2
  exit 1
fi
echo "Confirmed: tool_agent.py has no nudge/memo-checkpoint code (real v1 state)."

# Same infra-only fix as the v20 script -- irrelevant to what "v1" means,
# just needed so uv's Python resolves at all on this pod (confirmed same
# exact ==3.12.12 pin exists this far back too).
sed -i 's/requires-python = "==3.12.12"/requires-python = ">=3.12,<3.13"/' \
  "${ATLAS_SRC}/ARC3-Inference/pyproject.toml" \
  "${ATLAS_SRC}/tufa-arc-agi-framework/pyproject.toml" 2>/dev/null || true
grep -n "requires-python" "${ATLAS_SRC}/ARC3-Inference/pyproject.toml" "${ATLAS_SRC}/tufa-arc-agi-framework/pyproject.toml"

echo "=== [3/5] Re-point the editable install at the v1 code (torch/vllm/flashinfer untouched) ==="
rm -rf /root/.cache/uv
uv pip install --python "${VENV_PYTHON}" -e "${ATLAS_SRC}/tufa-arc-agi-framework" -e "${ATLAS_SRC}/ARC3-Inference"
"${VENV_PYTHON}" -c "
import importlib.util as u
for m in ('vllm','torch','arc_agi','taaf','inference'):
    print(m, '->', 'OK' if u.find_spec(m) else 'MISSING')
"

echo "=== [4/5] Confirm vLLM is still up (reused from the v20 test, not restarted) ==="
curl -sf http://127.0.0.1:1234/v1/models >/dev/null 2>&1 && echo "vLLM alive, reusing it." || { echo "FATAL: vLLM is not responding -- cannot reuse it, would need a full restart"; exit 1; }

echo "=== [5/5] Run 25 official games, v1 settings: concurrency=14, analyzer-timeout=180s, unbounded max-output, 2h budget ==="
mkdir -p "${EXPERIMENT_DIR}"
export MPLBACKEND=Agg
export LOCAL_ANALYZER_BASE_URL=http://127.0.0.1:1234/v1
export OPENAI_BASE_URL=http://127.0.0.1:1234/v1
export LOCAL_ANALYZER_PROVIDER=vllm
export OPENAI_PROVIDER=vllm
export LOCAL_ANALYZER_API_KEY=not-needed
export LOCAL_ANALYZER_MODEL_ID=Qwen/Qwen3.8-27B-FP8
export INFERENCE_ANALYZER_MODEL=Qwen/Qwen3.8-27B-FP8
export LOCAL_ANALYZER_APP_NAME='ARC3 Agent Harness'
export LOCAL_ANALYZER_CONTEXT_WINDOW=32768
# The actual v1 bug, reproduced on purpose: unbounded generation.
export LOCAL_ANALYZER_MAX_OUTPUT=0
export LOCAL_ANALYZER_TIMEOUT=180
export ANALYZER_TIMEOUT=180
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

cd "${ATLAS_SRC}/ARC3-Inference"
set +e
"${VENV_DIR}/bin/inference-taaf-run" \
  --include-tags official \
  --agent inference \
  --model Qwen/Qwen3.8-27B-FP8 \
  --analyzer-timeout 180 \
  --deployment-target inline \
  --concurrent-jobs 14 \
  --n-passes 1 \
  --max-runtime-minutes 60 \
  --max-experiment-runtime-hours 2 \
  --run-name "${RUN_NAME}" \
  --experiment-dir "${EXPERIMENT_DIR}" \
  2>&1 | tee "/workspace/${RUN_NAME}.log"
RUN_EXIT=${PIPESTATUS[0]}
set -e
echo "=== DONE (exit code: ${RUN_EXIT}). Results in ${EXPERIMENT_DIR} and /workspace/${RUN_NAME}.log ==="

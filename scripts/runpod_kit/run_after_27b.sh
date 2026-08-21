#!/usr/bin/env bash
# Queued follow-up for the 27B VL training (21.08): wait for its DONE marker,
# cancel the trainer's own self-stop wait, run the 27B stand (merge LoRA ->
# serve -> exam matrix), then stop the pod via the RunPod API.
#   RUNPOD_API_KEY=rpa_... POD_ID=... nohup bash run_after_27b.sh > /workspace/after27.log 2>&1 &
set -u
cd /workspace/runpod-kit
LOG=/workspace/full_vl27_console.log
echo "=== AFTER27 waiting for 27B DONE $(date -u +%H:%M:%S)"
until grep -qE "^=== DONE|^=== TRAIN EXIT [1-9]" "$LOG" 2>/dev/null; do sleep 60; done
if grep -qE "^=== TRAIN EXIT [1-9]" "$LOG"; then
  echo "=== AFTER27 training FAILED -> no stand; stopping pod"
else
  echo "=== AFTER27 training done $(date -u +%H:%M:%S); cancelling trainer self-stop wait"
  pkill -f "^bash run_full_vl.sh"; pkill -f "^bash run_chain_27.sh"; sleep 3
  cp /workspace/adapter_vl_final.tgz /workspace/adapter_vl27_final.tgz 2>/dev/null || true
  echo "=== AFTER27 stand start $(date -u +%H:%M:%S)"
  BASE_MODEL=Qwen/Qwen3.6-27B ADAPTER=/workspace/out_vl27/adapter_final MERGED=/workspace/student_vl27_merged \
    LABEL=student-vl-27b bash /workspace/stand-kit/serve_vl_and_run.sh > /workspace/stand_vl27_console.log 2>&1
  echo "=== AFTER27 stand exit $? $(date -u +%H:%M:%S)"
  pkill -f "vllm.entrypoints"; sleep 5
fi
echo "=== AFTER27 waiting up to 20 min for /workspace/FETCHED27 (VM pulls adapter + stand results)"
for i in $(seq 1 120); do [ -f /workspace/FETCHED27 ] && break; sleep 10; done
curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" -H "Content-Type: application/json" \
  -d "{\"query\":\"mutation { podStop(input: {podId: \\\"${POD_ID}\\\"}) { id desiredStatus } }\"}"
echo " === AFTER27 POD STOP REQUESTED $(date -u +%H:%M:%S)"

#!/usr/bin/env bash
# Tiny process controller for the pod, so ssh one-liners never have to carry
# pkill patterns (a pattern that matches the ssh command line kills the ssh
# shell itself -- happened three times on 21.08). Patterns are anchored to
# the process's own argv[0..1], so log tails / greps never match.
#   bash ctl.sh status   - what is running
#   bash ctl.sh killall  - stop chains, smokes, trainers
#   bash ctl.sh ab       - (re)start the VL A/B smoke chain (run_chain2.sh)
#   bash ctl.sh full     - start the full VL training (run_full_vl.sh)
set -u
cd /workspace/runpod-kit
case "${1:-status}" in
  status)
    ps -eo pid,etimes,args | grep -E "^ *[0-9]+ +[0-9]+ +(bash run_(chain2|chain|smoke|smoke_vl|full_vl)\.sh|python train(_vl)?\.py)" || echo "(nothing running)"
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
    ;;
  killall)
    pkill -f "^bash run_chain2.sh"; pkill -f "^bash run_chain.sh"; pkill -f "^bash run_full_vl.sh"
    pkill -f "^bash run_smoke_vl.sh"; pkill -f "^bash run_smoke.sh"
    pkill -f "^python train_vl.py"; pkill -f "^python train.py"
    sleep 4
    echo "after killall:"; bash "$0" status
    ;;
  ab)
    bash "$0" killall >/dev/null
    nohup bash run_chain2.sh > /workspace/chain2.log 2>&1 &
    sleep 2; echo "A/B chain started"; head -1 /workspace/chain2.log
    ;;
  full)
    bash "$0" killall >/dev/null
    nohup bash run_full_vl.sh > /workspace/full_vl_console.log 2>&1 &
    sleep 2; echo "full VL training started"
    ;;
  *) echo "unknown: $1"; exit 1;;
esac

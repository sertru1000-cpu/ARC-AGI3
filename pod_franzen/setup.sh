#!/usr/bin/env bash
# Подготовка пода RTX PRO 6000 под сервер Франзена. Запуск: nohup bash /workspace/pod_franzen/setup.sh > /workspace/setup.log 2>&1 &
set -euo pipefail
W=/workspace; mkdir -p $W/run $W/models $W/wheels
cd $W
echo "== 1. паспорт машины"; python3 $W/pod_franzen/machine_probe.py | tee $W/run/machine_probe.txt || true
nvidia-smi --query-gpu=name,driver_version,power.limit,memory.total --format=csv | tee -a $W/run/machine_probe.txt
free -g | tee -a $W/run/machine_probe.txt; df -h / $W | tee -a $W/run/machine_probe.txt
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
[ "$DRV" -ge 580 ] || { echo "ДРАЙВЕР $DRV < 580: колёса Франзена собраны под CUDA 13 — СТОП"; exit 1; }
pip install -q --break-system-packages uv kaggle "huggingface_hub[cli]" hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "== 2. колёса Pennyroyal (Kaggle dfranzen/pennyroyal-v253)"; export KAGGLE_API_TOKEN=$(cat /root/.kaggle/access_token)
[ -d $W/wheels/pennyroyal/wheels ] || { kaggle datasets download dfranzen/pennyroyal-v253 -p $W/wheels/pennyroyal --unzip; }
echo "== 3. модель Intel W4A16 и черновик albucino (HF)"
[ -f $W/models/intel/config.json ] || hf download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound --local-dir $W/models/intel
[ -d $W/models/albucino ] || hf download albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE --include "runtime/mtp-int4-g32/*" --local-dir $W/models/albucino
echo "== 4. python 3.12 для колёс cp312"; uv python install 3.12
echo "== готово: $(date)"; du -sh $W/models/* $W/wheels/*

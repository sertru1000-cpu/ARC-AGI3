#!/bin/bash
# Слово владельца 04.10 «собрать и после проверки поставить стенд»: пушит стенд events, как только у Kaggle
# освободится один из двух слотов GPU (сейчас заняты base4 и пробой TTT). Один успешный пуш — и выход.
cd "$(dirname "$0")/.."
LOG=runs/kaggle_push_events.log
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
for i in $(seq 1 600); do
  out=$(.venv/bin/kaggle kernels push -p kernels/graft_dfranzen_m2_events 2>&1 | tail -1)
  echo "$(TZ=Europe/Moscow date '+%d.%m %H:%M') $out" >> $LOG
  echo "$out" | grep -q "successfully pushed" && exit 0
  sleep 120
done

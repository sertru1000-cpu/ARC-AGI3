#!/bin/bash
# Сторож НА ПОДЕ: живёт на самой арендованной машине, поэтому работает и тогда, когда ноутбук выключен.
#
# Зачем именно на поде: сторож с мака (scripts/runpod_watch_and_kill.sh) молчит, если мак выключен,
# а под в это время жжёт $7.89/ч. Случай 23-24.09 стоил $4.70 -- повторять незачем.
#
# Что делает: ждёт, пока исчезнет процесс обучения, проверяет НАЛИЧИЕ ФАЙЛОВ адаптера (а не строку
# в логе -- лог врёт, файлы нет), пишет опись в /workspace/guard_report.txt и гасит под. По истечении
# предельного срока гасит БЕЗУСЛОВНО. Адаптер никуда не выгружается: сетевой том переживает удаление
# пода, файлы на нём остаются и забираются позже с любой машины.
#
# usage: bash pod_guard.sh <pod_id> <предельные_минуты>
set -uo pipefail
POD=$1; LIMIT=${2:-240}
KEY=$(cat /workspace/.runpod_key)
OUT=/workspace/out
REP=/workspace/guard_report.txt

kill_pod() {
  echo "$(date -u +%H:%MZ) гашу под $POD" | tee -a "$REP"
  curl -s -m 30 -X DELETE -H "Authorization: Bearer $KEY" \
    "https://rest.runpod.io/v1/pods/$POD" -w "  HTTP %{http_code}\n" | tee -a "$REP"
}

inventory() {   # опись результата: что именно осталось на томе
  { echo "--- опись $(date -u +%H:%MZ) ---"
    du -sh $OUT/lora_v1* 2>/dev/null
    ls -l $OUT/lora_v1*/adapter_model.safetensors 2>/dev/null
    tail -25 /workspace/train3.log | tr '\r' '\n' | grep -vE 'it/s\]$|^$'
  } >> "$REP" 2>&1
}

for i in $(seq 1 "$LIMIT"); do
  if ! pgrep -f "train_lora_bnb.py --no-quant" > /dev/null; then
    F=$(ls $OUT/lora_v1*/adapter_model.safetensors 2>/dev/null | wc -l)
    echo "$(date -u +%H:%MZ) обучение завершилось, файлов адаптера $F" | tee -a "$REP"
    inventory
    [ "$F" -eq 0 ] && echo "  ФАЙЛОВ АДАПТЕРА НЕТ -- обучение упало, гашу всё равно" | tee -a "$REP"
    kill_pod; exit 0
  fi
  sleep 60
done
echo "$(date -u +%H:%MZ) срок $LIMIT мин вышел -- гашу БЕЗУСЛОВНО" | tee -a "$REP"
inventory; kill_pod

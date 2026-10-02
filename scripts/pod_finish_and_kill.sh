#!/bin/bash
# Ждёт конца матрицы на поде и скачивает результаты. ПОД НЕ УДАЛЯЕТ (глобальное правило 02.10: удаление/остановка/аренда
# пода — только по прямой команде владельца; сторожей-убийц не ставить). Удаление делается вручную по слову.
# Запуск с мака так, чтобы пережить сессию: nohup bash scripts/pod_finish_and_kill.sh <ip> <port> <pod_id> <лог> <срок_UTC HH:MM> <куда> > log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
IP=$1; PORT=$2; POD=$3; LOGF=$4; DEADLINE=$5; DEST=$6
KEY=$(grep '^RUNPOD_API_KEY=' .env | cut -d= -f2-)
SSH="ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $PORT -i $HOME/.ssh/id_ed25519 root@$IP"

fetch() {
  mkdir -p "$DEST"
  scp -q -r -o ConnectTimeout=20 -P "$PORT" -i "$HOME/.ssh/id_ed25519" "root@$IP:/workspace/run/*" "$DEST/" \
    && echo "$(date +%H:%M) результаты скачаны в $DEST ($(du -sh "$DEST" | cut -f1))"
}
kill_pod() {   # отключено правилом 02.10: только сообщение, под не трогаем
  echo "$(date +%H:%M) под $POD НЕ удалён — ждёт команды владельца"
}

while :; do
  if $SSH "grep -q 'МАТРИЦА ГОТОВА' $LOGF" 2>/dev/null; then
    echo "$(date +%H:%M) матрица готова"; fetch; kill_pod; exit 0
  fi
  if [[ "$(date -u +%H:%M)" > "$DEADLINE" || "$(date -u +%H:%M)" == "$DEADLINE" ]]; then
    echo "$(date +%H:%M) срок $DEADLINE UTC вышел — качаю что есть; под НЕ трогаю"; fetch; kill_pod; exit 0
  fi
  sleep 60
done

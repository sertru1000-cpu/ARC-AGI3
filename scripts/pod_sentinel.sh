#!/bin/bash
# Сторож пода (НЕ зависит от сессии Claude): раз в 2 минуты проверяет, что на поде идёт работа, и громко сообщает
# о любом событии — конец прогона, ошибка в логе, простой карты, мало денег на счёте. Под НЕ трогает (правило 02.10).
# Сообщения — только строки в runs/pod_sentinel.log (владелец 04.10: «громко сообщать не надо»); читаю их сам.
# Запуск: nohup caffeinate -i bash scripts/pod_sentinel.sh <ip> <port> "<лог1> <лог2> ..." > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/.."
IP=$1; PORT=$2; LOGS=$3
OUT=runs/pod_sentinel.log
KEY=$(grep '^RUNPOD_API_KEY=' .env | cut -d= -f2-)
SSH="ssh -n -o ConnectTimeout=20 -o BatchMode=yes -p $PORT -i $HOME/.ssh/id_ed25519 root@$IP"
note() {
  echo "$(TZ=Europe/Moscow date '+%d.%m %H:%M') $*" >> "$OUT"
}
SEEN_DIR=$(mktemp -d)
idle=0
note "сторож запущен: $IP:$PORT, логи: $LOGS"
while :; do
  for L in $LOGS; do
    ev=$($SSH "grep -c 'ГОТОВ\|FAILED\|TIMEOUT\|Traceback\|Error:' $L 2>/dev/null" 2>/dev/null)
    [ -z "$ev" ] && continue
    key="$SEEN_DIR/$(echo "$L" | tr '/' '_')"
    prev=$(cat "$key" 2>/dev/null || echo "")
    if [ "$prev" != "$ev" ]; then
      last=$($SSH "grep 'ГОТОВ\|FAILED\|TIMEOUT\|Traceback\|Error:' $L | tail -1 | cut -c1-120" 2>/dev/null)
      { [ -n "$prev" ] || [ "$ev" != "0" ]; } && note "$(basename $L): $last"
      echo "$ev" > "$key"
    fi
  done
  util=$($SSH "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" 2>/dev/null | head -1)
  if [ -n "$util" ] && [ "$util" -lt 5 ]; then idle=$((idle+1)); else idle=0; fi
  [ "$idle" -eq 5 ] && note "карта простаивает 10 минут (загрузка ${util}%) — под оплачивается впустую"
  bal=$(curl -s -m 20 --noproxy '*' -H "Content-Type: application/json" -d '{"query":"query { myself { clientBalance currentSpendPerHr } }"}' "https://api.runpod.io/graphql?api_key=$KEY" | python3 -c "import json,sys;m=json.load(sys.stdin)['data']['myself'];print('%.1f'%(m['clientBalance']/max(m['currentSpendPerHr'],0.01)))" 2>/dev/null)
  if [ -n "$bal" ] && python3 -c "import sys;sys.exit(0 if float('$bal')<1.0 else 1)"; then
    note "денег на счёте RunPod меньше чем на час (${bal} ч)"
  fi
  sleep 120
done

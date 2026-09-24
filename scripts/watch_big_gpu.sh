#!/bin/bash
# Сторож крупных карт: ждёт появления B300 (288 ГБ) или двух H200 (282 ГБ) -- меньшего для обучения не хватает.
#
# Почему именно столько: эксперты боевой модели лежат СЛИТЫМ тензором (512,1280,2560) в собственном модуле
# Qwen4ExpTextExperts, и bitsandbytes их не сжимает (умеет только обычные линейные слои). Значит модель
# занимает ~258 ГБ в bf16, и одной H200 (141) или B200 (180) не хватает.
#
# usage: bash scripts/watch_big_gpu.sh [минут_между_проверками]
set -uo pipefail
GAP=${1:-10}
KEY=$(grep '^RUNPOD_API_KEY=' .env | cut -d= -f2-)
ask() {  # $1 = id карты, $2 = сколько карт
  curl -s -m 20 --noproxy '*' -X POST "https://api.runpod.io/graphql?api_key=$KEY" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"query { gpuTypes(input:{id:\\\"$1\\\"}) { lowestPrice(input:{gpuCount:$2}) { uninterruptablePrice stockStatus } } }\"}" \
  | python3 -c "
import json,sys
try:
    t=json.load(sys.stdin)['data']['gpuTypes']; p=(t[0].get('lowestPrice') or {}) if t else {}
    print('%s|%s' % (p.get('uninterruptablePrice'), p.get('stockStatus') or ''))
except Exception: print('|')"
}
while true; do
  # слово владельца 24.09: «сканируй только B300» -- одна карта на 288 ГБ проще и быстрее пары H200
  B3=$(ask "NVIDIA B300" 1)
  T=$(TZ=Europe/Moscow date +%H:%M)
  if [ -n "${B3#*|}" ]; then echo "$T НАШЛАСЬ B300: \$${B3%%|*}/час (${B3#*|}) -- можно обучать"; exit 0; fi
  echo "$T B300 нет"
  sleep $((GAP*60))
done

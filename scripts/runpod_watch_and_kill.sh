#!/bin/bash
# Сторож пода RunPod: гасит под, как только работа сделана, и ГАСИТ ВСЁ РАВНО по истечении срока.
#
# ЧТО СТОИЛО ДЕНЕГ (23→24.09, ~$4.70): первая версия ждала в логе метку "ГОТОВО". Скрипт скачивания
# умер на шаге сверки, метку не дописал, и под простоял девять часов впустую. Два вывода, оба здесь:
#   1. Проверять НАЛИЧИЕ РЕЗУЛЬТАТА (опись файлов), а не строку в логе -- лог врёт, файлы нет.
#   2. По истечении срока гасить БЕЗУСЛОВНО: сетевой том переживает удаление пода, скачанное остаётся,
#      новый под продолжит с того же места. Оставлять живой под "на всякий случай" -- чистый убыток.
#
# usage: bash scripts/runpod_watch_and_kill.sh <ip> <port> <pod_id> [предельные_минуты]
set -uo pipefail
IP=$1; PORT=$2; POD=$3; LIMIT=${4:-120}
KEY=$(grep '^RUNPOD_API_KEY=' .env | cut -d= -f2-)
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $PORT -i $HOME/.ssh/id_ed25519 root@$IP"

kill_pod() {   # ЛОВУШКА: API RunPod с этого мака доступен только в обход прокси
  curl -s -m 30 --noproxy '*' -X DELETE -H "Authorization: Bearer $KEY" \
    "https://rest.runpod.io/v1/pods/$POD" -w "  удаление пода: HTTP %{http_code}\n"
}

for i in $(seq 1 "$LIMIT"); do
  OUT=$($SSH '/workspace/venv312/bin/python - <<PY 2>/dev/null || python3 - <<PY
import json, os
d="/workspace/model"
try:
    idx=json.load(open(os.path.join(d,"model.safetensors.index.json")))
except Exception:
    print("NOINDEX"); raise SystemExit
need=sorted(set(idx["weight_map"].values()))
miss=[f for f in need if not os.path.exists(os.path.join(d,f))]
print("MISS=%d TOTAL=%d" % (len(miss), len(need)))
PY' 2>/dev/null | tail -1)
  echo "$(date +%H:%M) минута $i: $OUT"
  if [[ "$OUT" == MISS=0* ]]; then
    echo "  всё на месте -- гашу под"; kill_pod; exit 0
  fi
  sleep 60
done
echo "$(date +%H:%M) срок $LIMIT мин вышел -- гашу БЕЗУСЛОВНО (том и скачанное сохраняются)"
kill_pod

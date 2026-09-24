#!/bin/bash
# Сторож датацентра EU-NL-1: там лежит наш сетевой том с весами (450 ГБ, giant_silver_junglefowl),
# а том увезти нельзя -- он привязан к датацентру. Поэтому ждём ЛЮБУЮ карту именно здесь.
#
# 24.09, после гашения B300: в EU-NL-1 не оказалось НИ ОДНОЙ свободной карты из 45 типов -- ни дешёвых
# (A4000/A5000/L4/A40/L40S/A6000), ни A100/H100/H200/B200, ни даже MIG-раздела B300 на 34 ГБ.
# Процессорных машин тоже нет ни в одной из шести конфигураций.
#
# Пробует создание НА САМОМ ДЕЛЕ (а не запрос цены): наличие в gpuTypes не означает наличия в нужном
# датацентре с нужным томом. Подняв под, СРАЗУ выходит и печатает id -- гасить его нельзя без команды
# владельца (см. память feedback-never-kill-pod-without-command).
#
# usage: bash scripts/watch_eu_nl_1.sh [секунд_между_кругами]
set -o pipefail   # без -u: bash 3.2 в macOS падает на "${МАССИВ[@]}" при set -u
GAP=${1:-60}
KEY=$(grep '^RUNPOD_API_KEY=' .env | cut -d= -f2-)
PUB=$(cat "$HOME/.ssh/id_ed25519.pub")
VOL=5feahiplpp

# от дешёвых к дорогим; список строками, а не массивом -- см. про bash 3.2 выше
CARDS_LIST=$(cat <<'EOF'
NVIDIA RTX A4000
NVIDIA RTX A4500
NVIDIA RTX A5000
NVIDIA L4
NVIDIA RTX 4000 Ada Generation
NVIDIA A40
NVIDIA L40S
NVIDIA RTX A6000
NVIDIA A100 80GB PCIe
NVIDIA A100-SXM4-80GB
NVIDIA H100 PCIe
NVIDIA H100 NVL
NVIDIA H100 80GB HBM3
NVIDIA H200 NVL
NVIDIA H200
NVIDIA B300 SXM6 AC MIG 1g.34gb
NVIDIA B200
NVIDIA B300 SXM6 AC
EOF
)

try() {
  python3 - "$PUB" "$1" "$VOL" > /tmp/eu_nl_req.json <<'PY'
import json, sys
print(json.dumps({
  "name": "eu-nl-probe",
  "imageName": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
  "gpuTypeIds": [sys.argv[2]], "gpuCount": 1, "gpuTypePriority": "custom",
  "dataCenterIds": ["EU-NL-1"], "dataCenterPriority": "custom",
  "networkVolumeId": sys.argv[3], "volumeMountPath": "/workspace",
  "containerDiskInGb": 30, "ports": ["22/tcp"],
  "dockerEntrypoint": ["sleep"], "dockerStartCmd": ["infinity"],
  "env": {"PUBLIC_KEY": sys.argv[1]}}))
PY
  curl -s -m 60 --noproxy '*' -X POST -H "Authorization: Bearer $KEY" \
    -H "Content-Type: application/json" -d @/tmp/eu_nl_req.json https://rest.runpod.io/v1/pods
}

round=0
while true; do
  round=$((round+1))
  while IFS= read -r c; do
    [ -z "$c" ] && continue
    R=$(try "$c")
    case "$R" in
      *'"id"'*)
        echo "$(TZ=Europe/Moscow date +%H:%M) ПОДНЯЛСЯ под на «$c»:"
        echo "$R" | python3 -c "import sys,json;d=json.load(sys.stdin);print(' id=%s  цена=%s/ч' % (d['id'], d.get('costPerHr')))"
        exit 0;;
    esac
  done <<< "$CARDS_LIST"
  echo "$(TZ=Europe/Moscow date +%H:%M) круг $round: в EU-NL-1 свободных карт нет"
  sleep "$GAP"
done

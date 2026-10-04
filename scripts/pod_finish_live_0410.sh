#!/bin/bash
# По прямой команде владельца 04.10 ~16:40 «гаси сразу после окончания» (живой стенд base4 на поде).
# Удаляет под ТОЛЬКО если стенд закончился УСПЕШНО: в логе «ЖИВОЙ СТЕНД ГОТОВ ... код 0», нет Traceback,
# результаты скачаны на мак. Если стенд упал или итог неясен — под НЕ трогает (правило ~/.claude/CLAUDE.md:
# при изменившейся ситуации — спросить владельца) и пишет об этом в runs/pod_sentinel.log.
set -u
cd "$(dirname "$0")/.."
IP=81.27.69.178; PORT=30824; POD=aa3vk551mnpbf4
SSH="ssh -n -o ConnectTimeout=20 -o BatchMode=yes -p $PORT -i $HOME/.ssh/id_ed25519 root@$IP"
LOG=runs/pod_sentinel.log
note() { echo "$(TZ=Europe/Moscow date '+%d.%m %H:%M') $*" >> "$LOG"; }
note "ожидание конца живого стенда base4 (удаление пода только при успехе)"
until $SSH "grep -q 'ЖИВОЙ СТЕНД ГОТОВ' /workspace/kin/live_base4.log" 2>/dev/null; do sleep 60; done
END=$($SSH "grep 'ЖИВОЙ СТЕНД ГОТОВ' /workspace/kin/live_base4.log | tail -1")
TB=$($SSH "grep -c Traceback /workspace/kin/live_base4.log")
mkdir -p runs/pod_0410_live
scp -q -r -o ConnectTimeout=20 -P $PORT -i $HOME/.ssh/id_ed25519 root@$IP:/kaggle/working/summary.txt root@$IP:/kaggle/working/benchmark.json root@$IP:/workspace/kin/live_base4.log root@$IP:/workspace/run runs/pod_0410_live/ 2>/dev/null
if echo "$END" | grep -q "код 0" && [ "$TB" = "0" ] && [ -s runs/pod_0410_live/summary.txt ]; then
  KEY=$(grep '^RUNPOD_API_KEY=' .env | cut -d= -f2-)
  R=$(curl -s -m 30 --noproxy '*' -X DELETE -H "Authorization: Bearer $KEY" "https://rest.runpod.io/v1/pods/$POD" -w "HTTP %{http_code}")
  note "живой стенд закончен успешно ($END), результаты в runs/pod_0410_live; под удалён по команде владельца: $R"
  pkill -f "[p]od_sentinel.sh"
else
  note "живой стенд закончился НЕУСПЕШНО или итог неясен ($END, Traceback=$TB) — под НЕ удалён, нужен владелец"
fi

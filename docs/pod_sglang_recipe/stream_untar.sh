#!/usr/bin/env bash
# Веса без архива на диске: раздача Kaggle публичная и отдаёт куски, поэтому поток
# сразу уходит в распаковку. Пик занятости тома = только распакованные веса.
# Запускать, ЕСЛИ том расширить не удалось: архив при этом удаляется.
set -euo pipefail
URL="https://www.kaggle.com/api/v1/models/keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1/download"
D=/workspace/model
pkill -f "kaggle models instances versions download" || true
sleep 3
rm -f "$D"/*.tar.gz "$D"/*.kaggle-partial
mkdir -p "$D"
echo "=== $(date -u +%H:%M:%S) поток пошёл"
curl -sL --retry 5 --retry-delay 10 "$URL" | tar -xzf - -C "$D"
echo "=== $(date -u +%H:%M:%S) распаковано, файлов: $(ls "$D" | wc -l)"
test -f "$D/config.json" || { echo "FATAL: нет config.json" >&2; exit 1; }
du -sh "$D"

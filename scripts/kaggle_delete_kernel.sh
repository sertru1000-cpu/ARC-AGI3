#!/bin/bash
# Удаление кернела Kaggle. Узкая обёртка: берёт токен проекта и удаляет НАЗВАННЫЙ кернел.
# Защита: работают только слаги, начинающиеся с sergueimakarov/arc3- — чужие и не-ARC
# ноутбуки владельца этой командой не удалить.
set -euo pipefail
cd "$(dirname "$0")/.."
for slug in "$@"; do
  case "$slug" in
    sergueimakarov/arc3-*) ;;
    *) echo "ОТКАЗ: $slug — не кернел ARC этого проекта"; exit 1 ;;
  esac
done
for slug in "$@"; do
  printf '%-40s ' "$slug"
  KAGGLE_API_TOKEN=$(cat .kaggle/access_token) .venv/bin/kaggle kernels delete "$slug" --yes 2>&1 | tail -1
done

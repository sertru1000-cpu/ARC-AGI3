#!/usr/bin/env bash
# Сборка SGLang с поддержкой qwen4_exp: в выпусках PyPI её нет (проверены 0.5.17-0.5.19),
# нужна ветка qwen4-main-squashed (коммит 4ccff141db) — в ней загрузчик ModelOpt
# MIXED_PRECISION (PR 38121), файловая/закреплённая таблица PLE (PR 37068) и правки роутера.
set -euo pipefail
export UV_LINK_MODE=symlink UV_CACHE_DIR=/root/.cache/uv
# расширения на Rust — это отдельный маршрутизатор, серверу они не нужны, а cargo в образе нет
export SGLANG_BUILD_RUST_EXTS=none
SRC=/workspace/sglang_src; V=/workspace/venv_q4
step() { echo "=== [$(date -u +%H:%M:%S)] $*"; }
step "клон"
if [ ! -d "$SRC/.git" ]; then git clone -q https://github.com/sgl-project/sglang.git "$SRC"; fi
cd "$SRC"; git fetch -q origin 4ccff141db 2>/dev/null || git fetch -q origin
git checkout -q 4ccff141db
step "коммит: $(git rev-parse --short HEAD)"
test -f python/sglang/srt/models/qwen4_exp.py && echo "qwen4_exp.py НА МЕСТЕ" || { echo "FATAL: нет qwen4_exp.py"; exit 1; }
step "окружение"
uv venv "$V" --python 3.12
step "установка (долго)"
uv pip install --python "$V/bin/python" -e "$SRC/python"
step "проверка"
"$V/bin/python" -c "import sglang; print(\"sglang\", sglang.__version__); from sglang.srt.models import qwen4_exp; print(\"qwen4_exp импортируется\")"
step "ГОТОВО"

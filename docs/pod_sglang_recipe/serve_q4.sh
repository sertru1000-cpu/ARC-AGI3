#!/usr/bin/env bash
# Запуск Qwen3.8-Flash-Next-NVFP4 на ОДНОЙ RTX PRO 6000 (96 ГБ).
#
# Флаги взяты из проверенной ячейки поваренной книги SGLang для этой карты и этого
# экспорта весов (NVFP4 RDXA, высокая пропускная способность). Главный из них —
# --ple-offload-embedding: 47,7 ГиБ FP8-таблицы N-грамм уходят в закреплённую память
# хозяина, иначе 126 ГиБ чекпойнта на 96 ГБ карты не ложатся. Остальные 78 ГиБ — на карту.
#
# Разбор вызовов инструмента включён намеренно: наш синтезатор просит программу вызовом
# инструмента, и без разборщика ответы придут неструктурированной строкой.
set -euo pipefail
V=/workspace/venv_q4; MODEL_DIR=/workspace/model; PORT="${PORT:-8000}"
SERVED="${SERVED_NAME:-Qwen/Qwen3.8-Flash-Next-NVFP4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1
# libz3 приезжает питоновским пакетом z3-solver и лежит внутри venv, а грузит её через
# ctypes библиотека tilelang (её тянет индексатор QSA). Загрузчик про эту папку не знает,
# поэтому сервер падал на импорте. Показываем путь явно.
export LD_LIBRARY_PATH="$V/lib/python3.12/site-packages/z3/lib:${LD_LIBRARY_PATH:-}"
# Ядра собираются на лету: нужен ninja (ставится в окружение) и nvcc. Системного nvcc в
# образе нет, но он приезжает вместе с колёсами CUDA внутри venv — показываем оба явно,
# иначе планировщик падает на FileNotFoundError: ninja сразу после загрузки весов.
export CUDA_HOME="$V/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
nohup "$V/bin/python" -m sglang.launch_server \
  --model-path "$MODEL_DIR" --served-model-name "$SERVED" \
  --tp 1 --quantization modelopt_fp4 \
  --fp4-gemm-backend flashinfer_cutlass --moe-runner-backend flashinfer_cutlass \
  --ple-offload-embedding \
  --page-size 64 --mamba-track-interval 64 --chunked-prefill-size 4096 \
  --context-length 32768 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-running-requests 64 --max-mamba-cache-size 192 --mamba-ssm-dtype bfloat16 \
  --reasoning-parser qwen3 --tool-call-parser qwen25 \
  --mem-fraction-static 0.93 \
  --host 127.0.0.1 --port "$PORT" > /workspace/sglang_q4.log 2>&1 &
echo "сервер запущен, жду здоровья (до 40 мин: 78 ГиБ на карту плюс 47,7 ГиБ в закреплённую память)"
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "=== [$(date -u +%H:%M:%S)] СЕРВЕР ПОДНЯЛСЯ за $((i*10)) с"; exit 0; fi
  pgrep -f sglang.launch_server >/dev/null || { echo "=== СЕРВЕР УМЕР, хвост лога:"; tail -40 /workspace/sglang_q4.log; exit 1; }
  sleep 10
done
echo "=== НЕ ОТВЕТИЛ ЗА 40 МИН"; tail -40 /workspace/sglang_q4.log; exit 1

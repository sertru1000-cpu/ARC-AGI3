#!/usr/bin/env bash
# Пакетный синтез программ модели мира на арендованном поде.
#
# ЧТО ДЕЛАЕТ: ставит окружение, качает веса с HuggingFace, поднимает сервер vLLM, ждёт здоровья
# и запускает генерацию программ по готовым сводкам наблюдений. Игрового движка не требует.
#
# ПРАВИЛА, ВЫСТРАДАННЫЕ 26-27.08 (см. skill runpod-deploy и память arc-agi-3-runpod-lessons):
#   * UV_LINK_MODE=symlink ОБЯЗАТЕЛЕН: кэш uv и venv на разных файловых системах, иначе установка
#     torch ползёт часами вместо секунд;
#   * никакой отладки на счётчике: каждый шаг печатает время и падает громко;
#   * по окончании под ТЕРМИНИРОВАТЬ (не Stop), затем проверить осиротевшие тома.
#
# ПЕРЕМЕННЫЕ (задать перед запуском):
#   MODEL_ID      репозиторий HuggingFace с весами
#   SERVED_NAME   под каким именем сервер отдаёт модель
#   ATTEMPTS      попыток синтеза на игру (по умолчанию 20)
#   VLLM_EXTRA    дополнительные аргументы сервера
#
# usage (на поде):
#   export HF_TOKEN=...; export MODEL_ID=RadixArk/Qwen3.8-Flash-Next-NVFP4
#   nohup bash /root/runpod_synth_batch.sh > /root/deploy.log 2>&1 &
set -euo pipefail

MODEL_ID="${MODEL_ID:?нужен MODEL_ID}"
SERVED_NAME="${SERVED_NAME:-$MODEL_ID}"
ATTEMPTS="${ATTEMPTS:-20}"
WORK="${WORK:-/workspace}"
VENV="${WORK}/venv"
MODEL_DIR="${WORK}/model"
PORT="${PORT:-8000}"
# Для NVFP4 на Blackwell нужен свежий vLLM: августовский пин 0.19.0 этот формат не знает.
VLLM_SPEC="${VLLM_SPEC:-vllm}"        # без пина — берём последний
TORCH_SPEC="${TORCH_SPEC:-torch}"

step() { echo "=== [$(date +%H:%M:%S)] $*"; }

step "карта и диск"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h "${WORK}" | tail -1

step "окружение"
export DEBIAN_FRONTEND=noninteractive
command -v uv >/dev/null 2>&1 || pip install -q uv
uv venv "${VENV}" --python 3.12
export UV_LINK_MODE=symlink          # без этого установка torch идёт часами, а не секунды
export UV_CACHE_DIR=/root/.cache/uv  # кэш ДОЛЖЕН быть на локальном диске, не на сетевом томе
time uv pip install --python "${VENV}/bin/python" -q \
  "${TORCH_SPEC}" "${VLLM_SPEC}" "huggingface_hub[hf_transfer]" "kaggle"

step "веса ${MODEL_ID}"
# hf_transfer рвётся на сетевом томе («Background writer channel closed»), 09.09 — не включаем
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "${MODEL_DIR}"
time "${VENV}/bin/hf" download "${MODEL_ID}" --local-dir "${MODEL_DIR}" \
  --exclude "*.pth" --exclude "original/*" || {
    echo "FATAL: веса не скачались — проверь HF_TOKEN и доступ к репозиторию" >&2; exit 1; }
test -f "${MODEL_DIR}/config.json" || { echo "FATAL: нет config.json — выгрузка неполная" >&2; exit 1; }
du -sh "${MODEL_DIR}"

step "сервер vLLM"
nohup "${VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_DIR}" --served-model-name "${SERVED_NAME}" \
  --port "${PORT}" --max-model-len 32768 --max-num-seqs 32 \
  ${VLLM_EXTRA:-} > "${WORK}/vllm.log" 2>&1 &

step "ждём здоровья сервера (до 30 мин)"
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "сервер поднялся за $((i * 10)) с"; break
  fi
  if ! pgrep -f "vllm.entrypoints" >/dev/null; then
    echo "FATAL: сервер умер, последние строки лога:" >&2; tail -40 "${WORK}/vllm.log" >&2; exit 1
  fi
  sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || {
  echo "FATAL: сервер не ответил за 30 мин" >&2; tail -40 "${WORK}/vllm.log" >&2; exit 1; }

step "пробный запрос"
curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 300; echo

step "пакетный синтез: ${ATTEMPTS} попыток на игру"
time "${VENV}/bin/python" /root/wm_synth_generate.py \
  --digests /root/wm_digests.json --out "${WORK}/programs.json" \
  --attempts "${ATTEMPTS}" --base-url "http://127.0.0.1:${PORT}/v1" --model "${SERVED_NAME}"

step "готово. Забрать результат:"
echo "  scp -P <port> -i ~/.ssh/id_ed25519 root@<ip>:${WORK}/programs.json ."
echo "  затем локально: .venv/bin/python scripts/wm_eval_programs.py programs.json"
echo "ПОД ТЕРМИНИРОВАТЬ (не Stop), затем проверить осиротевшие тома."

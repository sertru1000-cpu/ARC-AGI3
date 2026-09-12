#!/usr/bin/env bash
# Полная Фаза A со слоем «модель мира» на арендованном поде: 25 публичных игр, 2,2 часа на игру.
#
# ЗАЧЕМ ОТДЕЛЬНО ОТ ПАКЕТНОГО СИНТЕЗА. Пакетный замер (runpod_synth_batch.sh) отвечает, какое
# правило приёма отбирает программы, по которым находится путь. Этот прогон отвечает на другой
# вопрос: окупается ли цепочка целиком — приём, цель, план, взятый уровень. Порядок именно такой,
# иначе повторится версия 13, где два часа измерили отсутствие материала.
#
# КАК ПОДКЛЮЧАЕТСЯ НАШ СЛОЙ. На Kaggle это ячейка ноутбука; здесь ноутбука нет, поэтому патч
# ставится через sitecustomize.py венва: любой процесс Python в этом окружении импортирует его
# при старте, и патчи ложатся на inference.agent.tool_agent до запуска движка. Хвост ячейки с
# настройками пробы вырезается — бюджеты на поде задаются флагами запуска.
#
# ПРАВИЛА ПОДА (skill runpod-deploy): UV_LINK_MODE=symlink обязателен; игры ТОЛЬКО офлайн через
# --environments-dir, иначе движка нет и половина механизмов молча мертва; по окончании под
# ТЕРМИНИРОВАТЬ, не Stop, и проверить осиротевшие тома.
#
# ПЕРЕМЕННЫЕ: MODEL_DIR (уже скачанные веса), SERVED_NAME, PATCH (общий патч),
#             CONC (конкурентность), CAP_MIN (минут на игру), EXP_MIN (общий предел),
#             WM_LAYER / WM_SCHED (что именно проверяем — ровно одно из двух).
#
# usage (на поде, после runpod_synth_batch.sh — окружение и веса уже на месте):
#   export PATCH=/root/wm_pod_patch.py SERVED_NAME=... ; nohup bash /root/runpod_phase_a_wm.sh > /root/phasea.log 2>&1 &
set -euo pipefail

WORK="${WORK:-/workspace}"
VENV="${VENV:-${WORK}/venv}"
MODEL_DIR="${MODEL_DIR:-${WORK}/model}"
SERVED_NAME="${SERVED_NAME:?нужно имя модели, под которым отдаёт сервер}"
PATCH="${PATCH:-/root/wm_pod_patch.py}"   # общий патч: модель мира + расписание
PORT="${PORT:-8000}"
# РОВНО ОДНО ИЗМЕНЕНИЕ ЗА ПРОГОН. Пакетный синтез решает, какое именно; смешивать нельзя,
# иначе результат не приписать причине. По умолчанию — расписание без слоя.
WM_LAYER="${WM_LAYER:-0}"   # 1 — слой «модель мира»
WM_SCHED="${WM_SCHED:-1}"   # 1 — снятие игры после 45 минут без уровня
export WM_LAYER WM_SCHED
CONC="${CONC:-18}"          # 18 игр одновременно: при 14 шесть игр из 25 не стартовали вовсе
# Потолок на игру равен окну: связывают только правило снятия и общий предел.
EXP_MIN="${EXP_MIN:-150}"
CAP_MIN="${CAP_MIN:-${EXP_MIN}}"
BUNDLE_REF="${BUNDLE_REF:-keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1}"
BUNDLE_DIR="${WORK}/bundle"
COMP_DIR="${WORK}/comp"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="wm_pod_${STAMP}"
EXP_DIR="${WORK}/runs/${RUN_NAME}"

step() { echo "=== [$(date +%H:%M:%S)] $*"; }

step "проверки перед стартом"
test -d "${MODEL_DIR}" || { echo "FATAL: нет весов в ${MODEL_DIR}" >&2; exit 1; }
test -f "${PATCH}" || { echo "FATAL: нет файла патча ${PATCH}" >&2; exit 1; }
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { echo "FATAL: сервер не отвечает" >&2; exit 1; }

step "бандл харнесса"
if [ ! -d "${BUNDLE_DIR}/src" ]; then
  mkdir -p "${BUNDLE_DIR}"
  "${VENV}/bin/kaggle" datasets download -d "${BUNDLE_REF}" -p "${BUNDLE_DIR}" --unzip -q
fi
SRC="${BUNDLE_DIR}/src"
test -d "${SRC}/ARC3-Inference" || { echo "FATAL: в бандле нет ARC3-Inference" >&2; exit 1; }

step "игры соревнования — ОФЛАЙН (без этого движка нет, урок 27.08)"
if [ ! -d "${COMP_DIR}/environment_files" ]; then
  mkdir -p "${COMP_DIR}"
  "${VENV}/bin/kaggle" competitions download -c arc-prize-2026-arc-agi-3 -p "${COMP_DIR}" -q
  (cd "${COMP_DIR}" && unzip -oq arc-prize-2026-arc-agi-3.zip)
fi
ENV_FILES_DIR="${COMP_DIR}/environment_files"
test -d "${ENV_FILES_DIR}" || { echo "FATAL: нет каталога игр ${ENV_FILES_DIR}" >&2; exit 1; }
echo "игр найдено: $(ls "${ENV_FILES_DIR}" | wc -l)"

step "наш слой -> sitecustomize"
SITE=$("${VENV}/bin/python" -c "import site; print(site.getsitepackages()[0])")
cp "${PATCH}" "${SITE}/wm_patch.py"
echo "патч скопирован: ${SITE}/wm_patch.py ($(wc -c < "${PATCH}") знаков)"
cat > "${SITE}/sitecustomize.py" <<'PY'
try:
    import wm_patch  # noqa: F401
except Exception as exc:  # патч не должен ронять процесс молча
    import sys
    print("WM PATCH НЕ ВСТАЛ: %r" % (exc,), file=sys.stderr, flush=True)
PY
"${VENV}/bin/python" -c "
import sys; sys.path.insert(0, '${SRC}/ARC3-Inference')
import os, wm_patch
import inference.agent.tool_agent as t, inference.framework.solver as s
layer = t.ToolAgent._run_python_tool.__name__ == '_wm_run'
sched = s._HarnessGameSession.runtime_limit_reached.__name__ == '_s_limit'
want_layer = os.environ.get('WM_LAYER') == '1'
want_sched = os.environ.get('WM_SCHED') == '1'
print('слой модели мира: %s (просили %s)' % (layer, want_layer))
print('расписание: %s (просили %s)' % (sched, want_sched))
assert layer == want_layer, 'слой встал не так, как просили'
assert sched == want_sched, 'расписание встало не так, как просили'
assert layer or sched, 'не включено ни одно изменение — прогон нечего измерять'
assert not (layer and sched), 'два изменения сразу — результат не приписать причине'
" || { echo "FATAL: патч встал не так, как просили — прогон бессмыслен" >&2; exit 1; }

step "окружение анализатора (как в Фазе A на Kaggle)"
export MPLBACKEND=Agg
export LOCAL_ANALYZER_BASE_URL="http://127.0.0.1:${PORT}/v1" OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export LOCAL_ANALYZER_PROVIDER=vllm OPENAI_PROVIDER=vllm
export LOCAL_ANALYZER_MODEL_ID="${SERVED_NAME}" INFERENCE_ANALYZER_MODEL="${SERVED_NAME}"
export LOCAL_ANALYZER_APP_NAME="ARC3 Agent Harness"
export LOCAL_ANALYZER_CONTEXT_WINDOW=32768 LOCAL_ANALYZER_MAX_OUTPUT=0
export LOCAL_ANALYZER_TOOL_STEPS=0 LOCAL_ANALYZER_TOOL_TIMEOUT=30 LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS=1024
export LOCAL_ANALYZER_YIELD_SECONDS=60 LOCAL_ANALYZER_TEMPERATURE=0.6 LOCAL_ANALYZER_TOP_P=0.95 LOCAL_ANALYZER_TOP_K=20
export LOCAL_ANALYZER_ENABLE_THINKING=true
export MULTIMODAL_CONTEXT=current_grid MULTIMODAL_UPSCALE=8
export ONLY_RESET_LEVELS=true TAAF_RUN_AS_SUBMISSION=0 TAAF_MINIMAL_DIAGNOSTICS=0
export LOCAL_ANALYZER_TIMEOUT=900

step "ПРОГОН: 25 публичных игр, conc ${CONC}, окно ${EXP_MIN} мин, WM_LAYER=${WM_LAYER} WM_SCHED=${WM_SCHED}"
mkdir -p "${EXP_DIR}"
cd "${SRC}/ARC3-Inference"
DRIVER="$(ls -1 inference*/driver.py 2>/dev/null | head -1 || echo inference/driver.py)"
set +e
"${VENV}/bin/python" -m inference.taaf_run \
  --include-tags official --agent inference --model "${SERVED_NAME}" \
  --analyzer-timeout 900 --deployment-target inline \
  --concurrent-jobs "${CONC}" --n-passes 1 \
  --max-runtime-minutes "${CAP_MIN}" --max-experiment-runtime-minutes "${EXP_MIN}" \
  --environments-dir "${ENV_FILES_DIR}" \
  --run-name "${RUN_NAME}" --experiment-dir "${EXP_DIR}" \
  2>&1 | tee "${WORK}/${RUN_NAME}.log"
RC=${PIPESTATUS[0]}
set -e
echo "код возврата движка: ${RC}"

step "проверки после прогона"
echo "игр с событиями: $(find "${EXP_DIR}" -name '*_events.jsonl' | wc -l) из 25 (несыгравшие — метрика расписания)"
echo "снято по простою: $(grep -c '\[SCHED\]' "${WORK}/${RUN_NAME}.log" || true)"
echo "строк слоя [WM] в логе: $(grep -c '\[WM\]' "${WORK}/${RUN_NAME}.log" || true)"
echo "принято программ: $(grep -c 'ПРИНЯТА ПО ГОРИЗОНТУ' "${WORK}/${RUN_NAME}.log" || true)"

step "упаковка"
cp "${WORK}/vllm.log" "${EXP_DIR}/" 2>/dev/null || true
cp "${WORK}/${RUN_NAME}.log" "${EXP_DIR}/" 2>/dev/null || true
tar -czf "${WORK}/${RUN_NAME}.tar.gz" -C "${WORK}/runs" "${RUN_NAME}"
ls -la "${WORK}/${RUN_NAME}.tar.gz"
step "ГОТОВО — забрать архив, затем ТЕРМИНИРОВАТЬ под (не Stop) и проверить тома"

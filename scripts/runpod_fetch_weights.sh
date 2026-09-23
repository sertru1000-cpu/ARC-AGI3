#!/bin/bash
# Скачивание боевых весов и установка окружения на СЕТЕВОЙ ТОМ RunPod (23.09).
#
# Замысел: всё тяжёлое и медленное делается на дешёвом поде, а карту B200 мы подключаем к тому же тому
# только под само обучение. Скачать 135 ГБ и поставить окружение -- это час, и платить за него $6/ч
# за простаивающую карту незачем.
#
# Запускать НА ПОДЕ:  nohup bash /root/runpod_fetch_weights.sh > /root/fetch.log 2>&1 &
# Следить:            tail -f /root/fetch.log
set -euo pipefail

VOL=/workspace
MODEL_DIR=$VOL/model
VENV=$VOL/venv312
REPO=RadixArk/Qwen3.8-Flash-Next-NVFP4
REV=7b719225242aacd3dbd3f9407468c2ee9a9d2594

echo "=== 1/4 диск и окружение"
df -h $VOL | tail -1
export UV_LINK_MODE=symlink          # см. навык runpod-deploy: иначе установка ползёт часами
export HF_HUB_ENABLE_HF_TRANSFER=1
pip -q install --upgrade "huggingface_hub[hf_transfer]" uv 2>&1 | tail -2

echo "=== 2/4 python-окружение на томе (переживёт смену пода)"
if [ ! -d "$VENV" ]; then
  uv venv --python 3.12 "$VENV"
fi
UV_LINK_MODE=symlink uv pip install --python "$VENV/bin/python" -q \
  "torch" "transformers>=5.8" "peft" "accelerate" "safetensors" "datasets" 2>&1 | tail -3
"$VENV/bin/python" -c "import torch,transformers,peft;print('torch',torch.__version__,'| transformers',transformers.__version__,'| peft',peft.__version__)"

echo "=== 3/4 веса модели (135 ГБ, только нужные файлы)"
mkdir -p "$MODEL_DIR"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id=os.environ.get("REPO", "RadixArk/Qwen3.8-Flash-Next-NVFP4"),
    revision=os.environ.get("REV"),
    local_dir=os.environ.get("MODEL_DIR", "/workspace/model"),
    max_workers=16,
    # видео-башня и файлы отчётов для обучения не нужны
    ignore_patterns=["*.mp4", "smoke_report.json", "validate_*report.json", "aime26_metrics.json", "gsm8k_metrics.json"],
)
print("скачано в", p)
PY

echo "=== 4/4 сверка"
du -sh "$MODEL_DIR"
ls "$MODEL_DIR" | head -5
echo "файлов: $(ls $MODEL_DIR | wc -l)"
python - <<'PY'
import json, os
d=os.environ.get("MODEL_DIR","/workspace/model")
idx=json.load(open(os.path.join(d,"model.safetensors.index.json")))
missing=[f for f in set(idx["weight_map"].values()) if not os.path.exists(os.path.join(d,f))]
print("недостающих файлов весов:", len(missing))
PY
echo "ГОТОВО. Теперь под можно погасить, том остаётся."

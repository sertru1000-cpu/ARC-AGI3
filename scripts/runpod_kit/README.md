# Боевой кит RunPod

## ▶ ЗАПУСК 22.08 — студент v3 (27B VL, датасет v3, 2 эпохи, r=32)

**Карта:** RTX Pro 6000 96 ГБ (пик ~78 ГБ) или H200. Container 40 ГБ, Volume 150 ГБ.
27B грузится в **bf16 без квантования** (`train_vl.py` строка ~151) — 54 ГБ весов
плюс ~24 ГБ активаций при 16k контекста; на 80 ГБ будет впритык.

```bash
# на поде, из распакованного runpod-kit/
export HF_HOME=/workspace/hf          # ИНАЧЕ 58 ГБ уедут на container-диск и пропадут при Stop
export HF_TOKEN=hf_...
export RUNPOD_API_KEY=rpa_... POD_ID=...       # чтобы под сам остановился после приёмки
BASE_MODEL=Qwen/Qwen3.6-27B \
DATA_DIR=./data_vision OUT_DIR=/workspace/out_vl27 \
EPOCHS=2 LORA_R=32 MAX_LENGTH=16384 \
  bash run_full_vl.sh
```

Дефолты уже такие (`BASE_MODEL=Qwen/Qwen3.6-27B`, `LORA_R=32`, `EPOCHS=2`) — переменные
выше нужны только чтобы это было видно в логе.

**Данные в ките:** `data_vision/` = 1633 обучающих примера (все с картинкой, 963 решающих,
22 игры) + 92 валидационных. Дублирования нет; отложены sc25/su15/m0r0.
`data/` — старый текстовый датасет, для `train.py`.

**Ожидаемо:** ~3.1 ч на эпоху на H200 (6.8 с/пример), две эпохи ≈ 6.2 ч ≈ $28.
На RTX Pro 6000 дешевле по часу, но скорость не мерена.

### Приёмка — выполняется САМА, до всякой Kaggle-квоты

`run_full_vl.sh` после обучения запускает `check_student.py` и пишет
`/workspace/check_student.log`. Он генерирует на отложенных промптах и сравнивает
с целями из того же файла:

| метрика | данные | студент v6 (провал) | ворота |
|---|---|---|---|
| длина ответа | ~1300 | 535 | ≥ 70% от данных |
| доля WORLD_MODEL | ~74% | 40% | ≥ 80% от данных |
| `action()` на ответ | ~1.5 | 0.5 | ≥ 70% от данных |

**FAIL — не разворачивать на Kaggle**, а учить дольше или поднимать ранг. Именно
эти три числа объясняли провал v6, и увидеть их можно не играя ни одной игры.

Запустить приёмку отдельно:
```bash
ADAPTER=/workspace/out_vl27/adapter_final DATA=./data_vision/valid.jsonl N=16 \
  python check_student.py
```

### Забрать результат
```bash
scp -P <port> root@<ip>:/workspace/adapter_vl_final.tgz data/student_v3/
scp -P <port> root@<ip>:/workspace/check_student.log data/student_v3/
ssh -p <port> root@<ip> touch /workspace/FETCHED     # под остановится сам
```

---


Дистилляция Gemini-учителя в Qwen3.6-27B. Рецепт проверен мак-репетицией;
здесь та же схема на полных данных (8 пар контекста, 16k контекст).

## 0. Под

RunPod → Deploy: **RTX Pro 6000 (96 GB)**, шаблон PyTorch 2.8 (cu128),
Container 30 GB, **Volume 200 GB** (persistent, /workspace; на нём HF-кэш 54 GB
+ merged 54 GB + FP8 28 GB + чекпоинты ~15 GB). SSH-ключ уже в аккаунте.

## 1. Закинуть кит и войти

С мака (архив пришлёт ассистент, команда подключения — из карточки пода):
```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 runpod-kit.tgz root@<IP>:/workspace/
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519
```

## 2. На поде

```bash
cd /workspace && tar xzf runpod-kit.tgz && cd runpod-kit
export HF_TOKEN=hf_...            # твой Write-токен, вводится вручную
export HF_REPO=<username>/arc3-student-v1   # приватный репозиторий результата
bash run_all.sh
```

Всё. Дальше скрипт сам: зависимости → проверка датасета → QLoRA (~2.5-3.5 ч)
→ merge в bf16 → FP8 → загрузка обоих артефактов в HF_REPO.

## Что смотреть в логе обучения

- `loss` в строках прогресса: старт ~1.5-2.5, должен устойчиво падать;
  `eval_loss` каждые 100 шагов — не должен расти при падающем train.
- Чекпоинты каждые 100 шагов в /workspace/out/ — смерть пода не страшна:
  повторный `bash run_all.sh` продолжит с последнего чекпоинта.
- OOM теоретически невозможен (пик ~35-40 GB из 96), но если что-то
  странное — лог целиком в /workspace/train.log.

## После успеха

1. Убедиться, что в HF-репозитории появились папки `fp8/` и `adapter/`.
2. **Terminate** пода (вместе с volume — всё ценное уже в HF).
3. Сказать ассистенту — он соберёт Kaggle-датасет из fp8/ и запустит
   репетицию студента (v23).

## Смета

~3-4 часа × $2.09 ≈ **$7-9** (+ ~$0.5 если под жил дольше на закачках).

## Student v2 (MoE, 35B-A3B) — этот кит НЕ применим как есть

Всё выше — рецепт для плотной 27B (проверен, student v1 обучен так 19.08). Для
MoE-базы (`Qwen/Qwen3.6-35B-A3B`, 256 экспертов) `train.py` теперь детектирует
MoE автоматически и ведёт себя иначе (см. докстринг файла), но **под и смета
другие**, ничего из раздела 0/«Смета» выше не переносить бездумно:

- **Не RTX Pro 6000 96GB.** bitsandbytes не умеет квантовать слитые
  `nn.Parameter`-веса экспертов (bitsandbytes-foundation/bitsandbytes#1849) —
  для MoE обучение идёт в bf16 без квантования, ~70GB только веса. Нужна
  карта с большим запасом (кандидат: H200 141GB) — дороже по часу.
- **`BASE_MODEL=Qwen/Qwen3.6-35B-A3B`** (bf16-репозиторий, БЕЗ `-FP8` —
  тот вариант только для инференса).
- **Ранг адаптера на эксперта = 1** (`r=16 // num_experts=256`, см. докстринг
  `train.py`) — не проверено, хватит ли; смотреть на loss в первых ~50 шагах,
  при подозрении на недообучение — поднять базовый `r`.
- **Ничего из этого не прогонялось живьём** — только код-ревью + юнит-логика.
  Первый реальный запуск должен начинаться с короткого смоук-теста (мало
  шагов, маленький под) ПЕРЕД полным 3-4-часовым прогоном на дорогой карте.

## Смоук-тест MoE-рецепта (обязателен ПЕРЕД полным прогоном) — добавлено 21.08

Цель: за ~$2-3 и 20-30 минут ответить на вопросы, на которых нельзя споткнуться
через 2 часа платного прогона: грузится ли 35B-A3B в bf16 на карте, цепляется
ли LoRA к слитым весам экспертов (`target_parameters`), двигается ли loss,
пиковая VRAM и секунды на шаг (→ стоимость полного прогона).

Под: **H100 NVL 94GB или RTX Pro 6000 96GB** (впритык: ~70GB веса); если OOM —
H200 141GB. Container 30GB, Volume 100GB. Шаблон PyTorch 2.8 (cu128).

```bash
# с мака
scp -P <PORT> -i ~/.ssh/id_ed25519 runpod-kit.tgz root@<IP>:/workspace/
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519
# на поде
cd /workspace && tar --no-same-owner -xzf runpod-kit.tgz && cd runpod-kit
export HF_TOKEN=hf_...
nohup bash run_smoke.sh > /workspace/smoke_console.log 2>&1 &
tail -f /workspace/smoke_console.log
```

Результат — последняя JSON-строка в `/workspace/smoke.log` (`"smoke": true`):
`loss_curve` должен идти вниз, `of which on experts` ≠ 0 (иначе LoRA на
экспертов не подцепилась — фолбэк `EXPERT_LORA=0`), `peak_vram_gb` < объём
карты − 8, `sec_per_step × (примеров × эпох / 8)` = длительность полного прогона.
Если OOM на 16k — `MAX_LENGTH=8192 bash run_smoke.sh` (тогда и датасет резать
до 8k: `build_sft_dataset.py --max-context-pairs 4`). После смоука — **Stop**,
не Terminate, если полный прогон пойдёт на том же сетапе (веса останутся на volume).
Полный прогон тем же китом: `bash run_all.sh` с `BASE_MODEL=Qwen/Qwen3.6-35B-A3B`
(и подменённым `data/` — см. `build_kit.sh <train> <valid>`).

# NOTE: shell scripts here MUST keep LF endings -- bash reads "set -euo pipefail"
# as an invalid option and the whole script dies on line 8. Patching them with
# Python on Windows silently rewrites LF to CRLF (write_text translates newlines);
# use write_bytes, and build_kit.sh verifies before packing.

#!/usr/bin/env bash
# Публикация датасета arc3-nextfork: подготовка, проверка, загрузка, сверка.
#
# ЗАЧЕМ СВЕРКА ПОСЛЕ ЗАГРУЗКИ. Kaggle CLI умеет отчитаться об успехе, загрузив
# не всё (02.09: `-r skip` молча выбросил src/ — 288 КБ вместо 2.6 МБ; 30.08:
# обрыв после первого файла из 77 при статусе ready). Здесь бандл — 4409
# файлов, и потеря одного каталога означает падение прогона на импорте через
# сорок минут после старта. Поэтому после загрузки список файлов на Kaggle
# сравнивается с локальным деревом — этим занят scripts/verify_nextfork_dataset.py
# (через CLI нельзя: `datasets files` отдаёт только первые 200 записей,
# сколько ни проси, и сверка объявила бы неполным нормальный пуш).
#
# Первая публикация (датасета ещё нет) делается вручную:
#   .venv/bin/kaggle datasets create -p nextfork --dir-mode tar
#
# usage: bash scripts/push_nextfork_dataset.sh "что изменилось"

set -u
cd "$(dirname "$0")/.."

MSG="${1:-}"
if [ -z "$MSG" ]; then
  echo "нужно сообщение: bash scripts/push_nextfork_dataset.sh \"что изменилось\""
  exit 1
fi

KAGGLE=.venv/bin/kaggle
SRC=nextfork
REF=sergueimakarov/arc3-nextfork

echo "== готовлю бандл (штамп версии + метаданные) =="
.venv/bin/python scripts/build_nextfork_dataset.py || exit 1

LOCAL=$(find "$SRC" -type f -not -path "*__pycache__*" -not -name "*.pyc" | wc -l)
echo "== локально файлов: $LOCAL =="
if [ "$LOCAL" -lt 1000 ]; then
  echo "СТОП: файлов подозрительно мало"
  exit 1
fi

echo "== загружаю (-r tar, НИКОГДА не skip) =="
$KAGGLE datasets version -p "$SRC" -m "$MSG" -r tar 2>&1 | tail -3

echo "== жду публикации =="
sleep 90

.venv/bin/python scripts/verify_nextfork_dataset.py || exit 1
echo "== пуш полный =="

echo "== проверьте кернел: пуш датасета иногда запускает фантомный реран =="
$KAGGLE kernels list -m -s arc3-nextfork --csv 2>&1 | sed -n '2p'

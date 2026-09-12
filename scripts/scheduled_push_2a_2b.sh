#!/bin/bash
# Автоматический пуш прогонов 2а и 2б в ночь на субботу 12.09.2026 (слово владельца 11.09:
# «в 3 ночи по Москве автоматически пуш одновременно 2a 2b»).
#
# Запускается launchd-агентом com.sergeimakarov.arc3.push2a2b в 03:15 МСК — через пятнадцать минут после
# обновления недельной квоты Kaggle (00:00 UTC субботы), чтобы не попасть в последние секунды старой недели.
#
# ЗАЩИТЫ: (1) ровно один запуск — метка runs/.scheduled_push_12_09.done ставится ДО первого пуша;
# (2) окно — только 12.09.2026 с 03:00 до 23:59 МСК; (3) пушится только проверенное — отпечатки
# sha256 ноутбуков и метаданных обязаны совпасть со снятыми 11.09 после тестов (2а 15 проверок,
# 2б 16); (4) после работы агент сам себя выгружает. DRY=1 — всё, кроме пуша (для проверки).
set -u
ROOT=/Users/sergeimakarov/Projects/ARC-AGI-3
LOG=$ROOT/runs/scheduled_push_12_09.log
DONE=$ROOT/runs/.scheduled_push_12_09.done
KAGGLE=$ROOT/.venv/bin/kaggle
LABEL=com.sergeimakarov.arc3.push2a2b
log() { echo "$(TZ=Europe/Moscow date '+%d.%m %H:%M:%S МСК') $*" >> "$LOG"; }
DRY=${DRY:-0}

NOW=${TEST_NOW:-$(TZ=Europe/Moscow date '+%Y-%m-%d %H:%M')}
[ "$DRY" != "1" ] && NOW=$(TZ=Europe/Moscow date '+%Y-%m-%d %H:%M')
log "запуск (DRY=$DRY), время $NOW"

if [ -e "$DONE" ]; then log "метка уже стоит — пуш был, повторно не пушу"; exit 0; fi
if [[ "$NOW" < "2026-09-12 03:00" ]]; then log "рано, окно с 12.09 03:00 МСК — выхожу"; exit 0; fi
if [[ "$NOW" > "2026-09-12 23:59" ]]; then log "окно 12.09 прошло — не пушу, нужен ручной пуш"; exit 0; fi

check() {  # $1 файл, $2 ожидаемый sha256
  local got; got=$(shasum -a 256 "$ROOT/$1" | cut -d' ' -f1)
  if [ "$got" = "$2" ]; then log "отпечаток ок: $1"; return 0; fi
  log "ОТПЕЧАТОК НЕ СОВПАЛ: $1 (ожидался $2, есть $got) — этот прогон не пушу"; return 1
}
OK2A=1; OK2B=1
check kernels/notebooks_stockflash_input/submission.ipynb 4e9334b6bbfc0231eb1efbfa1c6883a3e83495d0de726d67b4f221e0464c540a || OK2A=0
check kernels/notebooks_stockflash_input/kernel-metadata.json 486d7c4b20a1af93d8ec913d6bf2794f52934d5061c658df434251abe8838756 || OK2A=0
check kernels/notebooks_stockflash_carry/submission.ipynb 5f889d7f11b94354ed12fb060388b1d8d1f7b6d71faefcc99a44f525e750e80d || OK2B=0
check kernels/notebooks_stockflash_carry/kernel-metadata.json 272bc003317e93d77ba4da0f83d881ce072d7958acd2fbdcae9c5590e5d9a78e || OK2B=0

export KAGGLE_API_TOKEN="$(cat $ROOT/.kaggle/access_token)"
push() {  # $1 имя, $2 каталог; одна повторная попытка через минуту при сетевом сбое
  if [ "$DRY" = "1" ]; then log "DRY: пушил бы $1 из $2"; return 0; fi
  local out; out=$("$KAGGLE" kernels push -p "$ROOT/$2" 2>&1 | tail -2)
  log "$1: $out"
  if ! echo "$out" | grep -q "successfully pushed"; then
    sleep 60; out=$("$KAGGLE" kernels push -p "$ROOT/$2" 2>&1 | tail -2); log "$1, повтор: $out"
  fi
}
if [ "$OK2A" = "1" ] || [ "$OK2B" = "1" ]; then
  [ "$DRY" != "1" ] && touch "$DONE" && log "метка поставлена"
  [ "$OK2A" = "1" ] && push "2а (arc3-stock-flash-loop)" kernels/notebooks_stockflash_input
  [ "$OK2B" = "1" ] && push "2б (arc3-stock-flash-sched)" kernels/notebooks_stockflash_carry
fi

if [ "$DRY" = "1" ]; then
  log "DRY: статус loop сейчас: $("$KAGGLE" kernels status sergueimakarov/arc3-stock-flash-loop 2>&1 | tail -1)"
  log "DRY: проверка окончена, агент не выгружаю"; exit 0
fi
sleep 150
for k in arc3-stock-flash-loop arc3-stock-flash-sched; do
  log "статус через 2.5 мин: $("$KAGGLE" kernels status sergueimakarov/$k 2>&1 | tail -1)"
done
launchctl bootout gui/$(id -u)/$LABEL 2>/dev/null && log "агент выгружен" || log "агент уже не загружен"

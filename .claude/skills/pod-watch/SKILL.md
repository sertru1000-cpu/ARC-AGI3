---
name: pod-watch
description: "Whenever ANY job runs on a rented pod (RunPod) or a long Kaggle run is in flight — from the moment it is launched until the owner says to stop. Load right after launching pod work, at the start of every session while a pod exists, and before answering any status question. Prevents the pod sitting idle and paid for hours because nobody noticed a failure (04.10: live stand died 9 min in, noticed 2.5 h later, ~$5 wasted; same pattern several times before)."
metadata:
  author: user
  version: "1.0.0"
---

# Следить за подом — всегда, а не «когда вспомню»

**Почему.** Владелец 04.10: «ты уже который раз не следишь за pod». Живой стенд упал через 9 минут после старта (не было `imageio`), заметил я через 2.5 часа — под всё это время оплачивался впустую. Причина каждый раз одна: фоновое ожидание в сессии умирает при её перезапуске, а новую я не ставлю. Под при этом работает и стоит денег.

## Правило 1. Сторож вне сессии — с первой минуты работы пода

Сразу после запуска любой работы на поде (а не потом):

```bash
pkill -f "[p]od_sentinel.sh"
nohup caffeinate -i /bin/bash scripts/pod_sentinel.sh <ip> <port> "<лог1> <лог2>" > runs/pod_sentinel.nohup 2>&1 &
pgrep -fl "bin/bash scripts/pod_sentinel"     # убедиться, что живой
```

`scripts/pod_sentinel.sh` раз в 2 минуты: события в логах (ГОТОВ / FAILED / TIMEOUT / Traceback / Error:), простой карты 10 минут подряд, деньги на счёте меньше чем на час. Пишет ТОЛЬКО в `runs/pod_sentinel.log` (владелец: «громко сообщать не надо, просто следи»). Под НЕ трогает — глобальное правило ~/.claude/CLAUDE.md.

При новом логе (новый прогон) — перезапустить сторожа с новым списком логов. Скрипт для `/bin/bash` 3.2 на маке (без `declare -A` и прочего из bash 4).

## Правило 2. Проверять самому, не надеясь на фон

- **В начале каждой сессии**, пока под существует: `tail runs/pod_sentinel.log`, `pgrep -fl pod_sentinel`, затем прямо на поде — что идёт, нет ли ошибки, сколько денег (`clientBalance / currentSpendPerHr` через GraphQL RunPod). Если сторож мёртв — поднять.
- **При уведомлении о прерванной/завершённой фоновой задаче** — тут же проверить под, а не только прочитать вывод задачи.
- **На любой вопрос «статус?»** — свежие данные с пода, а не память.
- В сессии держать фоновое ожидание первого события (`run_in_background`) — но это дополнение к сторожу, не замена.

## Правило 3. Первые 10–15 минут нового прогона — смотреть руками

Большинство падений случается в начале: зависимости, пути, порты, память. Через 10–15 минут после старта зайти и убедиться, что прогон дошёл до РАБОТЫ (сервер поднялся, игры/нагрузка пошли, в логе нет Traceback), а не только что «процесс жив».

Проверка «до запуска сервера» не считается проверкой всего прогона: 04.10 предпроверка прошла, а `imageio`/`scipy` импортируются позже, при запуске игр. Перед долгим прогоном — проверить импорт ВСЕХ модулей того, что будет работать (обход пакетов с `importlib.import_module` и список `ModuleNotFoundError`).

## Правило 4. Деньги

Перед каждым длинным прогоном сравнить его длительность с остатком на счёте (`clientBalance / currentSpendPerHr`). Если не хватает — сказать владельцу ДО запуска. Пополнять, гасить, арендовать — только его слово.

## Связано

[[feedback-background-watch-dies-with-session]], [[feedback-never-kill-pod-without-command]], [[feedback-pod-costs-real-money]], навык `runpod-deploy`.

#!/usr/bin/env bash
# Остаток калибровочной серии: ждём прогон 2, качаем, пушим 3, ждём, качаем.
set -u
cd "c:/Users/sertru1000/Projects/ARC-AGI-3"
K=.venv/Scripts/kaggle.exe
PY=.venv/Scripts/python.exe
export PYTHONUTF8=1

wait_done () {
  # Кернел живёт ~2ч; опрашиваем раз в 5 минут, потолок 3.5 часа.
  for _ in $(seq 1 42); do
    sleep 300
    s=$($K kernels status sergueimakarov/arc3-atlas 2>&1 | tail -1)
    case "$s" in
      *COMPLETE*) echo "[$(date -u +%H:%M)] завершён"; return 0 ;;
      *ERROR*|*CANCEL*) echo "[$(date -u +%H:%M)] ОТКАЗ: $s"; return 1 ;;
    esac
  done
  echo "таймаут ожидания"; return 1
}

for i in 2 3; do
  echo "=== ждём прогон $i ==="
  wait_done || exit 1
  # Качаем ДО следующего пуша: новый пуш затирает выхлоп предыдущего.
  $K kernels output sergueimakarov/arc3-atlas -p "runs/calib_$i" -o >/dev/null 2>&1
  echo "--- конверсия прогона $i ---"
  $PY scripts/conversion.py "runs/calib_$i"
  if [ "$i" = "2" ]; then
    echo "=== пуш прогона 3 ==="
    $K kernels push -p kernels/notebooks_atlas 2>&1 | tail -1
  fi
done

echo
echo "=== СЕРИЯ ЗАВЕРШЕНА: три отдельных одноволновых прогона ==="
$PY scripts/conversion.py runs/calib_1 runs/calib_2 runs/calib_3

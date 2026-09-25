#!/usr/bin/env bash
# Push the arc3-atlas-src dataset: stage, verify, upload, verify again.
#
# WHY THIS SCRIPT EXISTS (02.09.2026). The dataset carries our whole forked
# source -- 25k lines, of which ~1k are ours -- plus the pickles, h_model and
# the own-games testbed. It is the only way a change to agent BEHAVIOUR
# reaches Kaggle; a notebook knob cannot patch a method body.
#
# Two ways to break it, both of which report success:
#
#   * `-r skip` means "ignore directories". Used it on the first push of
#     02.09 and the CLI cheerfully uploaded the 12 top-level files and
#     dropped src/ and our_games/ entirely -- 288 KB instead of 2.6 MB. The
#     dataset was live and broken; every run would have failed on import.
#     Correct mode is `-r tar` (or zip).
#   * a truncated upload. V37 died exactly this way on 30.08: the push
#     stopped after one file of 77 while the status said ready.
#
# So the file list is compared against the local tree AFTER the upload, and
# the only acceptable difference is dataset-metadata.json, which Kaggle keeps
# as a manifest and never stores as a file.
#
# usage:  bash scripts/push_dataset.sh "what changed"

set -u
cd "$(dirname "$0")/.."

MSG="${1:-}"
if [ -z "$MSG" ]; then
  echo "нужно сообщение: bash scripts/push_dataset.sh \"что изменилось\""
  exit 1
fi

KAGGLE=.venv/Scripts/kaggle.exe
SRC=atlas_src

# --- 1. stage the testbed into the dataset directory ----------------------
echo "== раскладываю полигон в датасет =="
rm -rf "$SRC/our_games"
mkdir -p "$SRC/our_games"
find our_games -type f -not -name "*.pyc" -not -path "*__pycache__*" | while read -r f; do
  rel="${f#our_games/}"
  mkdir -p "$SRC/our_games/$(dirname "$rel")"
  cp "$f" "$SRC/our_games/$rel"
done

printf 'commit: %s\nbuilt:  %s\ngames:  %s\n' \
  "$(git rev-parse HEAD)" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(ls "$SRC/our_games" | wc -l)" > "$SRC/ATLAS_VERSION.txt"
cat "$SRC/ATLAS_VERSION.txt"

# --- 2. sanity before uploading anything ---------------------------------
LOCAL=$(find "$SRC" -type f -not -path "*__pycache__*" | wc -l)
echo "== локально файлов: $LOCAL =="
if [ "$LOCAL" -lt 100 ]; then
  echo "СТОП: файлов подозрительно мало, ожидалось больше сотни"
  exit 1
fi

# --- 3. upload. -r tar, NEVER skip ---------------------------------------
echo "== загружаю (-r tar) =="
$KAGGLE datasets version -p "$SRC" -m "$MSG" -r tar 2>&1 | tail -2

echo "== жду публикации =="
sleep 60

# --- 4. verify what actually landed --------------------------------------
$KAGGLE datasets files sergueimakarov/arc3-atlas-src --page-size 500 2>&1 \
  | tail -n +3 | awk 'NF{print $1}' | sort > /tmp/atlas_kaggle_files.txt
(cd "$SRC" && find . -type f -not -path "*__pycache__*" | sed 's|^\./||' | sort) \
  > /tmp/atlas_local_files.txt

MISSING=$(comm -23 /tmp/atlas_local_files.txt /tmp/atlas_kaggle_files.txt \
          | grep -v '^dataset-metadata.json$' || true)
echo "== на Kaggle: $(wc -l < /tmp/atlas_kaggle_files.txt), локально: $(wc -l < /tmp/atlas_local_files.txt) =="
if [ -n "$MISSING" ]; then
  echo "ПУШ НЕПОЛНЫЙ, не уехало:"
  echo "$MISSING" | head -20
  exit 1
fi
echo "== пуш полный =="

# --- 5. the phantom rerun ------------------------------------------------
echo "== проверьте кернел: пуш датасета иногда запускает фантомный реран =="
$KAGGLE kernels list -m -s arc3-atlas --csv 2>&1 | sed -n '2p'
echo "   если lastRunTime только что изменился -- отменить прогон в веб-интерфейсе"

#!/usr/bin/env bash
# Pack the training kit into data/runpod-kit.tgz (Windows-safe: no owner
# metadata, extract on the pod with `tar --no-same-owner -xzf`).
# Usage: bash scripts/runpod_kit/build_kit.sh [path/to/train.jsonl path/to/valid.jsonl]
# With dataset args, the kit's data/ is replaced by those files first.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KIT="$ROOT/scripts/runpod_kit"
# Two positional args replace the TEXT dataset; --vision puts them in
# data_vision/ instead (train_vl.py reads DATA_DIR=./data_vision).
DEST="data"
if [ "${1:-}" = "--vision" ]; then DEST="data_vision"; shift; fi
if [ $# -ge 2 ]; then
  mkdir -p "$KIT/$DEST"
  cp "$1" "$KIT/$DEST/train.jsonl"; cp "$2" "$KIT/$DEST/valid.jsonl"
  echo "$DEST replaced: $(wc -l < "$1") train / $(wc -l < "$2") valid"
fi
"$ROOT/.venv/Scripts/python.exe" - "$KIT" <<'EOF'
import ast, sys, pathlib
for f in ("train.py", "convert.py", "train_vl.py", "check_student.py"):
    ast.parse(pathlib.Path(sys.argv[1], f).read_text(encoding="utf-8"))
print("syntax ok: train.py convert.py train_vl.py check_student.py")
EOF
# Shell scripts must ship with LF endings: bash reads a trailing CR as part
# of the option name and dies on `set -euo pipefail` (a failed pod launch,
# 22.08). Octal 015 is used below so this check can never itself carry a CR.
for f in "$KIT"/*.sh; do
  if [ "$(tr -cd '' < "$f" | wc -c)" != "0" ]; then
    echo "FATAL: CRLF in $(basename "$f") -- convert it to LF before packing"; exit 1
  fi
done
OUT="$ROOT/data/runpod-kit.tgz"
tar --owner=0 --group=0 --exclude='__pycache__' -czf "$OUT" \
    -C "$ROOT/scripts" --transform 's,^runpod_kit,runpod-kit,' runpod_kit
ls -la "$OUT"
tar -tzf "$OUT" | head -20

#!/usr/bin/env bash
# Pack the training kit into data/runpod-kit.tgz (Windows-safe: no owner
# metadata, extract on the pod with `tar --no-same-owner -xzf`).
# Usage: bash scripts/runpod_kit/build_kit.sh [path/to/train.jsonl path/to/valid.jsonl]
# With dataset args, the kit's data/ is replaced by those files first.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KIT="$ROOT/scripts/runpod_kit"
if [ $# -ge 2 ]; then
  cp "$1" "$KIT/data/train.jsonl"; cp "$2" "$KIT/data/valid.jsonl"
  echo "dataset replaced: $(wc -l < "$1") train / $(wc -l < "$2") valid"
fi
"$ROOT/.venv/Scripts/python.exe" - "$KIT" <<'EOF'
import ast, sys, pathlib
for f in ("train.py", "convert.py"):
    ast.parse(pathlib.Path(sys.argv[1], f).read_text(encoding="utf-8"))
print("syntax ok: train.py convert.py")
EOF
OUT="$ROOT/data/runpod-kit.tgz"
tar --owner=0 --group=0 --exclude='__pycache__' -czf "$OUT" \
    -C "$ROOT/scripts" --transform 's,^runpod_kit,runpod-kit,' runpod_kit
ls -la "$OUT"
tar -tzf "$OUT" | head -20

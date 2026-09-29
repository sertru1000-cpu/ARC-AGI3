"""Пересчёт хэша и размера serving_setup.py в nextfork/SOURCE_IDENTITY.json после правки (29.09).
Без этого serving_setup сам себя отвергнет: «Serving setup identity mismatch».
usage: .venv/bin/python scripts/sync_nextfork_identity.py
"""
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ident = ROOT / "nextfork/SOURCE_IDENTITY.json"
setup = ROOT / "nextfork/serving_setup.py"
d = json.loads(ident.read_text(encoding="utf-8"))
sha = hashlib.sha256(setup.read_bytes()).hexdigest()
old = d["serving_setup_sha256"]
d["serving_setup_sha256"] = sha
d["replacement_records"]["serving_setup.py"] = {"bytes": setup.stat().st_size, "sha256": sha}
ident.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("serving_setup.py: %s -> %s (%d байт)" % (old[:12], sha[:12], setup.stat().st_size))

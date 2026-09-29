"""Наши правки v3 поверх чужого бандла исходников — готовый к пушу датасет, НЕ пушится (26.09, к этапу 30.09).

Патч docs/graft/nextfork_v3.patch = разница keithtyser-база -> наш боевой v3 (4 файла: noop_guard.py новый,
prompts.py, tool_agent.py, solver.py; запрет повтора пустого хода с третьего раза + goalrule + precond + nostop).
26.09 проверено: ложится без конфликтов на базу keithtyser, на оригинальный Duck Tufa (reference/duck-source) и
на harness/duck (fuzz 3).
Шаги: скачать их датасет -> скопировать в datasets_graft/<slug>/ -> patch -p1 -F3 (сначала --dry-run) ->
dataset-metadata.json с нашим id sergueimakarov/arc3-graft-<slug>.
Дальше по слову владельца: kaggle datasets create -p datasets_graft/<slug> --dir-mode zip, затем
scripts/fork_published.py <их кернел> --swap <их датасет>=sergueimakarov/arc3-graft-<slug>.
usage: .venv/bin/python scripts/graft_bundle.py owner/their-dataset [--name short]
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K = str(ROOT / ".venv/bin/kaggle")
PATCH = ROOT / "docs/graft/nextfork_v3.patch"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("--name", default=""); a = ap.parse_args()
    slug = (a.name or a.dataset.split("/", 1)[1])[:40]
    tmp = Path(tempfile.mkdtemp(prefix="graftds_"))
    r = subprocess.run([K, "datasets", "download", a.dataset, "-p", str(tmp), "--unzip"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("не скачался: " + r.stderr[-400:])
    marker = list(tmp.rglob("taaf-kaggle-bundle.json"))
    src = marker[0].parent if marker else tmp
    print("бандл:", src, "(маркер TAAF %s)" % ("есть" if marker else "НЕТ — не Duck/TAAF, патч может не лечь"))
    out = ROOT / "datasets_graft" / slug
    shutil.rmtree(out, ignore_errors=True); shutil.copytree(src, out)
    dry = subprocess.run(["patch", "-p1", "--dry-run", "-F3"], cwd=out, stdin=open(PATCH), capture_output=True, text=True)
    print(dry.stdout.strip())
    if dry.returncode:
        raise SystemExit("патч НЕ ложится чисто — переносить правки руками (см. вывод выше)")
    subprocess.run(["patch", "-p1", "-F3", "--no-backup-if-mismatch"], cwd=out, stdin=open(PATCH), check=True, capture_output=True)
    (out / "GRAFT_NOTE.txt").write_text("их датасет: %s\nнаш патч: docs/graft/nextfork_v3.patch (v3: noop_guard N=3, goalrule, precond, nostop)\n" % a.dataset)
    meta = {"title": ("arc3 graft " + slug)[:50], "id": "sergueimakarov/arc3-graft-" + slug.lower().replace("_", "-"),
            "licenses": [{"name": "other"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    print("готово (НЕ отправлено):", out, "| id", meta["id"])


if __name__ == "__main__":
    main()

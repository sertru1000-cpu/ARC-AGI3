"""Готовность к прививке (26.09): что именно изменено в опубликованном решении относительно нашей базы.

Под ставку владельца: к 30.09 кто-то из первой тройки, вероятно, опубликует решение. Этот скрипт за минуты
отвечает на первые вопросы прививки, не тратя квоту:
  1. какие датасеты и модели кернел подключает (другая модель? другие веса? свой бандл?);
  2. чем его ноутбук отличается от нашей боевой базы (ячейки, настройки обвязки, env);
  3. если он подключает бандл исходников (taaf-kaggle-bundle.json), — какие файлы обвязки отличаются от
     нашего форка nextfork/ и насколько (строки +/−), с первыми строками разницы.
Датасеты скачиваются только если они меньше --max-gb (веса моделей не тянем).

usage: .venv/bin/python scripts/graft_inspect.py <owner/kernel-slug> [--max-gb 2]
"""
from __future__ import annotations
import argparse, difflib, json, os, shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K = str(ROOT / ".venv/bin/kaggle")
OUR_NB = ROOT / "kernels/notebooks_nextfork/submission.ipynb"
OUR_BUNDLE = ROOT / "reference/nextfork-v3"   # боевой v3 (локальный nextfork/ ушёл вперёд: согласование, проба старта)


def sh(*a):
    return subprocess.run([K, *a], capture_output=True, text=True)


def nb_cells(path):
    nb = json.load(open(path, encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def ds_size_gb(ref):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    tot = 0; tok = None
    while True:
        r = api.dataset_list_files(ref, page_token=tok, page_size=200)
        tot += sum(int(getattr(f, "total_bytes", 0) or 0) for f in r.files)
        tok = getattr(r, "next_page_token", None)
        if not tok or not r.files:
            return tot / 1e9


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("kernel"); ap.add_argument("--max-gb", type=float, default=2.0)
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="graft_")); print("рабочая папка:", work)
    r = sh("kernels", "pull", a.kernel, "-p", str(work / "kernel"), "-m")
    if r.returncode:
        raise SystemExit("не скачался кернел: " + r.stderr[-300:])
    meta = json.load(open(work / "kernel" / "kernel-metadata.json"))
    nbs = list((work / "kernel").glob("*.ipynb"))
    print("\n══ 1. ИСТОЧНИКИ ══")
    print("модели:  ", meta.get("model_sources")); print("датасеты:", meta.get("dataset_sources"))
    print("машина:  ", meta.get("machine_shape"), "| интернет:", meta.get("enable_internet"))
    ours = json.load(open(ROOT / "kernels/notebooks_nextfork/kernel-metadata.json"))
    print("у нас:    модели", ours.get("model_sources"), "| датасеты", ours.get("dataset_sources"))

    print("\n══ 2. НОУТБУК против нашей боевой базы ══")
    if nbs:
        theirs = "\n".join(nb_cells(nbs[0])); mine = "\n".join(nb_cells(OUR_NB))
        d = list(difflib.unified_diff(mine.splitlines(), theirs.splitlines(), "наш", "их", n=0, lineterm=""))
        add = sum(1 for l in d if l.startswith("+") and not l.startswith("+++"))
        rem = sum(1 for l in d if l.startswith("-") and not l.startswith("---"))
        print("строк кода: +%d / −%d" % (add, rem))
        env_like = [l for l in d if l[:1] in "+-" and any(k in l for k in ("environ", "TAAF_", "LOCAL_ANALYZER", "MULTIMODAL", "max_runtime", "concurrency", "VLLM"))]
        for l in env_like[:30]:
            print("  ", l[:170])
        (work / "notebook.diff").write_text("\n".join(d), encoding="utf-8"); print("полный дифф:", work / "notebook.diff")

    print("\n══ 3. БАНДЛ ИСХОДНИКОВ ══")
    for ref in meta.get("dataset_sources") or []:
        try:
            gb = ds_size_gb(ref)
        except Exception as e:
            print("  %-60s размер не узнать: %r" % (ref, e)); continue
        tag = "качаю" if gb <= a.max_gb else "пропуск (веса?)"
        print("  %-60s %7.2f ГБ  %s" % (ref, gb, tag))
        if gb > a.max_gb:
            continue
        dest = work / "ds" / ref.replace("/", "__")
        sh("datasets", "download", ref, "-p", str(dest), "--unzip")
        marker = list(dest.rglob("taaf-kaggle-bundle.json"))
        if not marker:
            continue
        bundle = marker[0].parent
        print("  → бандл исходников найден:", bundle)
        changed = []
        for f in sorted(bundle.rglob("*.py")):
            rel = f.relative_to(bundle); mine = OUR_BUNDLE / rel
            if not mine.is_file():
                changed.append(("НОВЫЙ", rel, sum(1 for _ in open(f, errors="ignore")), 0)); continue
            dl = list(difflib.unified_diff(open(mine, errors="ignore").read().splitlines(), open(f, errors="ignore").read().splitlines(), n=0, lineterm=""))
            if dl:
                changed.append(("изменён", rel, sum(1 for l in dl if l.startswith("+") and not l.startswith("+++")),
                                sum(1 for l in dl if l.startswith("-") and not l.startswith("---"))))
        for f in sorted(OUR_BUNDLE.rglob("*.py")):
            rel = f.relative_to(OUR_BUNDLE)
            if not (bundle / rel).is_file() and "sglang" not in str(rel):
                changed.append(("НЕТ У НИХ", rel, 0, 0))
        print("  файлов обвязки с отличиями: %d" % len(changed))
        for kind, rel, add, rem in sorted(changed, key=lambda x: -(x[2] + x[3]))[:25]:
            print("    %-9s +%-5d −%-5d %s" % (kind, add, rem, rel))
        # 26.09: ляжет ли наш патч v3 (запрет повтора пустого хода + три правки промпта) на их бандл
        trial = work / "patch_trial"; shutil.rmtree(trial, ignore_errors=True); shutil.copytree(bundle, trial)
        pr = subprocess.run(["patch", "-p1", "--dry-run", "-F3"], cwd=trial, capture_output=True, text=True,
                            stdin=open(ROOT / "docs/graft/nextfork_v3.patch"))
        bad = [l for l in pr.stdout.splitlines() if "FAILED" in l or "rej" in l or "Reversed" in l]
        print("  наш патч v3 на их бандл: %s" % ("ЛОЖИТСЯ ЧИСТО" if pr.returncode == 0 and not bad else "КОНФЛИКТЫ: " + "; ".join(bad[:6])))
    print("\nготово; всё лежит в", work)


if __name__ == "__main__":
    main()

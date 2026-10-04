"""Кернелы Франзена: пути к датасетам и моделям Kaggle монтирует то в /kaggle/input/datasets/<user>/<slug>,
то в /kaggle/input/<slug> (04.10: стенд events упал «cp: cannot stat .../datasets/dfranzen/taaf-kaggle-source-bundle-copy»,
см. также kaggle-competition-mount-path). Правка ячейки 4: после объявления путей — подобрать существующий вариант.
usage: .venv/bin/python scripts/patch_input_paths.py kernels/<кернел> [...]
"""
import json, sys
from pathlib import Path

MARK = "# NEXTFORK 04.10: пути /kaggle/input — подобрать смонтированный вариант"
FIX = MARK + r'''
import glob as _pg, os as _po
def _nf_resolve(path):
    if _po.path.exists(path):
        return path
    slug = path.rstrip('/').split('/')
    cands = []
    if '/datasets/' in path:
        name = slug[slug.index('datasets') + 2] if len(slug) > slug.index('datasets') + 2 else slug[-1]
        cands = [f'/kaggle/input/{name}'] + _pg.glob(f'/kaggle/input/**/{name}', recursive=False) + _pg.glob(f'/kaggle/input/*/{name}') + _pg.glob(f'/kaggle/input/*/*/{name}')
    elif '/models/' in path:
        name = slug[slug.index('models') + 2]
        cands = _pg.glob(f'/kaggle/input/{name}/**/1', recursive=True) + _pg.glob(f'/kaggle/input/**/{name}/**/1', recursive=True)
    for c in cands:
        if _po.path.exists(c):
            print(f'путь {path} не найден, беру {c}', flush=True)
            return c
    print(f'ВНИМАНИЕ: путь {path} не найден, смонтировано: {sorted(_po.listdir("/kaggle/input"))}', flush=True)
    return path
WHEELHOUSE_DIR = _nf_resolve(WHEELHOUSE_DIR)
MODEL_DIR = _nf_resolve(MODEL_DIR)
DRAFT_MODEL_DIR = _nf_resolve(DRAFT_MODEL_DIR)
ORIG_BUNDLE_DIR = _nf_resolve(ORIG_BUNDLE_DIR)
'''


def main(paths):
    for p in paths:
        f = Path(p) / "submission.ipynb"
        nb = json.loads(f.read_text())
        c = nb["cells"][4]; s = "".join(c["source"])
        if MARK in s:
            print("уже:", p); continue
        anchor = "ORIG_BUNDLE_DIR   = '/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy'"
        i = s.find(anchor); assert i >= 0, p
        j = s.find("\n", i) + 1
        rest_vars = ["WHEELHOUSE_DIR", "MODEL_DIR", "DRAFT_MODEL_DIR"]
        assert all(v in s[:j] for v in rest_vars), "переменные путей объявлены не до ORIG_BUNDLE_DIR"
        c["source"] = (s[:j] + FIX + s[j:]).splitlines(keepends=True)
        compile("".join(c["source"]).replace("!rm", "#!rm").replace("!cp", "#!cp"), "c4", "exec")
        f.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
        print("исправлено:", p)


if __name__ == "__main__":
    main(sys.argv[1:])

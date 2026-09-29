"""Сверка: всё ли из nextfork/ доехало на Kaggle (25.09).

Зачем отдельным скриптом. Kaggle CLI режет `datasets files` на страницы по
200 записей независимо от --page-size и молча отдаёт только первую. В бандле
4410 файлов, поэтому сверка через CLI объявила бы неполным совершенно
нормальный пуш. Здесь страницы обходятся до конца.

Зачем сверка вообще. Kaggle умеет отчитаться об успехе, загрузив не всё:
02.09 режим `-r skip` выбросил весь src/, 30.08 загрузка оборвалась после
первого файла из 77 при статусе ready. Потеря каталога означает падение
прогона на импорте через сорок минут после старта.

usage: .venv/bin/python scripts/verify_nextfork_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "nextfork"
REF = "sergueimakarov/arc3-nextfork"

# Без этих файлов кернел не поднимется — называем поимённо.
CRITICAL = [
    "NEXTFORK_VERSION.txt",
    "taaf-kaggle-bundle.json",
    "deploy_target.pkl",
    "benchmark_initial.pkl",
    "serving_setup.py",
    "src/ARC3-Inference/inference/agent/noop_guard.py",
    "src/ARC3-Inference/inference/agent/tool_agent.py",
    "src/ARC3-Inference/inference/framework/solver.py",
]


def remote_names() -> list[str]:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    names: list[str] = []
    token = None
    while True:
        page = api.dataset_list_files(REF, page_token=token, page_size=200)
        batch = [f.name for f in page.files]
        names += batch
        token = getattr(page, "next_page_token", None)
        if not token or not batch:
            return names


def local_names() -> set[str]:
    out = set()
    for path in BUNDLE.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
            continue
        rel = path.relative_to(BUNDLE).as_posix()
        if rel == "dataset-metadata.json":      # Kaggle держит его манифестом, а не файлом
            continue
        out.add(rel)
    return out


def main() -> None:
    remote = set(remote_names())
    local = local_names()
    missing = sorted(local - remote)
    print("на Kaggle: %d, локально: %d" % (len(remote), len(local)))
    for name in CRITICAL:
        print("  %-5s %s" % ("есть" if name in remote else "НЕТ", name))
    if missing:
        print("\nНЕ уехало: %d" % len(missing))
        for name in missing[:20]:
            print("   ", name)
        sys.exit(1)
    if any(name not in remote for name in CRITICAL):
        print("\nотсутствует обязательный файл")
        sys.exit(1)
    print("\nпуш полный")


if __name__ == "__main__":
    main()

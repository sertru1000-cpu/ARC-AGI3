"""Сторож: код, который харнесс дописывает в песочницу, обязан жить в её белом списке.

Повод — прогон wm v9 (08→09.09). Дописанный хвост ловил `except NameError`, а в
белом списке песочницы Duck (`python_tool_sandbox.py`, `SAFE_BUILTINS`, 53 имени)
есть `Exception`, `RuntimeError`, `TypeError`, `ValueError` и НЕТ `NameError`.
Обработчик падал сам: «name 'NameError' is not defined». Цена: 230 ходов из 294
вернули модели трассировку вместо отчёта, автопроверок ноль во всех 25 играх,
прогон 2,2 ч измерил мой дефект, а не метод.

Ошибка статическая и ловится за секунду — если её искать. Скрипт разбирает
ячейку собранного ноутбука, достаёт строковые константы с кодом для песочницы
(`_WM_HELPERS`, `_WM_TAIL` и любые другие, помеченные `--names`), и называет
имена, которые в песочнице не определены: ни в белом списке, ни среди
рантайм-переменных, ни в самом коде.

usage:
    .venv/bin/python scripts/check_injected_code.py kernels/notebooks_stockflash_wm10/submission.ipynb
    .venv/bin/python scripts/check_injected_code.py <ipynb> --names _WM_HELPERS _WM_TAIL _WM_DIGEST_CODE
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

SANDBOX = Path("atlas_src/src/ARC3-Inference/inference/agent/python_tool_sandbox.py")

# Имена, которые песочница кладёт в область видимости сама (см. runtime_globals в её исходнике).
RUNTIME = {
    "current_frame", "latest_frame", "previous_frame", "history", "transitions", "last_transition",
    "last_action", "last_action_frame", "last_action_result", "valid_actions", "action", "memo",
    "animation", "try_actions", "save_checkpoint", "rollback", "request",
    # имена, которые пишет сама модель и которые харнесс подставляет перед своим хвостом
    "predict", "state_of", "goal",
}


def whitelist(path: Path) -> set[str]:
    m = re.search(r"SAFE_BUILTINS = \{(.*?)\}", path.read_text(encoding="utf-8"), re.S)
    if not m:
        raise SystemExit(f"{path}: не найден SAFE_BUILTINS — проверять не с чем")
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def constants(cell: str, names: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in ast.parse(cell).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in names:
            try:
                value = ast.literal_eval(node.value)
            except Exception:
                continue
            if isinstance(value, str):
                out[node.targets[0].id] = value
    return out


def unknown_names(code: str, allowed: set[str]) -> tuple[set[str], set[str]]:
    """Возвращает (все неизвестные имена, из них использованные как исключения)."""
    tree = ast.parse(code)
    defined: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            defined.add(n.id)
        elif isinstance(n, ast.arg):
            defined.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                defined.add(al.asname or al.name.split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            defined.add(n.name)
    known = defined | allowed | RUNTIME
    bad = {n.id for n in ast.walk(tree)
           if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in known}
    bad_exc = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ExceptHandler) and n.type is not None:
            for t in ast.walk(n.type):
                if isinstance(t, ast.Name) and t.id not in allowed:
                    bad_exc.add(t.id)
    return bad, bad_exc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--names", nargs="*", default=["_WM_HELPERS", "_WM_TAIL", "_WM_DIGEST_CODE"])
    ap.add_argument("--sandbox", default=str(SANDBOX))
    a = ap.parse_args()

    allowed = whitelist(Path(a.sandbox))
    nb = json.load(open(a.notebook, encoding="utf-8"))
    cell = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    found = constants(cell, a.names)
    if not found:
        print(f"в ноутбуке нет ни одной из констант {a.names} — проверять нечего")
        return 0

    print(f"белый список песочницы: {len(allowed)} имён; исключения в нём: "
          f"{', '.join(sorted(n for n in allowed if n.endswith('Error') or n == 'Exception'))}")
    fail = False
    for name, code in sorted(found.items()):
        bad, bad_exc = unknown_names(code, allowed)
        if bad_exc:
            fail = True
            print(f"  {name}: ЗАПРЕЩЁННЫЕ ИСКЛЮЧЕНИЯ {sorted(bad_exc)} — обработчик упадёт сам")
        rest = sorted(bad - bad_exc)
        if rest:
            print(f"  {name}: имена вне песочницы {rest} — проверить, определены ли они рядом")
        if not bad:
            print(f"  {name}: ок, {len(code.splitlines())} строк")
    if fail:
        print("\nПУШИТЬ НЕЛЬЗЯ: дописанный код упадёт в песочнице раньше, чем сработает.")
        return 1
    print("\nок: дописанный код укладывается в белый список песочницы")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

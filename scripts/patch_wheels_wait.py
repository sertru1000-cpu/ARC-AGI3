"""Лечение падения «No matching distribution found for arc-agi» на старте кернела (26.09).

Что видели: twin v1, twin v2 и cons1h v1 упали в ячейке установки — на машине не было
/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels. Метаданные у них те же, что у
кернелов, которые стартовали нормально (vision, twin2, wmprobe); twin2 при этом нашёл колёса по обычному пути.
Значит, сбой случайный — ПРЕДПОЛОЖЕНИЕ: данные соревнования монтируются с задержкой, а установка идёт сразу.

Правка ячейки 5: ждать обычный путь до 10 минут; если не появился — искать arc_agi_3_wheels по всему
/kaggle/input и печатать, что смонтировано. Ячейка 15 берёт environment_files рядом с найденными колёсами.
usage: .venv/bin/python scripts/patch_wheels_wait.py kernels/notebooks_nextfork_open1h [...]
"""
import ast, json, sys

PRE = '''# 26.09: данные соревнования монтируются то в /kaggle/input/competitions/<slug>, то в /kaggle/input/<slug>
# (cons1h v2: второй путь; первая версия правки ждала первый путь 10 минут впустую). Сначала оба известных пути,
# затем поиск по /kaggle/input, и только если не нашлось нигде — ждать до 10 минут.
import time as _wt
from pathlib import Path as _WP
_WCANDS = ["/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
           "/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"]
_WHEELS = None
_w0 = _wt.time()
while _WHEELS is None:
    _WHEELS = next((c for c in _WCANDS if _WP(c).is_dir()), None)
    if _WHEELS is None:
        _found = [p for p in _WP("/kaggle/input").rglob("arc_agi_3_wheels") if p.is_dir()]
        _WHEELS = str(_found[0]) if _found else None
    if _WHEELS is None:
        if _wt.time() - _w0 > 600:
            print("WHEELS: не найдено за 600 с, смонтировано:", sorted(str(p) for p in _WP("/kaggle/input").glob("*")), flush=True)
            _WHEELS = _WCANDS[0]
            break
        _wt.sleep(10)
print("WHEELS: %s через %.0f с" % (_WHEELS, _wt.time() - _w0), flush=True)
'''
OLD5 = '"/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",'
OLD15 = 'Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent'


def patch(d):
    import glob
    p = (glob.glob(d.rstrip("/") + "/*.ipynb") or [d.rstrip("/") + "/submission.ipynb"])[0]
    nb = json.load(open(p, encoding="utf-8"))
    c5 = "".join(nb["cells"][5]["source"])
    if "_WCANDS" in c5:
        print("уже исправлен:", d); return
    if "_WHEELS" in c5:                      # первая версия правки (ждала 10 минут) -> снять и поставить новую
        c5 = c5[c5.index("# Install the ARC runtime"):] if "# Install the ARC runtime" in c5 else c5[c5.index("subprocess.check_call"):]
        c5 = c5.replace("_WHEELS,", '"/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",', 1)
    if c5.count(OLD5) != 1:
        raise SystemExit("ячейка 5 не та: " + d)
    nb["cells"][5]["source"] = (PRE + c5.replace(OLD5, "_WHEELS,")).splitlines(keepends=True)
    hit15 = 0
    for c in nb["cells"]:          # ячейка запуска: 15 у нас, 16 у Scott Le Grand — ищем по тексту
        s = "".join(c["source"])
        if c["cell_type"] == "code" and OLD15 in s:
            c["source"] = s.replace(OLD15, "Path(_WHEELS).parent").splitlines(keepends=True); hit15 += 1
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "c", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("исправлен:", d, "| ячейка 15:", "да" if hit15 else "пути не было")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        patch(d)

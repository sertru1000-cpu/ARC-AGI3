"""Проверка дозовой сборки: конкурентность падает ТОЛЬКО вне боя, остальное стоковое.

Правило проекта: боевая ветка не должна меняться ни одной сборкой-замером. Здесь это
проверяется исполнением самого текста патча в двух режимах (TRUE_SUBMISSION False и True)
на подставном `bm`, плюс побайтовым сравнением всех прочих ячеек со стоковым ноутбуком.

usage:  .venv/bin/python scripts/test_dose_patch.py
"""
import json
import types
from pathlib import Path

BUILT = Path("kernels/notebooks_stockflash_dose/submission.ipynb")
CELLF = Path("kernels/notebooks_stockflash_dose/cell15.py")
STOCK = Path("kernels/notebooks_stockflash/submission.ipynb")
CONC = 13

ok = lambda b, m: print(("ok   " if b else "СБОЙ ") + m)
fails = []


def check(b, m):
    ok(b, m)
    if not b:
        fails.append(m)


built = json.load(open(BUILT, encoding="utf-8"))
stock = json.load(open(STOCK, encoding="utf-8"))
cell = CELLF.read_text(encoding="utf-8")

# --- 1. изменена ровно одна ячейка ---
diff = [i for i in range(len(stock["cells"]))
        if "".join(built["cells"][i]["source"]) != "".join(stock["cells"][i]["source"])]
check(diff == [15], "изменена только ячейка 15 (различий: %s)" % diff)
check(len(built["cells"]) == len(stock["cells"]), "число ячеек не изменилось")

# --- 2. стоковые настройки на месте ---
c13 = "".join(stock["cells"][13]["source"])
check("bm.solver.concurrency = 28" in c13, "в стоковой ячейке 13 конкурентность 28 — её не трогаем")
b13 = "".join(built["cells"][13]["source"])
check(b13 == c13, "ячейка 13 в сборке побайтово стоковая")

# --- 3. патч вне боя: конкурентность 13, потолок на игру не тронут ---
def fake_bm():
    return types.SimpleNamespace(solver=types.SimpleNamespace(
        concurrency=28, max_runtime_s_per_game=7920.0, max_actions_per_game=None))


bm_off = fake_bm()
ns_off = {"TRUE_SUBMISSION": False, "bm": bm_off, "print": lambda *a, **k: None}
exec(compile(cell, "cell15", "exec"), ns_off)
check(bm_off.solver.concurrency == CONC, "вне боя: конкурентность %d" % CONC)
check(bm_off.solver.max_runtime_s_per_game == 7920.0, "вне боя: потолок на игру остался 7920 с")
check(ns_off["_DOSE_CONC"] == CONC, "константа дозы видна в пространстве имён")

# --- 4. боевая ветка не тронута ---
bm_on = fake_bm()
ns_on = {"TRUE_SUBMISSION": True, "bm": bm_on, "print": lambda *a, **k: None}
exec(compile(cell, "cell15", "exec"), ns_on)
check(bm_on.solver.concurrency == 28, "бой: конкурентность осталась стоковой 28")
check(bm_on.solver.max_runtime_s_per_game == 7920.0, "бой: потолок на игру остался 7920 с")

# --- 5. печать ворот: они должны быть в логе прогона, иначе разбор не с чем сверять ---
printed = []
bm_p = fake_bm()
exec(compile(cell, "cell15", "exec"),
     {"TRUE_SUBMISSION": False, "bm": bm_p, "print": lambda *a, **k: printed.append(" ".join(map(str, a)))})
line = " ".join(printed)
check(line.startswith("DOSE:"), "в лог печатается строка DOSE")
for frag in ("вызовов на игру >= 95", "130-180", "побед-поражений", "неразличима"):
    check(frag in line, "в строке DOSE есть условие: %s" % frag)
check("%" not in line.replace("+-5%", "").replace("%d", ""), "формат строки собран без остатков подстановки")

# --- 6. две волны при 25 играх ---
waves = -(-25 // CONC)
check(waves == 2, "25 игр при конкурентности %d дают %d волны" % (CONC, waves))

# --- 7. метаданные ---
meta = json.load(open("kernels/notebooks_stockflash_dose/kernel-metadata.json", encoding="utf-8"))
check(meta["id"] == "sergueimakarov/arc3-stock-flash-sched", "слаг рабочий: %s" % meta["id"])
check("arc-prize-2026-arc-agi-3" in str(meta.get("competition_sources")), "данные соревнования подключены")

print()
print("ПРОВАЛОВ: %d" % len(fails))
for f in fails:
    print("  -", f)
raise SystemExit(1 if fails else 0)

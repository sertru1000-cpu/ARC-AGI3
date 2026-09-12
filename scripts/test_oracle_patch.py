"""Проверка оракул-инъекции на НАСТОЯЩИХ классах бандла Duck.

Проверяется ровно то, что может сломаться и чего не видно глазами:
  * подсказка уходит ТОЛЬКО целевым играм, у остальных промпт побайтово стоковый;
  * до хода 50 промпт стоковый даже у целевой игры;
  * обёртка возвращает НАСТОЯЩИЙ агент со всеми стоковыми параметрами (шов _make_analyzer,
    а не analyzer_factory: тот вызывается без local_server и увёл бы агента мимо сервера);
  * боевая ветка не тронута;
  * текст подсказки добавляется один раз и в конец, стоковая часть не изменена;
  * учёт ведётся (строка [ORACLE] по одной на игру).

usage:  .venv/bin/python scripts/test_oracle_patch.py [--bundle /tmp/duckbundle]
"""
import argparse
import json
import sys
import types
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", default="/tmp/duckbundle")
ap.add_argument("--cell", default="kernels/notebooks_stockflash_oracle/cell15.py")
a = ap.parse_args()

SRC = Path(a.bundle) / "src" / "ARC3-Inference"
if not SRC.exists():
    raise SystemExit("нет бандла: %s" % SRC)
sys.path.insert(0, str(SRC))
extra = Path(a.bundle) / "src" / "tufa-arc-agi-framework" / "src"
if extra.exists():
    sys.path.insert(0, str(extra))

import inference.framework.solver as solver  # noqa: E402

ok = lambda b, m: print(("ok   " if b else "СБОЙ ") + m)
fails = []


def check(b, m):
    ok(b, m)
    if not b:
        fails.append(m)


print("настоящий класс:", solver.HarnessSolver.__module__ + "." + solver.HarnessSolver.__name__)
check(hasattr(solver.HarnessSolver, "_make_analyzer"), "шов _make_analyzer существует у настоящего класса")

# --- подставной агент: проверяем, что обёртка НЕ подменяет его, а только оборачивает промпт ---
BUILT = {"n": 0}


class FakeAgent:
    def __init__(self, marker):
        self.marker = marker

    def _build_user_prompt(self, action_num, **kw):
        return "STOCK PROMPT action_num=%s" % action_num


orig_make = solver.HarnessSolver._make_analyzer


def fake_make(self, game, index, local_server=None):
    BUILT["n"] += 1
    return FakeAgent(marker=("server" if local_server is not None else "no-server"))


solver.HarnessSolver._make_analyzer = fake_make

cell = Path(a.cell).read_text(encoding="utf-8")
lines = []
ns = {"TRUE_SUBMISSION": False, "print": lambda *x, **k: lines.append(" ".join(map(str, x)))}
exec(compile(cell, "cell15", "exec"), ns)

check(solver.HarnessSolver._make_analyzer is not fake_make, "патч встал поверх стокового шва")
check(any(s.startswith("ORACLE:") for s in lines), "в лог печатается строка ORACLE с порогами")


def game(gid):
    return types.SimpleNamespace(game_run=types.SimpleNamespace(game_id=gid + "-deadbeef"))


# --- целевая игра ---
ag = solver.HarnessSolver._make_analyzer(None, game("tn36"), 0, "LOCAL")
check(isinstance(ag, FakeAgent), "возвращён НАСТОЯЩИЙ агент стокового пути, а не подменыш")
check(ag.marker == "server", "local_server доехал до стокового конструктора (иначе агент мимо сервера)")
early = ag._build_user_prompt(10)
check(early == "STOCK PROMPT action_num=10", "до хода 50 промпт побайтово стоковый")
late = ag._build_user_prompt(50)
check(late.startswith("STOCK PROMPT action_num=50"), "стоковая часть промпта не изменена")
check("Known mechanic for this game" in late, "на ходу 50 правило выдано")
check(late.count("Known mechanic for this game") == 1, "правило добавлено ровно один раз")
check("switches in the strip" in late, "выдано правило ИМЕННО этой игры (tn36)")
check(len(late) > len(early), "правило дописано в конец, а не вместо")

# --- нецелевая игра: контроль ---
ag2 = solver.HarnessSolver._make_analyzer(None, game("ft09"), 1, "LOCAL")
p2 = ag2._build_user_prompt(120)
check(p2 == "STOCK PROMPT action_num=120", "нецелевая игра: промпт стоковый даже на 120-м ходу")

# --- учёт ---
stats = ns["_oracle_stats"]
check(stats["injected"] == 1, "счётчик инъекций ведётся: %d" % stats["injected"])
check(list(stats["games"]) == ["tn36"], "в учёте только целевые игры: %s" % list(stats["games"]))
oracle_lines = [s for s in lines if s.startswith("[ORACLE]")]
check(len(oracle_lines) == 1, "строка [ORACLE] печатается один раз на игру: %d" % len(oracle_lines))

# --- боевая ветка ---
solver.HarnessSolver._make_analyzer = fake_make
ns2 = {"TRUE_SUBMISSION": True, "print": lambda *x, **k: None}
exec(compile(cell, "cell15", "exec"), ns2)
check(solver.HarnessSolver._make_analyzer is fake_make, "бой: шов не патчится вовсе")
ag3 = solver.HarnessSolver._make_analyzer(None, game("tn36"), 0, "LOCAL")
check(ag3._build_user_prompt(200) == "STOCK PROMPT action_num=200", "бой: подсказка не выдаётся ни на каком ходу")

# --- состав целей ---
hints = ns["_ORACLE_HINTS"]
check(len(hints) == 9, "целей 9: %s" % ",".join(sorted(hints)))
base = json.load(open("runs/flash_v1_phaseA/benchmark.json", encoding="utf-8"))
gs = base["game_runs"] if isinstance(base.get("game_runs"), list) else list(base["game_runs"].values())
lv = {str(r["game_id"])[:4]: int(r["levels_completed"]) for r in gs}
check(all(lv.get(g, 9) <= 1 for g in hints), "все цели застряли в базе на 0-1 уровне")
check(all(len(v) > 120 for v in hints.values()), "каждое правило содержательно (>120 знаков)")
for forbidden in ("row=", "col=", "первый ход", "sequence:"):
    check(not any(forbidden in v for v in hints.values()), "в правилах нет координат и готовых ходов: %r" % forbidden)

solver.HarnessSolver._make_analyzer = orig_make
print()
print("ПРОВАЛОВ: %d" % len(fails))
for f in fails:
    print("  -", f)
raise SystemExit(1 if fails else 0)

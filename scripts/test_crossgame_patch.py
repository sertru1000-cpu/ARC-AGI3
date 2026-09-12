"""Проверка накопления знания между играми на НАСТОЯЩИХ классах бандла Duck.

Проверяется то, что может сломаться молча:
  * шов патчится ТОЛЬКО вне боя; в бою ни n_passes, ни _make_analyzer не трогаются;
  * обёртка возвращает настоящий агент стокового пути вместе с local_server (через
    штатный analyzer_factory агент ушёл бы мимо локального сервера — эту ловушку мы уже ловили);
  * стоковая часть промпта не изменена, знание дописывается В КОНЕЦ;
  * динамическая сводка не показывается, пока фактов меньше порога, и считает типы действий верно;
  * регистрация идёт из previous_step_summary (взятие уровня и конец игры);
  * лишних вызовов модели нет ни одного;
  * сбой учёта не ломает игру.

usage:  .venv/bin/python scripts/test_crossgame_patch.py [--bundle /tmp/duckbundle]
"""
import argparse
import sys
import types
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", default="/tmp/duckbundle")
ap.add_argument("--cell", default="kernels/notebooks_stockflash_xgame/cell15.py")
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


class FakeAgent:
    def __init__(self, marker):
        self.marker = marker

    def _build_user_prompt(self, action_num, **kw):
        return "STOCK action_num=%s" % action_num


def fake_make(self, game, index, local_server=None):
    return FakeAgent(marker=("server" if local_server is not None else "no-server"))


def game(gid):
    return types.SimpleNamespace(game_run=types.SimpleNamespace(game_id=gid + "-deadbeef"))


orig_make = solver.HarnessSolver._make_analyzer
solver.HarnessSolver._make_analyzer = fake_make

cell = Path(a.cell).read_text(encoding="utf-8")
check("_chat_completion" not in cell, "лишних вызовов модели в патче нет")

lines = []
bm = types.SimpleNamespace(n_passes=1, solver=types.SimpleNamespace(concurrency=28, max_runtime_s_per_game=7920.0))
ns = {"TRUE_SUBMISSION": False, "bm": bm, "print": lambda *x, **k: lines.append(" ".join(map(str, x)))}
exec(compile(cell, "cell15", "exec"), ns)

check(bm.n_passes == 2, "вне боя: два прохода (волны, иначе знанию некому достаться)")
check(any(s.startswith("XGAME:") for s in lines), "в лог печатается строка XGAME с порогами")

ag = solver.HarnessSolver._make_analyzer(None, game("tn36"), 0, "LOCAL")
check(isinstance(ag, FakeAgent), "возвращён настоящий агент стокового пути")
check(ag.marker == "server", "local_server доехал до стокового конструктора")

p0 = ag._build_user_prompt(3)
check(p0.startswith("STOCK action_num=3"), "стоковая часть промпта не изменена")
check("Cross-game notes" in p0, "статический словарь механик выдан")
check("So far in this run" not in p0, "динамическая сводка молчит, пока фактов нет")

# --- регистрация взятых уровней: три факта от трёх игр ---
ag._build_user_prompt(10, previous_step_summary={"level_transition": True, "executed_actions": ["UP"]})
ag2 = solver.HarnessSolver._make_analyzer(None, game("sp80"), 1, "LOCAL")
ag2._build_user_prompt(11, previous_step_summary={"level_transition": True, "executed_actions": ["LEFT", "SPACE"]})
ag3 = solver.HarnessSolver._make_analyzer(None, game("lf52"), 2, "LOCAL")
ag3._build_user_prompt(12, previous_step_summary={"level_transition": True,
                                                  "executed_actions": ["MOUSE(row=4, col=7)"]})
xg = ns["_xg"]
check(len(xg["levels"]) == 3, "записаны три взятия уровня: %d" % len(xg["levels"]))
check(xg["games_scored"] == {"tn36", "sp80", "lf52"}, "учтены три разные игры: %s" % sorted(xg["games_scored"]))

p1 = ag._build_user_prompt(20)
check("So far in this run" in p1, "после порога динамическая сводка выдана")
check("3 levels were completed across 3 games" in p1, "сводка считает уровни и игры верно")
check("arrow 1" in p1 and "space 1" in p1 and "mouse 1" in p1,
      "типы действий разнесены верно (стрелка/SPACE/мышь): %s" % p1.splitlines()[-1][:120])

# --- конец игры отмечается ---
ag._build_user_prompt(21, previous_step_summary={"game_over": True, "executed_actions": []})
check("tn36" in xg["games_done"], "конец игры зарегистрирован")

# --- сбой учёта не ломает игру ---
class Boom:
    def get(self, *a, **k):
        raise RuntimeError("порча сводки")


p2 = ag._build_user_prompt(22, previous_step_summary=Boom())
check(p2.startswith("STOCK action_num=22"), "сбой учёта не сломал ход: промпт построен")
check(any("[XGAME] сбой учёта" in s for s in lines), "сбой учёта назван в логе, а не проглочен")

# --- боевая ветка не тронута ---
solver.HarnessSolver._make_analyzer = fake_make
bm2 = types.SimpleNamespace(n_passes=1, solver=types.SimpleNamespace(concurrency=28, max_runtime_s_per_game=7920.0))
ns2 = {"TRUE_SUBMISSION": True, "bm": bm2, "print": lambda *x, **k: None}
exec(compile(cell, "cell15", "exec"), ns2)
check(bm2.n_passes == 1, "бой: проходов по-прежнему один")
check(solver.HarnessSolver._make_analyzer is fake_make, "бой: шов не патчится вовсе")
ag4 = solver.HarnessSolver._make_analyzer(None, game("tn36"), 0, "LOCAL")
check(ag4._build_user_prompt(99) == "STOCK action_num=99", "бой: знание не выдаётся ни на каком ходу")

# --- содержание словаря: только переносимое, без правил конкретных игр ---
prior = ns["_XG_PRIOR"]
for forbidden in ("tn36", "sp80", "row=", "col="):
    check(forbidden not in prior, "в словаре нет привязки к конкретной игре: %r" % forbidden)
check(len(prior) > 500, "словарь содержателен (%d знаков)" % len(prior))

solver.HarnessSolver._make_analyzer = orig_make
print()
print("ПРОВАЛОВ: %d" % len(fails))
for f in fails:
    print("  -", f)
raise SystemExit(1 if fails else 0)

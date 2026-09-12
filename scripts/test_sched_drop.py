"""Проверка правила снятия на заглушке: отсчёт от старта, сброс на каждом уровне, снятие после порога."""
import sys, types
from pathlib import Path

class _Sess:
    def __init__(self): self.game = types.SimpleNamespace(
        current_state=types.SimpleNamespace(levels_completed=0),
        game_run=types.SimpleNamespace(game_id="tst", solver_note=""))

mod = types.ModuleType("inference.framework.solver")
mod._HarnessGameSession = _Sess
mod._HarnessGameSession.runtime_limit_reached = lambda self: False
sys.modules["inference"] = types.ModuleType("inference")
sys.modules["inference.framework"] = types.ModuleType("inference.framework")
sys.modules["inference.framework.solver"] = mod

CLOCK = [0.0]
import time as _t
_t.monotonic = lambda: CLOCK[0]

TRUE_SUBMISSION = False
class _BM:
    class solver: concurrency = 28; max_runtime_s_per_game = 7920.0
bm = _BM()
exec(Path("kernels/notebooks_stockflash_sched/cell13.py").read_text(encoding="utf-8"))

ok = lambda b, m: print(("ok   " if b else "СБОЙ ") + m)
ok(bm.solver.concurrency == 18, "одновременно 18 игр")
ok(bm.solver.max_runtime_s_per_game is None, "потолка на игру нет")

s = _Sess()
CLOCK[0] = 0.0
ok(s.runtime_limit_reached() is False, "на старте игра не снимается")
CLOCK[0] = 2699.0
ok(s.runtime_limit_reached() is False, "за секунду до порога игра жива")
CLOCK[0] = 2701.0
ok(s.runtime_limit_reached() is True, "после 45 минут без уровня игра снята")
ok("снята" in s.game.game_run.solver_note, "причина записана в отчёт игры")

s2 = _Sess()
CLOCK[0] = 0.0; s2.runtime_limit_reached()
CLOCK[0] = 2000.0
s2.game.current_state.levels_completed = 1          # взяла уровень
ok(s2.runtime_limit_reached() is False, "новый уровень сбрасывает отсчёт")
CLOCK[0] = 4600.0
ok(s2.runtime_limit_reached() is False, "после уровня отсчёт идёт заново, 2600 с ещё мало")
CLOCK[0] = 4800.0
ok(s2.runtime_limit_reached() is True, "45 минут уже без НОВОГО уровня — снятие")

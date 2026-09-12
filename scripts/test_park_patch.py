"""Проверка патча парковки на НАСТОЯЩЕМ классе сессии из бандла Duck.

ЗАЧЕМ ИМЕННО ТАК. Урок 09.09 (версия 9): код, дописанный по памяти о чужом коде, падал
в бою на том, чего в нём нет. Поэтому здесь берётся скачанный бандл
(`keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1`), импортируется настоящий
`inference.framework.solver`, и патч применяется к настоящему `_HarnessGameSession`.
Проверяются: срабатывание порога, удержание слота, отсчёт от каждого уровня, снятие
парковки при вымершей волне, и то, что боевой порог отличается от пробного.

usage:  .venv/bin/python scripts/test_park_patch.py [--bundle /tmp/duckbundle]
"""

import argparse
import sys
import threading
import time
import types
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", default="/tmp/duckbundle")
ap.add_argument("--cell", default="kernels/notebooks_stockflash_park/cell15.py")
ap.add_argument("--probe-acts", type=int, default=25, help="ожидаемый порог вне боя")
ap.add_argument("--probe-cap", type=float, default=1500.0, help="ожидаемый потолок вне боя, с")
ap.add_argument("--probe-not-before", type=float, default=0.0,
                help="ожидаемый запрет ранней парковки вне боя, с (0 — запрета нет)")
a = ap.parse_args()

SRC = Path(a.bundle) / "src" / "ARC3-Inference"
if not SRC.exists():
    raise SystemExit("нет бандла: %s (скачать: kaggle datasets download -d "
                     "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1)" % SRC)
sys.path.insert(0, str(SRC))
for extra in (Path(a.bundle) / "src" / "tufa-arc-agi-framework" / "src",):
    if extra.exists():
        sys.path.insert(0, str(extra))

import inference.framework.solver as solver  # noqa: E402

Sess = solver._HarnessGameSession
print("настоящий класс:", Sess.__module__ + "." + Sess.__name__)
assert hasattr(Sess, "should_stop") and hasattr(Sess, "runtime_limit_reached")

ok = lambda b, m: print(("ok   " if b else "СБОЙ ") + m)

# --- сессия НАСТОЯЩЕГО класса: создаём без __init__ и заполняем поля сами.
# Подменять класс своим нельзя: патч зовёт методы сессии (runtime_limit_reached),
# и на чужом объекте они падают в обработчик — ровно эту ошибку тест и поймал. ---
def make(levels=0, acts=0, cap=None, age=0.0):
    """age — сколько секунд игра уже идёт: с запретом ранней парковки это существенно."""
    s = object.__new__(Sess)
    # action_count у настоящего класса — свойство len(game_run.history), присвоить нельзя
    s.game = types.SimpleNamespace(
        current_state=types.SimpleNamespace(levels_completed=levels),
        game_run=types.SimpleNamespace(game_id="tst", state="playing", solver_note=None,
                                       history=[None] * acts))
    s.stop_event = threading.Event()
    s.solver = types.SimpleNamespace(max_runtime_s_per_game=cap, max_actions_per_game=None)
    s.started_at = time.monotonic() - age
    return s


# исходный should_stop подменяем на «игра идёт», чтобы мерить ТОЛЬКО нашу добавку
CALLS = {"orig": 0}


def orig_stop(self):
    CALLS["orig"] += 1
    return False


Sess.should_stop = orig_stop
Sess.runtime_limit_reached = lambda self: (
    self.solver.max_runtime_s_per_game is not None
    and time.monotonic() - self.started_at >= self.solver.max_runtime_s_per_game)

TRUE_SUBMISSION = False
bm = types.SimpleNamespace(solver=types.SimpleNamespace(max_runtime_s_per_game=7920.0))
ns = {"TRUE_SUBMISSION": TRUE_SUBMISSION, "bm": bm, "print": print}
exec(compile(Path(a.cell).read_text(encoding="utf-8"), "cell15", "exec"), ns)

ok(bm.solver.max_runtime_s_per_game == a.probe_cap, "вне боя: потолок %.0f с выставлен" % a.probe_cap)
ok(ns["_PARK_ACTS"] == a.probe_acts, "вне боя: порог %d ходов (боевой 100)" % a.probe_acts)

patched = Sess.should_stop
# Сессии, которые ДОЛЖНЫ паркуваться, обязаны быть старше запрета ранней парковки,
# иначе тест меряет запрет, а не логику порога. Потолок при этом выше возраста —
# иначе цикл удержания слота выйдет по runtime_limit_reached сразу.
AGE = (a.probe_not_before + 60.0) if a.probe_not_before > 0 else 0.0
s = make(levels=0, acts=0, cap=AGE + 2.0, age=AGE)
patched.__get__(s, Sess)  # noqa
ok(patched(s) is False, "на старте игра не паркуется")
s.game.game_run.history = [None] * (a.probe_acts - 1)
ok(patched(s) is False, "за ход до порога игра играет")

s2 = make(levels=1, acts=a.probe_acts * 4, cap=AGE + 2.0, age=AGE)
patched(s2)
s2.game.game_run.history = [None] * (a.probe_acts * 4 + a.probe_acts // 2)
ok(patched(s2) is False, "после уровня отсчёт идёт заново")
s2.game.current_state.levels_completed = 2
s2.game.game_run.history = [None] * (a.probe_acts * 4 + a.probe_acts + 5)
ok(patched(s2) is False, "новый уровень снова сбрасывает отсчёт")

# срабатывание с УДЕРЖАНИЕМ слота: должно занять около потолка, а не вернуться мгновенно
s3 = make(levels=0, acts=0, cap=AGE + 1.5, age=AGE)
patched(s3)
s3.game.game_run.history = [None] * (a.probe_acts + 5)
t0 = time.time()
res = patched(s3)
held = time.time() - t0
ok(res is True, "после порога сессия завершается")
ok(held >= 1.0, "парковка ДЕРЖАЛА слот %.1f с (не отпустила сразу)" % held)
ok(s3.game.game_run.solver_note and "park" in s3.game.game_run.solver_note,
   "причина записана в отчёт игры")

# вымершая волна: если играющих не осталось, парковка снимается сразу
ns["_park"]["playing"] = 0
s4 = make(levels=0, acts=0, cap=AGE + 600.0, age=AGE)
patched(s4)
s4.game.game_run.history = [None] * (a.probe_acts + 15)
t0 = time.time()
patched(s4)
ok(time.time() - t0 < 8.0, "волна вымерла — парковка снялась за %.1f с, а не ждала потолка"
   % (time.time() - t0))

ok(ns["_park"]["fired"] >= 2, "счётчик срабатываний ведётся: %d" % ns["_park"]["fired"])
ok(CALLS["orig"] > 0, "исходный should_stop вызывается (%d раз)" % CALLS["orig"])

# --- ЗАПРЕТ РАННЕЙ ПАРКОВКИ (правка после прогона 12.09: поле вымирало до окна замера) ---
ok(ns.get("_PARK_NOT_BEFORE_S") == a.probe_not_before,
   "вне боя: запрет ранней парковки %.0f с (%.0f мин)" % (a.probe_not_before, a.probe_not_before / 60))
ns["_park"]["playing"] = 0          # волна пуста: парковка не будет ждать, тест меряет только решение
s5 = make(levels=0, acts=0, cap=None)
patched(s5)
s5.game.game_run.history = [None] * (a.probe_acts + 25)
if a.probe_not_before > 0:
    ok(patched(s5) is False,
       "игра за порогом, но моложе %.0f мин — НЕ паркуется" % (a.probe_not_before / 60))
    s5.started_at = time.monotonic() - (a.probe_not_before + 60.0)
    ok(patched(s5) is True, "та же игра после %.0f-й минуты — паркуется" % (a.probe_not_before / 60))
else:
    ok(patched(s5) is True, "запрета нет: игра за порогом паркуется сразу")

# --- боевая ветка: порог 100, потолок на игру НЕ трогаем ---
Sess.should_stop = orig_stop          # возвращаем исходный, чтобы патч лёг заново
bm2 = types.SimpleNamespace(solver=types.SimpleNamespace(max_runtime_s_per_game=7920.0))
ns2 = {"TRUE_SUBMISSION": True, "bm": bm2, "print": lambda *a, **k: None}
exec(compile(Path(a.cell).read_text(encoding="utf-8"), "cell15", "exec"), ns2)
ok(ns2["_PARK_ACTS"] == 100, "бой: порог 100 ходов")
ok(bm2.solver.max_runtime_s_per_game == 7920.0, "бой: потолок на игру остался стоковым 7920 с")
ok(ns2.get("_PARK_NOT_BEFORE_S", 0.0) == 0.0, "бой: запрета ранней парковки нет — ветка не тронута")

sb = make(levels=0, acts=0, cap=1.0)
Sess.should_stop(sb)
sb.game.game_run.history = [None] * 99
ok(Sess.should_stop(sb) is False, "бой: на 99 ходах игра ещё играет")
sb.game.game_run.history = [None] * 100
ns2["_park"]["playing"] = 0            # чтобы не ждать потолка в тесте
ok(Sess.should_stop(sb) is True, "бой: на 100 ходах без уровня — парковка")

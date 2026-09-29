"""Достройка симулятора игры по ходу партии (варианты пода a8 → a8b → a8c, 27–28.09) — в коде, ВЫКЛЮЧЕНО по умолчанию.

Включение: NEXTFORK_BUILDWM=1 (ставится из inference/framework/solver.py после AGENTFIX и памяти функций).
Порог подсказки: NEXTFORK_BUILDWM_STALL (по умолчанию 8 ходов на уровне).

Что делает:
  * в каждый вызов песочницы дописывает помощники из buildwm_sandbox/: level_transitions, check_step (разбор
    ошибки «ваш step изменил / на самом деле изменилось», полоса индикатора не считается), заготовки base_step и
    tpl_step (шаблоны правил), plan(step, goal) — поиск в ширину внутри симулятора, run_plan(p, step) — ходы по одному
    со сверкой кадра и остановкой на первом расхождении, check_goal(goal), diff_rows;
  * разрешает в песочнице numpy, hashlib, time и классы (песочница — наша, список разрешённого задаём мы);
    песочница стартует с `python -I -S`, поэтому путь к numpy добавляется явно;
  * после STALL ходов на уровне дописывает в сообщение модели подсказку построить симулятор и искать план в нём.

Замеры (под, 1 ч): a8 (порог 20, без заготовки) — модель ни разу не написала step (0 из 334 блоков кода);
a8b (порог 8 + заготовка) — step в 15 из 532 блоков, 8 игр, трижды 100% совпадение, но план почти не искала.
Части Tycho (NIMI-research/Tycho: validate_plan, verify_outcome, diff_text) добавлены 28.09.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import inference.agent.python_tool_sandbox as _sb
import inference.agent.tool_agent as _ta

_PARTS_DIR = Path(__file__).with_name("buildwm_sandbox")
_PARTS = ("helpers.py", "base_step.py", "tpl_step.py", "plan.py", "tycho.py")
HELPERS = "\n".join((_PARTS_DIR / name).read_text(encoding="utf-8") for name in _PARTS)
STALL = int(os.environ.get("NEXTFORK_BUILDWM_STALL", "8") or 8)

BLOCK = (
    "\nThis level is taking many moves. base_step(rows, action) is predefined: it replays transitions already seen on "
    "this level and otherwise repeats what the same action did last time (moves the same-colored object, repeats the "
    "same cell changes); tpl_step(rows, action) is a stronger learned version (object moves with blocking and sliding, "
    "click recolors). check_step(tpl_step) shows where it fails. Build (or repair) a simulator of it: def step(rows, "
    "action) -> rows, where rows is the board as a list of letter strings (like current_frame.ascii.split(chr(10))) and "
    "action is e.g. 'RIGHT' or 'MOUSE(row=R, col=C)'. Your functions persist between python calls, so improve it step "
    "by step. check_step(step) tests it on every transition of this level and prints counterexamples. When it "
    "reproduces all of them, call plan(step, goal) with goal(rows) -> True when the level is won (clicks=[(row, col), "
    "...] for click games): it searches action sequences INSIDE step for free and returns the shortest list; execute it "
    "with run_plan(p, step) — it stops at the first move that diverges from step. Check your goal first: "
    "check_goal(goal) (it must be False on every board already seen on this level)."
)

# --- песочница: numpy / hashlib / time и классы ---
if '"numpy"' not in _sb._SANDBOX_BOOTSTRAP:
    import numpy as _np

    _sb._SANDBOX_BOOTSTRAP = _sb._SANDBOX_BOOTSTRAP.replace(
        '"bisect",', '"bisect",\n        "numpy",\n        "hashlib",\n        "time",', 1)
    _sb._SANDBOX_BOOTSTRAP = _sb._SANDBOX_BOOTSTRAP.replace(
        '"abs",', '"abs",\n        "__build_class__",\n        "__name__",\n        "staticmethod",', 1)
    _sb._SANDBOX_BOOTSTRAP = (
        "import sys as _bwsys0\n_bwsys0.path.append(%r)\n" % os.path.dirname(os.path.dirname(_np.__file__))
        + _sb._SANDBOX_BOOTSTRAP
    )

# --- помощники перед кодом модели ---
_orig_sandbox = _ta.run_sandboxed_python


def _sandbox(*, code, **kw):
    return _orig_sandbox(code=HELPERS + "\n" + str(code), **kw)


_ta.run_sandboxed_python = _sandbox

# --- подсказка после STALL ходов на уровне ---
STATS = {"hints": 0}
_orig_prompt = _ta.ToolAgent._build_user_prompt


def _prompt(self, action_num, **kw):
    text = _orig_prompt(self, action_num, **kw)
    try:
        cur = kw.get("current_frame")
        hist = kw.get("history_entries") or []
        if cur is not None:
            n = sum(1 for e in hist if getattr(getattr(e, "frame", None), "level", None) == cur.level)
            if n >= STALL:
                text += BLOCK
                STATS["hints"] += 1
                if STATS["hints"] in (1, 50, 200):
                    print("NEXTFORK BUILDWM: подсказка №%d (ходов на уровне %d)" % (STATS["hints"], n),
                          file=sys.__stderr__, flush=True)
    except Exception as exc:  # noqa: BLE001
        print("NEXTFORK BUILDWM: сбой подсказки: %r" % (exc,), file=sys.__stderr__, flush=True)
    return text


_ta.ToolAgent._build_user_prompt = _prompt
print("NEXTFORK BUILDWM: помощники симулятора в песочнице, подсказка после %d ходов на уровне" % STALL, flush=True)

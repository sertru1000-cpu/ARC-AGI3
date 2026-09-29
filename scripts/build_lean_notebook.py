"""Сборка `arc3-nextfork-lean`: перенос пространственного счёта из рассуждения в код — ЯЧЕЙКОЙ поверх датасета v3 (26.09).

Что измерено. Рассуждение модели — 2212 знаков на вызов (медиана, night_nextfork-b5); пересказ известного
обвязке — лишь 12% строк, зато ~40% строк — арифметика координат прозой: границы объектов, счёт клеток,
«col0: blue 19 → expect 20, got 18», чтение ascii-фрагментов в уме. Учитель (Gemini) на тех же играх делает
ровно это КОДОМ и печатью, рассуждения не пишет вовсе, и играет не хуже (см. память arc-agi-3-teacher-vs-ours).
Задержка запроса у нас 146 с, из них 127 с очередь к насыщенному серверу: выходные токены — единственный рычаг
на число ходов. noreason (07.09, 2.91 против 9.43) убирал рассуждение целиком; здесь — только его счётную часть.

Как. Блок в конец системного промпта через подмену модульной функции _build_system_prompt (ЛОВУШКА 25.09:
это функция модуля, не метод ToolAgent). Мерить по знакам рассуждения на вызов, вызовам и ходам на игру, охвату —
не по баллу.
usage: .venv/bin/python scripts/build_lean_notebook.py
"""
import ast, json, os

CELL_TEMPLATE = r'''
# =====================================================================
# LEAN: пространственный счёт — в код, не в рассуждение (26.09). Наложено ячейкой на датасет v3.
# ~40% строк рассуждения — координаты и арифметика прозой; учитель делает это кодом и не рассуждает.
# Выходные токены — единственный рычаг на число ходов (87% задержки — очередь к серверу).
# =====================================================================
import inference.agent.tool_agent as _lnta

_LEAN_BLOCK = (
    "\n"
    "Reasoning discipline (this saves you moves):\n"
    "- Do NOT do spatial bookkeeping in your head: no computing coordinates, bounding boxes, cell counts, "
    "distances, or object positions in prose, and no reading ASCII crops mentally. Write a few lines of "
    "Python that compute and print exactly those facts, then reason only over the printed output.\n"
    "- Do NOT restate the board, the valid action list, or the last action result in words -- the harness "
    "already gives them to you. Spend your reasoning on hypotheses, rules and the next decision.\n"
    "- Keep reasoning short: state the rule you are testing, the one move (or short batch) that tests it, "
    "and what result would confirm or refute it. Then act.\n"
    "- Ready-made helpers are predefined in every python call (do not re-implement them): "
    "`objs(fr=None, color=None, min_pixels=1)` -> objects with r0,r1,c0,c1,h,w,center,pixels; "
    "`summ(fr=None)` -> one line per object; `crop(r0,r1,c0,c1,fr=None)` -> board fragment; "
    "`diff(a=None,b=None)` -> changed cells, moved/appeared/gone objects between frames (default previous->current); "
    "`effects()` -> per action: tried / changed the board, from history; `grid(fr=None)`, `bb(node)`.\n"
)
_ln_orig = _lnta._build_system_prompt


def _ln_build_system_prompt(**kw):
    try:
        text = _ln_orig(**kw)
        return text if "Reasoning discipline" in text else text + _LEAN_BLOCK
    except Exception as exc:                        # слой не имеет права ронять прогон
        print("LEAN: сбой слоя, отдаю стоковый промпт: %r" % (exc,), flush=True)
        return _ln_orig(**kw)


_lnta._build_system_prompt = _ln_build_system_prompt

# Помощники песочницы: подставляются ПЕРЕД кодом модели в каждом вызове python. 37% строк её кода
# повторялись дословно (обход segmentation['nodes'] с ручными границами, split ascii) — теперь они готовые.
_LEAN_HELPERS = __HELPERS_REPR__
_ln_orig_sandbox = _lnta.run_sandboxed_python


def _ln_sandbox(*, code, **kw):
    try:
        return _ln_orig_sandbox(code=_LEAN_HELPERS + "\n" + str(code), **kw)
    except TypeError:
        return _ln_orig_sandbox(code=code, **kw)


_lnta.run_sandboxed_python = _ln_sandbox
print("LEAN: помощники песочницы подставляются в каждый вызов (%d строк)" % len(_LEAN_HELPERS.splitlines()), flush=True)
print("LEAN: слой включён, блок +%d знаков в системный промпт, поверх датасета версии 3" % len(_LEAN_BLOCK), flush=True)
'''


HELPERS = open("kernels/notebooks_nextfork_lean/sandbox_helpers.py", encoding="utf-8").read()
CELL = CELL_TEMPLATE.replace("__HELPERS_REPR__", repr(HELPERS))


def main() -> None:
    src = json.load(open("kernels/notebooks_nextfork/submission.ipynb", encoding="utf-8"))
    nb = json.loads(json.dumps(src)); cell = nb["cells"][15]; body = "".join(cell["source"])
    anchor = "    seconds=budget - 600.0\n)"; at = body.find(anchor)
    if at < 0 or body.find("await bm.run(") < 0:
        raise SystemExit("не нашёл стоковый soft_end или запуск прогона — сборка остановлена")
    at += len(anchor); code = body[:at] + "\n" + CELL + body[at:]; cell["source"] = code.splitlines(keepends=True)
    out = "kernels/notebooks_nextfork_lean"; os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(CELL)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_nextfork/kernel-metadata.json")); meta["id"] = "sergueimakarov/arc3-nextfork-lean"; meta["title"] = "arc3 nextfork lean"
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    compile(code, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    print("ok   патч ДО запуска прогона:", code.find("_ln_build_system_prompt") < code.find("await bm.run("))
    print("ok   модульная функция, не метод:", "_lnta._build_system_prompt =" in CELL and "ToolAgent._build_system_prompt" not in CELL)
    print("собрано:", out, "| датасет", meta["dataset_sources"][0])


if __name__ == "__main__":
    main()

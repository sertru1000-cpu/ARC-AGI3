"""Сборка `arc3-stock-flash-deadmouse`: харнесс сообщает, что мышь в этой игре не работает.

НАХОДКА, РАДИ КОТОРОЙ ЭТО ДЕЛАЕТСЯ (25.09, по записям базы, ноль квоты).
В sk48 модель сделала 284 хода и не взяла ни одного уровня из восьми. Из них 165 (58%) — клики,
которые НЕ МЕНЯЮТ на доске ничего. При этом первый уровень берётся 14 ходами ОДНИМИ простыми
действиями (`runs/bfs_originals.json`, поиск в ширину). То есть агент тычет мышью там, где надо
нажимать стрелки, видит «отклик» (кадр перерисовывается) и продолжает.

ЧТО ДЕЛАЕТ ПАТЧ. Считает ПОДРЯД идущие клики, не изменившие ни одной клетки. Как только их
набирается 12, дописывает в конец пользовательского промпта одну строку: мышь в этой игре, судя по
всему, не действует; работай простыми действиями и учти, что действие может требовать предусловия
(в sk48 тянуть фигуру можно только после того, как её пронзили). Ничего не запрещает — сообщает факт.

ПОЧЕМУ БЕЗОПАСНО, ИЗМЕРЕНО ДО ПУСКА (`scripts/click_modality_bound.py`, 25 партий, 4380 ходов):
порог достигается ТОЛЬКО в sk48 и НИ В ОДНОЙ игре не запрещает ни одного полезного клика — там, где
мышь работает, серия пустых кликов никогда не дорастает до двенадцати. Это первое наше правило с
доказанно нулевым вредом.

ПОЧЕМУ ВО ВХОД, А НЕ В ЗАПРЕТ. Входной токен в 2317 раз дешевле выходного, блок весит десятки
токенов. Выход модели не трогаем вовсе: опыт `noreason` 07.09 стоил двух третей балла (2.91 против 9.43).

ЧТО СМОТРЕТЬ В РЕЗУЛЬТАТЕ. Не общий балл (30 минут — шумная точка), а sk48 отдельно: сработала ли
подсказка, и сколько кликов сделано ПОСЛЕ неё. Если подсказка была, а клики продолжились — модель её
игнорирует, слой закрыть.

usage:  .venv/bin/python scripts/build_deadmouse_notebook.py
"""

import ast
import json
import os

CELL = r'''
# =====================================================================
# МЁРТВАЯ МЫШЬ: харнесс сообщает, что клики в этой игре ничего не меняют.
#
# Основание: sk48 — 284 хода, 0 уровней из 8, из них 165 (58%) пустые клики; первый уровень
# берётся 14 ходами одними простыми действиями. Вред измерен на 25 партиях: порог достигается
# только там, полезных кликов не запрещает нигде.
# =====================================================================
import inference.agent.tool_agent as _dmta

_DM_AFTER = 12           # столько пустых кликов подряд -- и мы говорим об этом модели
_DM_STATS = {"fired": 0, "games": set(), "clicks_after": 0}
_DM_TEXT = (
    "\n\n[HOST] MOUSE has produced NO board change in the last {n} clicks of this game. "
    "In some games the mouse does nothing at all and levels are solved with ACTION1..ACTION5 only. "
    "Consider dropping MOUSE here and working the simple actions -- including their ORDER: "
    "an action may only take effect AFTER another one has been performed first."
)

_dm_orig_prompt = _dmta.ToolAgent._build_user_prompt


def _dm_build_user_prompt(self, action_num, **kw):
    text = _dm_orig_prompt(self, action_num, **kw)
    try:
        hist = list(getattr(self, "history_entries", None) or [])
        streak = 0
        for prev, cur in zip(hist, hist[1:]):
            action = (getattr(cur, "action", "") or "").strip()
            if not action.startswith("MOUSE("):
                continue
            before, after = getattr(prev, "frame", None), getattr(cur, "frame", None)
            if before is None or after is None:
                continue
            if getattr(before, "grid", None) != getattr(after, "grid", None):
                streak = 0
            else:
                streak += 1
        if streak >= _DM_AFTER:
            if _DM_STATS["fired"] == 0 or streak == _DM_AFTER:
                print("DEADMOUSE: подсказка выдана (ход %s, пустых кликов подряд %d)"
                      % (action_num, streak), flush=True)
            _DM_STATS["fired"] += 1
            return text + _DM_TEXT.format(n=streak)
    except Exception as exc:                      # слой не имеет права ронять прогон
        print("DEADMOUSE: сбой слоя, пропускаю: %r" % (exc,), flush=True)
    return text


_dmta.ToolAgent._build_user_prompt = _dm_build_user_prompt

if not TRUE_SUBMISSION:
    bm.solver.max_runtime_s_per_game = 1800.0    # проба 30 минут: точка сравнения base30 = 3.06

print("DEADMOUSE: слой включён (%s). Порог %d пустых кликов подряд. Вред измерен на 25 партиях: "
      "срабатывает только в sk48, полезных кликов не запрещает нигде."
      % ("бой" if TRUE_SUBMISSION else "проба 30 мин", _DM_AFTER), flush=True)
'''


def main() -> None:
    src = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))
    nb = json.loads(json.dumps(src))
    cell = nb["cells"][15]
    body = "".join(cell["source"])
    anchor = "    seconds=budget - 600.0\n)"
    at = body.find(anchor)
    if at < 0 or body.find("await bm.run(") < 0:
        raise SystemExit("не нашёл стоковый soft_end или запуск прогона — сборка остановлена")
    at += len(anchor)
    code = body[:at] + "\n" + CELL + body[at:]
    cell["source"] = code.splitlines(keepends=True)

    out = "kernels/notebooks_stockflash_deadmouse"
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(CELL)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    meta["id"] = "sergueimakarov/arc3-stock-flash-deadmouse"
    meta["title"] = "arc3 stock flash deadmouse"
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)

    compile(code, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    diff = [i for i in range(len(nb["cells"])) if "".join(nb["cells"][i]["source"]) != "".join(src["cells"][i]["source"])]
    new = "".join(nb["cells"][15]["source"])
    print("ok   изменена только ячейка 15:", diff == [15])
    print("ok   патч ПОСЛЕ стокового soft_end:", new.find("_DM_AFTER") > new.find("budget - 600.0"))
    print("ok   патч ДО запуска прогона:", new.find("_DM_AFTER") < new.find("await bm.run("))
    print("ok   выход модели не трогаем:", "_run_python_tool" not in CELL and "_chat_completion" not in CELL)
    print("ok   слой не роняет прогон при сбое:", "except Exception" in CELL)
    print("собрано:", out)


if __name__ == "__main__":
    main()

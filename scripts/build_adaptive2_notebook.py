"""Сборка `arc3-stock-flash-adaptive2`: адаптивная подсказка ПЛЮС снятая фраза про stop probing.

Отличие от `adaptive`: к подсказке по условию добавлена вторая правка — та самая замена стоковой
строки, что проверялась отдельно как `nostop`. Основание: на мёртвых играх nostop сработал не хуже
precond (g50t 0->2 против 0->1), а на здоровых он их не коснётся, потому что подсказка не появится.

ОТКУДА. Первая пара прогонов 25.09 (полные, 132 мин, против базы flash_v1_phaseA = 40 уровней, 9.43):

    precond  36 уровней, 8.26 | знаки +4/−8, p = 0.39 -> неразличимо
    nostop   36 уровней, 6.39 | знаки +5/−7, p = 0.77 -> неразличимо

Но в разбивке по играм видно то, чего не видно в сумме. Игры, где база брала НОЛЬ:

    игра     база  precond  nostop
    sk48      0       0       0
    tn36      0       0       0
    sp80      0       1       1
    g50t      0       1       2

В двух из четырёх мёртвых игр уровни появились, независимо в обоих прогонах. И это ровно те игры,
про которые перебор (`scripts/precondition_search.py`) заранее сказал, что там есть оживающие
действия: в g50t три действия срабатывают после ACTION1. Где предпосылки не было (tn36) — ноль остался.

Платят за это сильные игры: ar25 −3 и −2, re86 −2 и −3, ft09, m0r0. Инструкция заставляет тщательнее
перепроверять действия, и это отнимает ходы у тех, кто и так шёл хорошо.

ЧТО ДЕЛАЕМ. Тот же текст, но в ПОЛЬЗОВАТЕЛЬСКИЙ промпт и только когда игра застряла: прошло
ATLAS_STUCK_AFTER ходов, а уровень не сменился. Порог выбран по данным: до первого уровня у взявших
медиана 24 хода, три четверти укладываются в 32, максимум 81. При пороге 45 здоровые игры блок
не получат вовсе — а мы теряли именно на них.

usage:  .venv/bin/python scripts/build_adaptive_notebook.py
"""

import ast
import json
import os

CELL = r'''
# =====================================================================
# ПРЕДУСЛОВИЯ ПО УСЛОВИЮ: подсказка только застрявшей игре.
#
# Первая пара 25.09: блок всегда -> мёртвые игры ожили (sp80 0->1, g50t 0->2), но сильные просели
# (ar25 −3, re86 −3). Здоровые игры берут первый уровень за 24 хода (медиана), 75% за 32.
# Порог 45 ходов без смены уровня оставляет их нетронутыми.
# =====================================================================
import inference.agent.tool_agent as _adta

_AD_STUCK_AFTER = 45          # ходов на одном уровне -- после этого считаем, что игра встала
_AD_STATS = {"fired": 0, "games": 0}
_AD_BLOCK = (
    "\n\n[HOST] You have spent {n} actions on this level without clearing it. Before continuing, "
    "reconsider the actions you judged useless: an action that produced no visible change is NOT proven "
    "useless -- it may have an unmet PRECONDITION and start working after the board changes. Some actions "
    "only act on the second press. Pick one such action and retest it now, deliberately, after changing "
    "the board with another action. Think in terms of (PRECONDITION, ACTION) -> EFFECT, not ACTION -> EFFECT."
)

_ad_orig_prompt = _adta.ToolAgent._build_user_prompt


def _ad_build_user_prompt(self, action_num, **kw):
    text = _ad_orig_prompt(self, action_num, **kw)
    try:
        hist = list(getattr(self, "history_entries", None) or [])
        if not hist:
            return text
        cur_level = None
        frame = kw.get("current_frame")
        if frame is not None:
            cur_level = getattr(frame, "level", None)
        if cur_level is None:
            cur_level = getattr(getattr(hist[-1], "frame", None), "level", None)
        if cur_level is None:
            return text
        # сколько ходов подряд мы на этом уровне
        n = 0
        for e in reversed(hist):
            lv = getattr(getattr(e, "frame", None), "level", None)
            if lv != cur_level:
                break
            n += 1
        if n >= _AD_STUCK_AFTER:
            if _AD_STATS["fired"] == 0 or n == _AD_STUCK_AFTER:
                print("ADAPTIVE: подсказка выдана (уровень %s, ходов на нём %d)" % (cur_level, n), flush=True)
            _AD_STATS["fired"] += 1
            return text + _AD_BLOCK.format(n=n)
    except Exception as exc:                        # слой не имеет права ронять прогон
        print("ADAPTIVE: сбой слоя, пропускаю: %r" % (exc,), flush=True)
    return text


_adta.ToolAgent._build_user_prompt = _ad_build_user_prompt

# вторая правка: снять запрет на пробы для действий без НАБЛЮДЁННОГО эффекта (системный промпт)
_NS_OLD = ("Once the important state variables and action effects are sufficiently understood, "
           "stop probing and search in the inferred state space.")
_NS_NEW = ("Once the important state variables are understood, prefer searching in the inferred state "
           "space over further probing -- but only stop probing an ACTION whose effect you have actually "
           "OBSERVED at least once. An action that has produced no visible change anywhere is still "
           "UNKNOWN, not known-useless: it may have an unmet precondition, or it may need to be pressed "
           "twice. Keep such actions on the list of things to retest after the board changes.")
_ad_orig_system = _adta._build_system_prompt


def _ad_build_system_prompt(**kw):
    try:
        text = _ad_orig_system(**kw)
        if _NS_OLD in text:
            return text.replace(_NS_OLD, _NS_NEW)
        return text
    except Exception as exc:
        print("ADAPTIVE2: сбой правки системного промпта: %r" % (exc,), flush=True)
        return _ad_orig_system(**kw)


_adta._build_system_prompt = _ad_build_system_prompt


if not TRUE_SUBMISSION:
    bm.solver.max_runtime_s_per_game = 7920.0       # полный прогон 132 мин; сравнение flash_v1_phaseA = 9.43

print("ADAPTIVE2: два слоя включены (%s). Порог %d ходов на уровне без его взятия. Здоровые игры берут "
      "первый уровень за 24 хода (медиана), 75%% за 32 -- их подсказка не коснётся. Плюс снята стоковая фраза про stop probing."
      % ("бой" if TRUE_SUBMISSION else "полный прогон 132 мин", _AD_STUCK_AFTER), flush=True)
'''


def main() -> None:
    # С 25.09 слои накатываются поверх НАШЕГО форка (kernels/notebooks_nextfork),
    # а не поверх чужой стоковой сборки: в форке живёт запрет повторного
    # пустого хода, и мерить надстройку надо относительно него.
    src = json.load(open("kernels/notebooks_nextfork/submission.ipynb", encoding="utf-8"))
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

    out = "kernels/notebooks_nextfork_adaptive2"
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(CELL)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_nextfork/kernel-metadata.json"))
    meta["id"] = "sergueimakarov/arc3-nextfork-adaptive2"
    meta["title"] = "arc3 nextfork adaptive2"
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)

    compile(code, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    diff = [i for i in range(len(nb["cells"])) if "".join(nb["cells"][i]["source"]) != "".join(src["cells"][i]["source"])]
    new = "".join(nb["cells"][15]["source"])
    print("ok   изменена только ячейка 15:", diff == [15])
    print("ok   патч ДО запуска прогона:", new.find("_AD_BLOCK") < new.find("await bm.run("))
    print("ok   правим пользовательский промпт (нужна адаптивность):", "_build_user_prompt" in CELL)
    print("ok   обе правки в сборке:", "_AD_BLOCK" in CELL and "_NS_OLD" in CELL)
    print("ok   слой не роняет прогон:", "except Exception" in CELL)
    print("собрано:", out)


if __name__ == "__main__":
    main()

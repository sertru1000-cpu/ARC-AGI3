"""Сборка `arc3-stock-flash-precond`: действие без эффекта — не бесполезное, а с невыполненным условием.

НАХОДКА, РАДИ КОТОРОЙ ЭТО ДЕЛАЕТСЯ (25.09, перебором на локальном движке, ноль квоты).
Взяли действия, которые в стартовом состоянии НЕ МЕНЯЮТ кадр, и перебором проверили, оживают ли они
после коротких последовательностей других действий (`scripts/precondition_search.py`):

    sk48   мёртвых 2 -> ожило 1   (ACTION2 работает после ACTION1)
    g50t   мёртвых 3 -> ожило 3   (все три работают после ACTION1)
    tn36   мёртвых 5 -> ожило 0
    итого  10 -> 4, то есть 40% мёртвых действий оживают ПОСЛЕ ОДНОГО ХОДА.

В g50t показательный случай: ACTION1 не действует с первого нажатия и действует со второго —
переключатель. Модель, попробовав его один раз, записывает в бесполезные и больше не возвращается.

ЧТО МЕНЯЕМ В ПРОМПТЕ. Стоковый системный промпт учит связке «действие -> эффект» и прямо велит
прекращать пробы, когда эффекты «достаточно поняты» (строки 62-63 стокового текста). Мы дописываем
абзац, меняющий ЕДИНИЦУ ГИПОТЕЗЫ с «A даёт E» на «A даёт E при условии P»: пустой исход не доказывает
бесполезность, у действия может быть невыполненное предусловие, и его надо искать, а действие
перепроверять после смены состояния.

ПОЧЕМУ ЭТО НЕ ПОВТОРЯЕТ 22 ЗАКРЫТЫЕ ВЕТКИ. Там мы меняли КОЛИЧЕСТВО информации у модели (история,
подсказки о повторах, перенос правил, список испробованного). Здесь количество информации то же —
меняется форма причинного поиска. Разбор внешнего критика 25.09.

ЧТО СМОТРЕТЬ. Полный прогон 132 мин, прямая точка сравнения — runs/flash_v1_phaseA: RHAE 9.43,
40 уровней, первый уровень в 21 игре из 25. Ключевое — ЧИСЛО УРОВНЕЙ (балл упирается только в него)
и судьба tn36/g50t/sk48/sp80, где сейчас голый ноль.

usage:  .venv/bin/python scripts/build_precond_notebook.py
"""

import ast
import json
import os

CELL = r'''
# =====================================================================
# ПРЕДУСЛОВИЯ: пустой исход не доказывает бесполезность действия.
#
# Измерено перебором на локальном движке (25.09): 40% действий, не меняющих кадр в стартовом
# состоянии, оживают ПОСЛЕ ОДНОГО ХОДА (sk48: ACTION2 после ACTION1; g50t: три действия после
# ACTION1 — переключатель, срабатывающий со второго нажатия). Стоковый промпт учит связке
# «действие -> эффект» и велит прекращать пробы, когда эффекты поняты; из-за этого отвергнутое
# действие больше не проверяется никогда.
# =====================================================================
import inference.agent.tool_agent as _pcta

_PC_BLOCK = (
    "\n"
    "Conditional effects (important):\n"
    "- An action that produces NO visible change is NOT proven useless. It may have an unmet "
    "PRECONDITION: the same action can become effective after the state changes.\n"
    "- Model rules as (PRECONDITION P, ACTION A) -> EFFECT E, not as A -> E. When A yields nothing, "
    "record the condition C under which it did nothing, and keep the hypothesis that some P makes A work.\n"
    "- Before writing an action off, retest it at least once after the board has changed in a way you "
    "caused deliberately -- in particular after pressing each of the other simple actions once.\n"
    "- Some actions toggle: the first press arms something invisible and the second press acts. "
    "If an action seems dead, try it twice in a row before discarding it.\n"
    "- 'Stop probing' applies only to actions whose effect you have OBSERVED. An action with no "
    "observed effect anywhere is still unknown, not known-useless.\n"
)

_pc_orig_system = _pcta._build_system_prompt


def _pc_build_system_prompt(**kw):
    try:
        text = _pc_orig_system(**kw)
        if "Conditional effects (important)" in text:
            return text
        return text + _PC_BLOCK
    except Exception as exc:                       # слой не имеет права ронять прогон
        print("PRECOND: сбой слоя, отдаю стоковый промпт: %r" % (exc,), flush=True)
        return _pc_orig_system(**kw)


_pcta._build_system_prompt = _pc_build_system_prompt

if not TRUE_SUBMISSION:
    bm.solver.max_runtime_s_per_game = 7920.0     # ПОЛНЫЙ прогон 132 мин: точка сравнения
                                                  # runs/flash_v1_phaseA = 9.43, 40 уровней, первый в 21/25

print("PRECOND: слой включён (%s). Основание: перебором на локальном движке 40%% мёртвых действий "
      "оживают после одного хода (sk48 ACTION2 после ACTION1; g50t три действия после ACTION1). "
      "Блок в системный промпт: +%d знаков."
      % ("бой" if TRUE_SUBMISSION else "полный прогон 132 мин", len(_PC_BLOCK)), flush=True)
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

    out = "kernels/notebooks_stockflash_precond"
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(CELL)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    meta["id"] = "sergueimakarov/arc3-stock-flash-precond"
    meta["title"] = "arc3 stock flash precond"
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)

    # ЛОВУШКА 25.09: _build_system_prompt -- МОДУЛЬНАЯ ФУНКЦИЯ, а не метод ToolAgent.
    # Патч по ToolAgent._build_system_prompt уронил два прогона через 17 минут каждый.
    _ta = open("atlas_src/src/ARC3-Inference/inference/agent/tool_agent.py", encoding="utf-8").read()
    _is_module_fn = "\ndef _build_system_prompt(" in _ta
    print("ok   _build_system_prompt — функция модуля:", _is_module_fn)
    if not _is_module_fn:
        raise SystemExit("в исходнике это не модульная функция — проверьте, как патчить")
    print("ok   патчим функцию модуля, не метод класса:", ".ToolAgent._build_system_prompt" not in CELL)

    compile(code, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    diff = [i for i in range(len(nb["cells"])) if "".join(nb["cells"][i]["source"]) != "".join(src["cells"][i]["source"])]
    new = "".join(nb["cells"][15]["source"])
    print("ok   изменена только ячейка 15:", diff == [15])
    print("ok   патч ПОСЛЕ стокового soft_end:", new.find("_PC_BLOCK") > new.find("budget - 600.0"))
    print("ok   патч ДО запуска прогона:", new.find("_PC_BLOCK") < new.find("await bm.run("))
    print("ok   трогаем только системный промпт:", "_build_user_prompt" not in CELL)
    print("ok   выход модели не трогаем:", "_run_python_tool" not in CELL and "_chat_completion" not in CELL)
    print("ok   слой не роняет прогон:", "except Exception" in CELL)
    print("собрано:", out)


if __name__ == "__main__":
    main()

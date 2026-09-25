"""Сборка `arc3-stock-flash-nostop`: правим стоковую фразу «перестань пробовать».

ОДНА ПЕРЕМЕННАЯ. Это второй из двух опытов по разбору внешнего критика 25.09, и он намеренно НЕ
содержит блока про предусловия (тот проверяется отдельно, `arc3-stock-flash-precond`). Здесь меняется
ровно одно предложение стокового системного промпта:

  было:  "Once the important state variables and action effects are sufficiently understood,
          stop probing and search in the inferred state space."

  стало: то же, но «перестать пробовать» разрешено только для действий, эффект которых УЖЕ НАБЛЮДАЛИ;
         действие без наблюдённого эффекта остаётся неизвестным, а не признанным бесполезным.

ПОЧЕМУ ИМЕННО ЭТА ФРАЗА. Критик указал на неё как на источник преждевременного закрытия: модель
получает прямое указание прекращать пробы, когда «эффекты достаточно поняты», а пустой исход она
засчитывает за понимание. Наш перебор на локальном движке это подтверждает численно
(`scripts/precondition_search.py`): 40% действий, не меняющих кадр в стартовом состоянии, оживают
после одного хода — sk48 (ACTION2 после ACTION1), g50t (три действия после ACTION1, переключатель).

ЕСЛИ ЗАМЕНА НЕ НАЙДЁТ СВОЮ СТРОКУ — сборка падает, а не тихо ставит слой-пустышку: молча
не сработавший патч мы уже проходили (dead-mouse, 0 срабатываний из-за несовпавших условий).

usage:  .venv/bin/python scripts/build_nostop_notebook.py
"""

import ast
import json
import os

CELL = r'''
# =====================================================================
# БЕЗ ПРЕЖДЕВРЕМЕННОГО ЗАКРЫТИЯ: «перестань пробовать» — только про наблюдённые эффекты.
#
# Стоковая строка велит прекращать пробы, когда эффекты «достаточно поняты». Пустой исход модель
# засчитывает за понимание и больше к действию не возвращается. Перебор на локальном движке (25.09):
# 40% мёртвых в старте действий оживают после ОДНОГО хода.
# =====================================================================
import inference.agent.tool_agent as _nsta

_NS_OLD = ("Once the important state variables and action effects are sufficiently understood, "
           "stop probing and search in the inferred state space.")
_NS_NEW = ("Once the important state variables are understood, prefer searching in the inferred state "
           "space over further probing -- but only stop probing an ACTION whose effect you have actually "
           "OBSERVED at least once. An action that has produced no visible change anywhere is still "
           "UNKNOWN, not known-useless: it may have an unmet precondition, or it may need to be pressed "
           "twice. Keep such actions on the list of things to retest after the board changes.")
_NS_DONE = {"ok": False}

_ns_orig_system = _nsta.ToolAgent._build_system_prompt


def _ns_build_system_prompt(*args, **kw):
    try:
        text = _ns_orig_system(*args, **kw)
        if _NS_OLD in text:
            if not _NS_DONE["ok"]:
                print("NOSTOP: строка найдена и заменена", flush=True)
                _NS_DONE["ok"] = True
            return text.replace(_NS_OLD, _NS_NEW)
        if not _NS_DONE["ok"]:
            print("NOSTOP: ВНИМАНИЕ — стоковая строка НЕ найдена, промпт отдан как есть", flush=True)
            _NS_DONE["ok"] = True
        return text
    except Exception as exc:                        # слой не имеет права ронять прогон
        print("NOSTOP: сбой слоя, отдаю стоковый промпт: %r" % (exc,), flush=True)
        return _ns_orig_system(*args, **kw)


_nsta.ToolAgent._build_system_prompt = _ns_build_system_prompt

if not TRUE_SUBMISSION:
    bm.solver.max_runtime_s_per_game = 7920.0       # ПОЛНЫЙ прогон 132 мин; сравнение с flash_v1_phaseA = 9.43

print("NOSTOP: слой включён (%s). Меняем ОДНО предложение стокового промпта: запрет прекращать пробы "
      "для действий без наблюдённого эффекта." % ("бой" if TRUE_SUBMISSION else "полный прогон 132 мин"),
      flush=True)
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

    # проверка, что заменяемая строка ДЕЙСТВИТЕЛЬНО есть в настоящем промпте прогона
    ref = "runs/deadmouse_flash/prompts"
    import glob
    found = False
    for f in glob.glob(ref + "/*.log")[:3]:
        if "stop probing and search in the inferred state space" in open(f, encoding="utf-8", errors="ignore").read():
            found = True
            break
    print("ok   заменяемая строка найдена в настоящем промпте прогона:", found)
    if not found:
        raise SystemExit("строки нет в реальном промпте — сборка остановлена, иначе слой был бы пустышкой")

    out = "kernels/notebooks_stockflash_nostop"
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(CELL)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    meta["id"] = "sergueimakarov/arc3-stock-flash-nostop"
    meta["title"] = "arc3 stock flash nostop"
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)

    compile(code, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    diff = [i for i in range(len(nb["cells"])) if "".join(nb["cells"][i]["source"]) != "".join(src["cells"][i]["source"])]
    new = "".join(nb["cells"][15]["source"])
    print("ok   изменена только ячейка 15:", diff == [15])
    print("ok   патч ДО запуска прогона:", new.find("_NS_OLD") < new.find("await bm.run("))
    print("ok   блока про предусловия здесь НЕТ (одна переменная):", "Conditional effects" not in CELL)
    print("ok   слой не роняет прогон:", "except Exception" in CELL)
    print("собрано:", out)


if __name__ == "__main__":
    main()

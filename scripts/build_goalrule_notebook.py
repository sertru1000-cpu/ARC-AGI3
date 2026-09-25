"""Сборка `arc3-stock-flash-goalrule`: цель уровня как ПРАВИЛО, названное вслух и перепроверенное.

ЧТО ИМЕННО ПРОВЕРЯЕМ И ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ ЗАКРЫТОГО.

Закрыто 19.09 (`scripts/goal_progress_test.py`): цель как КАДР. Брали кадр победы и считали расстояние
до него как меру прогресса. С настоящей целью выигрышный ход угадывался в 44% случаев против 19%
случайного; с ПЕРЕНЕСЁННОЙ — 19% при случайном 18%, и в 53% состояний мера одинакова для всех ходов.
Вывод тогда: связка «цель -> выбор хода» сигнала не несёт, узкое место — ОПОЗНАНИЕ цели.

Здесь другое: цель как ПРАВИЛО, словами («совместить фигуры одного цвета»), и работает она не на выбор
хода, а на опознание — то есть ровно на то место, которое сентябрьский замер назвал узким.

Закрыто 17.09 (`carry`): перенос ПУТИ прохождения уровня. RHAE 4.27 против баз 6.11/5.40/3.91,
знаки p = 0.30/0.45/1.00 — неотличимо. Здесь путь не переносится вовсе.

ЧТО В СТОКЕ СЕЙЧАС. Цель упомянута вскользь, одним пунктом внутри «рабочей модели мира»
(«what the goal likely is», строка 60 стокового промпта). Нет ни требования назвать цель отдельной
строкой, ни указания перепроверить её при смене уровня.

ОСНОВАНИЕ, ЧТО ПЕРЕНОС В ПРИНЦИПЕ РАБОТАЕТ: эффективность по номерам уровней у нас 1.10 / 1.00 / 1.47 /
1.17 / 2.17 — поздние уровни проходятся БЫСТРЕЕ, то есть знание между уровнями уже переносится.
Значит усиливать надо не сам перенос, а точность формулировки.

ЧЕСТНОЕ ОЖИДАНИЕ. Скромное. До вторых уровней доходят 9 игр из 25, до третьих 5 — правка помогает
только тем, кто дошёл. Порог тот же: устойчивое смещение по знакам парных разностей, а не общий балл.

usage:  .venv/bin/python scripts/build_goalrule_notebook.py
"""

import ast
import json
import os

CELL = r'''
# =====================================================================
# ЦЕЛЬ КАК ПРАВИЛО: назвать вслух и перепроверить на новом уровне.
#
# Цель-картинка (кадр победы) закрыта 19.09: перенесённая даёт 19% против 18% случайного.
# Цель-правило словами — другое: она работает на ОПОЗНАНИЕ цели, а это и есть измеренное узкое место.
# =====================================================================
import inference.agent.tool_agent as _grta

_GR_BLOCK = (
    "\n"
    "Level goal as a rule (state it, then re-check it):\n"
    "- Keep ONE short sentence in your world model called GOAL RULE: what has to become true for the "
    "level to be cleared, phrased as a rule about the board, not as a list of moves. "
    "Examples of the right shape: 'bring every coloured shape onto the matching coloured tile', "
    "'remove all blocks of the same colour', 'move the agent to the exit after opening it'.\n"
    "- The GOAL RULE is about the board, never about which buttons you pressed. Do not carry move "
    "sequences between levels -- layouts change, rules usually do not.\n"
    "- When the level number changes, do NOT assume the rule still holds and do NOT relearn it from "
    "scratch either. State it as a hypothesis first: 'previous GOAL RULE was X; does the new board fit "
    "X?'. Confirm or correct it from the first frames, then continue.\n"
    "- If the new board clearly cannot fit the previous rule, say so explicitly and write a new GOAL "
    "RULE before planning any move.\n"
)

_gr_orig_system = _grta._build_system_prompt


def _gr_build_system_prompt(**kw):
    try:
        text = _gr_orig_system(**kw)
        if "Level goal as a rule" in text:
            return text
        return text + _GR_BLOCK
    except Exception as exc:                        # слой не имеет права ронять прогон
        print("GOALRULE: сбой слоя, отдаю стоковый промпт: %r" % (exc,), flush=True)
        return _gr_orig_system(**kw)


_grta._build_system_prompt = _gr_build_system_prompt

if not TRUE_SUBMISSION:
    bm.solver.max_runtime_s_per_game = 7920.0       # полный прогон 132 мин; сравнение flash_v1_phaseA = 9.43

print("GOALRULE: слой включён (%s). Цель уровня как ПРАВИЛО о доске, названное одной строкой и "
      "перепроверяемое при смене уровня. Путь прохождения НЕ переносится (это carry, закрыт 17.09). "
      "Блок +%d знаков." % ("бой" if TRUE_SUBMISSION else "полный прогон 132 мин", len(_GR_BLOCK)),
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

    out = "kernels/notebooks_stockflash_goalrule"
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(CELL)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    meta["id"] = "sergueimakarov/arc3-stock-flash-goalrule"
    meta["title"] = "arc3 stock flash goalrule"
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
    print("ok   патч ДО запуска прогона:", new.find("_GR_BLOCK") < new.find("await bm.run("))
    print("ok   трогаем только системный промпт:", "_build_user_prompt" not in CELL)
    # проверяем СОБРАННЫЙ текст блока, а не исходник: в исходнике фраза разбита между кавычками
    _ns = {}
    exec(CELL[CELL.index("_GR_BLOCK = ("):CELL.index("_gr_orig_system")], _ns)
    _blk = _ns["_GR_BLOCK"]
    print("ok   путь прохождения НЕ переносим (это carry, закрыт):",
          "Do not carry move sequences between levels" in _blk)
    print("ok   цель сформулирована как правило о ДОСКЕ:", "rule about the board" in _blk)
    print("ok   есть перепроверка при смене уровня:", "does the new board fit" in _blk)
    print("ok   блока про предусловия здесь НЕТ (одна переменная):", "Conditional effects" not in CELL)
    print("ok   слой не роняет прогон:", "except Exception" in CELL)
    print("собрано:", out)


if __name__ == "__main__":
    main()

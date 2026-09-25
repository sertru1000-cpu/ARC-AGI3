"""Сборка `arc3-stock-flash-retest`: обвязка напоминает, какое пустое действие пора перепроверить.

ОСНОВАНИЕ — ТРИ ИЗМЕРЕНИЯ 25.09, все на записях базы (runs/flash_v1_phaseA, 25 игр, 4380 ходов):

1. Что модель делает СРАЗУ после пустого хода (`scripts/after_noop_behaviour.py`):
       то же самое один в один   208 (61.0%)
       другой клик той же мышью   61 (17.9%)
       сменила орудие             72 (21.1%)
   Четыре хода из пяти после пустого результата — перебор, а не эксперимент.

2. Возврат к пустому действию после изменения доски окупается: 140 возвратов, из них
   ЗАРАБОТАЛО 117 — это 84%.

3. Но происходит он поздно и случайно (`промежуток между пробой и удачным возвратом`):
       медиана 4 хода, среднее 11.8, максимум 118; четверть возвратов позже 16-го хода.

ЧТО ДЕЛАЕТ СЛОЙ. Никаких новых вызовов модели — только детерминированный учёт в обвязке, то есть
упаковка информации, за которую УЖЕ заплачено реальными ходами. Обвязка помнит, какие действия дали
пустой результат и при какой доске. Как только доска существенно изменилась, она дописывает в промпт
короткую строку: такое-то действие не работало N ходов назад, с тех пор доска изменилась, имеет смысл
проверить его снова. Решение остаётся за моделью.

ЧЕМ ОТЛИЧАЕТСЯ ОТ ЗАКРЫТОГО. Подсказка «ты здесь уже был» (12.09) и «ты пробовал это сочетание» (25.09)
сообщали о ПОВТОРЕ — то есть отговаривали. Здесь наоборот: обвязка ПОДТАЛКИВАЕТ вернуться к брошенному
действию, и это подкреплено долей успеха 84%. Такого слоя у нас не было.

ПОЧЕМУ НЕ ОТДЕЛЬНЫЙ АГЕНТ-ИССЛЕДОВАТЕЛЬ. В бою нет снимков среды (проверено 27.08), поэтому любая
проба — реальный ход; а лишний вызов модели отнимает ходы у основного агента при жёстких 900 тыс.
токенов в час. Внешний критик, узнав об этих двух ограничениях, сам снял исследователя с первого места.

usage:  .venv/bin/python scripts/build_retest_notebook.py
"""

import ast
import json
import os

CELL = r'''
# =====================================================================
# НАПОМИНАНИЕ О ПЕРЕПРОВЕРКЕ: доска изменилась — пора вернуться к брошенному действию.
#
# Измерено 25.09 на записях базы: после пустого хода модель в 61% случаев повторяет его один в один,
# а возврат к нему ПОСЛЕ изменения доски срабатывает в 84% случаев (117 из 140), но случается поздно
# (медиана 4 хода, четверть позже 16-го). Слой ничего не спрашивает у модели и не делает вызовов —
# только напоминает о том, что уже наблюдалось.
# =====================================================================
import inference.agent.tool_agent as _rtta

_RT_MIN_GAP = 3           # не напоминать раньше, чем через столько ходов (половина возвратов и так в 4)
_RT_MAX_ITEMS = 2         # не больше двух напоминаний за раз, чтобы не раздувать промпт
_RT_STATS = {"fired": 0}


def _rt_board(frame):
    return getattr(frame, "ascii", None) or getattr(frame, "grid", None)


def _rt_build_user_prompt(self, action_num, **kw):
    text = _rt_orig_prompt(self, action_num, **kw)
    try:
        hist = list(getattr(self, "history_entries", None) or [])
        if len(hist) < 2:
            return text
        cur = kw.get("current_frame") or getattr(hist[-1], "frame", None)
        cur_level = getattr(cur, "level", None)
        # собираем: действие -> (номер хода, доска при пустом исходе)
        dead = {}
        for i in range(1, len(hist)):
            prev, now = hist[i - 1], hist[i]
            act = (getattr(now, "action", "") or "").strip()
            if not act:
                continue
            if getattr(getattr(now, "frame", None), "level", None) != cur_level:
                dead.clear()                      # другой уровень — прежний опыт не про эту доску
                continue
            before, after = _rt_board(getattr(prev, "frame", None)), _rt_board(getattr(now, "frame", None))
            if before is None or after is None:
                continue
            if before == after:
                dead[act] = (i, before)           # пусто: запоминаем, при какой доске
            else:
                dead.pop(act, None)               # сработало — больше не кандидат
        if not dead:
            return text
        now_board = _rt_board(cur)
        ready = []
        for act, (i, board_then) in dead.items():
            gap = len(hist) - i
            if gap >= _RT_MIN_GAP and now_board is not None and board_then != now_board:
                ready.append((gap, act))
        if not ready:
            return text
        ready.sort(reverse=True)                  # сначала самые давно брошенные
        items = "; ".join("%s (%d actions ago)" % (a, g) for g, a in ready[:_RT_MAX_ITEMS])
        _RT_STATS["fired"] += 1
        if _RT_STATS["fired"] <= 3:
            print("RETEST: напоминание выдано (ход %s): %s" % (action_num, items), flush=True)
        return text + (
            "\n\n[HOST] These actions produced NO change earlier, but the board has changed since: %s. "
            "An action that did nothing before can work once its precondition is met. If one of them fits "
            "your current hypothesis, retest it now instead of repeating what already failed." % items)
    except Exception as exc:                      # слой не имеет права ронять прогон
        print("RETEST: сбой слоя, пропускаю: %r" % (exc,), flush=True)
    return text


_rt_orig_prompt = _rtta.ToolAgent._build_user_prompt
_rtta.ToolAgent._build_user_prompt = _rt_build_user_prompt

if not TRUE_SUBMISSION:
    bm.solver.max_runtime_s_per_game = 7920.0     # полный прогон 132 мин; сравнение flash_v1_phaseA = 9.43

print("RETEST: слой включён (%s). Никаких новых вызовов модели — только учёт в обвязке. "
      "Основание: возврат к пустому действию после изменения доски срабатывает в 84%% случаев "
      "(117 из 140), но модель в 61%% случаев вместо этого повторяет пустой ход."
      % ("бой" if TRUE_SUBMISSION else "полный прогон 132 мин"), flush=True)
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

    out = "kernels/notebooks_stockflash_retest"
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(CELL)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    meta["id"] = "sergueimakarov/arc3-stock-flash-retest"
    meta["title"] = "arc3 stock flash retest"
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)

    compile(code, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    diff = [i for i in range(len(nb["cells"])) if "".join(nb["cells"][i]["source"]) != "".join(src["cells"][i]["source"])]
    new = "".join(nb["cells"][15]["source"])
    print("ok   изменена только ячейка 15:", diff == [15])
    print("ok   патч ДО запуска прогона:", new.find("_RT_MIN_GAP") < new.find("await bm.run("))
    print("ok   НИ ОДНОГО нового вызова модели:", "_chat_completion" not in CELL and "client." not in CELL)
    print("ok   системный промпт не трогаем:", "_build_system_prompt" not in CELL)
    print("ok   слой не роняет прогон:", "except Exception" in CELL)
    print("собрано:", out)


if __name__ == "__main__":
    main()

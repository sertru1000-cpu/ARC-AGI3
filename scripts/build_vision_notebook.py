"""Сборка `arc3-nextfork-vision`: картинка доски в ×4 большем разрешении — ЯЧЕЙКОЙ поверх датасета v3 (26.09).

ПОПРАВКА к первой версии этого описания: картинку обвязка ПОСЫЛАЛА всегда — бандл keithtyser выставляет
MULTIMODAL_CONTEXT=current_grid в serving_setup.py, метрики vLLM ночного прогона: 9588 обращений к кэшу изображений
на 1341 запрос (~7 картинок на вызов: текущая доска плюс история). В транскриптах их не видно, потому что туда
пишется текст запроса до приложения картинки. Ошибочный вывод «модель слепая» был сделан по транскриптам.

Настоящая находка: MULTIMODAL_UPSCALE=4. Доска 64×64 -> картинка 256×256, патч зрения 16 px -> каждый патч сливает
4×4 клетки доски; объекты меньше четырёх клеток на картинке неразличимы. ×4 — выбор самих Tufa для локальной модели (configs/inference.json), ×16 они давали frontier-моделям через API
(configs/inference.openrouter.json). Эта ячейка — эксперимент против их выбора, не возврат потерянной настройки.

Что делает ячейка:
  * MULTIMODAL_UPSCALE=16 — клетка = патч, ~1024 токена на картинку вместо ~64;
  * картинки ВЫНИМАЮТСЯ из сохраняемой истории: семь картинок ×1024 переполнили бы окно 32768; оценка токенов
    обвязки («JSON/3») к тому же насчитала бы ~2500 на каждую. Итог: окно +~1000 токенов за вызов вместо +~450.
Мерить: охват первых уровней, уровни, ходов на игру.
usage: .venv/bin/python scripts/build_vision_notebook.py
"""
import ast, json, os

CELL = r'''
# =====================================================================
# VISION: картинка доски в разрешении ×16 вместо ×4 (26.09). Ячейкой на датасет v3.
# Было ×4: патч зрения 16 px сливал 4×4 клетки. Картинки из истории вынимаются (окно).
# =====================================================================
import os as _vos, sys as _vsys
import inference.agent.tool_agent as _vta
_vos.environ["MULTIMODAL_CONTEXT"] = "current_grid"
_vos.environ["MULTIMODAL_UPSCALE"] = "16"
_v_orig_persist = _vta.ToolAgent._persistent_history_messages


def _v_strip_images(msg):
    c = msg.get("content")
    if not isinstance(c, list):
        return msg
    texts = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
    return dict(msg, content="\n".join(t for t in texts if t).replace("\n\nCurrent grid image:", ""))


def _v_persist(self, messages, *a, **kw):
    try:
        messages = [_v_strip_images(m) if m.get("role") == "user" else m for m in messages]
    except Exception as exc:
        print("VISION: сбой очистки истории: %r" % (exc,), file=_vsys.__stderr__, flush=True)
    return _v_orig_persist(self, messages, *a, **kw)


_vta.ToolAgent._persistent_history_messages = _v_persist
print("VISION: картинка текущей доски в каждый вызов (×16), из истории вынимается", flush=True)
'''


def main() -> None:
    src = json.load(open("kernels/notebooks_nextfork/submission.ipynb", encoding="utf-8"))
    nb = json.loads(json.dumps(src)); cell = nb["cells"][15]; body = "".join(cell["source"])
    anchor = "    seconds=budget - 600.0\n)"; at = body.find(anchor)
    if at < 0 or body.find("await bm.run(") < 0:
        raise SystemExit("не нашёл стоковый soft_end или запуск прогона — сборка остановлена")
    at += len(anchor); code = body[:at] + "\n" + CELL + body[at:]; cell["source"] = code.splitlines(keepends=True)
    out = "kernels/notebooks_nextfork_vision"; os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(CELL)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_nextfork/kernel-metadata.json"))
    meta["id"] = "sergueimakarov/arc3-nextfork-vision"; meta["title"] = "arc3 nextfork vision"
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    compile(code, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    print("ok   патч ДО запуска прогона:", code.find("_v_persist") < code.find("await bm.run("))
    print("собрано:", out, "| датасет", meta["dataset_sources"][0])


if __name__ == "__main__":
    main()

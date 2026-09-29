
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
_LEAN_HELPERS = '# --- Готовые помощники песочницы (26.09). Модель переписывала их сама по 30-70 раз за прогон:\n# 37% строк её кода повторялись дословно (обход segmentation[\'nodes\'] с ручными границами, split ascii).\n# Только безопасные встроенные и collections; переопределение моделью своих одноимённых функций безвредно.\ndef grid(fr=None):\n    """Доска как список строк символов (fr по умолчанию — current_frame)."""\n    fr = current_frame if fr is None else fr\n    a = getattr(fr, "ascii", "") or ""\n    if a.strip():\n        return a.split("\\n")\n    # ascii обвязка передаёт всегда; запасной путь — сырая сетка цифрами (палитра песочнице не видна)\n    return ["".join(str(v) for v in row) for row in (getattr(fr, "_grid", None) or [])]\n\ndef bb(node):\n    """(r0, r1, c0, c1) по границе узла segmentation."""\n    b = node["boundary"]; rs = [p[0] for p in b]; cs = [p[1] for p in b]\n    return (min(rs), max(rs), min(cs), max(cs))\n\ndef objs(fr=None, color=None, min_pixels=1):\n    """Объекты кадра: dict(id,color,pixels,r0,r1,c0,c1,h,w,center), отсортированы сверху-вниз, слева-направо."""\n    fr = current_frame if fr is None else fr\n    out = []\n    for n in fr.segmentation["nodes"]:\n        if color is not None and n["color"] != color: continue\n        if n["pixels"] < min_pixels: continue\n        r0, r1, c0, c1 = bb(n)\n        out.append({"id": n["id"], "color": n["color"], "pixels": n["pixels"], "r0": r0, "r1": r1, "c0": c0, "c1": c1,\n                    "h": r1 - r0 + 1, "w": c1 - c0 + 1, "center": ((r0 + r1) / 2, (c0 + c1) / 2)})\n    return sorted(out, key=lambda o: (o["r0"], o["c0"]))\n\ndef crop(r0, r1, c0, c1, fr=None):\n    """Фрагмент доски строками, границы включительно."""\n    return "\\n".join(row[c0:c1 + 1] for row in grid(fr)[r0:r1 + 1])\n\ndef summ(fr=None, min_pixels=1):\n    """Одна строка на объект: цвет, пикселей, строки, столбцы."""\n    return "\\n".join("%s %dpx r%d-%d c%d-%d" % (o["color"], o["pixels"], o["r0"], o["r1"], o["c0"], o["c1"]) for o in objs(fr, None, min_pixels))\n\ndef diff(a=None, b=None):\n    """Что изменилось между кадрами a->b (по умолчанию previous_frame->current_frame):\n    cells — изменившиеся клетки, moved — сдвиги объектов (по цвету и форме), appeared/gone — новые/исчезнувшие."""\n    a = previous_frame if a is None else a; b = current_frame if b is None else b\n    if a is None or b is None: return {"n_cells": 0, "cells": [], "moved": [], "appeared": [], "gone": []}\n    ga, gb = grid(a), grid(b)\n    cells = [(r, c) for r in range(min(len(ga), len(gb))) for c in range(min(len(ga[r]), len(gb[r]))) if ga[r][c] != gb[r][c]]\n    key = lambda o: (o["color"], o["pixels"], o["h"], o["w"])\n    A = {}; B = {}\n    for o in objs(a): A.setdefault(key(o), []).append(o)\n    for o in objs(b): B.setdefault(key(o), []).append(o)\n    moved = []; appeared = []; gone = []\n    for k in set(A) | set(B):\n        la, lb = A.get(k, []), B.get(k, [])\n        for oa, ob in zip(la, lb):\n            if (oa["r0"], oa["c0"]) != (ob["r0"], ob["c0"]):\n                moved.append({"color": ob["color"], "pixels": ob["pixels"], "from": (oa["r0"], oa["c0"]), "to": (ob["r0"], ob["c0"]),\n                              "d": (ob["r0"] - oa["r0"], ob["c0"] - oa["c0"])})\n        appeared += lb[len(la):]; gone += la[len(lb):]\n    return {"n_cells": len(cells), "cells": cells[:200], "moved": moved, "appeared": appeared, "gone": gone}\n\ndef effects():\n    """Таблица по истории: действие -> сколько раз пробовали и сколько раз доска менялась."""\n    tot = {}; ch = {}\n    for t in transitions:\n        a = str(t.action); tot[a] = tot.get(a, 0) + 1\n        ch[a] = ch.get(a, 0) + (1 if (t.result or {}).get("board_changed") else 0)\n    return {a: {"tried": tot[a], "changed": ch[a]} for a in tot}\n# --- конец помощников ---\n'
_ln_orig_sandbox = _lnta.run_sandboxed_python


def _ln_sandbox(*, code, **kw):
    try:
        return _ln_orig_sandbox(code=_LEAN_HELPERS + "\n" + str(code), **kw)
    except TypeError:
        return _ln_orig_sandbox(code=code, **kw)


_lnta.run_sandboxed_python = _ln_sandbox
print("LEAN: помощники песочницы подставляются в каждый вызов (%d строк)" % len(_LEAN_HELPERS.splitlines()), flush=True)
print("LEAN: слой включён, блок +%d знаков в системный промпт, поверх датасета версии 3" % len(_LEAN_BLOCK), flush=True)

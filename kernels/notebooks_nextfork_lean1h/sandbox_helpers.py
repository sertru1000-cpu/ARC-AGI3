# --- Готовые помощники песочницы (26.09). Модель переписывала их сама по 30-70 раз за прогон:
# 37% строк её кода повторялись дословно (обход segmentation['nodes'] с ручными границами, split ascii).
# Только безопасные встроенные и collections; переопределение моделью своих одноимённых функций безвредно.
def grid(fr=None):
    """Доска как список строк символов (fr по умолчанию — current_frame)."""
    fr = current_frame if fr is None else fr
    a = getattr(fr, "ascii", "") or ""
    if a.strip():
        return a.split("\n")
    # ascii обвязка передаёт всегда; запасной путь — сырая сетка цифрами (палитра песочнице не видна)
    return ["".join(str(v) for v in row) for row in (getattr(fr, "_grid", None) or [])]

def bb(node):
    """(r0, r1, c0, c1) по границе узла segmentation."""
    b = node["boundary"]; rs = [p[0] for p in b]; cs = [p[1] for p in b]
    return (min(rs), max(rs), min(cs), max(cs))

def objs(fr=None, color=None, min_pixels=1):
    """Объекты кадра: dict(id,color,pixels,r0,r1,c0,c1,h,w,center), отсортированы сверху-вниз, слева-направо."""
    fr = current_frame if fr is None else fr
    out = []
    for n in fr.segmentation["nodes"]:
        if color is not None and n["color"] != color: continue
        if n["pixels"] < min_pixels: continue
        r0, r1, c0, c1 = bb(n)
        out.append({"id": n["id"], "color": n["color"], "pixels": n["pixels"], "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                    "h": r1 - r0 + 1, "w": c1 - c0 + 1, "center": ((r0 + r1) / 2, (c0 + c1) / 2)})
    return sorted(out, key=lambda o: (o["r0"], o["c0"]))

def crop(r0, r1, c0, c1, fr=None):
    """Фрагмент доски строками, границы включительно."""
    return "\n".join(row[c0:c1 + 1] for row in grid(fr)[r0:r1 + 1])

def summ(fr=None, min_pixels=1):
    """Одна строка на объект: цвет, пикселей, строки, столбцы."""
    return "\n".join("%s %dpx r%d-%d c%d-%d" % (o["color"], o["pixels"], o["r0"], o["r1"], o["c0"], o["c1"]) for o in objs(fr, None, min_pixels))

def diff(a=None, b=None):
    """Что изменилось между кадрами a->b (по умолчанию previous_frame->current_frame):
    cells — изменившиеся клетки, moved — сдвиги объектов (по цвету и форме), appeared/gone — новые/исчезнувшие."""
    a = previous_frame if a is None else a; b = current_frame if b is None else b
    if a is None or b is None: return {"n_cells": 0, "cells": [], "moved": [], "appeared": [], "gone": []}
    ga, gb = grid(a), grid(b)
    cells = [(r, c) for r in range(min(len(ga), len(gb))) for c in range(min(len(ga[r]), len(gb[r]))) if ga[r][c] != gb[r][c]]
    key = lambda o: (o["color"], o["pixels"], o["h"], o["w"])
    A = {}; B = {}
    for o in objs(a): A.setdefault(key(o), []).append(o)
    for o in objs(b): B.setdefault(key(o), []).append(o)
    moved = []; appeared = []; gone = []
    for k in set(A) | set(B):
        la, lb = A.get(k, []), B.get(k, [])
        for oa, ob in zip(la, lb):
            if (oa["r0"], oa["c0"]) != (ob["r0"], ob["c0"]):
                moved.append({"color": ob["color"], "pixels": ob["pixels"], "from": (oa["r0"], oa["c0"]), "to": (ob["r0"], ob["c0"]),
                              "d": (ob["r0"] - oa["r0"], ob["c0"] - oa["c0"])})
        appeared += lb[len(la):]; gone += la[len(lb):]
    return {"n_cells": len(cells), "cells": cells[:200], "moved": moved, "appeared": appeared, "gone": gone}

def effects():
    """Таблица по истории: действие -> сколько раз пробовали и сколько раз доска менялась."""
    tot = {}; ch = {}
    for t in transitions:
        a = str(t.action); tot[a] = tot.get(a, 0) + 1
        ch[a] = ch.get(a, 0) + (1 if (t.result or {}).get("board_changed") else 0)
    return {a: {"tried": tot[a], "changed": ch[a]} for a in tot}
# --- конец помощников ---

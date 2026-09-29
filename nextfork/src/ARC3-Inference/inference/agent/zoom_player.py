"""Вторая картинка — окрестность игрока 16x16 клеток при x32 (nextfork, 27.09; вариант a7 на поде). ВЫКЛЮЧЕНО по умолчанию: NEXTFORK_ZOOM=1.

Зрение модели: патч 16 px, склейка 2x2 -> токен на 32x32 px; при x12 токен смазывает ~2.7x2.7 клетки. Здесь окрестность объектов,
двигавшихся на последних ходах уровня, рисуется по клетке на токен (512x512, ~256 токенов); вся доска остаётся последней картинкой.
Под 27.09 (один прогон): 19 уровней — больше всех за день (контроли 17/10/16), осмотров не меньше. Повтор — 27.09 вечером.
"""

# ===== a7 ZOOM: вторая картинка — окрестность игрока 16x16 клеток при x32 (27.09) =====
# Зрение модели: патч 16 px, склейка 2x2 -> один токен на 32x32 px. При x12 (Скотт) токен смазывает ~2.7x2.7 клетки,
# поэтому модель печатает куски доски буквами (51-75% осмотров). Здесь окрестность объектов, двигавшихся на последних
# ходах уровня (игрок), рисуется по одной клетке на токен: 16*32 = 512 px, ~256 токенов. Остальная доска — как у Скотта.
import sys as _zmsys, types as _zmtypes
import inference.agent.tool_agent as _zmta
from inference.agent.vision_context import frame_to_png_data_url as _zm_png
from inference.utils.segmentation import segment_layer as _zm_seg
from inference.utils.grid_utils import ARC_COLOR_CHARS as _zm_cc
_ZM_CELLS = 16; _ZM_SCALE = 32
_zm_stat = {"zooms": 0, "errors": 0}
def _zm_objs(fr):
    out = []
    for n in _zm_seg([list(r) for r in fr.grid], _zm_cc)["nodes"]:
        rs = [p[0] for p in n["boundary"]]; cs = [p[1] for p in n["boundary"]]
        out.append((n["hash"], n["pixels"], min(rs), max(rs), min(cs), max(cs)))
    return out
def _zm_box(cur, hist):
    boxes = []
    for i in range(max(1, len(hist) - 4), len(hist)):
        a, b = getattr(hist[i - 1], "frame", None), getattr(hist[i], "frame", None)
        if a is None or b is None or a.level != cur.level or b.level != cur.level:
            continue
        before = {h: (r0, c0) for h, px, r0, r1, c0, c1 in _zm_objs(a)}
        for h, px, r0, r1, c0, c1 in _zm_objs(b):
            if h in before and before[h] != (r0, c0) and px <= 400:
                boxes.append((r0, r1, c0, c1))
    if not boxes:
        return None
    H, W = len(cur.grid), len(cur.grid[0])
    cy = (min(b[0] for b in boxes) + max(b[1] for b in boxes)) // 2
    cx = (min(b[2] for b in boxes) + max(b[3] for b in boxes)) // 2
    r0 = min(max(0, cy - _ZM_CELLS // 2), max(0, H - _ZM_CELLS)); c0 = min(max(0, cx - _ZM_CELLS // 2), max(0, W - _ZM_CELLS))
    return r0, min(H, r0 + _ZM_CELLS) - 1, c0, min(W, c0 + _ZM_CELLS) - 1
_zm_orig_prompt = _zmta.ToolAgent._build_user_prompt
def _zm_prompt(self, action_num, **kw):
    text = _zm_orig_prompt(self, action_num, **kw)
    try:
        cur = kw.get("current_frame"); hist = kw.get("history_entries") or []
        self._zm_box = _zm_box(cur, hist) if cur is not None else None
    except Exception as exc:
        self._zm_box = None; _zm_stat["errors"] += 1
        if _zm_stat["errors"] <= 3:
            print("ZOOM: сбой рамки: %r" % (exc,), file=_zmsys.__stderr__, flush=True)
    return text
_zmta.ToolAgent._build_user_prompt = _zm_prompt
_zm_orig_msg = _zmta.ToolAgent._build_user_message
def _zm_msg(self, user_prompt, current_frame):
    msg = _zm_orig_msg(self, user_prompt, current_frame)
    box = self.__dict__.get("_zm_box")
    try:
        if box and current_frame is not None and isinstance(msg.get("content"), list):
            r0, r1, c0, c1 = box
            sub = [list(row[c0:c1 + 1]) for row in current_frame.grid[r0:r1 + 1]]
            part = {"type": "image_url", "image_url": {"url": _zm_png(_zmtypes.SimpleNamespace(grid=sub), upscale=_ZM_SCALE)}}
            label = {"type": "text", "text": "\nZoomed image of the board around the objects that moved recently: rows %d-%d, cols %d-%d "
                                             "(one board cell = %dx%d px; the image's top-left cell is row %d, col %d):" % (r0, r1, c0, c1, _ZM_SCALE, _ZM_SCALE, r0, c0)}
            content = list(msg["content"])
            content[-1:-1] = [label, part]                            # картинка всей доски остаётся ПОСЛЕДНЕЙ
            msg = dict(msg, content=content); _zm_stat["zooms"] += 1
            if _zm_stat["zooms"] in (1, 50, 300):
                print("ZOOM: %d увеличенных вырезок" % _zm_stat["zooms"], file=_zmsys.__stderr__, flush=True)
    except Exception as exc:
        _zm_stat["errors"] += 1
        if _zm_stat["errors"] <= 3:
            print("ZOOM: сбой картинки: %r" % (exc,), file=_zmsys.__stderr__, flush=True)
    return msg
_zmta.ToolAgent._build_user_message = _zm_msg
print("NEXTFORK ZOOM: окрестность игрока %dx%d клеток при x%d (%d px) второй картинкой" % (_ZM_CELLS, _ZM_CELLS, _ZM_SCALE, _ZM_CELLS * _ZM_SCALE), flush=True)

"""Наблюдение в сообщении обвязки v2 (nextfork, 27.09; вариант a6b на поде). ВЫКЛЮЧЕНО по умолчанию: NEXTFORK_OBSERVE=1.

Обвязка сама кладёт в сообщение модели объекты доски (до 25 крупнейших), что изменилось после последнего хода (сдвиги,
появилось/исчезло, изменило форму), вырезку доски буквами вокруг объектов, двигавшихся на последних ходах, и строку
«действуй в этом же вызове»; копии блока из прошлых сообщений истории вынимаются.
Под 27.09 (один прогон, 30 мин): вызовов-осмотров 95 против ~140 у контроля, 1.98 хода на запрос (лучшее за день),
но ответ модели +35% -> ходов игры столько же; балл 2.75 при контролях 2.60/1.45/2.98. Повтор — 27.09 вечером.
"""

# ===== a6b OBSERVE v2 (27.09): копии из истории вынимаются, вырезка вокруг двигавшихся объектов, «действуй в этом же вызове» =====
# ===== a6 OBSERVE: наблюдение в сообщении — объекты, что изменилось, вырезка доски (27.09) =====
# Аудит осмотров: ~половина вызовов модели — только осмотр (кусок доски 51-75%, объекты 34-61%, разница 20%),
# каждый ~2 мин очереди. Обвязка кладёт это в сообщение сама, чтобы модель могла ходить в том же запросе.
import sys as _obsys
import inference.agent.tool_agent as _obta
from inference.utils.segmentation import segment_layer as _ob_seg
from inference.utils.grid_utils import ARC_COLOR_CHARS as _ob_cc
_OB_MAX_OBJ = 25; _OB_MAX_CROP = 24; _OB_MAX_MOVES = 10
_ob_stat = {"prompts": 0, "chars": 0, "errors": 0}
def _ob_objs(fr):
    out = []
    for n in _ob_seg([list(r) for r in fr.grid], _ob_cc)["nodes"]:
        rs = [p[0] for p in n["boundary"]]; cs = [p[1] for p in n["boundary"]]
        out.append({"color": n["color"], "px": n["pixels"], "hash": n["hash"], "r0": min(rs), "r1": max(rs), "c0": min(cs), "c1": max(cs)})
    return out
def _ob_block(cur, prev):
    objs = sorted(_ob_objs(cur), key=lambda o: -o["px"])
    lines = ["Harness observation of the current board (computed by the harness; you do not need python to get this):",
             "Objects, largest first (color pixels rows cols), %d of %d:" % (min(len(objs), _OB_MAX_OBJ), len(objs))]
    lines += ["  %s %dpx r%d-%d c%d-%d" % (o["color"], o["px"], o["r0"], o["r1"], o["c0"], o["c1"]) for o in objs[:_OB_MAX_OBJ]]
    if prev is None:
        lines.append("Change since previous board: (no previous board)")
        return "\n".join(lines)
    g0, g1 = prev.grid, cur.grid
    cells = [(r, c) for r in range(min(len(g0), len(g1))) for c in range(min(len(g0[r]), len(g1[r]))) if g0[r][c] != g1[r][c]]
    if not cells:
        lines.append("Change since previous board: nothing changed.")
        return "\n".join(lines)
    before = {}
    for o in _ob_objs(prev):
        before.setdefault(o["hash"], []).append(o)
    moved, appeared, moved_boxes = [], [], []
    for o in objs:
        cand = before.get(o["hash"]) or []
        if not cand:
            appeared.append(o); continue
        same = [b for b in cand if (b["r0"], b["c0"]) == (o["r0"], o["c0"])]
        if same:
            cand.remove(same[0])
        else:
            b = min(cand, key=lambda b: abs(b["r0"] - o["r0"]) + abs(b["c0"] - o["c0"])); cand.remove(b)
            moved_boxes += [(b["r0"], b["r1"], b["c0"], b["c1"]), (o["r0"], o["r1"], o["c0"], o["c1"])]
            moved.append("%s %dpx (%d,%d)->(%d,%d) d(%+d,%+d)" % (o["color"], o["px"], b["r0"], b["c0"], o["r0"], o["c0"], o["r0"] - b["r0"], o["c0"] - b["c0"]))
    gone = [b for v in before.values() for b in v]
    reshaped = []                                   # тот же цвет, пересекающиеся границы — форма изменилась на месте
    for o in list(appeared):
        g = next((b for b in gone if b["color"] == o["color"] and not (b["r1"] < o["r0"] or o["r1"] < b["r0"] or b["c1"] < o["c0"] or o["c1"] < b["c0"])), None)
        if g is not None:
            appeared.remove(o); gone.remove(g)
            reshaped.append("%s %dpx->%dpx r%d-%d c%d-%d" % (o["color"], g["px"], o["px"], o["r0"], o["r1"], o["c0"], o["c1"]))
    r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells); c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
    lines.append("Change since previous board: %d cells changed in rows %d-%d cols %d-%d." % (len(cells), r0, r1, c0, c1))
    if moved: lines.append("  moved: " + "; ".join(moved[:_OB_MAX_MOVES]))
    if appeared: lines.append("  appeared: " + "; ".join("%s %dpx at (%d,%d)" % (o["color"], o["px"], o["r0"], o["c0"]) for o in appeared[:_OB_MAX_MOVES]))
    if gone: lines.append("  gone: " + "; ".join("%s %dpx from (%d,%d)" % (o["color"], o["px"], o["r0"], o["c0"]) for o in gone[:_OB_MAX_MOVES]))
    if reshaped: lines.append("  changed shape in place: " + "; ".join(reshaped[:_OB_MAX_MOVES]))
    if moved_boxes:                                 # вырезка — по сдвинувшимся объектам (было+стало), а не по всем клеткам: полоса-счётчик раздувала область
        r0 = min(b[0] for b in moved_boxes); r1 = max(b[1] for b in moved_boxes); c0 = min(b[2] for b in moved_boxes); c1 = max(b[3] for b in moved_boxes)
    R0, R1, C0, C1 = max(0, r0 - 1), min(len(g1) - 1, r1 + 1), max(0, c0 - 1), min(len(g1[0]) - 1, c1 + 1)
    if R1 - R0 + 1 <= _OB_MAX_CROP and C1 - C0 + 1 <= _OB_MAX_CROP:
        rows = cur.ascii.split("\n")
        lines.append("Changed region of the current board (rows %d-%d, cols %d-%d):" % (R0, R1, C0, C1))
        lines += ["  %2d %s" % (r, rows[r][C0:C1 + 1]) for r in range(R0, R1 + 1)]
    return "\n".join(lines)
_ob_orig = _obta.ToolAgent._build_user_prompt
def _ob_prompt(self, action_num, **kw):
    text = _ob_orig(self, action_num, **kw)
    try:
        cur = kw.get("current_frame"); hist = kw.get("history_entries") or []
        if cur is None:
            return text
        prev = hist[-2].frame if len(hist) >= 2 and getattr(hist[-2], "frame", None) is not None else None
        if prev is not None and prev.level != cur.level:
            prev = None                                     # новый уровень — сравнивать не с чем
        block = _ob_block(cur, prev)
        _ob_stat["prompts"] += 1; _ob_stat["chars"] += len(block)
        if _ob_stat["prompts"] in (1, 50, 200, 500):
            print("OBSERVE: %d сообщений, в среднем %d знаков блока" % (_ob_stat["prompts"], _ob_stat["chars"] // _ob_stat["prompts"]), file=_obsys.__stderr__, flush=True)
        return text + "\n" + block
    except Exception as exc:
        _ob_stat["errors"] += 1
        if _ob_stat["errors"] <= 3:
            print("OBSERVE: сбой, сообщение без блока: %r" % (exc,), file=_obsys.__stderr__, flush=True)
        return text
_obta.ToolAgent._build_user_prompt = _ob_prompt
print("ARM a6b OBSERVE v2: наблюдение в каждом сообщении (объекты до %d, изменения, вырезка до %dx%d)" % (_OB_MAX_OBJ, _OB_MAX_CROP, _OB_MAX_CROP), flush=True)

# --- v2: (1) вырезка вокруг объектов, двигавшихся на последних ходах уровня (игрок), даже если последний ход был пустым;
#         (2) из прошлых сообщений истории блок наблюдения вынимается (в v1 было ~7 копий на запрос);
#         (3) строка «действуй в этом же вызове».
_ob_block_v1 = _ob_block
def _ob_player_crop(cur, hist):
    boxes = []
    for i in range(max(1, len(hist) - 4), len(hist)):
        a, b = getattr(hist[i - 1], "frame", None), getattr(hist[i], "frame", None)
        if a is None or b is None or a.level != cur.level or b.level != cur.level:
            continue
        before = {o["hash"]: o for o in _ob_objs(a)}
        for o in _ob_objs(b):
            p = before.get(o["hash"])
            if p is not None and (p["r0"], p["c0"]) != (o["r0"], o["c0"]) and o["px"] <= 400:
                boxes.append((o["r0"], o["r1"], o["c0"], o["c1"]))
    if not boxes:
        return ""
    H, W = len(cur.grid), len(cur.grid[0])
    r0 = max(0, min(b[0] for b in boxes) - 3); r1 = min(H - 1, max(b[1] for b in boxes) + 3)
    c0 = max(0, min(b[2] for b in boxes) - 3); c1 = min(W - 1, max(b[3] for b in boxes) + 3)
    if r1 - r0 + 1 > _OB_MAX_CROP or c1 - c0 + 1 > _OB_MAX_CROP:
        cy, cx = (r0 + r1) // 2, (c0 + c1) // 2; h = _OB_MAX_CROP // 2
        r0, r1, c0, c1 = max(0, cy - h), min(H - 1, cy + h - 1), max(0, cx - h), min(W - 1, cx + h - 1)
    rows = cur.ascii.split("\n")
    return "\n".join(["Board around the objects that moved recently (rows %d-%d, cols %d-%d):" % (r0, r1, c0, c1)] +
                     ["  %2d %s" % (r, rows[r][c0:c1 + 1]) for r in range(r0, r1 + 1)])
_OB_ACT = ("Use this observation to act in this same python call; inspect in python only what it does not show.")
_ob_prompt_v1 = _obta.ToolAgent._build_user_prompt
def _ob_prompt_v2(self, action_num, **kw):
    text = _ob_prompt_v1(self, action_num, **kw)
    try:
        cur = kw.get("current_frame"); hist = kw.get("history_entries") or []
        if cur is not None and "Harness observation" in text:
            crop = _ob_player_crop(cur, hist)
            text = text + ("\n" + crop if crop else "") + "\n" + _OB_ACT
    except Exception as exc:
        print("OBSERVE v2: сбой вырезки: %r" % (exc,), file=_obsys.__stderr__, flush=True)
    return text
_obta.ToolAgent._build_user_prompt = _ob_prompt_v2
_OB_START = "\nHarness observation of the current board"
def _ob_strip_text(t):
    i = t.find(_OB_START)
    if i < 0:
        return t
    tail = "\n\nCurrent grid image:" if t.endswith("\n\nCurrent grid image:") else ""
    return t[:i] + tail
def _ob_strip_msg(m):
    c = m.get("content")
    if isinstance(c, str):
        return dict(m, content=_ob_strip_text(c))
    if isinstance(c, list):
        return dict(m, content=[dict(p, text=_ob_strip_text(p.get("text", ""))) if isinstance(p, dict) and p.get("type") == "text" else p for p in c])
    return m
_ob_orig_persist = _obta.ToolAgent._persistent_history_messages
def _ob_persist(self, messages, *a, **kw):
    hist = _ob_orig_persist(self, messages, *a, **kw)
    try:
        return [_ob_strip_msg(m) if m.get("role") == "user" else m for m in hist]
    except Exception:
        return hist
_obta.ToolAgent._persistent_history_messages = _ob_persist
print("NEXTFORK OBSERVE v2: v2 включён — вырезка вокруг двигавшихся объектов, копии из истории вынимаются, строка «действуй в этом же вызове»", flush=True)

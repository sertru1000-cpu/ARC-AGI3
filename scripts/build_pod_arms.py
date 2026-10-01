"""Варианты обвязки Скотта для прогонов на поде подряд, без перезапуска сервера (27.09).

База — kernels/pub_scott_cap3600_r2 (его ноутбук как есть + поиск колёс по обоим путям). Для пода:
  * 30 мин на игру;
  * ячейка настройки: при POD_REUSE_SERVER=1 и сохранённом /root/pod_setup_env.json сервер не поднимается заново;
    после первой настройки окружение сохраняется туда;
  * остановка сервера в конце отключается при POD_KEEP_SERVER=1.
Варианты (основание — аудиты 26–27.09, feedback-audit-harness-plumbing-first):
  a0 control   — как есть (сверка пода с Kaggle: 30 мин 3.83 / 3.63 / 2.14 / 2.49);
  a1 ledger    — F7 Скотта: журнал эффектов ходов по уровню во входе (колея: 83% развилок — нужный ход уже пробовали);
  a2 dedup     — F11 Скотта: из прошлых сообщений вынута повторяющаяся инструкция (~25% входа);
  a3 noreason  — в историю не возвращаются прошлые рассуждения (~25% входа; блок модели мира после починки держит выводы);
  a4 persist   — функции и классы модели сохраняются между вызовами в пределах игры (32–45% строк кода — повтор);
  a5 helpers   — готовые функции песочницы (как в lean, БЕЗ блока дисциплины) + разрешены difflib и time.
usage: .venv/bin/python scripts/build_pod_arms.py   ->  pod_arms/<вариант>.ipynb
"""
import ast, glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = glob.glob(os.path.join(ROOT, "kernels/pub_scott_cap3600_r2/*.ipynb"))[0]
OUT = os.path.join(ROOT, "pod_arms")

SETUP_OLD = '''env = _command_env()
for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):
    print(f"taaf.kaggle: setup command: {command}", flush=True)
    subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)
    # Re-read in case the command persisted new env keys.
    env = _command_env()
    os.environ.update(env)
'''
SETUP_NEW = '''_POD_SAVED = Path("/root/pod_setup_env.json")
if os.environ.get("POD_REUSE_SERVER") == "1" and _POD_SAVED.is_file():
    _cur = json.loads(SETUP_ENV_PATH.read_text()); _sv = json.loads(_POD_SAVED.read_text())
    _sv.update({k: v for k, v in _cur.items() if k.startswith("TAAF_KAGGLE_")})   # пути к датасетам — ТЕКУЩЕГО запуска
    SETUP_ENV_PATH.write_text(json.dumps(_sv, indent=2, sort_keys=True))
    env = _command_env()
    os.environ.update(env)
    print("POD: сервер уже поднят, настройка пропущена", flush=True)
else:
    env = _command_env()
    for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):
        print(f"taaf.kaggle: setup command: {command}", flush=True)
        subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)
        env = _command_env()
        os.environ.update(env)
    _POD_SAVED.write_text(SETUP_ENV_PATH.read_text())
'''
TEARDOWN_OLD = 'for command in json.loads((BUNDLE_DIR / "teardown_commands.json").read_text()):'
TEARDOWN_NEW = 'for command in ([] if os.environ.get("POD_KEEP_SERVER") == "1" else json.loads((BUNDLE_DIR / "teardown_commands.json").read_text())):'
ENV_ANCHOR = 'os.environ["LOCAL_ANALYZER_CONTEXT_WINDOW"] = "16384"'

NOREASON = r'''
# ===== a3 NOREASON: прошлые рассуждения не возвращаются в историю (27.09) =====
import sys as _nrsys
import inference.agent.tool_agent as _nrta
_nr_orig = _nrta.ToolAgent._persistent_history_messages
_nr_stat = {"stripped": 0}
def _nr_persist(self, messages, *a, **kw):
    hist = _nr_orig(self, messages, *a, **kw)
    out = []
    for m in hist:
        if m.get("role") == "assistant" and m.get("reasoning"):
            m = dict(m); m.pop("reasoning", None); _nr_stat["stripped"] += 1
        out.append(m)
    return out
_nrta.ToolAgent._persistent_history_messages = _nr_persist
print("ARM a3 NOREASON: прошлые рассуждения вынимаются из истории", flush=True)
'''

PERSIST = r'''
# ===== a4 PERSIST: функции модели живут до конца игры (27.09; классы песочница запрещает) =====
import ast as _pmast, sys as _pmsys
import inference.agent.tool_agent as _pmta
_PM_LIMIT = 8000
_PM_SAFE = {"bisect", "collections", "copy", "fractions", "functools", "heapq", "itertools", "json", "math", "operator", "random", "re",
            "statistics", "string"}   # ровно SAFE_MODULES песочницы: чужой импорт в префиксе ронял бы КАЖДЫЙ следующий вызов
_pm_orig_run = _pmta.ToolAgent._run_python_tool
def _pm_store(self, code):
    """Функции, импорты (белый список песочницы) и константы-литералы верхнего уровня — всё, на что опираются функции.
    27.09 v2: в первом прогоне a4 сохранялись только функции -> NameError на deque/Counter/collections и константах (8 из 26 падений)."""
    try:
        tree = _pmast.parse(code)
    except Exception:
        return
    st = self.__dict__.setdefault("_pm_defs", {})
    imp = self.__dict__.setdefault("_pm_imports", {})
    const = self.__dict__.setdefault("_pm_consts", {})
    for n in tree.body:
        src = _pmast.get_source_segment(code, n)
        if not src:
            continue
        if isinstance(n, _pmast.FunctionDef):          # классы песочница запрещает (__build_class__) — их не хранить
            st.pop(n.name, None); st[n.name] = src
        elif isinstance(n, (_pmast.Import, _pmast.ImportFrom)):
            mods = [a.name for a in n.names] if isinstance(n, _pmast.Import) else [n.module or ""]
            if all(m.split(".")[0] in _PM_SAFE for m in mods):
                imp[src] = True
        elif isinstance(n, _pmast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], _pmast.Name):
            try:
                _pmast.literal_eval(n.value)
                const.pop(n.targets[0].id, None); const[n.targets[0].id] = src
            except Exception:
                pass
    while st and sum(len(v) for v in st.values()) > _PM_LIMIT:
        st.pop(next(iter(st)))                                        # старейшее — вон
    while const and sum(len(v) for v in const.values()) > 3000:
        const.pop(next(iter(const)))
def _pm_run(self, state_path, arguments):
    try:
        key = str(getattr(state_path, "parent", state_path))
        if self.__dict__.get("_pm_game") != key:
            self._pm_game = key; self._pm_defs = {}; self._pm_imports = {}; self._pm_consts = {}
        code = str((arguments or {}).get("code", ""))
        parts = list(self.__dict__.get("_pm_imports", {})) + list(self.__dict__.get("_pm_consts", {}).values()) + \
                list(self.__dict__.get("_pm_defs", {}).values())
        prefix = ("\n".join(parts) + "\n\n") if parts else ""
        res = _pm_orig_run(self, state_path, dict(arguments or {}, code=prefix + code))
        _pm_store(self, code)
        return res
    except Exception as exc:
        print("PERSIST: сбой, вызов без слоя: %r" % (exc,), file=_pmsys.__stderr__, flush=True)
        return _pm_orig_run(self, state_path, arguments)
_pmta.ToolAgent._run_python_tool = _pm_run
_pm_orig_prompt = _pmta.ToolAgent._build_user_prompt
def _pm_prompt(self, action_num, **kw):
    text = _pm_orig_prompt(self, action_num, **kw)
    names = list(self.__dict__.get("_pm_defs", {}))
    return text + ("\nFunctions you define in `python` are kept for the rest of this game and are predefined in "
                   "every later call (redefine one to change it). Currently kept: %s." % (", ".join(names[-20:]) or "none"))
_pmta.ToolAgent._build_user_prompt = _pm_prompt
print("ARM a4 PERSIST: определения модели сохраняются между вызовами (до %d знаков)" % _PM_LIMIT, flush=True)
'''


OBSERVE = r"""
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
print("ARM a6 OBSERVE: наблюдение в каждом сообщении (объекты до %d, изменения, вырезка до %dx%d)" % (_OB_MAX_OBJ, _OB_MAX_CROP, _OB_MAX_CROP), flush=True)
"""

OBSERVE_V2 = OBSERVE.replace("# ===== a6 OBSERVE:", "# ===== a6b OBSERVE v2 (27.09): копии из истории вынимаются, вырезка вокруг двигавшихся объектов, «действуй в этом же вызове» =====\n# ===== a6 OBSERVE:").replace(
    "ARM a6 OBSERVE", "ARM a6b OBSERVE v2") + r"""
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
print("ARM a6b: v2 включён — вырезка вокруг двигавшихся объектов, копии из истории вынимаются, строка «действуй в этом же вызове»", flush=True)
"""

ZOOM = r"""
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
print("ARM a7 ZOOM: окрестность игрока %dx%d клеток при x%d (%d px) второй картинкой" % (_ZM_CELLS, _ZM_CELLS, _ZM_SCALE, _ZM_CELLS * _ZM_SCALE), flush=True)
"""

_BW_HELPERS_SRC = open(os.path.join(ROOT, "nextfork/src/ARC3-Inference/inference/agent/buildwm_sandbox/helpers.py"), encoding="utf-8").read()
BUILDWM = PERSIST.replace("# ===== a4 PERSIST:", "# ===== a8 BUILDWM (27.09): память функций (a4 v2) + check_step + подсказка достраивать симулятор, когда уровень затянулся =====\n# ===== a4 PERSIST:").replace(
    'print("ARM a4 PERSIST:', 'print("ARM a8 (база a4) PERSIST:') + r"""
import sys as _bwsys
import inference.agent.tool_agent as _bwta
_BW_STALL = 20                     # ходов на уровне без взятия -> подсказка достраивать симулятор
_BW_HELPERS = __HELPERS_REPR__
_bw_orig_sandbox = _bwta.run_sandboxed_python
def _bw_sandbox(*, code, **kw):
    return _bw_orig_sandbox(code=_BW_HELPERS + "\n" + str(code), **kw)
_bwta.run_sandboxed_python = _bw_sandbox
_BW_BLOCK = ("\nThis level is taking many moves. Build (or repair) a simulator of it: def step(rows, action) -> rows, where rows is "
             "the board as a list of letter strings (like current_frame.ascii.split(chr(10))) and action is e.g. 'RIGHT' or "
             "'MOUSE(row=R, col=C)'. Your functions persist between python calls, so improve it step by step. check_step(step) "
             "tests it on every transition of this level and prints counterexamples. When it reproduces all of them, search "
             "for a plan inside step (try action sequences there, they cost nothing) before executing real actions.")
_bw_stat = {"hints": 0}
_bw_orig_prompt = _bwta.ToolAgent._build_user_prompt
def _bw_prompt(self, action_num, **kw):
    text = _bw_orig_prompt(self, action_num, **kw)
    try:
        cur = kw.get("current_frame"); hist = kw.get("history_entries") or []
        if cur is not None:
            n = sum(1 for e in hist if getattr(getattr(e, "frame", None), "level", None) == cur.level)
            if n >= _BW_STALL:
                text += _BW_BLOCK; _bw_stat["hints"] += 1
                if _bw_stat["hints"] in (1, 50, 200):
                    print("BUILDWM: подсказка №%d (ходов на уровне %d)" % (_bw_stat["hints"], n), file=_bwsys.__stderr__, flush=True)
    except Exception as exc:
        print("BUILDWM: сбой: %r" % (exc,), file=_bwsys.__stderr__, flush=True)
    return text
_bwta.ToolAgent._build_user_prompt = _bw_prompt
print("ARM a8 BUILDWM: check_step в песочнице, подсказка после %d ходов на уровне" % _BW_STALL, flush=True)
""".replace("__HELPERS_REPR__", repr(_BW_HELPERS_SRC))


_BWB_HELPERS_SRC = open(os.path.join(ROOT, "nextfork/src/ARC3-Inference/inference/agent/buildwm_sandbox/base_step.py"), encoding="utf-8").read()
_BWT_HELPERS_SRC = open(os.path.join(ROOT, "nextfork/src/ARC3-Inference/inference/agent/buildwm_sandbox/tpl_step.py"), encoding="utf-8").read()
_BWB_SANDBOX = r"""
import inference.agent.python_tool_sandbox as _bwsb          # a8b: песочница — наша; шаблонам нужны numpy, hashlib и классы
if '"numpy"' not in _bwsb._SANDBOX_BOOTSTRAP:
    _bwsb._SANDBOX_BOOTSTRAP = _bwsb._SANDBOX_BOOTSTRAP.replace('"bisect",', '"bisect",\n        "numpy",\n        "hashlib",', 1)
    _bwsb._SANDBOX_BOOTSTRAP = _bwsb._SANDBOX_BOOTSTRAP.replace('"abs",', '"abs",\n        "__build_class__",\n        "__name__",\n        "staticmethod",', 1)
    import os as _bwos, numpy as _bwnp                      # песочница стартует с -I -S: пакеты окружения ей не видны
    _bwsb._SANDBOX_BOOTSTRAP = "import sys as _bwsys0\n_bwsys0.path.append(%r)\n" % _bwos.path.dirname(_bwos.path.dirname(_bwnp.__file__)) + _bwsb._SANDBOX_BOOTSTRAP
"""
# a8b (27.09, собран заранее, без прогона): a8 не включался — порог 20 ходов на уровне почти не достигался за 30 мин
# (игра делает 5–43 хода), а где включался, модель ни разу не написала step. Порог 8 и готовая заготовка base_step.
BUILDWM_B = BUILDWM.replace("# ===== a8 BUILDWM (27.09)", "# ===== a8b BUILDWM (27.09, порог 8 + заготовка base_step)").replace(
    "_BW_STALL = 20 ", "_BW_STALL = 8 ").replace(
    "_BW_HELPERS = " + repr(_BW_HELPERS_SRC), "_BW_HELPERS = " + repr(_BW_HELPERS_SRC + "\n" + _BWB_HELPERS_SRC + "\n" + _BWT_HELPERS_SRC)).replace(
    '_BW_BLOCK = ("\\nThis level is taking many moves. Build (or repair) a simulator',
    '_BW_BLOCK = ("\\nThis level is taking many moves. base_step(rows, action) is predefined: it replays transitions already '
    'seen on this level and otherwise repeats what the same action did last time (moves the same-colored object, repeats '
    'the same cell changes); tpl_step(rows, action) is a stronger learned version (object moves with blocking and sliding, '
    'click recolors). check_step(tpl_step) shows where it fails. Build (or repair) a simulator').replace(
    '"ARM a8 BUILDWM:', '"ARM a8b BUILDWM:') + _BWB_SANDBOX


_BWP_HELPERS_SRC = open(os.path.join(ROOT, "nextfork/src/ARC3-Inference/inference/agent/buildwm_sandbox/plan.py"), encoding="utf-8").read()
_BWTY_HELPERS_SRC = open(os.path.join(ROOT, "nextfork/src/ARC3-Inference/inference/agent/buildwm_sandbox/tycho.py"), encoding="utf-8").read()
# a8c (28.09, собран заранее, без прогона): a8b — модель впервые писала step (15 блоков, 8 игр; ar25 довела до 11/11),
# но план в симуляторе почти не искала (step после написания — в 5 блоках). a8c = a8b + готовый поиск plan(step, goal).
BUILDWM_C = BUILDWM_B.replace("# ===== a8b BUILDWM (27.09, порог 8 + заготовка base_step)",
                              "# ===== a8c BUILDWM (28.09, a8b + планировщик plan)").replace(
    "_BW_HELPERS = " + repr(_BW_HELPERS_SRC + "\n" + _BWB_HELPERS_SRC + "\n" + _BWT_HELPERS_SRC),
    "_BW_HELPERS = " + repr(_BW_HELPERS_SRC + "\n" + _BWB_HELPERS_SRC + "\n" + _BWT_HELPERS_SRC + "\n" + _BWP_HELPERS_SRC + "\n" + _BWTY_HELPERS_SRC)).replace(
    "When it reproduces all of them, search \"\n             \"for a plan inside step (try action sequences there, they cost nothing) before executing real actions.",
    "When it reproduces all of them, call plan(step, goal) with goal(rows) -> True when the level \"\n             \"is won (clicks=[(row, col), ...] for click games): it searches action sequences INSIDE step for free and returns \"\n             \"the shortest list; execute it with run_plan(p, step) — it stops at the first move that diverges from step. \"\n             \"Check your goal first: check_goal(goal) (it must be False on every board already seen on this level).").replace(
    '"ARM a8b BUILDWM:', '"ARM a8c BUILDWM:').replace('\\n        "hashlib",', '\\n        "hashlib",\\n        "time",')


def helpers_cell():
    lean = open(os.path.join(ROOT, "kernels/notebooks_nextfork_lean/cell15.py"), encoding="utf-8").read()
    start = lean.index("_LEAN_HELPERS = "); end = lean.index("\n", start)
    helpers_line = lean[start:end]
    return r'''
# ===== a5 HELPERS: готовые функции песочницы (как в lean, без блока дисциплины) + difflib/time (27.09) =====
import sys as _hpsys
import inference.agent.tool_agent as _hpta
import inference.agent.python_tool_sandbox as _hpsb
''' + helpers_line + r'''
_hp_orig_sandbox = _hpta.run_sandboxed_python
def _hp_sandbox(*, code, **kw):
    return _hp_orig_sandbox(code=_LEAN_HELPERS + "\n" + str(code), **kw)
_hpta.run_sandboxed_python = _hp_sandbox
if '"difflib"' not in _hpsb._SANDBOX_BOOTSTRAP:
    _hpsb._SANDBOX_BOOTSTRAP = _hpsb._SANDBOX_BOOTSTRAP.replace('"bisect",', '"bisect",\n        "difflib",\n        "time",', 1)
_hp_orig_prompt = _hpta.ToolAgent._build_user_prompt
def _hp_prompt(self, action_num, **kw):
    return _hp_orig_prompt(self, action_num, **kw) + (
        "\nPredefined in every python call (do not re-implement): objs(fr=None, color=None, min_pixels=1) -> objects with "
        "r0,r1,c0,c1,h,w,center,pixels; summ(fr=None) -> one line per object; crop(r0,r1,c0,c1,fr=None); "
        "diff(a=None,b=None) -> changed cells and moved/appeared/gone objects (default previous->current); "
        "effects() -> per action: tried / changed the board; grid(fr=None); bb(node). Modules difflib and time are allowed.")
_hpta.ToolAgent._build_user_prompt = _hp_prompt
print("ARM a5 HELPERS: %d строк помощников, difflib/time разрешены: %s" % (len(_LEAN_HELPERS.splitlines()), '"difflib"' in _hpsb._SANDBOX_BOOTSTRAP), flush=True)
'''


ARMS = {
    "a0_control": ({}, ""),
    "a1_ledger": ({"AGENTFIX_LEDGER": "1"}, ""),        # 27.09: журнал молча НЕ включается без F2+F5 — фактически второй контроль
    "a1b_ledger": ({"AGENTFIX_LEDGER": "1", "AGENTFIX_RESULT": "1", "AGENTFIX_NOIMPACT": "1"}, ""),   # журнал едет на F2+F5
    "a2_dedup": ({"AGENTFIX_DEDUP": "1"}, ""),
    "a3_noreason": ({}, NOREASON),
    "a4_persist": ({}, PERSIST),
    "a5_helpers": ({}, None),
    "a6_observe": ({}, OBSERVE),
    "a6b_observe": ({}, OBSERVE_V2),
    "a7_zoom": ({}, ZOOM),
    "a8_buildwm": ({}, BUILDWM),
    "a8b_buildwm": ({}, BUILDWM_B),                               # собран заранее; прогон — только по слову
    "a8c_buildwm": ({}, BUILDWM_C),                               # собран заранее 28.09; прогон — только по слову
    "a9_persist_observe": ({}, PERSIST + "\n" + OBSERVE_V2),     # a4 режет ответ, a6b режет осмотры, но удлиняет ответ
}


def build_v4(names=("v4_a", "v4_b"), extra=(), tag="v4pod", cap=1800, seqs=16):
    """Боевой кандидат: наш стоковый ноутбук (build_nextfork_notebook --seqs 16 --cap 1800), датасет nextfork из локальной папки.
    extra — флаги сборщика: ("--no-v3",) даёт v4-lite (Скотт + память функций, без запрета пустого хода и правок промпта v3)."""
    import subprocess
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts/build_nextfork_notebook.py"), "--seqs", str(seqs), "--cap", str(cap),
                    "--name", tag, *extra], check=True, capture_output=True)
    src = os.path.join(ROOT, "kernels/notebooks_nextfork_%s/submission.ipynb" % tag)
    for name in names:
        nb = json.load(open(src, encoding="utf-8")); hits = {"setup": 0, "teardown": 0, "bundle": 0}
        for c in nb["cells"]:
            t = "".join(c["source"])
            if SETUP_OLD in t: t = t.replace(SETUP_OLD, SETUP_NEW); hits["setup"] += 1
            if TEARDOWN_OLD in t: t = t.replace(TEARDOWN_OLD, TEARDOWN_NEW); hits["teardown"] += 1
            if BUNDLE_OLD in t: t = t.replace(BUNDLE_OLD, BUNDLE_NEW); hits["bundle"] += 1
            c["source"] = t.splitlines(keepends=True)
        assert all(v == 1 for v in hits.values()), (name, hits)
        full = "".join("".join(c["source"]) for c in nb["cells"])
        assert '"TAAF_VLLM_MAX_NUM_SEQS": "%d"' % seqs in full and "max_runtime_s_per_game = %.1f" % cap in full, name
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                compile("".join(c["source"]), name, "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        json.dump(nb, open(os.path.join(OUT, name + ".ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("%-12s боевой кандидат v4 (наш ноутбук, 16 мест, %d мин, бандл nextfork)" % (name, cap // 60))


BUNDLE_OLD = """    for marker in Path("/kaggle/input").rglob(DATASET_BUNDLE_MARKER):
        return marker.parent"""
BUNDLE_NEW = """    _markers = list(Path("/kaggle/input").rglob(DATASET_BUNDLE_MARKER))
    for marker in _markers:                      # под: рядом лежит бандл keithtyser — брать бандл со штампом nextfork
        if (marker.parent / "NEXTFORK_VERSION.txt").is_file():
            return marker.parent
    for marker in _markers:
        return marker.parent"""


BUNDLE_OLD_KEITH = """    _markers = list(Path("/kaggle/input").rglob(DATASET_BUNDLE_MARKER))
    for marker in _markers:                      # под: рядом лежит наш nextfork — брать бандл keithtyser (без штампа)
        if not (marker.parent / "NEXTFORK_VERSION.txt").is_file():
            return marker.parent
    for marker in _markers:
        return marker.parent"""


def build():
    os.makedirs(OUT, exist_ok=True)
    for name, (env, cell) in ARMS.items():
        nb = json.load(open(BASE, encoding="utf-8"))
        hits = {"cap": 0, "setup": 0, "teardown": 0, "env": 0}
        for c in nb["cells"]:
            s = "".join(c["source"])
            if "bm.solver.max_runtime_s_per_game = 3600.0" in s:
                s = s.replace("bm.solver.max_runtime_s_per_game = 3600.0", "bm.solver.max_runtime_s_per_game = 1800.0"); hits["cap"] += 1
            if SETUP_OLD in s:
                s = s.replace(SETUP_OLD, SETUP_NEW); hits["setup"] += 1
            if TEARDOWN_OLD in s:
                s = s.replace(TEARDOWN_OLD, TEARDOWN_NEW); hits["teardown"] += 1
            if ENV_ANCHOR in s:
                lines = "".join('\nos.environ["%s"] = "%s"   # вариант %s' % (k, v, name) for k, v in env.items())
                s = s.replace(ENV_ANCHOR, ENV_ANCHOR + lines, 1); hits["env"] += 1
            c["source"] = s.splitlines(keepends=True)
        assert all(v == 1 for v in hits.values()), (name, hits)
        for c in nb["cells"]:                        # под: рядом лежит наш nextfork — ноутбукам Скотта брать бандл БЕЗ штампа nextfork
            t = "".join(c["source"])
            if BUNDLE_OLD in t:
                c["source"] = t.replace(BUNDLE_OLD, BUNDLE_OLD_KEITH).splitlines(keepends=True)
        code = helpers_cell() if cell is None else cell
        if code:
            at = next(i for i, c in enumerate(nb["cells"]) if "AGENTFIX LIVE" in "".join(c["source"]))
            nb["cells"].insert(at + 1, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                                        "source": code.splitlines(keepends=True)})
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                compile("".join(c["source"]), name, "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        json.dump(nb, open(os.path.join(OUT, name + ".ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("%-12s ячеек %d | env %s | своя ячейка %s" % (name, len(nb["cells"]), env or "-", "да" if code else "нет"))


if __name__ == "__main__":
    import sys
    build()
    build_v4()
    build_v4(("v4lite_a",), ("--no-v3",), "v4litepod")

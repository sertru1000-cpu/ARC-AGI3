
# =====================================================================
# РАЗБОР ПЕРЕХОДА ВО ВХОДЕ: что изменила последняя пачка действий, посчитано харнессом точно.
# Проба 11.09 на записанных ходах: генерация 1502 -> 1253 токена (-17%), p = 0.108.
# Выход модели не трогаем. Перезаход того же шага получает тот же разбор.
# =====================================================================
import collections as _dcol
import inference.agent.tool_agent as _dta

_DIFF_LETTERS = "WwgGcBMPRbSYOrNp"   # легенда системного промпта, сверена с board_ascii
_DIFF_MAX_LIST = 8
_DIFF_BIG_CELLS = 300
_DIFF_BIG_OBJS = 12
_DIFF_STATS = {"notes": 0, "summaries": 0, "noop": 0, "reuse": 0, "fail": 0}
_DIFF_INSTRUCTION = (
    "Use this diff directly as ground truth for what the last action sequence changed. "
    "Do NOT restate coordinates, re-list objects, or re-describe the board in your reasoning - "
    "spend reasoning only on hypotheses about the mechanics and on choosing the next action.")


def _diff_comps(g, bg, only):
    h, w = len(g), len(g[0])
    seen = set()
    out = []
    for r0, c0 in only:
        if (r0, c0) in seen or g[r0][c0] == bg:
            continue
        col = g[r0][c0]
        stack, cells = [(r0, c0)], []
        seen.add((r0, c0))
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen and g[ny][nx] == col:
                    seen.add((ny, nx))
                    stack.append((ny, nx))
        ry = min(y for y, _ in cells); rx = min(x for _, x in cells)
        hh = max(y for y, _ in cells) - ry + 1; ww = max(x for _, x in cells) - rx + 1
        shape = tuple(sorted((y - ry, x - rx) for y, x in cells))
        edge = min(hh, ww) == 1 and max(hh, ww) >= 8 and (ry == 0 or rx == 0 or ry + hh == h or rx + ww == w)
        out.append({"col": col, "shape": shape, "pos": (ry, rx), "size": (hh, ww), "n": len(cells), "edge": edge})
    return out


def _diff_name(o):
    full = o["n"] == o["size"][0] * o["size"][1]
    return "%s %dx%d%s" % (_DIFF_LETTERS[o["col"] % 16], o["size"][0], o["size"][1], "" if full else " shape")


def _diff_text(before, after, executed, lv_before, lv_after):
    lines = ["HARNESS DIFF (computed exactly by the harness from the frames before and after your last action sequence):",
             "- executed: %s" % (", ".join(executed) if executed else "none")]
    if lv_before != lv_after or len(before) != len(after) or len(before[0]) != len(after[0]):
        lines.append("- level changed: the board was replaced by a new level, per-object diff omitted")
        return "\n".join(lines + [_DIFF_INSTRUCTION])
    h, w = len(after), len(after[0])
    changed = [(r, c) for r in range(h) for c in range(w) if before[r][c] != after[r][c]]
    if not changed:
        _DIFF_STATS["noop"] += 1
        lines.append("- board changed: NO - the sequence was a complete no-op on the board")
        return "\n".join(lines + [_DIFF_INSTRUCTION])
    rs = [r for r, _ in changed]; cs = [c for _, c in changed]
    lines.append("- board changed: %d cells, rows %d-%d, cols %d-%d" % (len(changed), min(rs), max(rs), min(cs), max(cs)))
    bg = _dcol.Counter(v for row in after for v in row).most_common(1)[0][0]
    A = _diff_comps(before, bg, changed)
    B = _diff_comps(after, bg, changed)
    edges = [o for o in A + B if o["edge"]]
    A = [o for o in A if not o["edge"]]; B = [o for o in B if not o["edge"]]
    used, moved, gone = set(), [], []
    for a in A:
        cand = [(abs(b["pos"][0] - a["pos"][0]) + abs(b["pos"][1] - a["pos"][1]), i) for i, b in enumerate(B)
                if i not in used and b["col"] == a["col"] and b["shape"] == a["shape"]]
        if cand:
            _, i = min(cand); used.add(i); b = B[i]
            dr, dc = b["pos"][0] - a["pos"][0], b["pos"][1] - a["pos"][1]
            if (dr, dc) != (0, 0):
                moved.append((a, b, dr, dc))
        else:
            gone.append(a)
    appeared = [b for i, b in enumerate(B) if i not in used]
    if edges:
        seen_e = sorted({"%s at (%d,%d)" % (_diff_name(o), *o["pos"]) for o in edges})
        lines.append("- edge bars changed (likely HUD/timer, not puzzle objects): %s" % "; ".join(seen_e[:4]))
    n_obj = len(moved) + len(appeared) + len(gone)
    if len(changed) > _DIFF_BIG_CELLS or n_obj > _DIFF_BIG_OBJS:
        _DIFF_STATS["summaries"] += 1
        big = sorted([b for _, b, _, _ in moved] + appeared + gone, key=lambda o: -o["n"])[:3]
        lines.append("- large change: %d objects moved, %d appeared, %d disappeared - likely a scene redraw; largest: %s"
                     % (len(moved), len(appeared), len(gone),
                        "; ".join("%s at (%d,%d)" % (_diff_name(o), *o["pos"]) for o in big) or "-"))
    else:
        if moved:
            lines.append("- moved: %s" % "; ".join("%s at (%d,%d) -> (%d,%d) [%+d,%+d]" % (_diff_name(a), *a["pos"], *b["pos"], dr, dc)
                                                for a, b, dr, dc in moved[:_DIFF_MAX_LIST]))
        if appeared:
            lines.append("- appeared or reshaped: %s" % "; ".join("%s at (%d,%d)" % (_diff_name(o), *o["pos"]) for o in appeared[:_DIFF_MAX_LIST]))
        if gone:
            lines.append("- disappeared or reshaped: %s" % "; ".join("%s at (%d,%d)" % (_diff_name(o), *o["pos"]) for o in gone[:_DIFF_MAX_LIST]))
    return "\n".join(lines + [_DIFF_INSTRUCTION])


_diff_orig_prompt = _dta.ToolAgent._build_user_prompt


def _diff_build_user_prompt(self, action_num, **kw):
    text = _diff_orig_prompt(self, action_num, **kw)
    try:
        frame = kw.get("current_frame")
        grid = getattr(frame, "grid", None) if frame is not None else None
        if not grid:
            return text
        step = int(getattr(frame, "step", 0) or 0)
        level = int(getattr(frame, "level", 0) or 0)
        cache = getattr(self, "_diff_cache", None)
        if cache is not None and cache[0] == step:          # перезаход того же шага
            _DIFF_STATS["reuse"] += 1
            return text + ("\n\n" + cache[1] if cache[1] else "")
        prev = getattr(self, "_diff_prev", None)
        note = ""
        if prev is not None and prev[0] < step:
            summary = kw.get("previous_step_summary") or {}
            acts = [str(a).strip() for a in (summary.get("executed_actions") or []) if str(a).strip()] \
                if isinstance(summary, dict) else []
            note = _diff_text(prev[2], grid, acts[:10], prev[1], level)
            _DIFF_STATS["notes"] += 1
        self._diff_prev = (step, level, grid)
        self._diff_cache = (step, note)
        return text + ("\n\n" + note if note else "")
    except Exception as _exc:
        _DIFF_STATS["fail"] += 1
        print("[DIFF] сбой разбора: %r" % (_exc,), flush=True)
        return text


_dta.ToolAgent._build_user_prompt = _diff_build_user_prompt

# =====================================================================
# ПРОВЕРКА НА ПЕТЛЮ: харнесс сообщает, что эта доска уже была.
#
# Считаем хеш доски вместе с номером уровня. При возврате в виденное состояние дописываем
# в конец промпта: на каком шаге оно было, что из него делали, сколько раз повторялось.
# Ничего не запрещаем — только сообщаем факт, который модель по своей истории не восстановит.
# Выход модели не трогаем: см. noreason 07.09 (2.91 против 9.43).
# =====================================================================
import hashlib as _lh
import inference.agent.tool_agent as _lta

_LOOP_KEEP_ACTS = 6      # сколько действий прошлого захода показывать
_LOOP_STATS = {"hits": 0, "states": 0, "fail": 0, "reentry": 0}

_loop_orig_prompt = _lta.ToolAgent._build_user_prompt


def _loop_build_user_prompt(self, action_num, **kw):
    text = _loop_orig_prompt(self, action_num, **kw)
    try:
        frame = kw.get("current_frame")
        board = getattr(frame, "ascii", None) if frame is not None else None
        if not board:
            return text
        seen = getattr(self, "_loop_seen", None)
        if seen is None:
            seen = {}
            self._loop_seen = seen
            self._loop_last = None
        step = int(getattr(frame, "step", 0) or 0)
        key = (int(getattr(frame, "level", 0) or 0),
               _lh.md5(str(board).encode("utf-8", "ignore")).hexdigest())
        # ПЕРЕЗАХОД ТОГО ЖЕ ШАГА — НЕ ВОЗВРАТ. Солвер Duck повторно входит в тот же analysis_step
        # с той же доской при yielded_control (раз в 60 с) и при сбое запроса: это 36-38% всех
        # построений промпта. Версия 1 принимала их за петлю — 120 ложных тревог из 132 в пробе
        # 10.09. Пропускаем целиком и ДО записи «что делали», иначе туда попадут ходы, которые
        # в это состояние ПРИВЕЛИ.
        rec0 = seen.get(key)
        if rec0 is not None and rec0["step"] == step:
            _LOOP_STATS["reentry"] += 1
            return text
        # чем закончился прошлый заход из прошлого состояния
        summary = kw.get("previous_step_summary")
        last = getattr(self, "_loop_last", None)
        if last is not None and last != key and isinstance(summary, dict):
            acts = [str(a).strip() for a in (summary.get("executed_actions") or []) if str(a).strip()]
            prev = seen.get(last)
            if prev is not None and not prev["after"] and acts:
                prev["after"] = acts[:_LOOP_KEEP_ACTS]
        rec = seen.get(key)
        if rec is None:
            seen[key] = {"step": step, "after": [], "hits": 0}
            _LOOP_STATS["states"] += 1
        else:
            rec["hits"] += 1
            _LOOP_STATS["hits"] += 1
            note = ["", "LOOP CHECK (computed by the harness from the recorded frames, not by you):",
                    "This exact board (level %d) was already seen at step %d; you are back on it "
                    "for the %s time." % (key[0], rec["step"], "%d-th" % (rec["hits"] + 1))]
            if rec["after"]:
                note.append("From that state you then executed: %s." % ", ".join(rec["after"]))
                note.append("That path has already been tried and led back here. Repeating it "
                            "cannot produce new information -- choose an action you have NOT "
                            "tried from this board, or re-examine what the goal actually is.")
            else:
                note.append("Whatever you did from it led back here. Choose an action you have "
                            "NOT tried from this board.")
            text = text + "\n" + "\n".join(note)
            print("[LOOP] шаг %d: возврат в состояние шага %d (повтор %d), всего возвратов %d"
                  % (step, rec["step"], rec["hits"], _LOOP_STATS["hits"]), flush=True)
        self._loop_last = key
    except Exception as _exc:
        _LOOP_STATS["fail"] += 1
        print("[LOOP] сбой учёта: %r" % (_exc,), flush=True)
    return text


_lta.ToolAgent._build_user_prompt = _loop_build_user_prompt


print("INPUT LEVERS: разбор перехода во входе + проверка на петлю v2 (%s); потолок на игру %s с. "
      "Проба промпта 11.09: генерация 1502 -> 1253 (-17%%). Пороги 2а: генерация <= 1280, действий на игру >= 195."
      % ("бой" if TRUE_SUBMISSION else "Фаза A", bm.solver.max_runtime_s_per_game), flush=True)

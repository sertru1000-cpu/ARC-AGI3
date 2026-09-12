"""Сборка прогона 2а: правки ТОЛЬКО во входе промпта — разбор перехода + проверка на петлю v2.

ЧТО ВНУТРИ (одна ячейка поверх базы `arc3-stock-flash`, после стокового soft_end):
  1. Разбор перехода, считаемый ВНУТРИ харнесса. В пробе 11.09 он строился по записанным доскам;
     здесь — из сетки кадра прошлого шага и текущего кадра при построении промпта. Перезаход
     солвера в тот же шаг (yielded_control / сбой, 36-38% построений) получает ТОТ ЖЕ разбор,
     а не «ничего не изменилось».
  2. Проверка на петлю v2 — текст ячейки `scripts/build_loop_notebook.py` без пробного потолка
     (прошла пробу 10.09: 15 предупреждений, все настоящие, ноль сбоев).

ПРАВКИ РАЗБОРА ПО НАВОДКЕ ПРОБЫ. Сильнее всего укорачивались ответы там, где разбор чистый, и
удлинялись там, где шумный (ar25 — 359 клеток и гора одноклеточных «сдвигов», bp35 — 1232 клетки).
Поэтому: объекты сопоставляются с БЛИЖАЙШИМ одинаковым, а не с первым; тонкие полосы по краю поля
выводятся отдельной строкой как вероятный HUD/таймер; при больших изменениях (больше 300 клеток
или больше 12 объектов) вместо списка — короткая сводка с тремя крупнейшими объектами.

ВЫХОД МОДЕЛИ НЕ ТРОГАЕТСЯ: ни `_run_python_tool`, ни `_chat_completion` в ячейке нет (проверяется).
Потолок на игру стоковый 7920 с — это полный прогон Фазы A, не проба.

Пушить ВЕРСИЕЙ 4 в `arc3-stock-flash-loop` (новый слаг 11.09 стартовал без данных соревнования).

usage:  .venv/bin/python scripts/build_input_levers_notebook.py
"""
import ast
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_loop_notebook  # noqa: E402

DIFF_CELL = r'''
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
'''


def loop_cell_without_probe_cap() -> str:
    cell = build_loop_notebook.CELL
    cut = cell.index("if not TRUE_SUBMISSION:")
    return cell[:cut]


FINAL = r'''
print("INPUT LEVERS: разбор перехода во входе + проверка на петлю v2 (%s); потолок на игру %s с. "
      "Проба промпта 11.09: генерация 1502 -> 1253 (-17%%). Пороги 2а: генерация <= 1280, действий на игру >= 195."
      % ("бой" if TRUE_SUBMISSION else "Фаза A", bm.solver.max_runtime_s_per_game), flush=True)
'''


def main() -> None:
    cell_text = DIFF_CELL + loop_cell_without_probe_cap() + FINAL
    src = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))
    nb = json.loads(json.dumps(src))
    body = "".join(nb["cells"][15]["source"])
    anchor = "    seconds=budget - 600.0\n)"
    at = body.find(anchor)
    if at < 0 or body.find("await bm.run(") < 0:
        raise SystemExit("не нашёл стоковый soft_end или запуск прогона")
    at += len(anchor)
    code = body[:at] + "\n" + cell_text + body[at:]
    nb["cells"][15]["source"] = code.splitlines(keepends=True)
    out = "kernels/notebooks_stockflash_input"
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(cell_text)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash_loop/kernel-metadata.json"))
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    compile(code, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    diff = [i for i in range(18) if "".join(nb["cells"][i]["source"]) != "".join(src["cells"][i]["source"])]
    new = "".join(nb["cells"][15]["source"])
    print("ok   изменена только ячейка 15:", diff == [15])
    print("ok   патч после стокового soft_end и до запуска:",
          new.find("_DIFF_LETTERS") > new.find("budget - 600.0") and new.find("_LOOP_KEEP_ACTS") < new.find("await bm.run("))
    print("ok   разбор ставится ПЕРВЫМ, петля оборачивает его:", cell_text.find("_diff_orig_prompt =") < cell_text.find("_loop_orig_prompt ="))
    print("ok   выход модели не трогаем:", "_run_python_tool" not in cell_text and "_chat_completion" not in cell_text)
    print("ok   пробного потолка нет (полный прогон):", "max_runtime_s_per_game = 1500" not in cell_text)
    print("ok   слаг для пуша:", meta["id"])
    print("ok   компилируется, %d символов" % len(code))


if __name__ == "__main__":
    main()

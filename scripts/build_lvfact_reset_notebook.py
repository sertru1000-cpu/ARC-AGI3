"""Три слоя без выходных токенов для стокового Duck (план 13.09, пункты 1 и 3). Каждый — свой ноутбук.

1. ФАКТ О ЗАВЕРШЕНИИ УРОВНЯ (lvfact). При переходе на новый уровень обвязка сама сообщает модели,
   ЧЕМ был взят предыдущий: завершивший ход, последние ходы перед ним, сколько ходов ушло, и доска
   перед завершающим ходом (один раз, при первом промпте нового уровня; дальше — одна строка).
   Только входные токены. Повод: первый уровень берут 21 игры из 25, второй — 9; слот «заметки
   между уровнями» модель заполняет в 0.3% ответов.

2. RESET ПРИ ЗАСТРЕВАНИИ С СОХРАНЕНИЕМ ПАМЯТИ (reset). После %(stall)d ходов на уровне без взятия
   обвязка сама делает RESET уровня (доска чистая, взятые уровни сохраняются -- ONLY_RESET_LEVELS)
   и кладёт во вход сжатую сводку: какие ходы пробовались, сколько различных досок видела, сколько
   раз возвращалась в виденное. Не амнезия (история не режется -- та резала и вредила), а сброс
   доски. Не больше %(max_resets)d сбросов на уровень. Повод: признак «100 ходов без уровня»
   предсказывает застревание 8/9; круг ходов кормится состоянием доски.

3. СКРИПТОВЫЙ ИССЛЕДОВАТЕЛЬ (explore). Тот же признак застревания, но вместо сброса обвязка сама, без
   модели, пробует каждую стрелку по 3 раза и кликает по центрам объектов, которых ещё не кликали
   (<= 16), и кладёт отчёт «что меняло доску» во вход. Ноль выходных токенов, ходы движка.

Швы: _build_user_prompt (все), _run_python_tool (reset/explore -- ПОСЛЕ ходов модели, чтобы
следующий промпт видел чистую доску вместе со сводкой). Только оффлайн. Пороги (до пуска), против
базы 10.25 знаковым тестом: польза -- победы−поражения >= +8 и медиана >= +8; вред -- <= −6;
механизм: генерация 1400–1500 (слои не должны удлинять ответ), число срабатываний в логе.

usage:  .venv/bin/python scripts/build_lvfact_reset_notebook.py
"""
import ast
import json
import os

STALL_MOVES, STALL_CALLS, MAX_RESETS, NOTE_TURNS = 100, 40, 2, 3   # второй триггер (13.09 12:30): 40 вызовов без уровня

LVFACT_CELL = r'''
# =====================================================================
# ФАКТ О ЗАВЕРШЕНИИ УРОВНЯ (13.09): при переходе на новый уровень обвязка сообщает, чем был взят
# предыдущий. Только входные токены. Только оффлайн.
# =====================================================================
import inference.agent.tool_agent as _wta
_lf_stats = {"games": 0, "levels": 0, "facts": 0, "turns": 0}

def _lf_fact(history_entries, new_level):
    """Завершивший ход, последние ходы перед ним, число ходов на уровне, доска перед завершением."""
    ents = list(history_entries or [])
    idx = None
    for i, e in enumerate(ents):
        try:
            if int(getattr(e.frame, "level", 0)) >= int(new_level):
                idx = i; break
        except Exception:
            continue
    if idx is None or idx == 0:
        return None, None
    start = 0
    for j in range(idx - 1, -1, -1):
        try:
            if int(getattr(ents[j].frame, "level", 0)) < int(new_level) - 1:
                start = j + 1; break
        except Exception:
            continue
    fin = str(getattr(ents[idx], "action", "") or "?")
    before = [str(getattr(e, "action", "") or "?") for e in ents[max(start, idx - 8):idx]]
    n_moves = idx - start + 1
    prev_ascii = getattr(ents[idx - 1].frame, "ascii", "") or ""
    short = ("LEVEL %d FACT (from the harness log): level %d was completed after %d moves; the completing move was %s; "
             "the moves right before it were: %s." % (new_level, new_level - 1, n_moves, fin, ", ".join(before) or "-"))
    full = short + ("\nThe board right BEFORE the completing move (level %d) looked like this -- use it to infer what "
                    "condition completes a level in this game:\n%s" % (new_level - 1, prev_ascii))
    return short, full

if not TRUE_SUBMISSION:
    _lf_orig_prompt = _wta.ToolAgent._build_user_prompt
    def _lf_prompt(self, action_num, *args, **kwargs):
        text = _lf_orig_prompt(self, action_num, *args, **kwargs)
        try:
            st = getattr(self, "_lf_state", None)
            if st is None:
                st = {"level": None, "short": None, "full_left": 0}; self._lf_state = st; _lf_stats["games"] += 1
            _lf_stats["turns"] += 1
            lv = getattr(kwargs.get("current_frame"), "level", None)
            if lv is not None:
                lv = int(lv)
                if st["level"] is not None and lv > st["level"]:
                    _lf_stats["levels"] += 1
                    short, full = _lf_fact(kwargs.get("history_entries"), lv)
                    if short:
                        st["short"], st["full"], st["full_left"] = short, full, 1
                        _lf_stats["facts"] += 1
                        print("[LVFACT] уровень %d: факт выдан (%s)" % (lv, short[:90]), flush=True)
                st["level"] = lv
            if st.get("short"):
                if st["full_left"] > 0:
                    st["full_left"] -= 1
                    return st["full"] + "\n\n" + text
                return st["short"] + "\n\n" + text
            return text
        except Exception as _e:
            print("[LVFACT] сбой: %r" % (_e,), flush=True)
            return text
    _wta.ToolAgent._build_user_prompt = _lf_prompt
    print("LVFACT: факт о завершении уровня во входе при переходе на новый уровень (полный один раз, дальше строка). "
          "ПОРОГИ против базы 10.25: польза -- победы-поражения >= +8 и медиана >= +8; вред -- <= -6; генерация 1400-1500.", flush=True)
'''

RESET_CELL = r'''
# =====================================================================
# RESET ПРИ ЗАСТРЕВАНИИ С СОХРАНЕНИЕМ ПАМЯТИ (13.09): после %(stall)d ходов на уровне без взятия обвязка
# сама сбрасывает уровень (взятые уровни сохраняются) и кладёт во вход сводку испробованного.
# Не больше %(max_resets)d сбросов на уровень. Только оффлайн.
# =====================================================================
import hashlib as _rs_hashlib
import inference.agent.tool_agent as _wta
_RS_STALL, _RS_CALLS, _RS_MAX, _RS_NOTE_TURNS = %(stall)d, %(stall_calls)d, %(max_resets)d, %(note_turns)d
_rs_stats = {"games": 0, "resets": 0, "turns": 0, "blocked_max": 0, "levels": 0}

def _rs_summary(history_entries, level):
    """Сводка по ходам текущего уровня: гистограмма ходов, различные доски, возвраты в виденное."""
    ents = [e for e in (history_entries or []) if int(getattr(e.frame, "level", 0) or 0) == int(level)]
    hist = {}
    seen = set(); returns = 0
    for e in ents:
        a = str(getattr(e, "action", "") or "?").split("(")[0]
        hist[a] = hist.get(a, 0) + 1
        h = _rs_hashlib.blake2b((getattr(e.frame, "ascii", "") or "").encode(), digest_size=8).hexdigest()
        if h in seen:
            returns += 1
        seen.add(h)
    top = ", ".join("%%s x%%d" %% (k, v) for k, v in sorted(hist.items(), key=lambda kv: -kv[1])[:6])
    return len(ents), top, len(seen), returns

if not TRUE_SUBMISSION:
    _rs_orig_prompt = _wta.ToolAgent._build_user_prompt
    def _rs_prompt(self, action_num, *args, **kwargs):
        text = _rs_orig_prompt(self, action_num, *args, **kwargs)
        try:
            st = getattr(self, "_rs_state", None)
            if st is None:
                st = {"level": None, "level_start": int(action_num or 0), "resets": 0, "last_reset": None,
                      "note": None, "note_left": 0, "entries": None}; self._rs_state = st; _rs_stats["games"] += 1
            _rs_stats["turns"] += 1
            st["entries"] = kwargs.get("history_entries")
            lv = getattr(kwargs.get("current_frame"), "level", None)
            if lv is not None:
                lv = int(lv)
                if st["level"] is None or lv > st["level"]:
                    if st["level"] is not None:
                        _rs_stats["levels"] += 1
                    st.update({"level": lv, "level_start": int(action_num or 0), "resets": 0, "last_reset": None,
                               "calls": 0, "calls_at_reset": 0})
            st["action_num"] = int(action_num or 0)
            if st.get("note") and st.get("note_left", 0) > 0:
                st["note_left"] -= 1
                return st["note"] + "\n\n" + text
            return text
        except Exception as _e:
            print("[RESET] сбой промпта: %%r" %% (_e,), flush=True)
            return text
    _wta.ToolAgent._build_user_prompt = _rs_prompt

    _rs_orig_run = _wta.ToolAgent._run_python_tool
    def _rs_run(self, state_path, arguments):
        out = _rs_orig_run(self, state_path, arguments)
        try:
            st = getattr(self, "_rs_state", None)
            if st is not None:
                st["calls"] = int(st.get("calls", 0)) + 1   # один запуск кода = один вызов модели (промптов меньше: tool-loop)
            cb = getattr(self, "_step_env_callback", None)
            sess = getattr(cb, "__self__", None) if cb is not None else None
            if st is None or sess is None or st.get("level") is None:
                return out
            last = getattr(self, "_last_action_result", None) or {}
            cur_num = int(last.get("action_num") or st.get("action_num") or 0)
            if last.get("level_completed") or last.get("game_over") or last.get("done"):
                return out
            since_start = cur_num - int(st.get("level_start") or 0)
            since_reset = cur_num - int(st["last_reset"]) if st.get("last_reset") is not None else since_start
            calls_since = int(st.get("calls", 0)) - int(st.get("calls_at_reset", 0))
            by_moves = since_start >= _RS_STALL and since_reset >= _RS_STALL
            by_calls = calls_since >= _RS_CALLS
            if by_moves or by_calls:
                if st["resets"] >= _RS_MAX:
                    _rs_stats["blocked_max"] += 1
                    return out
                n, top, boards, returns = _rs_summary(st.get("entries"), st["level"])
                sess._execute_auto_reset()
                st["resets"] += 1; st["last_reset"] = cur_num; st["calls_at_reset"] = int(st.get("calls", 0)); _rs_stats["resets"] += 1
                _rs_stats["by_calls"] = _rs_stats.get("by_calls", 0) + int(by_calls and not by_moves)
                st["note"] = ("HARNESS RESET (attempt %%d of %%d): this level was RESET by the harness after %%d moves and %%d model calls without "
                              "completing it -- the board is back to the level's start; completed levels are kept and your history "
                              "is kept. What you tried on this level: %%d moves (%%s); %%d distinct boards; returned to an already-seen "
                              "board %%d times. Do NOT repeat the same sequences from the start: choose a different mechanic or "
                              "a different target first." %% (st["resets"], _RS_MAX, since_start, calls_since, n, top or "-", boards, returns))
                st["note_left"] = _RS_NOTE_TURNS
                print("[RESET] уровень %%d: сброс %%d/%%d после %%d ходов / %%d вызовов (досок %%d, возвратов %%d)%%s"
                      %% (st["level"], st["resets"], _RS_MAX, since_start, calls_since, boards, returns, " [по вызовам]" if (by_calls and not by_moves) else ""), flush=True)
        except Exception as _e:
            print("[RESET] сбой: %%r" %% (_e,), flush=True)
        return out
    _wta.ToolAgent._run_python_tool = _rs_run
    print("RESET: сброс уровня обвязкой после %%d ходов ИЛИ %%d вызовов без взятия, не больше %%d на уровень, сводка испробованного во входе %%d хода. "
          "ПОРОГИ против базы 10.25: польза -- победы-поражения >= +8 и медиана >= +8; вред -- <= -6; генерация 1400-1500."
          %% (_RS_STALL, _RS_CALLS, _RS_MAX, _RS_NOTE_TURNS), flush=True)
''' % {"stall": STALL_MOVES, "stall_calls": STALL_CALLS, "max_resets": MAX_RESETS, "note_turns": NOTE_TURNS}

EXPLORE_CELL = r"""
# =====================================================================
# СКРИПТОВЫЙ ИССЛЕДОВАТЕЛЬ ПРИ ЗАСТРЕВАНИИ (13.09, пункт 2): после %(stall)d ходов на уровне без взятия
# обвязка сама, без модели, перебирает классы ходов (каждую стрелку по 3 раза, клик по центру каждого
# объекта, которого ещё не кликали) и кладёт отчёт во вход. Ноль выходных токенов. Не больше
# %(max_resets)d исследований на уровень. Только оффлайн.
# =====================================================================
import inference.agent.tool_agent as _wta
_EX_STALL, _EX_CALLS, _EX_MAX, _EX_NOTE_TURNS, _EX_MAX_CLICKS = %(stall)d, %(stall_calls)d, %(max_resets)d, %(note_turns)d, 16
_ex_stats = {"games": 0, "explorations": 0, "moves": 0, "levels_during": 0, "turns": 0, "blocked_max": 0}

def _ex_objects(grid, max_cells=400):
    # Связные компоненты одного цвета (4-соседство) без фона; центры крупнейших -- цели для клика.
    rows = [list(r) for r in (grid or [])]
    H = len(rows); W = max((len(r) for r in rows), default=0)
    if not H or not W:
        return []
    counts = {}
    for r in rows:
        for v in r:
            counts[v] = counts.get(v, 0) + 1
    bg = max(counts, key=counts.get)
    seen = [[False] * W for _ in range(H)]
    objs = []
    for i in range(H):
        for j in range(len(rows[i])):
            if seen[i][j] or rows[i][j] == bg:
                continue
            col = rows[i][j]; stack = [(i, j)]; seen[i][j] = True; cells = []
            while stack:
                r, c = stack.pop(); cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < len(rows[nr]) and not seen[nr][nc] and rows[nr][nc] == col:
                        seen[nr][nc] = True; stack.append((nr, nc))
            if len(cells) <= max_cells:
                rr = sum(c[0] for c in cells) // len(cells); cc = sum(c[1] for c in cells) // len(cells)
                objs.append({"color": col, "size": len(cells), "row": rr, "col": cc})
    objs.sort(key=lambda o: -o["size"])
    return objs

def _ex_run(agent, sess, state_path, st):
    # Исследование: стрелки x3, клики по объектам. Возвращает текст отчёта.
    valid = [str(v).upper() for v in (getattr(agent, "_current_valid_actions", None) or [])]
    cb = agent._step_env_callback
    report = []; moves = 0; levels = 0; stop = False
    def step(actions):
        nonlocal moves, levels, stop
        res = cb({"actions": actions}) or {}
        moves += int(res.get("executed_count") or 1)
        if res.get("level_completed"):
            levels += 1; stop = True
        if res.get("game_over"):
            try:
                sess._execute_auto_reset()
            except Exception:
                pass
        return res
    for a in ("UP", "DOWN", "LEFT", "RIGHT", "SPACE"):
        if stop or (valid and a not in valid):
            continue
        changed = 0
        for _ in range(3):
            r = step([{"action": a}])
            changed += int(bool(r.get("board_changed")))
            if stop or r.get("game_over"):
                break
        report.append("%%s changed the board %%d/3" %% (a, changed))
    if not stop and (not valid or "MOUSE" in valid or "CLICK" in valid):
        try:
            frame, _h = _wta.load_runtime_state(state_path)
            pl = _wta._ascii_frame_view_payload(frame) or {}
            objs = _ex_objects(pl.get("grid"))
        except Exception as _e:
            objs = []; report.append("segmentation failed: %%r" %% (_e,))
        clicked = st.setdefault("clicked", set())
        n = 0
        for o in objs:
            if stop or n >= _EX_MAX_CLICKS:
                break
            key = (o["row"] // 2, o["col"] // 2)
            if key in clicked:
                continue
            clicked.add(key); n += 1
            r = step([{"action": "MOUSE", "row": o["row"], "col": o["col"]}])
            report.append("click (row %%d, col %%d) on colour %%s (%%d cells): %%s" %% (o["row"], o["col"], o["color"], o["size"],
                          "CHANGED the board" if r.get("board_changed") else "no change"))
    _ex_stats["moves"] += moves; _ex_stats["levels_during"] += levels
    head = ("HARNESS EXPLORATION (attempt %%d of %%d): after %%d moves without completing this level the harness itself spent %%d moves "
            "probing the controls; %%s. Results: " %% (st["explorations"], _EX_MAX, st["since_start"], moves,
            ("a LEVEL WAS COMPLETED during probing -- the board is now the next level" if levels else "no level completed")))
    return head + "; ".join(report) + ". Use these facts: prefer the controls and objects that changed the board and were not part of your earlier sequences."

if not TRUE_SUBMISSION:
    _ex_orig_prompt = _wta.ToolAgent._build_user_prompt
    def _ex_prompt(self, action_num, *args, **kwargs):
        text = _ex_orig_prompt(self, action_num, *args, **kwargs)
        try:
            st = getattr(self, "_ex_state", None)
            if st is None:
                st = {"level": None, "level_start": int(action_num or 0), "explorations": 0, "last": None,
                      "note": None, "note_left": 0, "clicked": set(), "since_start": 0}; self._ex_state = st; _ex_stats["games"] += 1
            _ex_stats["turns"] += 1
            lv = getattr(kwargs.get("current_frame"), "level", None)
            if lv is not None:
                lv = int(lv)
                if st["level"] is None or lv > st["level"]:
                    st.update({"level": lv, "level_start": int(action_num or 0), "explorations": 0, "last": None, "clicked": set(),
                               "calls": 0, "calls_at_last": 0})
            st["action_num"] = int(action_num or 0)
            if st.get("note") and st.get("note_left", 0) > 0:
                st["note_left"] -= 1
                return st["note"] + "\n\n" + text
            return text
        except Exception as _e:
            print("[EXPLORE] сбой промпта: %%r" %% (_e,), flush=True)
            return text
    _wta.ToolAgent._build_user_prompt = _ex_prompt

    _ex_orig_run = _wta.ToolAgent._run_python_tool
    def _ex_run_tool(self, state_path, arguments):
        out = _ex_orig_run(self, state_path, arguments)
        try:
            st = getattr(self, "_ex_state", None)
            if st is not None:
                st["calls"] = int(st.get("calls", 0)) + 1   # один запуск кода = один вызов модели (промптов меньше: tool-loop)
            cb = getattr(self, "_step_env_callback", None)
            sess = getattr(cb, "__self__", None) if cb is not None else None
            if st is None or cb is None or st.get("level") is None:
                return out
            last = getattr(self, "_last_action_result", None) or {}
            if last.get("level_completed") or last.get("game_over") or last.get("done"):
                return out
            cur_num = int(last.get("action_num") or st.get("action_num") or 0)
            since_start = cur_num - int(st.get("level_start") or 0)
            since_last = cur_num - int(st["last"]) if st.get("last") is not None else since_start
            calls_since = int(st.get("calls", 0)) - int(st.get("calls_at_last", 0))
            by_moves = since_start >= _EX_STALL and since_last >= _EX_STALL
            by_calls = calls_since >= _EX_CALLS
            if by_moves or by_calls:
                if st["explorations"] >= _EX_MAX:
                    _ex_stats["blocked_max"] += 1
                    return out
                st["explorations"] += 1; st["last"] = cur_num; st["since_start"] = since_start; st["calls_at_last"] = int(st.get("calls", 0)); _ex_stats["explorations"] += 1
                _ex_stats["by_calls"] = _ex_stats.get("by_calls", 0) + int(by_calls and not by_moves)
                st["note"] = _ex_run(self, sess, state_path, st); st["note_left"] = _EX_NOTE_TURNS
                print("[EXPLORE] уровень %%d: исследование %%d/%%d после %%d ходов / %%d вызовов%%s" %% (st["level"], st["explorations"], _EX_MAX, since_start, calls_since, " [по вызовам]" if (by_calls and not by_moves) else ""), flush=True)
        except Exception as _e:
            print("[EXPLORE] сбой: %%r" %% (_e,), flush=True)
        return out
    _wta.ToolAgent._run_python_tool = _ex_run_tool
    print("EXPLORE: скриптовый исследователь после %%d ходов ИЛИ %%d вызовов без взятия (стрелки x3, клики по объектам, <= %%d раз на уровень), отчёт во входе %%d хода. "
          "ПОРОГИ против базы 10.25: польза -- победы-поражения >= +8 и медиана >= +8; вред -- <= -6; генерация 1400-1500."
          %% (_EX_STALL, _EX_CALLS, _EX_MAX, _EX_NOTE_TURNS), flush=True)
""" % {"stall": STALL_MOVES, "stall_calls": STALL_CALLS, "max_resets": MAX_RESETS, "note_turns": NOTE_TURNS}


def build(cell: str, out: str, slug: str, title: str, marker_var: str) -> None:
    src = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))
    nb = json.loads(json.dumps(src))
    c = nb["cells"][15]
    body = "".join(c["source"])
    anchor = "    seconds=budget - 600.0\n)"
    at = body.find(anchor); run_at = body.find("await bm.run(")
    if at < 0 or run_at < 0:
        raise SystemExit("не нашёл стоковый soft_end или запуск прогона")
    at += len(anchor)
    code = body[:at] + "\n" + cell + body[at:]
    c["source"] = code.splitlines(keepends=True)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "cell15.py"), "w", encoding="utf-8").write(cell)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    meta["id"] = slug; meta["title"] = title
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    compile(code, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    diff = [i for i in range(18) if "".join(nb["cells"][i]["source"]) != "".join(src["cells"][i]["source"])]
    print("ok   %s: изменена только ячейка 15: %s; патч после soft_end и до запуска: %s; слаг %s"
          % (title, diff == [15], 0 < code.find(anchor) < code.find(marker_var) < code.find("await bm.run("), slug))


def main() -> None:
    build(LVFACT_CELL, "kernels/notebooks_stockflash_lvfact", "sergueimakarov/arc3-stock-flash-lvfact", "arc3 stock flash lvfact", "_lf_stats = ")
    build(RESET_CELL, "kernels/notebooks_stockflash_reset", "sergueimakarov/arc3-stock-flash-reset", "arc3 stock flash reset", "_rs_stats = ")
    build(EXPLORE_CELL, "kernels/notebooks_stockflash_explore", "sergueimakarov/arc3-stock-flash-explore", "arc3 stock flash explore", "_ex_stats = ")


if __name__ == "__main__":
    main()

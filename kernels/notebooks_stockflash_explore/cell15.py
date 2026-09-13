
# =====================================================================
# СКРИПТОВЫЙ ИССЛЕДОВАТЕЛЬ ПРИ ЗАСТРЕВАНИИ (13.09, пункт 2): после 100 ходов на уровне без взятия
# обвязка сама, без модели, перебирает классы ходов (каждую стрелку по 3 раза, клик по центру каждого
# объекта, которого ещё не кликали) и кладёт отчёт во вход. Ноль выходных токенов. Не больше
# 2 исследований на уровень. Только оффлайн.
# =====================================================================
import inference.agent.tool_agent as _wta
_EX_STALL, _EX_CALLS, _EX_MAX, _EX_NOTE_TURNS, _EX_MAX_CLICKS = 100, 40, 2, 3, 16
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
        report.append("%s changed the board %d/3" % (a, changed))
    if not stop and (not valid or "MOUSE" in valid or "CLICK" in valid):
        try:
            frame, _h = _wta.load_runtime_state(state_path)
            pl = _wta._ascii_frame_view_payload(frame) or {}
            objs = _ex_objects(pl.get("grid"))
        except Exception as _e:
            objs = []; report.append("segmentation failed: %r" % (_e,))
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
            report.append("click (row %d, col %d) on colour %s (%d cells): %s" % (o["row"], o["col"], o["color"], o["size"],
                          "CHANGED the board" if r.get("board_changed") else "no change"))
    _ex_stats["moves"] += moves; _ex_stats["levels_during"] += levels
    head = ("HARNESS EXPLORATION (attempt %d of %d): after %d moves without completing this level the harness itself spent %d moves "
            "probing the controls; %s. Results: " % (st["explorations"], _EX_MAX, st["since_start"], moves,
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
            st["calls"] = int(st.get("calls", 0)) + 1
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
            print("[EXPLORE] сбой промпта: %r" % (_e,), flush=True)
            return text
    _wta.ToolAgent._build_user_prompt = _ex_prompt

    _ex_orig_run = _wta.ToolAgent._run_python_tool
    def _ex_run_tool(self, state_path, arguments):
        out = _ex_orig_run(self, state_path, arguments)
        try:
            st = getattr(self, "_ex_state", None)
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
                print("[EXPLORE] уровень %d: исследование %d/%d после %d ходов / %d вызовов%s" % (st["level"], st["explorations"], _EX_MAX, since_start, calls_since, " [по вызовам]" if (by_calls and not by_moves) else ""), flush=True)
        except Exception as _e:
            print("[EXPLORE] сбой: %r" % (_e,), flush=True)
        return out
    _wta.ToolAgent._run_python_tool = _ex_run_tool
    print("EXPLORE: скриптовый исследователь после %d ходов ИЛИ %d вызовов без взятия (стрелки x3, клики по объектам, <= %d раз на уровень), отчёт во входе %d хода. "
          "ПОРОГИ против базы 10.25: польза -- победы-поражения >= +8 и медиана >= +8; вред -- <= -6; генерация 1400-1500."
          % (_EX_STALL, _EX_CALLS, _EX_MAX, _EX_NOTE_TURNS), flush=True)

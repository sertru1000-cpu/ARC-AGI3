
# =====================================================================
# ПЕРЕНОС МОДЕЛИ МИРА ЧЕРЕЗ ВЗЯТИЕ УРОВНЯ + ПОЛНЫЙ ПУТЬ УРОВНЯ (17.09). Выключатель CARRY_LAYER=0.
# =====================================================================
import os as _cr_os, json as _cr_json, atexit as _cr_atexit, re as _cr_re
import inference.agent.tool_agent as _cr_wta
_CR_KEEP = ("world_model", "goal_model", "action_model")
_CR_PATH_MAX = 150
_CR_TAG = _cr_re.compile(r"^\[carried from level \d+ -- re-check on this board\] ")
_cr_stats = {"transitions": 0, "carried_nonempty": 0, "carried_fields": 0, "paths": 0, "path_moves": [], "level_moves": [],
             "prompts_with_path": 0, "errors": 0}

def _cr_rle(acts):
    out = []
    for a in acts:
        if out and out[-1][0] == a:
            out[-1][1] += 1
        else:
            out.append([a, 1])
    return ", ".join(a if n == 1 else "%s x%d" % (a, n) for a, n in out)

def _cr_path(history_entries, new_level):
    """(ходов на уровне всего, ходы от последнего RESET до завершающего включительно) для уровня new_level-1."""
    ents = list(history_entries or [])
    idx = None
    for i, e in enumerate(ents):
        try:
            if int(getattr(e.frame, "level", 0) or 0) >= int(new_level):
                idx = i; break
        except Exception:
            continue
    if idx is None or idx == 0:
        return None
    # кадр записи -- ПОСЛЕ хода: первая запись с уровнем new_level-1 -- это завершающий ход предыдущего уровня
    # (или стартовый кадр без хода), поэтому путь начинается со следующей за ней.
    j = idx - 1
    while j - 1 >= 0 and int(getattr(ents[j - 1].frame, "level", 0) or 0) == int(new_level) - 1:
        j -= 1
    start = j + 1
    acts = [str(getattr(e, "action", "") or "").strip() for e in ents[start:idx + 1]]
    acts = [a for a in acts if a]
    total = len(acts)
    last_reset = max([k for k, a in enumerate(acts) if a.upper().startswith("RESET")], default=-1)
    eff = acts[last_reset + 1:]
    return total, eff

def _cr_block(level_done, total, eff):
    shown = eff[-_CR_PATH_MAX:]
    cut = len(eff) - len(shown)
    head = ("LEVEL %d SOLUTION PATH (exact, from the harness log): level %d was completed by this move sequence "
            "(%d moves since the last RESET of that level; %d moves spent on the level in total; the final move completed it):"
            % (level_done, level_done, len(eff), total))
    body = ("(first %d moves omitted) " % cut if cut > 0 else "") + _cr_rle(shown) + "."
    tail = ("Your world/goal/action model from level %d is carried into the world model below, marked [carried]. "
            "Check on the new board whether the same goal and mechanics still hold and whether an analogous sequence applies, "
            "then act; revise the carried model where the new board contradicts it." % level_done)
    return head + "\n" + body + "\n" + tail

def _cr_block_bfs(rec):
    level_done = int(rec.get("level_done") or 0); path = list(rec.get("path") or [])
    shown = path[-_CR_PATH_MAX:]; cut = len(path) - len(shown)
    head = ("LEVEL %d SOLVED BY HARNESS SEARCH (not by your own moves): while you were stuck, the harness searched the real game and "
            "completed level %d with this move sequence from the start of that level (right after RESET; %d moves in the path, "
            "%d search moves in total). You are now on level %d -- ignore any line saying you are still on the same level:"
            % (level_done, level_done, len(path), int(rec.get("moves") or 0), level_done + 1))
    body = ("(first %d moves omitted) " % cut if cut > 0 else "") + _cr_rle(shown) + "."
    tail = ("Study what this sequence did to infer the goal and mechanics of level %d, update the world model below (your model from level %d "
            "is carried, marked [carried]), then check whether the same goal holds on the new board and act." % (level_done, level_done))
    return head + "\n" + body + "\n" + tail

def _cr_carry_fields(self, prev_level, wipe_rest):
    """world/goal/action -- с пометкой уровня; при wipe_rest findings/questions/plan стираются (как в стоке при переходе)."""
    know = getattr(self, "_summarized_knowledge", None)
    if not isinstance(know, dict):
        return 0
    n = 0
    for k in _CR_KEEP:
        v = _CR_TAG.sub("", str(know.get(k, "") or ""))
        if v.strip():
            know[k] = "[carried from level %d -- re-check on this board] %s" % (prev_level, v); n += 1
    if wipe_rest:
        for k in ("recent_findings", "open_questions", "current_plan"):
            know[k] = ""
    return n

if _cr_os.environ.get("CARRY_LAYER", "1") != "0":
    _cr_orig_update = _cr_wta.ToolAgent._update_summarized_knowledge_from_step_summary
    def _cr_update(self):
        s = getattr(self, "_last_step_summary", None) or {}
        if not (s.get("level_transition") and not s.get("run_complete") and not s.get("game_over")):
            return _cr_orig_update(self)
        try:
            know = getattr(self, "_summarized_knowledge", None) or {}
            saved = {k: _CR_TAG.sub("", str(know.get(k, "") or "")) for k in _CR_KEEP}
        except Exception as _e:
            _cr_stats["errors"] += 1; print("[CARRY] сбой чтения модели мира: %r" % (_e,), flush=True)
            return _cr_orig_update(self)
        out = _cr_orig_update(self)
        try:
            try:
                new_level = int(s.get("level"))
            except (TypeError, ValueError):
                new_level = None
            prev = (new_level - 1) if new_level else 0
            n = 0
            for k, v in saved.items():
                if v.strip():
                    self._summarized_knowledge[k] = "[carried from level %d -- re-check on this board] %s" % (prev, v); n += 1
            _cr_stats["transitions"] += 1; _cr_stats["carried_fields"] += n
            if n:
                _cr_stats["carried_nonempty"] += 1
            print("[[CARRY]] переход на уровень %s: перенесено полей %d (%s)" % (new_level, n, ", ".join(k for k, v in saved.items() if v.strip()) or "-"), flush=True)
            _cr_dump()
        except Exception as _e:
            _cr_stats["errors"] += 1; print("[CARRY] сбой переноса: %r" % (_e,), flush=True)
        return out
    _cr_wta.ToolAgent._update_summarized_knowledge_from_step_summary = _cr_update

    _cr_orig_prompt = _cr_wta.ToolAgent._build_user_prompt
    def _cr_prompt(self, action_num, *args, **kwargs):
        st = getattr(self, "_cr_state", None)
        if st is None:
            st = {"level": None, "block": None}; self._cr_state = st
        try:
            # уровень, взятый перебором (слой BFS, если стоит): разбираем ДО сборки стокового промпта, чтобы перенесённая
            # модель мира и стёртый план попали в этот же промпт
            cb = getattr(self, "_step_env_callback", None); sess = getattr(cb, "__self__", None)
            bf = list(getattr(sess, "_bf_found", []) or []) if sess is not None else []
            if len(bf) > int(st.get("bf_seen", 0)):
                rec = bf[-1]; st["bf_seen"] = len(bf)
                st["block"] = _cr_block_bfs(rec)
                n = _cr_carry_fields(self, int(rec.get("level_done") or 0), True)
                st["level"] = max(int(st["level"] or 0), int(rec.get("level_done") or 0) + 1)
                _cr_stats["bfs_paths"] = int(_cr_stats.get("bfs_paths", 0)) + 1; _cr_stats["transitions"] += 1; _cr_stats["carried_fields"] += n
                if n:
                    _cr_stats["carried_nonempty"] += 1
                print("[[CARRY]] путь перебора уровня %d во входе: %d ходов, перенесено полей %d" % (int(rec.get("level_done") or 0), len(rec.get("path") or []), n), flush=True)
                _cr_dump()
        except Exception as _e:
            _cr_stats["errors"] += 1; print("[CARRY] сбой записи перебора: %r" % (_e,), flush=True)
        text = _cr_orig_prompt(self, action_num, *args, **kwargs)
        try:
            lv = getattr(kwargs.get("current_frame"), "level", None)
            if lv is not None:
                lv = int(lv)
                if st["level"] is not None and lv > st["level"]:
                    st["block"] = None
                    got = _cr_path(kwargs.get("history_entries"), lv)
                    if got is not None:
                        total, eff = got
                        st["block"] = _cr_block(lv - 1, total, eff)
                        _cr_stats["paths"] += 1; _cr_stats["path_moves"].append(len(eff)); _cr_stats["level_moves"].append(total)
                        print("[[CARRY]] путь уровня %d: %d ходов от последнего RESET, всего %d" % (lv - 1, len(eff), total), flush=True)
                        _cr_dump()
                st["level"] = lv if st["level"] is None else max(st["level"], lv)
            if st.get("block"):
                _cr_stats["prompts_with_path"] += 1
                return st["block"] + "\n\n" + text
            return text
        except Exception as _e:
            _cr_stats["errors"] += 1; print("[CARRY] сбой промпта: %r" % (_e,), flush=True)
            return text
    _cr_wta.ToolAgent._build_user_prompt = _cr_prompt

    def _cr_dump():
        try:
            _cr_os.makedirs("/kaggle/working", exist_ok=True)
            _cr_json.dump(_cr_stats, open("/kaggle/working/carry_stats.json", "w"), indent=1)
        except Exception:
            pass
    _cr_atexit.register(_cr_dump)
    print("[[CARRY]] слой установлен: модель мира (world/goal/action) переносится через взятие уровня, путь уровня во входе "
          "(до %d ходов); TRUE_SUBMISSION=%s" % (_CR_PATH_MAX, TRUE_SUBMISSION), flush=True)

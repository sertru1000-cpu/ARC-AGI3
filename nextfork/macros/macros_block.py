        # ---- NEXTFORK macros (02.10): a series of moves in one call, with early stop ----
        # Every step is a separate action([...]) call, so all of the solver's
        # guards still apply per move. A macro stops at the first step that
        # completes a level, ends the attempt, changes nothing, or is refused by
        # a guard, and prints a one-line report of what it executed.
        _MACRO_ALIAS = {"U": "UP", "D": "DOWN", "L": "LEFT", "R": "RIGHT",
                        "S": "SPACE", "Z": "UNDO"}
        _MACRO_GUARDS = (KnownNoOpActionError, KnownDeathActionError,
                         StaleStateActionError, RepeatedActionInStateError,
                         TerminalStateActionError)

        def _macro_entry(token):
            tok = str(token).strip()
            up = tok.upper()
            if up.startswith("C") and "," in up and up[1:].replace(",", "").replace(" ", "").isdigit():
                r, c = up[1:].split(",", 1)
                return {"action": "MOUSE", "row": int(r), "col": int(c)}, 1, "C%s,%s" % (int(r), int(c))
            count = 1
            if up.endswith("*"):
                count, up = None, up[:-1]
            else:
                digits = ""
                while up and up[-1].isdigit():
                    digits, up = up[-1] + digits, up[:-1]
                if digits:
                    count = int(digits)
            name = _MACRO_ALIAS.get(up, up)
            if not name:
                raise ValueError("empty macro token %r" % (token,))
            return name, count, up + ("*" if count is None else (str(count) if count != 1 else ""))

        def _macro_tokens(plan):
            if isinstance(plan, str):
                return [t for t in plan.replace(";", " ").split() if t]
            if isinstance(plan, dict):
                return [plan]
            return list(plan)

        def _macro_step(entry):
            try:
                res = action([entry])
            except _MACRO_GUARDS as exc:
                return "guard:" + type(exc).__name__.replace("ActionError", ""), None
            for flag in ("level_completed", "run_complete", "done", "game_over"):
                if res.get(flag):
                    return flag, res
            if res.get("gameplay_changed") is False:
                return "no_change", res
            return "", res

        def _macro_label(entry):
            if isinstance(entry, dict):
                if str(entry.get("action", "")).upper() == "MOUSE":
                    return "C%s,%s" % (entry.get("row"), entry.get("col"))
                return str(entry.get("action"))
            return str(entry)

        def run(plan, max_steps=30, star_max=12, stop_on_noop=True):
            # run("U3 R2 S D* C12,30 Z"): letters U D L R S Z, a count, '*' = repeat
            # until the board stops changing (at most star_max); C<row>,<col> clicks. Dicts and full
            # action names are accepted too.
            done_steps, stop, trail = 0, "", []
            for token in _macro_tokens(plan):
                if isinstance(token, dict):
                    entry, count, label = token, 1, _macro_label(token)
                else:
                    entry, count, label = _macro_entry(token)
                n = 0
                while (n < star_max) if count is None else (n < count):
                    if done_steps >= max_steps:
                        stop = "max_steps"
                        break
                    reason, _ = _macro_step(entry)
                    if reason.startswith("guard:"):
                        stop = reason + " at " + _macro_label(entry)
                        break
                    done_steps += 1
                    n += 1
                    if reason == "no_change" and (stop_on_noop or count is None):
                        if count is None:
                            n -= 1  # the probing move that hit the wall
                            break
                        stop = "no_change at " + _macro_label(entry)
                        break
                    if reason and reason != "no_change":
                        stop = reason + " at " + _macro_label(entry)
                        break
                trail.append("%s:%d" % (label, n))
                if count is None and n >= star_max and not stop:
                    stop = "star_max at " + label
                if stop:
                    break
            merged = []
            for item in trail:
                lab, cnt = item.rsplit(":", 1)
                if merged and merged[-1][0] == lab:
                    merged[-1][1] += int(cnt)
                else:
                    merged.append([lab, int(cnt)])
            report = "run %s -> %d moves [%s]%s" % (
                plan if isinstance(plan, str) else "list", done_steps,
                " ".join("%s:%d" % (lab, cnt) for lab, cnt in merged),
                (" STOP " + stop) if stop else " ok")
            print(report)

        def until(step, cond, max_steps=20):
            # until("R", lambda f: <test on frame f>): repeat one move until cond(current_frame).
            entry = _macro_entry(step)[0] if not isinstance(step, dict) else step
            n, stop = 0, ""
            if cond(runtime_globals["current_frame"]):
                stop = "condition already true"
            while not stop and n < max_steps:
                reason, _ = _macro_step(entry)
                if reason.startswith("guard:"):
                    stop = reason
                    break
                n += 1
                if reason:
                    stop = reason
                    break
                if cond(runtime_globals["current_frame"]):
                    stop = "condition true"
            report = "until %s -> %d moves, %s" % (_macro_label(entry), n, stop or "max_steps")
            print(report)

        def _macro_targets(color=None, min_pixels=1, max_pixels=None):
            frame = runtime_globals["current_frame"]
            grid = frame._grid
            out = []
            for node in frame.segmentation.get("nodes", []):
                if color is not None and node.get("color") != color:
                    continue
                if node.get("pixels", 0) < min_pixels or (max_pixels and node.get("pixels", 0) > max_pixels):
                    continue
                rows = [p[0] for p in node.get("boundary", [])]
                cols = [p[1] for p in node.get("boundary", [])]
                if not rows:
                    continue
                r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
                want = COLOR_CHARS.index(node["color"]) if node["color"] in COLOR_CHARS else None
                cell = None
                cr, cc = (r0 + r1) // 2, (c0 + c1) // 2
                if want is None or grid[cr][cc] == want:
                    cell = (cr, cc)
                else:
                    for rr in range(r0, r1 + 1):
                        for c2 in range(c0, c1 + 1):
                            if grid[rr][c2] == want:
                                cell = (rr, c2)
                                break
                        if cell:
                            break
                if cell:
                    out.append(cell)
            return out

        def click_each(color, min_pixels=1, max_pixels=None, max_steps=30):
            # click_each("r"): click the centre of every object of that color, in reading order.
            targets = _macro_targets(color, min_pixels, max_pixels)
            plan = [{"action": "MOUSE", "row": r, "col": c} for r, c in targets[:max_steps]]
            if not plan:
                report = "click_each %s -> no objects of that color" % (color,)
                print(report)
                return None
            n, stop = 0, ""
            for entry in plan:
                reason, _ = _macro_step(entry)
                if reason.startswith("guard:"):
                    stop = reason + " at " + _macro_label(entry)
                    break
                n += 1
                if reason and reason != "no_change":
                    stop = reason + " at " + _macro_label(entry)
                    break
            report = "click_each %s -> %d/%d clicks%s" % (color, n, len(plan), (" STOP " + stop) if stop else " ok")
            print(report)

        runtime_globals["run"] = run
        runtime_globals["until"] = until
        runtime_globals["click_each"] = click_each
        # ---- end NEXTFORK macros ----

        # ---- NEXTFORK events (04.10): compact event report printed right after every action() ----
        # Half of all model calls execute no move: they only look at the board. With NEXTFORK_EVENTS=1 the
        # sandbox prints what the call did at once (cells changed, objects moved/appeared/gone, level,
        # whether this board was seen before), so the model does not need a separate look-only call.
        import os as _nf_os
        _nf_seen_boards = runtime_globals.setdefault("_nf_seen_boards", set())

        def _nf_board_key(frame):
            # без полосы у края (счётчик ходов, таймер): иначе любая доска «новая»
            try:
                g = frame._grid
                return hash(tuple(tuple(r[4:-4]) for r in g[4:-4]))
            except Exception:
                return None

        def _nf_obj(o):
            c = o.get("color", "?")
            if "from" in o and "to" in o:
                return f"{c}{o['from']}->{o['to']}"
            if "bbox" in o:
                return f"{c}@{o['bbox'][:2]}"
            return str(c)

        def _nf_events(before, after, result):
            parts = []
            try:
                d = compute_frame_diff(before._grid, after._grid, before.segmentation.get("nodes", []),
                                       after.segmentation.get("nodes", [])) if before.shape == after.shape else None
            except Exception:
                d = None
            if d is None:
                parts.append("board redrawn")
            else:
                n = d.get("changed_cell_count", 0)
                parts.append(f"{n} cells changed" if n else "no cell changed")
                for key, label in (("moved", "moved"), ("appeared", "new"), ("disappeared", "gone"),
                                   ("changed_color", "recolored"), ("rotated", "rotated"), ("resized", "resized")):
                    items = d.get(key) or []
                    if items:
                        shown = ", ".join(_nf_obj(o) for o in items[:4])
                        more = f" +{len(items) - 4}" if len(items) > 4 else ""
                        parts.append(f"{label}: {shown}{more}")
            lb, la = getattr(before, "level", None), getattr(after, "level", None)
            if lb is not None and la is not None and la != lb:
                parts.append(f"LEVEL {lb}->{la}")
            if result.get("game_over"):
                parts.append("GAME OVER (level restarted)")
            k = _nf_board_key(after)
            if k is not None:
                parts.append("board seen before" if k in _nf_seen_boards else "new board")
                _nf_seen_boards.add(k)
            return "[events] " + " | ".join(parts)

        if _nf_os.environ.get("NEXTFORK_EVENTS", "0").strip().lower() in ("1", "true", "on"):
            _nf_inner_action = action

            def action(actions):
                if not _nf_seen_boards:   # песочница пересоздаётся на каждый вызов: засеять из истории игры
                    for _h in runtime_globals.get("history") or []:
                        _k = _nf_board_key(getattr(_h, "frame", None)) if getattr(_h, "frame", None) is not None else None
                        if _k is not None:
                            _nf_seen_boards.add(_k)
                _before = runtime_globals.get("current_frame")
                if _before is not None:
                    k0 = _nf_board_key(_before)
                    if k0 is not None:
                        _nf_seen_boards.add(k0)
                result = _nf_inner_action(actions)
                _after = runtime_globals.get("current_frame")
                if _before is not None and _after is not None:
                    try:
                        print(_nf_events(_before, _after, result if isinstance(result, dict) else {}))
                    except Exception as _e:
                        print(f"[events] unavailable: {_e}")
                return result

            runtime_globals["action"] = action

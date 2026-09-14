
_MC_MAX_MOVES, _MC_MAX_CLICKS = 40, 16
def _mc_stop(res):
    return (not res.get("executed")) or bool(res.get("level_completed") or res.get("done") or res.get("game_over") or res.get("run_complete"))
def _mc_report(out):
    print("[[MACRO]] " + repr({"macro": out.get("macro"), "moves": out.get("moves"), "level_completed": out.get("level_completed")}))
    return out
def _mc_center(node):
    pts = node.get("boundary") or []
    rs = [int(p[0]) for p in pts]; cs = [int(p[1]) for p in pts]
    if not rs or not cs:
        return None
    return (min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2
def move_until_stuck(move, max_moves=40):
    """Repeat ONE move (e.g. "RIGHT") while the board keeps changing. Stops at the first move that changes nothing
    (wall/edge), when a level completes, or after max_moves. Returns a summary dict."""
    move = str(move).strip().upper()
    if move == "MOUSE":
        raise ValueError("move_until_stuck takes a non-MOUSE move name")
    n = 0; changed = 0; last = {}; why = "max_moves"
    for _ in range(max(1, min(int(max_moves), _MC_MAX_MOVES))):
        last = action(move); n += 1
        if last.get("board_changed"):
            changed += 1
        if _mc_stop(last):
            why = "level_completed" if last.get("level_completed") else "game_end"; break
        if not last.get("board_changed"):
            why = "no_change"; break
    return _mc_report({"macro": "move_until_stuck", "move": move, "moves": n, "changed_moves": changed, "stopped_by": why,
                       "level_completed": bool(last.get("level_completed")), "level": last.get("level")})
def click_objects(color=None, min_pixels=None, max_pixels=None, ids=None, stop_on_change=False, max_clicks=16):
    """Click the center of every object of current_frame.segmentation that matches the filters (color letter,
    pixel-count range, or explicit node ids). Skips the background (largest object). Reports which clicks changed the
    board; stops when a level completes, or at the first change if stop_on_change=True. Returns a list of dicts."""
    nodes = list((current_frame.segmentation or {}).get("nodes") or [])
    if nodes:
        bg = max(nodes, key=lambda n: int(n.get("pixels") or 0))
        nodes = [n for n in nodes if n is not bg]
    if ids is not None:
        want = set(int(i) for i in ids); nodes = [n for n in nodes if int(n.get("id", -1)) in want]
    if color is not None:
        nodes = [n for n in nodes if str(n.get("color")) == str(color)]
    if min_pixels is not None:
        nodes = [n for n in nodes if int(n.get("pixels") or 0) >= int(min_pixels)]
    if max_pixels is not None:
        nodes = [n for n in nodes if int(n.get("pixels") or 0) <= int(max_pixels)]
    out = []; last = {}; lvl = False
    for n in nodes[:max(1, min(int(max_clicks), _MC_MAX_CLICKS))]:
        c = _mc_center(n)
        if c is None:
            continue
        last = action({"action": "MOUSE", "row": c[0], "col": c[1]})
        out.append({"id": n.get("id"), "color": n.get("color"), "pixels": n.get("pixels"), "row": c[0], "col": c[1],
                    "changed": bool(last.get("board_changed")), "level_completed": bool(last.get("level_completed"))})
        lvl = lvl or bool(last.get("level_completed"))
        if _mc_stop(last) or (stop_on_change and last.get("board_changed")):
            break
    _mc_report({"macro": "click_objects", "moves": len(out), "level_completed": lvl})
    return out
def sweep_moves():
    """Try each valid non-MOUSE move ONCE and report which of them changed the board. Returns {move: changed}."""
    res = {}; lvl = False
    for name in [str(a) for a in (valid_actions or [])]:
        if name.upper() in ("MOUSE", "RESET"):
            continue
        last = action(name)
        res[name] = bool(last.get("board_changed"))
        lvl = lvl or bool(last.get("level_completed"))
        if _mc_stop(last):
            break
    _mc_report({"macro": "sweep_moves", "moves": len(res), "level_completed": lvl})
    return res

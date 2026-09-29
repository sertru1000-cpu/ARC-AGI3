# --- Twin-помощники песочницы (26.09): исполняемая модель мира с реплей-проверкой и планом.
# По статьям Twin (arXiv:2608.14490) и Tycho (arXiv:2607.28287): модель пишет step(grid, action) -> grid,
# ход по плану делается только когда step воспроизводит ВСЕ наблюдённые переходы уровня; расхождения
# возвращаются модели контрпримерами; поиск пути идёт внутри симулятора, бесплатно по ходам.
# Только безопасные встроенные; ни одного импорта.

def parse_action(a):
    """'UP' -> ('UP', None, None); 'MOUSE(row=3, col=5)' -> ('MOUSE', 3, 5)."""
    s = str(a).strip()
    if s.startswith("MOUSE"):
        nums = []
        cur = ""
        for ch in s:
            if ch.isdigit():
                cur += ch
            elif cur:
                nums.append(int(cur)); cur = ""
        if cur:
            nums.append(int(cur))
        return ("MOUSE", nums[0] if nums else None, nums[1] if len(nums) > 1 else None)
    return (s, None, None)


def _tw_grid(fr):
    g = getattr(fr, "_grid", None)
    return [list(r) for r in (g or [])]


def level_transitions(level=None):
    """Наблюдённые переходы текущего уровня: список (before_grid, action_str, after_grid)."""
    lv = current_frame.level if level is None else level
    out = []
    for t in transitions:
        b, a = t.before_frame, t.after_frame
        if b is None or a is None or b.level != lv or a.level != lv:
            continue
        out.append((_tw_grid(b), str(t.action), _tw_grid(a)))
    return out


def validate(step, level=None, show=3):
    """Реплей: прогнать step по всем переходам уровня. Печатает итог и первые контрпримеры
    (ход, сколько клеток не сошлось, первые клетки r,c: предсказано -> наблюдено).
    Возвращает число расхождений (0 = симулятор воспроизводит всё наблюдённое)."""
    ts = level_transitions(level)
    bad = 0
    shown = 0
    for i, (b, a, after) in enumerate(ts):
        try:
            pred = step([row[:] for row in b], a)
        except Exception as e:
            bad += 1
            if shown < show:
                print("  counterexample #%d %s: step raised %r" % (i, a, e)); shown += 1
            continue
        if pred != after:
            bad += 1
            if shown < show:
                cells = []
                for r in range(min(len(pred or []), len(after))):
                    for c in range(min(len(pred[r]), len(after[r]))):
                        if pred[r][c] != after[r][c]:
                            cells.append((r, c, pred[r][c], after[r][c]))
                print("  counterexample #%d %s: %d cells differ, e.g. %s (r,c,predicted,observed)"
                      % (i, a, len(cells), cells[:6])); shown += 1
    print("validate: %d/%d transitions reproduced" % (len(ts) - bad, len(ts)))
    return bad


def plan(step, goal, actions=None, max_depth=60, max_nodes=60000):
    """Поиск в ширину ВНУТРИ симулятора от текущей доски до goal(grid)==True.
    actions — список строк ходов; по умолчанию клавиатурные из valid_actions (клики передавайте явно,
    например ['MOUSE(row=10, col=20)', ...]). Возвращает список ходов или None."""
    if actions is None:
        actions = [a for a in valid_actions if not str(a).startswith("MOUSE")]
    start = _tw_grid(current_frame)
    key = lambda g: tuple(tuple(r) for r in g)
    seen = {key(start)}
    frontier = [(start, [])]
    nodes = 0
    for depth in range(max_depth):
        nxt = []
        for g, path in frontier:
            for a in actions:
                nodes += 1
                if nodes > max_nodes:
                    print("plan: node budget %d exhausted at depth %d" % (max_nodes, depth)); return None
                try:
                    g2 = step([row[:] for row in g], a)
                except Exception:
                    continue
                k = key(g2)
                if k in seen:
                    continue
                seen.add(k)
                p2 = path + [a]
                try:
                    if goal(g2):
                        print("plan: found %d moves, %d nodes" % (len(p2), nodes)); return p2
                except Exception:
                    pass
                nxt.append((g2, p2))
        frontier = nxt
        if not frontier:
            break
    print("plan: no path within depth %d (%d nodes)" % (max_depth, nodes))
    return None


def to_actions(path):
    """Путь из plan() -> аргумент для action([...])."""
    out = []
    for a in path or []:
        n, r, c = parse_action(a)
        out.append({"action": "MOUSE", "row": r, "col": c} if n == "MOUSE" else {"action": n})
    return out
# --- конец Twin-помощников ---

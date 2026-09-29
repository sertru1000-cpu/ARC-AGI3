# --- a8c: планировщик внутри симулятора модели (28.09) ---
def _pl_to_action(a):
    """'MOUSE(row=R, col=C)' -> {'action': 'MOUSE', 'row': R, 'col': C} для action(); клавиши — как есть."""
    s = str(a)
    if s.upper().startswith("MOUSE"):
        nums = [int(t) for t in "".join(ch if ch.isdigit() else " " for ch in s).split()]
        if len(nums) >= 2:
            return {"action": "MOUSE", "row": nums[0], "col": nums[1]}
    return s


def plan(step, goal, actions=None, clicks=None, max_depth=300, max_states=20000, seconds=12.0):
    """Кратчайшая последовательность ходов ВНУТРИ симулятора: поиск в ширину от текущей доски.
    step(rows, action) -> rows — ваш симулятор; goal(rows) -> True, когда уровень пройден (или достигнута подцель).
    actions — список ходов-строк (по умолчанию все клавиши из valid_actions); clicks — список (row, col) для кликов.
    Возвращает список, готовый для action(...), или None; печатает, сколько состояний просмотрено.
    Настоящих ходов не тратит. План верен ровно настолько, насколько верен step: сначала check_step(step)."""
    import time
    t0 = time.time()
    start = tuple(str(r) for r in current_frame.ascii.split("\n"))
    acts = list(actions) if actions else [a for a in valid_actions if not str(a).upper().startswith("MOUSE")]
    acts += ["MOUSE(row=%d, col=%d)" % (int(r), int(c)) for r, c in (clicks or [])]
    if not acts:
        print("plan: нет ходов для перебора (для кликов передайте clicks=[(row, col), ...])")
        return None
    try:
        if goal(list(start)):
            print("plan: цель уже достигнута на текущей доске")
            return []
    except Exception as e:
        print("plan: goal упал на текущей доске: %r" % (e,))
        return None
    parent = {start: None}
    frontier = [start]
    depth = 0; fails = 0
    while frontier and depth < max_depth:
        depth += 1
        nxt = []
        for s in frontier:
            for a in acts:
                try:
                    n = tuple(str(r) for r in step(list(s), a))
                except Exception:
                    fails += 1
                    continue
                if n in parent:
                    continue
                parent[n] = (s, a)
                try:
                    hit = goal(list(n))
                except Exception:
                    hit = False
                if hit:
                    path = []
                    cur = n
                    while parent[cur] is not None:
                        cur, act = parent[cur][0], parent[cur][1]
                        path.append(act)
                    path.reverse()
                    print("plan: найдено за %d ходов, просмотрено состояний %d, %.1f с" % (len(path), len(parent), time.time() - t0))
                    return [_pl_to_action(x) for x in path]
                nxt.append(n)
                if len(parent) >= max_states or time.time() - t0 > seconds:
                    print("plan: предел (%d состояний, %.1f с, глубина %d), цели нет; падений step: %d" % (len(parent), time.time() - t0, depth, fails))
                    return None
        frontier = nxt
    print("plan: цель недостижима в симуляторе (состояний %d, глубина %d, падений step %d)" % (len(parent), depth, fails))
    return None
# --- конец планировщика a8c ---

# --- a8b: готовая заготовка симулятора base_step (27.09) ---
def _bs_cells(b, a):
    """Клетки, где доска изменилась: список (r, c, было, стало)."""
    out = []
    for r in range(min(len(b), len(a))):
        for c in range(min(len(b[r]), len(a[r]))):
            if b[r][c] != a[r][c]:
                out.append((r, c, b[r][c], a[r][c]))
    return out


def _bs_shift(b, a):
    """Если ход сдвинул один цвет целиком на (dr, dc): вернуть (цвет, dr, dc, чем заполнить освободившееся), иначе None."""
    ch = _bs_cells(b, a)
    if not ch:
        return None
    for col in sorted({x[2] for x in ch} | {x[3] for x in ch}):
        src = [(r, c) for r in range(len(b)) for c in range(len(b[r])) if b[r][c] == col]
        dst = set((r, c) for r in range(len(a)) for c in range(len(a[r])) if a[r][c] == col)
        if not src or len(src) != len(dst) or set(src) == dst:
            continue
        r0, c0 = min(src); r1, c1 = min(dst); dr, dc = r1 - r0, c1 - c0
        if set((r + dr, c + dc) for r, c in src) == dst:
            gone = [a[r][c] for r, c in src if (r, c) not in dst]
            return (col, dr, dc, max(set(gone), key=gone.count) if gone else None)
    return None


def base_step(rows, action):
    """Заготовка: (1) переход уже видели на этом уровне -> вернуть увиденное; (2) иначе повторить то, что этот же
    ход сделал в последний раз: сдвиг объекта одного цвета и/или те же изменения клеток, если там стояло то же."""
    rows = [str(x) for x in rows]
    ts = level_transitions()
    for b, a, af in ts:
        if a == action and b == rows:
            return list(af)
    last = [t for t in ts if t[1] == action]
    if not last:
        return rows
    b, a, af = last[-1]
    g = [list(x) for x in rows]
    sh = _bs_shift(b, af)
    if sh:
        col, dr, dc, fill = sh
        src = [(r, c) for r in range(len(g)) for c in range(len(g[r])) if g[r][c] == col]
        ok = all(0 <= r + dr < len(g) and 0 <= c + dc < len(g[r + dr]) for r, c in src)
        if src and ok and fill is not None:
            for r, c in src:
                g[r][c] = fill
            for r, c in src:
                g[r + dr][c + dc] = col
            return ["".join(x) for x in g]
    for r, c, was, now in _bs_cells(b, af):
        if r < len(g) and c < len(g[r]) and g[r][c] == was:
            g[r][c] = now
    return ["".join(x) for x in g]
# --- конец заготовки a8b ---

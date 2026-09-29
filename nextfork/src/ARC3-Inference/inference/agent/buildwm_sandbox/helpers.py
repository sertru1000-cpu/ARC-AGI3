# --- a8: помощники песочницы для достройки симулятора (27.09) ---
def level_transitions():
    """Переходы ТЕКУЩЕГО уровня: список (rows_before, action, rows_after); rows — список строк букв доски."""
    lv = current_frame.level
    out = []
    for t in transitions:
        b, a = t.before_frame, t.after_frame
        if b is None or a is None or b.level != lv or a.level != lv:
            continue
        out.append((b.ascii.split("\n"), str(t.action), a.ascii.split("\n")))
    return out


def check_step(step, show=3):
    """Прогнать step(rows, action) -> rows по ВСЕМ переходам текущего уровня; печатает контрпримеры
    (номер, ход, клеток мимо, первые (строка, столбец, предсказано, увидено)); возвращает число расхождений."""
    ts = level_transitions()
    bad = 0
    shown = 0
    for i, (b, a, af) in enumerate(ts):
        try:
            p = step([r for r in b], a)
        except Exception as e:
            bad += 1
            if shown < show:
                print("  #%d %s: step raised %r" % (i, a, e)); shown += 1
            continue
        if p != af:
            bad += 1
            if shown < show:
                cells = []
                for r in range(len(af)):
                    for c in range(len(af[r])):
                        pv = p[r][c] if (r < len(p) and c < len(p[r])) else None
                        if pv != af[r][c]:
                            cells.append((r, c, pv, af[r][c]))
                print("  #%d %s: %d cells wrong, e.g. %s (row, col, predicted, observed)" % (i, a, len(cells), cells[:6])); shown += 1
    print("check_step: %d/%d transitions of this level reproduced" % (len(ts) - bad, len(ts)))
    return bad
# --- конец помощников a8 ---

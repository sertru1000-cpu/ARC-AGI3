# --- a8c: части Tycho (28.09): сжатый дифф, разбор ошибки check_step, проверка цели, план со сверкой кадра ---
def _ty_runs(cols):
    cols = sorted(cols); out = []; i = 0
    while i < len(cols):
        j = i
        while j + 1 < len(cols) and cols[j + 1] == cols[j] + 1:
            j += 1
        out.append(str(cols[i]) if i == j else "%d-%d" % (cols[i], cols[j]))
        i = j + 1
    return ",".join(out)


def diff_rows(a, b, mask=None):
    """Сжатый дифф двух досок (списки строк): группы «старый->новый цвет», соседние клетки — диапазонами,
    соседние строки с одинаковыми столбцами — одной строкой (как workspace.diff_text в Tycho)."""
    d = []
    for r in range(min(len(a), len(b))):
        for c in range(min(len(a[r]), len(b[r]))):
            if a[r][c] != b[r][c] and not (mask and (r, c) in mask):
                d.append((r, c, a[r][c], b[r][c]))
    if not d:
        return "no cells changed"
    by = {}
    for r, c, o, n in d:
        by.setdefault((o, n), {}).setdefault(r, []).append(c)
    lines = []
    for (o, n) in sorted(by):
        rows = by[(o, n)]
        spec = [(r, _ty_runs(rows[r])) for r in sorted(rows)]
        i = 0
        while i < len(spec):
            r0, s = spec[i]; j = i
            while j + 1 < len(spec) and spec[j + 1][1] == s and spec[j + 1][0] == spec[j][0] + 1:
                j += 1
            lines.append("  %s->%s %s, cols %s" % (o, n, ("rows %d-%d" % (r0, spec[j][0])) if j > i else ("row %d" % r0), s))
            i = j + 1
    rs = [x[0] for x in d]; cs = [x[1] for x in d]
    return "%d cells changed; region rows %d-%d, cols %d-%d\n%s" % (len(d), min(rs), max(rs), min(cs), max(cs), "\n".join(lines[:12]))


def hud_mask():
    """Клетки полосы индикатора: строки/столбцы у края (3 клетки), менявшиеся в >= половине переходов уровня."""
    ts = level_transitions()
    if len(ts) < 4:
        return set()
    H = len(ts[0][0]); W = len(ts[0][0][0]) if H else 0
    cnt = {}
    for b, a, af in ts:
        rows = set(); cols = set()
        for r in range(min(len(b), len(af))):
            for c in range(min(len(b[r]), len(af[r]))):
                if b[r][c] != af[r][c]:
                    if r <= 2 or r >= H - 3:
                        rows.add(r)
                    if c <= 2 or c >= W - 3:
                        cols.add(c)
        for r in rows:
            cnt[("r", r)] = cnt.get(("r", r), 0) + 1
        for c in cols:
            cnt[("c", c)] = cnt.get(("c", c), 0) + 1
    m = set()
    for (k, i), v in cnt.items():
        if v >= 0.5 * len(ts):
            if k == "r":
                m |= {(i, c) for c in range(W)}
            else:
                m |= {(r, i) for r in range(H)}
    return m


def _ty_same(p, q, mask):
    if len(p) != len(q):
        return False
    for r in range(len(q)):
        if len(p[r]) != len(q[r]):
            return False
        if p[r] != q[r]:
            for c in range(len(q[r])):
                if p[r][c] != q[r][c] and (r, c) not in mask:
                    return False
    return True


def check_step(step, show=2):
    """step(rows, action) -> rows против ВСЕХ переходов текущего уровня (полоса индикатора не считается).
    На первых расхождениях печатает рядом: что изменил ваш step и что изменилось на самом деле."""
    ts = level_transitions(); mask = hud_mask(); bad = 0; shown = 0
    for i, (b, a, af) in enumerate(ts):
        try:
            p = [str(x) for x in step([r for r in b], a)]
        except Exception as e:
            bad += 1
            if shown < show:
                print("  #%d %s: step raised %r" % (i, a, e)); shown += 1
            continue
        if not _ty_same(p, af, mask):
            bad += 1
            if shown < show:
                print("  #%d %s mismatch\n  your step changed: %s\n  actually changed:  %s" % (
                    i, a, diff_rows(b, p, mask).replace("\n", "\n    "), diff_rows(b, af, mask).replace("\n", "\n    ")))
                shown += 1
    print("check_step: %d/%d transitions of this level reproduced%s" % (len(ts) - bad, len(ts), " (HUD band ignored)" if mask else ""))
    return bad


def check_goal(goal, step=None):
    """goal(rows) -> True только на выигранной доске. Проверка по истории: на всех уже увиденных досках текущего
    уровня goal обязана быть ложной (уровень там не завершился). Если передан step — ещё и на прошлых уровнях:
    step(доска перед победой, победный ход) должен давать доску, где goal истинна."""
    ts = level_transitions(); bad = []
    seen = [ts[0][0]] + [t[2] for t in ts] if ts else [current_frame.ascii.split("\n")]
    for k, rows in enumerate(seen):
        try:
            if goal(list(rows)):
                bad.append(k)
        except Exception as e:
            print("check_goal: goal raised %r on board %d" % (e, k)); return False
    wins = [t for t in transitions if t.before_frame is not None and t.after_frame is not None
            and t.after_frame.level > t.before_frame.level]
    win_ok = None
    if step is not None and wins:
        win_ok = 0
        for t in wins:
            try:
                win_ok += bool(goal(step(t.before_frame.ascii.split("\n"), str(t.action))))
            except Exception:
                pass
    msg = "check_goal: goal is True on %d of %d boards already seen on this level (must be 0)" % (len(bad), len(seen))
    if win_ok is not None:
        msg += "; step+goal recognise %d of %d past winning moves" % (win_ok, len(wins))
    print(msg)
    return not bad and (win_ok is None or win_ok == len(wins))


def _ty_act_str(a):
    if isinstance(a, dict):
        return "MOUSE(row=%s, col=%s)" % (a.get("row"), a.get("col")) if str(a.get("action", "")).upper() == "MOUSE" else str(a.get("action"))
    return str(a)


def run_plan(plan_actions, step):
    """Исполнить план по одному ходу, сверяя каждую новую доску с предсказанием step (без полосы индикатора).
    Останавливается на первом расхождении (печатает, что разошлось), на завершении уровня и на конце игры.
    Возвращает число сделанных ходов."""
    mask = hud_mask(); done = 0
    for a in plan_actions or []:
        before = current_frame.ascii.split("\n"); lv = current_frame.level
        try:
            pred = [str(x) for x in step(list(before), _ty_act_str(a))]
        except Exception as e:
            print("run_plan: step raised %r before move %d — stopped" % (e, done)); return done
        action([a]); done += 1
        now = current_frame.ascii.split("\n")
        if current_frame.level != lv:
            print("run_plan: level changed after move %d — stopped" % done); return done
        if not _ty_same(pred, now, mask):
            print("run_plan: move %d (%s) diverged from your step — stopped\n  predicted: %s\n  observed:  %s" % (
                done, _ty_act_str(a), diff_rows(before, pred, mask).replace("\n", "\n    "), diff_rows(before, now, mask).replace("\n", "\n    ")))
            return done
    print("run_plan: all %d moves matched the simulator" % done)
    return done
# --- конец частей Tycho ---

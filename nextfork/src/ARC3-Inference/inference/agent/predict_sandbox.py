# --- NEXTFORK_PREDICT: предсказание перед каждым ходом (30.09; идея arc-skill, реализация своя) ---
# Выполняется в песочнице ПЕРЕД кодом модели: оборачивает action(...), оценивает предсказания по кадрам
# до/после, печатает ✓/✗ и короткий рассказ о переходе; notes(text) сохраняет страницу заметок.
_pd_raw_action = action


def _pd_rows(frame):
    return frame.ascii.split("\n") if frame is not None else []


def _pd_volatile(level):
    """Клетки, меняющиеся почти на каждом ходу этого уровня (счётчики, полосы) — в noop/change/region не считаются."""
    cnt = {}
    n = 0
    for t in transitions:
        b, a = t.before_frame, t.after_frame
        if b is None or a is None or b.level != level or a.level != level:
            continue
        n += 1
        rb, ra = _pd_rows(b), _pd_rows(a)
        for r in range(min(len(rb), len(ra))):
            if rb[r] == ra[r]:
                continue
            for c in range(min(len(rb[r]), len(ra[r]))):
                if rb[r][c] != ra[r][c]:
                    cnt[(r, c)] = cnt.get((r, c), 0) + 1
    if n < 4:
        return set()
    return {k for k, v in cnt.items() if v >= 0.8 * n}


def _pd_diff(rb, ra, skip):
    out = []
    for r in range(min(len(rb), len(ra))):
        if rb[r] == ra[r]:
            continue
        for c in range(min(len(rb[r]), len(ra[r]))):
            if rb[r][c] != ra[r][c] and (r, c) not in skip:
                out.append((r, c, rb[r][c], ra[r][c]))
    return out


def _pd_background(rows):
    cnt = {}
    for row in rows:
        for ch in row:
            cnt[ch] = cnt.get(ch, 0) + 1
    return max(cnt, key=cnt.get) if cnt else None


def _pd_component(rows, r, c, cap=900):
    if not (0 <= r < len(rows) and 0 <= c < len(rows[r])):
        return None, set()
    ch = rows[r][c]
    seen = {(r, c)}
    stack = [(r, c)]
    while stack and len(seen) <= cap:
        y, x = stack.pop()
        for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if (yy, xx) not in seen and 0 <= yy < len(rows) and 0 <= xx < len(rows[yy]) and rows[yy][xx] == ch:
                seen.add((yy, xx))
                stack.append((yy, xx))
    return ch, seen


def _pd_ints(text, n):
    parts = [p for p in text.replace(",", " ").split() if p]
    if len(parts) != n:
        raise ValueError(text)
    return [int(p) for p in parts]


PREDICT_FORMS = ("noop | change | cell R,C=X | move R,C DR,DC | vanish R,C | region R0:R1,C0:C1 | level+1 | win | "
                 "gameover   (R=row, C=col, X=color letter as in current_frame.ascii; join several with ';')")


def _pd_grade(claim, rb, ra, res, skip):
    """(ok, текст) для одного утверждения."""
    s = claim.strip()
    low = s.lower()
    diff = _pd_diff(rb, ra, skip)
    lvl_up = bool(res.get("level_completed"))
    if low in ("level+1", "level +1", "levelup", "level up"):
        return lvl_up, "level completed" if lvl_up else "level NOT completed"
    if low == "win":
        ok = bool(res.get("run_complete") or res.get("done")) and not res.get("game_over")
        return ok, "game won" if ok else "game NOT won"
    if low in ("gameover", "game over", "game_over"):
        ok = bool(res.get("game_over"))
        return ok, "game over" if ok else "no game over"
    if low == "noop":
        return (not diff and not lvl_up), ("nothing changed" if not diff else "%d cells changed" % len(diff))
    if low == "change":
        return (bool(diff) or lvl_up), ("%d cells changed" % len(diff) if diff else "nothing changed")
    head, _, rest = s.partition(" ")
    head = head.lower()
    if head == "cell":
        pos, _, val = rest.partition("=")
        r, c = _pd_ints(pos, 2)
        val = val.strip()
        seen = ra[r][c] if 0 <= r < len(ra) and 0 <= c < len(ra[r]) else "?"
        return seen == val, "cell %d,%d is %s" % (r, c, seen)
    if head == "region":
        a, _, b = rest.partition(",")
        r0, r1 = _pd_ints(a.replace(":", " "), 2)
        c0, c1 = _pd_ints(b.replace(":", " "), 2)
        inside = [d for d in diff if r0 <= d[0] <= r1 and c0 <= d[1] <= c1]
        return bool(inside), "%d changed cells inside, %d outside" % (len(inside), len(diff) - len(inside))
    if head in ("move", "vanish"):
        bits = rest.split()
        r, c = _pd_ints(bits[0], 2)
        bg = _pd_background(rb)
        ch, comp = _pd_component(rb, r, c)
        if ch is None:
            return False, "cell %d,%d is off the board" % (r, c)
        if ch == bg or len(comp) > 900:
            return False, "cell %d,%d is background (%s), not an object" % (r, c, ch)
        if head == "vanish":
            left = sum(1 for (y, x) in comp if 0 <= y < len(ra) and 0 <= x < len(ra[y]) and ra[y][x] == ch)
            return left == 0, ("object %s gone" % ch) if left == 0 else ("%d of %d cells of %s still there" % (left, len(comp), ch))
        dr, dc = _pd_ints(bits[1], 2)
        moved = all(0 <= y + dr < len(ra) and 0 <= x + dc < len(ra[y + dr]) and ra[y + dr][x + dc] == ch for (y, x) in comp)
        if moved and (dr or dc):
            vacated = [(y, x) for (y, x) in comp if (y - dr, x - dc) not in comp]
            moved = any(ra[y][x] != ch for (y, x) in vacated if 0 <= y < len(ra) and 0 <= x < len(ra[y]))
        if moved:
            return True, "object %s moved by %d,%d" % (ch, dr, dc)
        # где объект на самом деле
        best = None
        for ddr in range(-8, 9):
            for ddc in range(-8, 9):
                if all(0 <= y + ddr < len(ra) and 0 <= x + ddc < len(ra[y + ddr]) and ra[y + ddr][x + ddc] == ch for (y, x) in comp):
                    if best is None or abs(ddr) + abs(ddc) < abs(best[0]) + abs(best[1]):
                        best = (ddr, ddc)
        return False, ("object %s moved by %d,%d" % (ch, best[0], best[1])) if best else ("object %s did not keep its shape" % ch)
    raise ValueError("unknown claim %r" % s)


def _pd_story(rb, ra, res, skip):
    diff = _pd_diff(rb, ra, skip)
    parts = []
    if res.get("level_completed"):
        parts.append("LEVEL COMPLETED")
    if res.get("game_over"):
        parts.append("GAME OVER")
    if not diff:
        parts.append("no board change" + (" (ignoring %d always-changing cells)" % len(skip) if skip else ""))
    else:
        rs = [d[0] for d in diff]
        cs = [d[1] for d in diff]
        flips = {}
        for d in diff:
            flips[(d[2], d[3])] = flips.get((d[2], d[3]), 0) + 1
        top = sorted(flips.items(), key=lambda kv: -kv[1])[:4]
        parts.append("%d cells changed in rows %d-%d cols %d-%d; %s" % (
            len(diff), min(rs), max(rs), min(cs), max(cs), ", ".join("%s->%s x%d" % (a, b, n) for (a, b), n in top)))
    return "; ".join(parts)


def _pd_one(act, claims):
    before = current_frame
    rb = _pd_rows(before)
    skip = _pd_volatile(before.level) if before is not None else set()
    res = _pd_raw_action([act])
    ra = _pd_rows(current_frame)
    if before is not None and current_frame is not None and current_frame.level != before.level:
        skip = set()
    ok_all = True
    lines = []
    for cl in [x for x in str(claims).split(";") if x.strip()]:
        try:
            ok, why = _pd_grade(cl, rb, ra, res if isinstance(res, dict) else {}, skip)
        except Exception:
            ok, why = False, "claim not understood; forms: " + PREDICT_FORMS
        ok_all = ok_all and ok
        lines.append("%s %s  (%s)" % ("OK " if ok else "MISS", cl.strip(), why))
    name = act if isinstance(act, str) else "%s(%s)" % (act.get("action"), ",".join("%s=%s" % (k, v) for k, v in act.items() if k != "action"))
    print("PREDICT %s %s: %s | %s" % ("✓" if ok_all else "✗", name, " ; ".join(lines), _pd_story(rb, ra, res if isinstance(res, dict) else {}, skip)))
    return res, ok_all


def action(actions, predict=None, because=None):
    """Как прежде, но КАЖДЫЙ ход требует предсказания: predict='move 12,5 0,-1' для одного хода или список
    предсказаний по одному на ход. Пачка останавливается на первом промахе (остальные ходы не тратятся)."""
    items = [actions] if isinstance(actions, (str, dict)) else list(actions)
    if predict is None or predict == "" or predict == []:
        raise ValueError("REFUSED (no move spent): every action(...) needs predict=... saying what the move will do "
                         "to the board, e.g. action(['LEFT'], predict='move 12,5 0,-1'). Forms: " + PREDICT_FORMS)
    preds = [predict] * len(items) if isinstance(predict, str) and len(items) == 1 else predict
    if isinstance(preds, str):
        raise ValueError("REFUSED (no move spent): %d actions need a LIST of %d predictions, one per move "
                         "(the batch stops at the first miss)." % (len(items), len(items)))
    preds = list(preds)
    if len(preds) != len(items):
        raise ValueError("REFUSED (no move spent): %d actions but %d predictions." % (len(items), len(preds)))
    last = {}
    for i, (act, cl) in enumerate(zip(items, preds)):
        last, ok = _pd_one(act, cl)
        if isinstance(last, dict) and (last.get("level_completed") or last.get("game_over") or last.get("done")):
            break
        if not ok and i < len(items) - 1:
            print("PREDICT batch stopped after move %d of %d: reality differs from your model; %d moves NOT spent."
                  % (i + 1, len(items), len(items) - i - 1))
            break
    return last


def notes(text):
    """Одна страница заметок на игру (Verified / Assumed / Plan). Показывается в каждом следующем сообщении."""
    text = str(text).strip()
    print("@@NOTES@@\n%s\n@@END_NOTES@@" % text[:3000])
# --- конец NEXTFORK_PREDICT ---

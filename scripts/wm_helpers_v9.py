WM_HELPERS = r'''
from collections import deque as _wm_deque

def _wm_ascii(f):
    a = getattr(f, "ascii", None)
    if a is None: return ""
    return a if isinstance(a, str) else str(a)

def wm_level(frame=None):
    """Номер уровня кадра. Клади его в состояние: без него цель не проверить."""
    f = frame if frame is not None else current_frame
    try: return int(getattr(f, "level", 0) or 0)
    except Exception: return 0

def wm_objects(frame=None):
    """Типизированные объекты кадра. `t` — механический тип: подпись цвета и формы
    без привязки к позиции. Правило для одного объекта типа обязано работать для всех."""
    f = frame if frame is not None else current_frame
    seg = getattr(f, "segmentation", None) or {}
    out = []
    for n in (seg.get("nodes") or []):
        b = n.get("boundary") or []
        xs = [p[1] for p in b] or [0]; ys = [p[0] for p in b] or [0]
        out.append({"id": n.get("id"), "t": n.get("hash"), "color": n.get("color"),
                    "x": min(xs), "y": min(ys), "w": max(xs)-min(xs)+1, "h": max(ys)-min(ys)+1,
                    "pixels": n.get("pixels")})
    return out

def _wm_norm(x):
    if isinstance(x, dict): return tuple(sorted((str(k), _wm_norm(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)): return tuple(_wm_norm(v) for v in x)
    if isinstance(x, set): return tuple(sorted(_wm_norm(v) for v in x))
    return x

def _wm_pairs(k=400, level=None):
    """Общий буфер: наблюдённые переходы (до, действие, после) ТЕКУЩЕГО уровня.

    Почему фильтр: доска меняется на новом уровне целиком, и одна программа не может
    воспроизвести переходы двух разных досок. Без фильтра точный приём после первого
    взятого уровня недостижим в принципе. Переход, ВОШЕДШИЙ в текущий уровень, оставляем:
    на нём проверяется предикат цели."""
    if level is None:
        try: level = wm_level(current_frame)
        except Exception: level = None
    out = []
    for tr in (transitions or [])[-k:]:
        b = getattr(tr, "before_frame", None); a = getattr(tr, "after_frame", None)
        act = str(getattr(tr, "action", "") or "").strip()
        if b is None or a is None or not act: continue
        if level is not None:
            try:
                if wm_level(b) != level and wm_level(a) != level: continue
            except Exception: pass
        out.append((b, act, a))
    return out

_WM_RAN = [False]

def wm_check(predict, state_of=None, goal=None, k=400, verbose=True):
    """ПРИЁМ ТОЧНЫЙ: модель принимается, только если воспроизводит КАЖДЫЙ записанный
    переход. Непокрытый переход (predict вернул None) — тоже отказ.
    Программа гоняется ДВАЖДЫ на каждом переходе: расхождение прогонов — отказ
    (так отсеиваются модели со скрытым состоянием, ломающие планировщик).
    Переход, ВЗЯВШИЙ УРОВЕНЬ, особый: кадр после него — уже новая доска, её предсказывать
    нельзя. Он проверяется через предикат цели: goal(predict(до, действие)) обязан быть True.
    Без goal такие переходы не считаются (отчёт назовёт их число)."""
    _WM_RAN[0] = True
    st = state_of if state_of is not None else (lambda f: _wm_ascii(f))
    pairs = _wm_pairs(k)
    correct = 0; checked = 0; cex = None; miss = {}; unstable = None; reward = 0; reward_unchecked = 0
    for b, act, a in pairs:
        try: sb = st(b); sa = st(a)
        except Exception as exc:
            cex = cex or {"action": act, "error": "state_of упал: %r" % (exc,)}; continue
        try:
            p1 = predict(sb, act); p2 = predict(sb, act)
        except Exception as exc:
            cex = cex or {"action": act, "error": "predict упал: %r" % (exc,)}; continue
        if wm_level(a) != wm_level(b):
            reward += 1
            if goal is None:
                reward_unchecked += 1; continue
            checked += 1
            try: hit = (p1 is not None) and bool(goal(p1))
            except Exception: hit = False
            if hit: correct += 1
            elif cex is None:
                cex = {"action": act, "before": sb, "predicted": p1,
                       "actual": "УРОВЕНЬ ВЗЯТ — goal(predict(...)) должна быть True, а она False"}
            continue
        if p1 is None:
            miss[act] = miss.get(act, 0) + 1; continue
        checked += 1
        if _wm_norm(p1) != _wm_norm(p2):
            unstable = unstable or act; continue
        if _wm_norm(p1) == _wm_norm(sa): correct += 1
        elif cex is None:
            cex = {"action": act, "before": sb, "predicted": p1, "actual": sa}
    total = len(pairs) - reward_unchecked
    admitted = bool(total >= 1 and checked == total and correct == total and unstable is None)
    acc = (correct / checked) if checked else 0.0
    if verbose:
        print("WM_CHECK admitted=%d correct=%d checked=%d total=%d acc=%.3f not_covered=%s%s reward=%d%s"
              % (1 if admitted else 0, correct, checked, total, acc, sorted(miss.items()),
                 (" unstable=%s" % unstable) if unstable else "", reward,
                 (" reward_unchecked=%d" % reward_unchecked) if reward_unchecked else ""), flush=True)
        if reward_unchecked:
            print("WM_REWARD переходов, взявших уровень: %d — они не проверены, потому что нет goal(state). "
                  "Напиши def goal(state) и проверь wm_goal_check(goal, predict, state_of)" % reward_unchecked, flush=True)
        if unstable:
            print("WM_UNSTABLE: два прогона predict дали разное на '%s' — в модели скрытое состояние, планировать по ней нельзя" % unstable, flush=True)
        elif cex is not None:
            print("WM_COUNTEREXAMPLE %r" % (cex,), flush=True)
        elif not admitted and miss:
            print("WM_COUNTEREXAMPLE непокрытые действия: %s" % sorted(miss.items()), flush=True)
        elif not admitted and total == 0:
            print("WM_COUNTEREXAMPLE переходов ещё нет — сначала походи и набери наблюдения", flush=True)
    return {"admitted": admitted, "correct": correct, "checked": checked, "total": total,
            "accuracy": acc, "counterexample": cex, "not_covered": miss, "unstable": unstable,
            "reward": reward, "reward_unchecked": reward_unchecked}


def wm_horizon(predict, state_of, k=400, max_steps=12, verbose=True):
    """ГОРИЗОНТ РАСКАТКИ: сколько шагов программа ведёт состояние сама, не расходясь с записью.

    Меряется ровно то, чем пользуется планировщик: цепочка предсказаний, а не одиночный переход.
    Основание (замер 09.09 на 21 программе с взятым уровнем): среди программ с медианным
    горизонтом >= 8 поиск находил путь к взятию уровня в 6 случаях из 9, среди программ с
    горизонтом <= 2 — ни в одном из 7. Точный приём этих двух классов НЕ различает: и программа
    с горизонтом 12, и программа с горизонтом 0 проходят его одинаково.
    """
    _WM_RAN[0] = True
    pairs = _wm_pairs(k)
    hor = []
    # Считаем только старты, где ВПЕРЕДИ есть max_steps переходов: иначе медиана падает не от
    # качества программы, а от того, что раскатываться некуда (поймано проверкой 09.09).
    for i in range(max(0, len(pairs) - max_steps + 1)):
        try: s = state_of(pairs[i][0])
        except Exception: continue
        steps = 0
        for j in range(i, min(i + max_steps, len(pairs))):
            b, act, a = pairs[j]
            try:
                n = predict(s, act); t = state_of(a)
            except Exception:
                n = None
            if n is None or _wm_norm(n) != _wm_norm(t): break
            steps += 1; s = n
        hor.append(steps)
    hor.sort()
    if len(hor) < 3:
        if verbose:
            print("WM_HORIZON med=-1 n=%d pairs=%d max=%d — наблюдений мало, горизонт не измерить"
                  % (len(hor), len(pairs), max_steps), flush=True)
        return {"median": -1, "rollouts": len(hor), "pairs": len(pairs)}
    med = hor[len(hor) // 2]
    if verbose:
        print("WM_HORIZON med=%d n=%d pairs=%d max=%d" % (med, len(hor), len(pairs), max_steps), flush=True)
        if med == 0:
            print("WM_HORIZON программа расходится с записью на первом же шаге — планировать по ней нельзя", flush=True)
    return {"median": med, "rollouts": len(hor), "pairs": len(pairs)}

def wm_goal_check(goal, predict, state_of, k=400, verbose=True):
    """Проверка ПРЕДИКАТА ЦЕЛИ тем же точным повтором (статья, раздел 2: цель распознаётся по
    наблюдённой награде, а затем синтезированный булев предикат проверяется на записи).
    Требования: (1) на каждом переходе, взявшем уровень, goal(predict(до, действие)) == True;
    (2) на всех состояниях ДО взятия, на том же уровне, goal == False — иначе предикат
    срабатывает раньше времени и план остановится не там."""
    _WM_RAN[0] = True
    pairs = _wm_pairs(k)
    hits = 0; rewards = 0; false_alarms = 0; negatives = 0; err = None
    for b, act, a in pairs:
        try:
            sb = state_of(b)
            if wm_level(a) != wm_level(b):
                rewards += 1
                p = predict(sb, act)
                if p is not None and bool(goal(p)): hits += 1
            else:
                negatives += 1
                if bool(goal(sb)): false_alarms += 1
                sa = state_of(a)
                if bool(goal(sa)): false_alarms += 1
        except Exception as exc:
            err = err or repr(exc)
    ok = bool(rewards >= 1 and hits == rewards and false_alarms == 0 and err is None)
    if verbose:
        if rewards == 0:
            print("WM_GOAL ok=0 reason=награда ещё не наблюдалась — цель проверить не на чем", flush=True)
        else:
            print("WM_GOAL ok=%d hits=%d/%d false_alarms=%d negatives=%d%s"
                  % (1 if ok else 0, hits, rewards, false_alarms, negatives, (" error=%s" % err) if err else ""), flush=True)
    return {"ok": ok, "hits": hits, "rewards": rewards, "false_alarms": false_alarms, "error": err}

_WM_SIG_ATTRS = ("x", "y", "pixels")

def _wm_sig(before, after):
    """Сигнатура эффекта: КАКИЕ атрибуты изменились, без самих значений."""
    if before is None: return "born"
    if after is None: return "gone"
    ch = [b for b in _WM_SIG_ATTRS if before.get(b) != after.get(b)]
    return ",".join(ch) if ch else "no change"

def wm_ontology(k=400, top=5, verbose=True):
    """Онтологическая ошибка: насколько текущие типы объясняют наблюдённые эффекты.
    Каждый переход объекта попадает в строку (тип, действие); по строке считается
    распределение сигнатур со сглаживанием Дирихле и нормированная энтропия.
    Строки с высокой энтропией — там, где типов не хватает: туда и надо ходить."""
    import math as _m
    rows = {}; alphabet = set()
    for b, act, a in _wm_pairs(k):
        ob = {}; oa = {}
        for o in wm_objects(b): ob.setdefault((o["t"], o["x"], o["y"]), o)
        bt = {}; at = {}
        for o in wm_objects(b): bt.setdefault(o["t"], []).append(o)
        for o in wm_objects(a): at.setdefault(o["t"], []).append(o)
        for t, lst in bt.items():
            after = sorted(at.get(t, []), key=lambda o: (o["y"], o["x"]))
            before = sorted(lst, key=lambda o: (o["y"], o["x"]))
            for i, o in enumerate(before):
                sig = _wm_sig(o, after[i] if i < len(after) else None)
                alphabet.add(sig)
                rows.setdefault((t, act), {}).setdefault(sig, 0)
                rows[(t, act)][sig] += 1
        for t, lst in at.items():
            if t not in bt:
                for o in lst:
                    alphabet.add("born"); rows.setdefault((t, act), {}).setdefault("born", 0)
                    rows[(t, act)]["born"] += 1
    m = max(2, len(alphabet)); a0 = 1.0
    scored = []
    for j, cnt in rows.items():
        n = sum(cnt.values()); denom = m * a0 + n
        H = 0.0
        for e in alphabet:
            q = (a0 + cnt.get(e, 0)) / denom
            if q > 0: H -= q * _m.log(q)
        scored.append((H / _m.log(m), n, j, dict(cnt)))
    scored.sort(reverse=True)
    eta = sum(s[0] for s in scored) / len(scored) if scored else 1.0
    if verbose:
        print("WM_ONTOLOGY eta=%.3f rows=%d alphabet=%s" % (eta, len(scored), sorted(alphabet)), flush=True)
        for H, n, j, cnt in scored[:top]:
            print("   неопределённость %.2f  тип %s действие %s  наблюдений %d  эффекты %s" % (H, j[0], j[1], n, cnt), flush=True)
    return {"eta": eta, "rows": [{"type": j[0], "action": j[1], "H": H, "n": n, "effects": cnt} for H, n, j, cnt in scored]}

def wm_plan(predict, goal, actions=None, state=None, state_of=None, max_depth=8, max_nodes=6000, verbose=True):
    """Ограниченный поиск вперёд ВНУТРИ принятой модели: кратчайшая цепочка действий,
    после которой goal(state) истинно. Настоящих ходов не тратит."""
    st = state_of if state_of is not None else (lambda f: _wm_ascii(f))
    s0 = state if state is not None else st(current_frame)
    acts = list(actions) if actions else list(valid_actions or [])
    try:
        if goal(s0):
            if verbose: print("WM_PLAN уже в цели", flush=True)
            return {"plan": [], "nodes": 0, "reason": "already"}
    except Exception as exc:
        print("WM_PLAN цель упала: %r" % (exc,), flush=True)
        return {"plan": None, "nodes": 0, "reason": "goal_error"}
    seen = {_wm_norm(s0)}; q = _wm_deque([(s0, [])]); nodes = 0
    while q:
        s, path = q.popleft()
        if len(path) >= max_depth: continue
        for a in acts:
            nodes += 1
            if nodes > max_nodes:
                if verbose: print("WM_PLAN исчерпан лимит узлов nodes=%d" % nodes, flush=True)
                return {"plan": None, "nodes": nodes, "reason": "limit"}
            try: ns = predict(s, a)
            except Exception: continue
            if ns is None: continue
            key = _wm_norm(ns)
            if key in seen: continue
            seen.add(key); np_ = path + [a]
            try: hit = goal(ns)
            except Exception: hit = False
            if hit:
                if verbose: print("WM_PLAN found len=%d nodes=%d plan=%s" % (len(np_), nodes, np_), flush=True)
                return {"plan": np_, "nodes": nodes, "reason": "found"}
            q.append((ns, np_))
    if verbose: print("WM_PLAN пути нет nodes=%d" % nodes, flush=True)
    return {"plan": None, "nodes": nodes, "reason": "exhausted"}

def wm_validate_plan(predict, goal, state_of, level=None, verbose=True):
    """Проверка планировщика ОФФЛАЙН: планируем к награде из входного состояния уже
    взятого уровня. Планировщик, не прошедший её, в живую игру не пускаем."""
    lv = level
    ent = None
    for h in (history or []):
        f = getattr(h, "frame", None)
        if f is None: continue
        l = wm_level(f)
        if lv is None: lv = l
        if l == lv: ent = f; break
    if ent is None:
        if verbose: print("WM_VALIDATE нет входного состояния уровня %s" % (lv,), flush=True)
        return {"ok": False, "reason": "no_entry"}
    r = wm_plan(predict, goal, state=state_of(ent), state_of=state_of, verbose=False)
    ok = bool(r.get("plan"))
    if verbose:
        print("WM_VALIDATE level=%s ok=%d plan=%s" % (lv, 1 if ok else 0, r.get("plan")), flush=True)
    return {"ok": ok, "plan": r.get("plan"), "reason": r.get("reason")}

def wm_execute(predict, actions, state_of=None):
    """План исполняется по одному шагу. Совпало — идём дальше; разошлось — план обрывается,
    расхождение записывается контрпримером, управление возвращается разведке."""
    st = state_of if state_of is not None else (lambda f: _wm_ascii(f))
    done = []
    for a in list(actions):
        before = st(current_frame)
        try: pred = predict(before, a)
        except Exception as exc:
            print("WM_EXEC predict упал: %r" % (exc,), flush=True); break
        action(a); done.append(a)
        got = st(current_frame)
        if pred is None or _wm_norm(pred) != _wm_norm(got):
            print("WM_EXEC mismatch на шаге %d действие %s" % (len(done), a), flush=True)
            print("WM_COUNTEREXAMPLE %r" % ({"action": a, "before": before, "predicted": pred, "actual": got},), flush=True)
            return {"executed": done, "ok": False}
    print("WM_EXEC ok шагов=%d" % len(done), flush=True)
    return {"executed": done, "ok": True}
'''

"""Общий патч для прогона на поде: слой «модель мира» (v13) + расписание со снятием.

Собирается из тех же ячеек, что уезжают в Kaggle, поэтому код на поде и в кернеле один и тот же.
Отличие одно: бюджеты (окно прогона, конкурентность, потолок на игру) на поде задаются флагами
движка, а не присваиваниями в ячейке. Именно эта разница 09.09 стоила 2 ч 20 мин квоты —
в ноутбуке стоковая ячейка затирала наше окно, на поде такого места нет.
"""

TRUE_SUBMISSION = True   # ноутбучных веток на поде нет; нужно только для совместимости кода


# =====================================================================
# WM v13 — приём по ГОРИЗОНТУ РАСКАТКИ: 6 из 9 программ с горизонтом >= 8 давали план,
# и ни одна из 7 с горизонтом <= 2. Точный приём эти классы не различает.
# Синтез снова с первого уровня: ограничение по взятому уровню дало 2 запуска за прогон.
#
# v11 показала: канал починен (9 программ из 11 попыток), но принято 0 — семь программ
# возвращали None на все переходы при буфере уровня в 1-19 переходов, два ответа пришли
# без вызова инструмента с 87-92 тыс. знаков рассуждений. Отсюда три правки ниже.
#
# Мера, ради которой версия существует: в v10 программы писал ИГРАЮЩИЙ агент, и генерация на
# запрос выросла 1436 -> 1974, очередь 122 -> 137 с, ходов на игру 9.0 против ~12.6 у базы,
# действий 9.7 против 20.2 у v8. Написание модели должно уйти с игровых ходов — это и есть
# второй агент. Он же дважды молчал, и причины теперь найдены и починены (см. заголовок сборщика).
#
# Пять правок синтезатора: тот же инструмент `python`; цепочка чтения ответа из четырёх каналов;
# штатный таймаут вместо моих 300 с плюс один повтор на разрыв; ограничитель на 4 работника;
# канал ответа и первые 300 символов — в лог.
# Играющий агент: моделей НЕ пишет, обязан действовать каждый ход, после взятого уровня пишет
# только предикат `goal` (одна строка) — без него план не включается никогда.
# Работает в обеих ветках.
# =====================================================================
# ВЫКЛЮЧАТЕЛИ. Прогон на поде проверяет РОВНО ОДНО изменение: либо слой «модель мира»,
# либо расписание. Смешивать нельзя — иначе результат не приписать причине.
#   WM_LAYER=1/0  слой модели мира
#   WM_SCHED=1/0  снятие игры после простоя
import os as _wm_os
_WM_LAYER_ON = _wm_os.environ.get("WM_LAYER", "1") == "1"
_WM_SCHED_ON = _wm_os.environ.get("WM_SCHED", "1") == "1"

import ast as _wast
import json as _wjson
import re as _wre
import threading as _wthr
import inference.agent.tool_agent as _wta
from inference.agent.python_tool_sandbox import run_sandboxed_python as _wm_sandbox

_WM_HELPERS = r'''
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
_WM_REJECTS   = 2      # отклонений хода без действия на уровень
_WM_SYNTH_CAP = 20     # потолок попыток синтеза на игру
_WM_MIN_LEVEL_OBS = 10 # переходов на уровне до синтеза: горизонту нужно куда раскатываться
_WM_HOR_STEPS = 8      # потолок раскатки
_WM_HOR_MIN   = 8      # принимаем, если медианный горизонт не ниже потолка
_WM_MAX_NOPROG = 2     # ответов без программы на игру, дальше синтез для неё прекращаем
_WM_PARALLEL  = 4      # синтезаторов на ВЕСЬ прогон (v8: очередь 197 -> 128 с)
_WM_SEM = _wthr.BoundedSemaphore(_WM_PARALLEL)

# Хвост из v10: в белом списке песочницы (SAFE_BUILTINS) NameError НЕТ, ловить только Exception;
# всё под стражем, чтобы ошибка НАШЕГО кода не возвращалась модели трассировкой.
_WM_TAIL = (
    "\n# --- автопроверка харнесса: буфер уже вырос на сделанный ход ---\n"
    "try:\n"
    "    try:\n"
    "        _wm_p = predict; _wm_s = state_of\n"
    "    except Exception:\n"
    "        _wm_p = None\n"
    "    if _wm_p is not None:\n"
    "        try: _wm_g = goal\n"
    "        except Exception: _wm_g = None\n"
    "        if not _WM_RAN[0]:\n"
    "            wm_check(_wm_p, state_of=_wm_s, goal=_wm_g)\n"
    "            wm_horizon(_wm_p, _wm_s, max_steps=8)\n"
    "            if _wm_g is not None:\n"
    "                wm_goal_check(_wm_g, _wm_p, _wm_s)\n"
    "except Exception as _e:\n"
    "    print('WM_TAIL сбой харнесса: %r' % (_e,))\n"
)

_WM_DIGEST_CODE = (
    "print('ВАЛИДНЫЕ ДЕЙСТВИЯ:', valid_actions)\n"
    "print('УРОВЕНЬ:', wm_level(current_frame))\n"
    "print('ТЕКУЩИЙ КАДР:'); print(current_frame.ascii)\n"
    "print('НАБЛЮДЁННЫЕ ПЕРЕХОДЫ (действие -> что изменилось у объектов):')\n"
    "_pp = _wm_pairs(12)\n"
    "for _b, _a, _af in _pp:\n"
    "    _ob = {}; _oa = {}\n"
    "    for _o in wm_objects(_b): _ob.setdefault(_o['t'], []).append(_o)\n"
    "    for _o in wm_objects(_af): _oa.setdefault(_o['t'], []).append(_o)\n"
    "    _ch = []\n"
    "    for _t, _lst in _ob.items():\n"
    "        _bs = sorted(_lst, key=lambda o: (o['y'], o['x']))\n"
    "        _as_ = sorted(_oa.get(_t, []), key=lambda o: (o['y'], o['x']))\n"
    "        for _i, _o in enumerate(_bs):\n"
    "            if _i >= len(_as_): _ch.append((_t, 'исчез')); continue\n"
    "            _n = _as_[_i]\n"
    "            _d = (_n['x'] - _o['x'], _n['y'] - _o['y'], (_n['pixels'] or 0) - (_o['pixels'] or 0))\n"
    "            if any(_d): _ch.append((_t, 'dx=%d dy=%d dpix=%d' % _d))\n"
    "    print('  %-8s %s' % (_a, _ch[:8] if _ch else 'без изменений'))\n"
    "print('ЧИСЛО ПЕРЕХОДОВ В БУФЕРЕ:', len(_wm_pairs(400)))\n"
    "wm_ontology(top=4)\n"
)

_WM_SYNTH_SYS = (
    "Ты — агент-синтезатор модели мира. Ты НЕ играешь: твоя единственная работа — написать на "
    "Python программу, воспроизводящую наблюдённые переходы игры.\n"
    "ОТВЕЧАЙ ВЫЗОВОМ ИНСТРУМЕНТА `python`, положив программу в аргумент `code`. Ничего кроме "
    "вызова не нужно.\n"
    "В программе ровно две функции:\n"
    "  def state_of(frame): ...   # структурное состояние кадра; ОБЯЗАТЕЛЬНО включи wm_level(frame)\n"
    "  def predict(state, action): ...   # следующее состояние или None, если случай не покрыт\n"
    "Доступны помощники: wm_level(frame), wm_objects(frame) — объекты с полем 't' (подпись цвета и\n"
    "формы без привязки к позиции), 'x', 'y', 'w', 'h', 'pixels'; frame.ascii — сетка символов.\n"
    "Правило пиши на ТИП объекта, а не на координаты: тогда оно переносится на все такие объекты.\n"
    "Приём ТОЧНЫЙ: программа принимается, только если воспроизводит КАЖДЫЙ переход. Непокрытый\n"
    "переход — тоже отказ, поэтому покрывай все действия из списка, включая клики MOUSE.\n"
    "Не больше 60 строк, без объяснений и примеров. Никаких вызовов action(...) — они недоступны."
)

# Тот же инструмент, что у играющего агента: это единственный канал со стопроцентной надёжностью.
_WM_TOOLS = [{
    "type": "function",
    "function": {
        "name": "python",
        "description": "Верни программу модели мира: две функции state_of(frame) и predict(state, action).",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python-программа из двух функций."}},
            "required": ["code"],
        },
    },
}]

_WM_GATHER = (
    "[МОДЕЛЬ МИРА] Ход обязан содержать действие. Первый уровень бери как обычно, ничего\n"
    "писать не надо: модель этой игры начнёт строиться после того, как уровень будет взят.\n"
    "Пробуй РАЗНЫЕ действия и не трать ход на разбор без хода.\n"
)
_WM_AFTER = (
    "[МОДЕЛЬ МИРА] Уровень взят. Второй агент прямо сейчас строит модель ЭТОЙ доски по твоим\n"
    "наблюдениям — тебе писать её не надо. Ход обязан содержать действие: буфер этого уровня\n"
    "растёт только от ходов, и чем он полнее, тем быстрее появится модель.\n"
)
_WM_KEEP = (
    "[МОДЕЛЬ МИРА] Второй агент построил модель, воспроизводящую все %d записанных переходов.\n"
    "Её определения `state_of` и `predict` УЖЕ подставлены в начало твоего кода, писать их не надо;\n"
    "проверка идёт сама после твоих действий. Ход обязан содержать действие. Награды ещё не было —\n"
    "ходи туда, где модель знает меньше всего: `wm_ontology()` печатает самые неопределённые строки.\n"
)
_WM_GOAL = (
    "[МОДЕЛЬ МИРА] Уровень взят — награда наблюдалась, теперь цель можно ЗАПИСАТЬ. Модель второго\n"
    "агента принята (%d переходов) и подставлена. В ТОМ ЖЕ ходу, вместе с действием, напиши ОДНУ\n"
    "короткую функцию:\n"
    "  def goal(state): ...   # True ровно в том состоянии, которое взяло уровень\n"
    "Модель мира писать НЕ надо, её пишет второй агент. Харнесс проверит цель сам тем же точным\n"
    "повтором: goal(predict(до, ход_взявший_уровень)) обязана быть True, а на всех состояниях до\n"
    "взятия — False. Отчёт WM_GOAL скажет: попадания и ложные срабатывания.\n"
)
_WM_PLAN = (
    "[МОДЕЛЬ МИРА] Модель принята (%d переходов) и ЦЕЛЬ ПРОВЕРЕНА. `state_of`, `predict`, `goal`\n"
    "подставлены. Планируй и исполняй в одном ходу:\n"
    "  p = wm_plan(predict, goal, state_of=state_of, max_depth=10)\n"
    "  if p['plan']: wm_execute(predict, p['plan'], state_of=state_of)\n"
    "Поиск идёт внутри модели и ходов не тратит; `wm_execute` делает настоящие ходы по одному и\n"
    "обрывает план при первом расхождении. Плана нет — всё равно сделай ход, по `wm_ontology()`.\n"
)
_WM_REJ = (
    "[МОДЕЛЬ МИРА] Ход отклонён: в коде нет ни action(...), ни wm_execute(...). Ничего не исполнено.\n"
    "Разбор доски без хода не двигает игру и не растит буфер. Модель пишет второй агент,\n"
    "тебе на неё отвлекаться не нужно. Сделай ход. Осталось попыток: %d."
)

def _wm_defs(code):
    """Только определения: вызовы (в том числе action(...)) в хендофф не попадают."""
    try: tree = _wast.parse(code)
    except Exception: return ""
    keep = []
    for node in tree.body:
        if isinstance(node, (_wast.FunctionDef, _wast.ClassDef, _wast.Import,
                             _wast.ImportFrom, _wast.Assign, _wast.AnnAssign)):
            try: seg = _wast.get_source_segment(code, node)
            except Exception: seg = None
            if seg and "action(" not in seg and "wm_execute(" not in seg:
                keep.append(seg)
    src = "\n".join(keep)
    return src if ("def predict" in src and "def state_of" in src) else ""

def _wm_st(self):
    st = getattr(self, "_wm_stats", None)
    if st is None:
        st = self._wm_stats = {"turns": 0, "acted": 0, "rejected": 0, "auto": 0, "admitted": 0,
                               "cex": 0, "reuse": 0, "goal_try": 0, "goal_ok": 0, "levels": 0,
                               "plans": 0, "plans_found": 0, "exec_ok": 0, "exec_bad": 0, "onto": 0,
                               "tail_fail": 0, "synth_run": 0, "synth_ok": 0, "synth_bad": 0,
                               "synth_bad_prog": 0, "too_early": 0, "hor_sum": 0, "hor_n": 0,
                               "hor_last": -1, "also_exact": 0, "exact_only": 0,
                               "synth_err": 0, "synth_acted": 0, "sem_busy": 0, "same_buf": 0,
                               "ch_tool": 0, "ch_markup": 0, "ch_content": 0, "ch_reason": 0, "ch_none": 0}
    return st

def _wm_state_payload(self, state_path):
    frame, hist = _wta.load_runtime_state(state_path)
    return {"current_frame": _wta._ascii_frame_view_payload(frame),
            "history": _wta._ascii_history_view_payload(hist),
            "valid_actions": [str(a) for a in (self._current_valid_actions or [])],
            "last_action_result": {}}

def _wm_offline(self, state_path, code):
    """Песочница БЕЗ права ходить: синтезатор не может сделать ход даже намеренно."""
    st = _wm_st(self)
    def _refuse(actions):
        st["synth_acted"] += 1
        raise RuntimeError("синтезатору ходить нельзя")
    return _wm_sandbox(code=code, timeout_seconds=self._python_timeout,
                       initial_state=_wm_state_payload(self, state_path), action_handler=_refuse)

def _wm_fence(text):
    """Последний блок кода из текста: рассуждающая модель кладёт вывод в конец."""
    if not text: return ""
    blocks = _wre.findall(r"```(?:python)?\s*(.*?)```", text, _wre.S)
    if blocks: return blocks[-1].strip()
    return text.strip() if ("def predict" in text and "def state_of" in text) else ""

def _wm_from_tool_calls(calls):
    for call in (calls or []):
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        if str(fn.get("name", "")).strip() not in ("python", ""): continue
        args = fn.get("arguments", "")
        if isinstance(args, str):
            try: args = _wjson.loads(args or "{}")
            except Exception: args = {}
        if isinstance(args, dict) and args.get("code"):
            return str(args["code"])
    return ""

def _wm_read_reply(self, res):
    """ЧЕТЫРЕ КАНАЛА по убыванию надёжности. Причина двух немых прогонов была ровно здесь:
    читали только `content`, а он у этой модели пуст в 79% ответов."""
    st = _wm_st(self)
    msg = res.message if isinstance(getattr(res, "message", None), dict) else {}
    content = _wta._normalize_message_content(msg.get("content", "") or "")
    try: reasoning = _wta._extract_reasoning_text(msg)
    except Exception: reasoning = ""
    code = _wm_from_tool_calls(msg.get("tool_calls"))
    if code: st["ch_tool"] += 1; return code, "вызов инструмента", content, reasoning
    try: recovered = _wta._recover_tool_calls_from_markup(reasoning, content)
    except Exception: recovered = []
    code = _wm_from_tool_calls(recovered)
    if code: st["ch_markup"] += 1; return code, "разметка вызова в тексте", content, reasoning
    code = _wm_fence(content)
    if code: st["ch_content"] += 1; return code, "content", content, reasoning
    code = _wm_fence(reasoning)
    if code: st["ch_reason"] += 1; return code, "канал рассуждений", content, reasoning
    st["ch_none"] += 1
    return "", "нет кода ни в одном канале", content, reasoning

def _wm_ask(self, digest, cex):
    """Запрос синтезатора: штатный таймаут (НЕ мои 300 с) и один повтор на разрыв соединения."""
    ask = "Наблюдения:\n" + digest
    if cex:
        ask += ("\n\nПрежняя программа опровергнута вот этим переходом — новая обязана "
                "воспроизвести и его, и все прежние:\n" + str(cex)[:600])
    messages = [{"role": "system", "content": _WM_SYNTH_SYS}, {"role": "user", "content": ask}]
    last = None
    for attempt in (1, 2):
        try:
            return self._chat_completion(messages, tools=_WM_TOOLS)
        except Exception as exc:
            last = exc
            if attempt == 1 and ("Connection" in repr(exc) or "RemoteDisconnected" in repr(exc)):
                print("[WM] синтезатор: разрыв соединения, повтор", flush=True)
                continue
            raise
    raise last

def _wm_synth_body(self, state_path):
    st = _wm_st(self)
    try:
        st["synth_run"] += 1
        d = _wm_offline(self, state_path, _WM_HELPERS + "\n" + _WM_DIGEST_CODE)
        digest = str(d.get("stdout", "") or "")[:7000]
        res = _wm_ask(self, digest, getattr(self, "_wm_cex", None))
        code, channel, content, reasoning = _wm_read_reply(self, res)
        if "def predict" not in code or "def state_of" not in code:
            st["synth_bad"] += 1; st["synth_bad_prog"] += 1
            probe = (code or content or reasoning or "")[:300].replace("\n", " ")
            print("[WM] синтезатор: не программа (канал: %s; content %d, рассуждений %d; таких у игры %d из %d) | %s"
                  % (channel, len(content), len(reasoning), st["synth_bad_prog"], _WM_MAX_NOPROG, probe), flush=True)
            return
        out = _wm_offline(self, state_path,
                          _WM_HELPERS + "\n" + code
                          + "\nwm_check(predict, state_of=state_of)\n"
                          + "\nwm_horizon(predict, state_of, max_steps=%d)\n" % _WM_HOR_STEPS)
        so = str(out.get("stdout", "") or "") + str(out.get("error", "") or "")
        m = _wre.search(r"WM_CHECK admitted=(\d) correct=(\d+) checked=(\d+) total=(\d+)", so)
        mh = _wre.search(r"WM_HORIZON med=(-?\d+) n=(\d+) pairs=(\d+)", so)
        med = int(mh.group(1)) if mh else -1
        if med >= 0: st["hor_sum"] += med; st["hor_n"] += 1
        if mh and med >= _WM_HOR_MIN:
            defs = _wm_defs(code)
            if defs:
                self._wm_prog = defs; self._wm_cex = None
                st["synth_ok"] += 1
                if m: self._wm_ntr = int(m.group(4))
                st["hor_last"] = med
                if m and m.group(1) == "1": st["also_exact"] += 1
                print("[WM] СИНТЕЗАТОР: ПРИНЯТА ПО ГОРИЗОНТУ %d из %d (канал: %s; старым правилом "
                      "прошла бы: %s; попыток %d, принято %d)"
                      % (med, _WM_HOR_STEPS, channel, "да" if (m and m.group(1) == "1") else "нет",
                         st["synth_run"], st["synth_ok"]), flush=True)
                return
        st["synth_bad"] += 1
        if m and m.group(1) == "1": st["exact_only"] += 1
        print("[WM] синтезатор: НЕ принята — горизонт %d при пороге %d (старым правилом прошла бы: %s; "
              "канал: %s) | %s"
              % (med, _WM_HOR_MIN, "да" if (m and m.group(1) == "1") else "нет", channel,
                 code[:220].replace("\n", " ")), flush=True)
    except Exception as exc:
        st["synth_err"] += 1
        print("[WM] синтезатор упал (игра продолжается): %r" % (exc,), flush=True)
    finally:
        self._wm_busy = False
        try: _WM_SEM.release()
        except ValueError: pass

def _wm_synth_maybe(self, state_path):
    """Синтез только ПОСЛЕ взятого уровня, когда модели нет или она опровергнута,
    буфер вырос и свободен слот.

    Условие «только после взятого уровня» (v12) снято: оно дало ДВА запуска синтезатора за весь
    прогон, а замер v11 показал, что синтез почти ничего не стоит (312 запросов против 314 у базы).
    Осталось требование к числу наблюдений на уровне — горизонту нужно куда раскатываться."""
    st = _wm_st(self)
    if getattr(self, "_wm_busy", False): return
    if st["synth_run"] >= _WM_SYNTH_CAP: return
    if st["synth_bad_prog"] >= _WM_MAX_NOPROG: return   # игра уже дважды ответила без программы
    nobs = int(getattr(self, "_wm_level_obs", 0))       # наблюдения ТЕКУЩЕГО уровня, не всей игры
    if nobs < _WM_MIN_LEVEL_OBS:
        st["too_early"] += 1; return
    if getattr(self, "_wm_prog", "") and not getattr(self, "_wm_cex", None): return
    if nobs <= int(getattr(self, "_wm_lastobs", -1)):
        st["same_buf"] += 1; return
    if not _WM_SEM.acquire(blocking=False):
        st["sem_busy"] += 1; return
    self._wm_lastobs = nobs
    self._wm_busy = True
    _wthr.Thread(target=_wm_synth_body, args=(self, state_path), daemon=True).start()

_wm_orig_prompt = _wta.ToolAgent._build_user_prompt
def _wm_prompt(self, action_num, *args, **kwargs):
    text = _wm_orig_prompt(self, action_num, *args, **kwargs)
    try:
        hist = kwargs.get("history_entries") or []
        self._wm_nobs = sum(1 for h in hist if str(getattr(h, "action", "") or "").strip())
        lv = getattr(kwargs.get("current_frame"), "level", None)
        prev = getattr(self, "_wm_lvl", None)
        if prev is not None and lv is not None and lv != prev:
            self._wm_rej = 0
            self._wm_level_obs = 0        # буфер уровня начинается заново вместе с доской
            try:
                if int(lv) > int(prev):
                    self._wm_cleared = True; _wm_st(self)["levels"] += 1
            except Exception: pass
        self._wm_lvl = lv
        n = int(getattr(self, "_wm_ntr", 0) or 0)
        cleared = bool(getattr(self, "_wm_cleared", False))
        if not cleared:
            head = _WM_GATHER                      # до первого уровня версия = сток
        elif not getattr(self, "_wm_prog", ""):
            head = _WM_AFTER                       # уровень взят, модель ещё строится
        elif not getattr(self, "_wm_goal_ok", False):
            head = _WM_GOAL % n
        else:
            head = _WM_PLAN % n
        return head + "\n" + text
    except Exception:
        return text
if _WM_LAYER_ON:
    _wta.ToolAgent._build_user_prompt = _wm_prompt

_wm_orig_run = _wta.ToolAgent._run_python_tool
def _wm_run(self, state_path, arguments):
    code = str((arguments or {}).get("code", "") or "")
    st = _wm_st(self); st["turns"] += 1
    acts = ("action(" in code) or ("wm_execute(" in code)
    rej = int(getattr(self, "_wm_rej", 0) or 0)
    if not acts and rej < _WM_REJECTS:
        self._wm_rej = rej + 1; st["rejected"] += 1
        left = _WM_REJECTS - self._wm_rej
        print("[WM] ход без действия отклонён (осталось попыток: %d)" % left, flush=True)
        _wm_synth_maybe(self, state_path)
        return _wta._ToolDispatchResult(content=_WM_REJ % left, step_executed=False)
    if acts:
        st["acted"] += 1
        self._wm_level_obs = int(getattr(self, "_wm_level_obs", 0)) + 1
    if "wm_plan(" in code: st["plans"] += 1
    if "wm_ontology(" in code: st["onto"] += 1
    wrote_goal = bool(_wre.search(r"def\s+goal\s*\(", code))
    if wrote_goal: st["goal_try"] += 1
    prog = getattr(self, "_wm_prog", "") or ""
    inject = prog if (prog and "def predict" not in code and len(prog) < 6000) else ""
    if inject: st["reuse"] += 1
    arguments = dict(arguments)
    arguments["code"] = _WM_HELPERS + "\n" + (inject + "\n" if inject else "") + code + _WM_TAIL
    out = _wm_orig_run(self, state_path, arguments)
    try:
        text = getattr(out, "content", "") or ""
        if "WM_TAIL сбой харнесса" in text: st["tail_fail"] += 1
        if "WM_PLAN found" in text: st["plans_found"] += 1
        if "WM_EXEC ok" in text: st["exec_ok"] += 1
        if "WM_EXEC mismatch" in text: st["exec_bad"] += 1
        m = None
        for m in _wre.finditer(r"WM_CHECK admitted=(\d) correct=(\d+) checked=(\d+) total=(\d+)", text):
            pass
        if m:
            st["auto"] += 1; self._wm_ntr = int(m.group(4))
            if m.group(1) == "1":
                st["admitted"] += 1; self._wm_cex = None
                if wrote_goal:
                    merged = (inject + "\n" + code) if inject else code
                    cand = _wm_defs(merged)
                    if cand: self._wm_prog = cand          # хендофф вместе с проверенной целью
            else:
                cx = _wre.search(r"WM_COUNTEREXAMPLE (.{0,400})", text)
                if cx: self._wm_cex = cx.group(1).strip(); st["cex"] += 1
        g = _wre.search(r"WM_GOAL ok=(\d)", text)
        if g:
            if g.group(1) == "1":
                if not getattr(self, "_wm_goal_ok", False): st["goal_ok"] += 1
                self._wm_goal_ok = True
            else:
                self._wm_goal_ok = False
        print("[WM] ход %d: действие=%s модель=%s цель=%s переходов=%s | действий %d, автопроверок %d, "
              "принятий %d, контрпримеров %d, синтезов %d (принято %d, не программа %d, сбоев %d), "
              "каналы t/m/c/r/нет %d/%d/%d/%d/%d, целей %d/%d, планов %d/%d, исполнений %d/%d, "
              "уровней %d, отклонений %d, слот занят %d, рано %d, горизонт последней %d, "
              "принято и старым правилом %d, прошло бы только старым %d, СБОЕВ ХВОСТА %d"
              % (st["turns"], acts, bool(getattr(self, "_wm_prog", "")), bool(getattr(self, "_wm_goal_ok", False)),
                 getattr(self, "_wm_ntr", 0), st["acted"], st["auto"], st["admitted"], st["cex"],
                 st["synth_run"], st["synth_ok"], st["synth_bad"], st["synth_err"],
                 st["ch_tool"], st["ch_markup"], st["ch_content"], st["ch_reason"], st["ch_none"],
                 st["goal_ok"], st["goal_try"], st["plans_found"], st["plans"], st["exec_ok"], st["exec_bad"],
                 st["levels"], st["rejected"], st["sem_busy"], st["too_early"], st["hor_last"],
                 st["also_exact"], st["exact_only"], st["tail_fail"]), flush=True)
    except Exception as exc:
        print("[WM] учёт не сработал (игнорируем): %r" % (exc,), flush=True)
    _wm_synth_maybe(self, state_path)
    return out
if _WM_LAYER_ON:
    _wta.ToolAgent._run_python_tool = _wm_run
print("wm v13 %s: ПРИЁМ ПО ГОРИЗОНТУ (медиана >= %d из %d), синтез с первого уровня после "
      "%d переходов, не больше %d ответов без программы на игру"
      % ("установлен" if _WM_LAYER_ON else "ВЫКЛЮЧЕН выключателем WM_LAYER=0",
         _WM_HOR_MIN, _WM_HOR_STEPS, _WM_MIN_LEVEL_OBS, _WM_MAX_NOPROG), flush=True)



# --- расписание: снятие игры после простоя ---

# =====================================================================
# РАСПИСАНИЕ: 18 игр одновременно, без потолка на игру, снятие после 45 минут без уровня.
#
# Слот держит тот, кто движется. Игра, не взявшая нового уровня 45 минут (отсчёт от старта
# игры или от последнего уровня), снимается — её место занимает следующая из очереди.
# Модель об этом ничего не знает: снятие делает движок, счёт за него не штрафует, взятые
# уровни сохраняются.
#
# Только оффлайн: в боевой ветке ничего не меняется.
# =====================================================================
import time as _st, datetime as _sd
import inference.framework.solver as _ss

_SCHED_STALL_S = 2700.0     # 45 минут без нового уровня — снятие
_SCHED_CONC    = 18
_SCHED_WINDOW  = 5400.0     # полтора часа на весь прогон

_sched_stats = {"dropped": 0, "started": 0}

_s_orig_limit = _ss._HarnessGameSession.runtime_limit_reached
def _s_limit(self):
    if _s_orig_limit(self):
        return True
    now = _st.monotonic()
    try:
        lv = int(self.game.current_state.levels_completed)
    except Exception:
        return False
    mark = getattr(self, "_s_mark", None)
    if mark is None or lv > mark[0]:
        self._s_mark = (lv, now)              # отсчёт заново от старта и от каждого уровня
        return False
    idle = now - mark[1]
    if idle < _SCHED_STALL_S:
        return False
    gid = getattr(getattr(self.game, "game_run", None), "game_id", "?")
    reason = "sched: снята — %.0f с без нового уровня (уровней %d)" % (idle, lv)
    try:
        run = self.game.game_run
        if run is not None and not run.solver_note:
            run.solver_note = reason
    except Exception:
        pass
    _sched_stats["dropped"] += 1
    print("[SCHED] %s: %s; снято всего %d" % (gid, reason, _sched_stats["dropped"]), flush=True)
    return True
if _WM_SCHED_ON:
    _ss._HarnessGameSession.runtime_limit_reached = _s_limit

print("SCHED %s: одновременно %d игр, потолка на игру НЕТ, снятие после %.0f с без уровня, "
      "окно прогона %.0f с; точка сравнения — база, обрезанная на 90 мин: RHAE 7.78, "
      "первый уровень 21/25, действий на игру 111.2"
      % ("установлено" if _WM_SCHED_ON else "ВЫКЛЮЧЕНО выключателем WM_SCHED=0",
         _SCHED_CONC, _SCHED_STALL_S, _SCHED_WINDOW), flush=True)

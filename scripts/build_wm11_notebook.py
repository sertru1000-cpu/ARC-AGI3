"""Сборка wm v11: два агента, но синтезатор наконец отвечает тем каналом, которым модель говорит.

ЧТО ПОКАЗАЛИ ТРИ ПРОГОНА (все числа измерены, источники в docs/plan_top10_by_3009.md):
  * v10 (09.09): хвост починен, механизм работает — 59 проверок, 11 принятых программ в 7 играх,
    сбоев хвоста 0. Но балл 0.58 при базе-30 3.32: генерация на запрос 1436 (база) -> 1974,
    очередь 122 -> 137 с, ходов на игру 9.0 против ~12.6. Платит не проверка, а то, что программы
    пишет ИГРАЮЩИЙ агент: длинный ответ отнимает ходы у всех игр сразу (8 мест на 25 игр).
  * v8: играющий агент программ не писал, и действий на игру стало 20.2 против 9.7 у v10 —
    архитектура двух агентов сама по себе темп сохраняет.
  * Но синтезатор не дал НИ ОДНОЙ программы: v7 — 38 попыток (11 ответов не программой,
    16 разрывов соединения, 11 таймаутов), v8 — 24 попытки (20 таймаутов по МОЕМУ лимиту 300 с,
    4 разрыва). Цена провала посчитана по счётчикам v8: сервер выработал 547 тыс. токенов,
    на завершённые запросы пришлось 302 тыс. — 245 тыс. сожгли оборванные попытки, ~10 тыс. на каждую.

ПОЧЕМУ ОН МОЛЧАЛ — три причины, и главная не таймаут:
  1. ЧИТАЛИ НЕ ТОТ КАНАЛ. У Flash-Next поле `content` пустое в 180 ответах из 228 в прогоне v10
     (79%), канал рассуждений непустой во всех 228. Синтезатор брал только `content`.
     У проекта это уже было 02.09 с чистильщиком контекста (19 пустых из 23).
  2. НЕ ДАЛИ ИНСТРУМЕНТА. Звали с tools=None и просили блок ```python в тексте, тогда как
     играющий агент в этом харнессе отвечает вызовом инструмента в 100% ходов.
  3. ТАЙМАУТ И ОТСУТСТВИЕ ПОВТОРА. 300 с в v8 (мой), и один POST без ретрая: разрыв соединения
     («Remote end closed connection without response») убивает попытку целиком.

ЧТО ЗДЕСЬ (пять правок, ровно под эти причины):
  1. Синтезатор зовётся С ТЕМ ЖЕ инструментом `python`, что и играющий агент, и код берётся из
     аргументов вызова.
  2. Цепочка чтения ответа: вызов инструмента -> разметка вызова в тексте/рассуждениях
     (`_recover_tool_calls_from_markup` бандла) -> `content` -> канал рассуждений
     (`_extract_reasoning_text`), из последних двух вынимается блок кода.
  3. Штатный таймаут вместо 300 с плюс ОДИН повтор на разрыв соединения.
  4. Ограничитель на 4 синтезатора за прогон — измеренно помог (очередь 197 -> 128 с).
  5. В лог пишется канал ответа и первые 300 символов. Без этого причина не находилась два прогона.

РАЗДЕЛЕНИЕ ТРУДА: `state_of`/`predict` пишет синтезатор (дорого, вне игровых ходов),
`goal` пишет играющий агент одной строкой после взятого уровня (дёшево, и без него план мёртв).
Хвост проверки — из v10: `except Exception` (в белом списке песочницы NameError НЕТ), под стражем.

usage:  .venv/bin/python scripts/build_wm11_notebook.py
"""

import json
import os

S = os.path.dirname(os.path.abspath(__file__))
ns = {}
exec(open(os.path.join(S, "wm_helpers_v9.py"), encoding="utf-8").read(), ns)
HELPERS = ns["WM_HELPERS"]
Q3 = "'" * 3

CELL = r'''
# =====================================================================
# WM v11 — два агента; синтезатор отвечает тем каналом, которым модель говорит.
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
import ast as _wast
import json as _wjson
import re as _wre
import threading as _wthr
import inference.agent.tool_agent as _wta
from inference.agent.python_tool_sandbox import run_sandboxed_python as _wm_sandbox

_WM_HELPERS = @@@HELPERS@@@
_WM_REJECTS   = 2      # отклонений хода без действия на уровень
_WM_SYNTH_CAP = 20     # потолок попыток синтеза на игру
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
            st["synth_bad"] += 1
            probe = (code or content or reasoning or "")[:300].replace("\n", " ")
            print("[WM] синтезатор: не программа (канал: %s; content %d, рассуждений %d) | %s"
                  % (channel, len(content), len(reasoning), probe), flush=True)
            return
        out = _wm_offline(self, state_path,
                          _WM_HELPERS + "\n" + code + "\nwm_check(predict, state_of=state_of)\n")
        so = str(out.get("stdout", "") or "") + str(out.get("error", "") or "")
        m = _wre.search(r"WM_CHECK admitted=(\d) correct=(\d+) checked=(\d+) total=(\d+)", so)
        if m and m.group(1) == "1":
            defs = _wm_defs(code)
            if defs:
                self._wm_prog = defs; self._wm_cex = None
                st["synth_ok"] += 1; self._wm_ntr = int(m.group(4))
                print("[WM] СИНТЕЗАТОР: программа принята на %s переходах (канал: %s; попыток %d, принято %d)"
                      % (m.group(4), channel, st["synth_run"], st["synth_ok"]), flush=True)
                return
        st["synth_bad"] += 1
        print("[WM] синтезатор: программа не принята (канал: %s; %s)"
              % (channel, m.group(0) if m else "проверка не отчиталась"), flush=True)
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

    Условие «после уровня» — из разбора внешнего критика 09.09 и нашей же арифметики:
    до первого уровня модель мира ничего не даёт (цель ещё не наблюдалась, план не включается),
    а токены синтеза вычитаются из игры один к одному. Пока уровень не взят, версия обязана
    быть неотличима от стока — это её безопасный низ. После уровня работа идёт на глубину,
    где у базы за полчаса второй уровень лишь в 3 играх из 25."""
    st = _wm_st(self)
    if not getattr(self, "_wm_cleared", False): return
    if getattr(self, "_wm_busy", False): return
    if st["synth_run"] >= _WM_SYNTH_CAP: return
    nobs = int(getattr(self, "_wm_nobs", 0))
    if nobs < 3: return
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
    if acts: st["acted"] += 1
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
              "уровней %d, отклонений %d, слот занят %d, СБОЕВ ХВОСТА %d"
              % (st["turns"], acts, bool(getattr(self, "_wm_prog", "")), bool(getattr(self, "_wm_goal_ok", False)),
                 getattr(self, "_wm_ntr", 0), st["acted"], st["auto"], st["admitted"], st["cex"],
                 st["synth_run"], st["synth_ok"], st["synth_bad"], st["synth_err"],
                 st["ch_tool"], st["ch_markup"], st["ch_content"], st["ch_reason"], st["ch_none"],
                 st["goal_ok"], st["goal_try"], st["plans_found"], st["plans"], st["exec_ok"], st["exec_bad"],
                 st["levels"], st["rejected"], st["sem_busy"], st["tail_fail"]), flush=True)
    except Exception as exc:
        print("[WM] учёт не сработал (игнорируем): %r" % (exc,), flush=True)
    _wm_synth_maybe(self, state_path)
    return out
_wta.ToolAgent._run_python_tool = _wm_run
print("wm v11 installed: синтезатор отвечает вызовом инструмента, ответ читается из четырёх каналов, "
      "штатный таймаут и повтор на разрыв; играющий агент моделей не пишет, только цель", flush=True)

# Проба на 30 минут: только оффлайн — бой сохраняет 7920 с.
if not TRUE_SUBMISSION:
    bm.solver.max_runtime_s_per_game = 1800.0
    print("WM11 PROBE: потолок 1800 с на игру (только оффлайн); точка сравнения — база-30 по формуле "
          "харнесса: RHAE 3.32, первый уровень в 13 играх из 25", flush=True)
'''

CELL = CELL.replace("@@@HELPERS@@@", "r" + Q3 + HELPERS + Q3)
open(os.path.join(S, os.pardir, "kernels", "notebooks_stockflash_wm11", "cell13.py"), "w",
     encoding="utf-8") if os.path.isdir(os.path.join(S, os.pardir, "kernels", "notebooks_stockflash_wm11")) else None

src = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))
nb = json.loads(json.dumps(src))
c = nb["cells"][13]
code = "".join(c["source"]) + "\n" + CELL
c["source"] = code.splitlines(keepends=True)
os.makedirs("kernels/notebooks_stockflash_wm11", exist_ok=True)
open("kernels/notebooks_stockflash_wm11/cell13.py", "w", encoding="utf-8").write(CELL)
json.dump(nb, open("kernels/notebooks_stockflash_wm11/submission.ipynb", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
m = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
m["id"] = "sergueimakarov/arc3-stock-flash-wm11"
m["title"] = "arc3 stock flash wm11"
json.dump(m, open("kernels/notebooks_stockflash_wm11/kernel-metadata.json", "w"), indent=2)
compile(code, "c13", "exec")
diff = [i for i in range(18) if "".join(nb["cells"][i]["source"]) != "".join(src["cells"][i]["source"])]
print("ok   изменена только ячейка 13:", diff == [13])
print("ok   компилируется, %d символов" % len(code))
print("ok   бой не затронут:", code.count("if not TRUE_SUBMISSION") == 1)

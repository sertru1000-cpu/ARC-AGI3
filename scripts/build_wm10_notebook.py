import json, os
S = os.path.dirname(os.path.abspath(__file__))  # scripts/
ns = {}; exec(open(os.path.join(S, "wm_helpers_v9.py"), encoding="utf-8").read(), ns); HELPERS = ns["WM_HELPERS"]
Q3 = "'" * 3

CELL = r'''
# =====================================================================
# WM v10 — та же v9, но её механизм наконец МОЖЕТ сработать.
#
# ЧТО ПОКАЗАЛ ПРОГОН v9 (runs/flash_wm_v9, разобран 09.09 ночью): механизм не провалился —
# он ни разу не запускался, из-за дефекта в МОЁМ дописанном хвосте. В песочнице Duck белый
# список имён (python_tool_sandbox.py: SAFE_BUILTINS, 53 имени) содержит Exception, RuntimeError,
# TypeError, ValueError — и НЕ содержит NameError. Хвост же ловил `except NameError`, поэтому
# сам обработчик падал: «name 'NameError' is not defined». Итог прогона: модель писала
# state_of/predict в 79 ходах во всех 25 играх, а отчёт проверки не получила НИ РАЗУ
# (автопроверок 0, принятий 0), и вместо отчёта ей возвращался наш traceback — 230 ходов из 294.
# Балл 1.00 против сопоставимой базы-30 3.32 измерил сломанный хвост, а не метод.
#
# ПРАВКИ ЗДЕСЬ, ровно три и все про это:
#  1. `except Exception` вместо `except NameError` (дважды) — воспроизведено и проверено в
#     настоящей песочнице бандла: до правки хвост падает, после — печатает WM_CHECK admitted=1
#     на буфере, выросшем на сделанный ход.
#  2. Весь хвост обёрнут одним стражем: любая будущая ошибка НАШЕГО кода печатает строку
#     «WM_TAIL сбой харнесса», а не traceback в ответ модели.
#  3. Счётчик tail_fail в строке [WM] — чтобы такая поломка была видна в логе сразу.
#     Прошлый раз она выглядела как «модель не пишет программ», хотя модель их писала.
#
# На чём стоит остальное (измерено 08.09, runs/flash_wm_v5, scripts/llm_program_generalization.py):
#  * программы, которые пишет наша модель, ОБОБЩАЮТ: 33 принятые программы v5 проверены на
#    ходах, которых при приёме не существовало — 120 из 148 верно (81%), а на переходах с реально
#    изменившимся состоянием 89 из 103 (86%). Синтезатор-программа без модели давал 24%.
#  * провалы v5 (0.50) и v7 (0.30) — цена интеграции: роли по ходам отбирали действия,
#    параллельный синтезатор топил очередь карты (8 мест на 25 игр);
#  * планировщик ни разу не получал шанса: по статье план включается ТОЛЬКО после первого
#    взятого уровня, а цели у нас не было вовсе.
#
# Что здесь:
#  1. ИНТЕГРАЦИЯ С НУЛЕВОЙ ЦЕНОЙ (v6): каждый ход обязан действовать; принятая программа
#     подставляется харнессом (только определения, вызовы вырезаны разбором синтаксиса);
#     проверка дописывается ПОСЛЕ действий хода — буфер к этому моменту уже вырос, и
#     проверка на нём сама находит контрпример. Лишних запросов к карте нет, отдельных
#     ходов на модель нет.
#  2. НАГРАДА И ЦЕЛЬ (статья, раздел 2 и 3.5): переход, взявший уровень, показывает уже
#     новую доску — его нельзя требовать предсказать. Он проверяется через предикат цели:
#     goal(predict(до, действие)) обязан быть True. После первого взятого уровня агент пишет
#     def goal(state), и wm_goal_check проверяет его тем же точным повтором: попадание на
#     каждой награде, ни одного срабатывания раньше времени.
#  3. ПЛАН НА УРОВНЯХ 2+: с принятой программой и проверенной целью — wm_plan внутри модели
#     (ходов не тратит) и wm_execute по одному шагу с обрывом на первом расхождении. Где
#     это должно платить: база на полной длине доходит до L2+ в 9 играх из 25.
# Отклоняется ровно одно — ход без действия (не больше 2 отклонений на уровень).
# Работает в обеих ветках.
# =====================================================================
import ast as _wast
import re as _wre
import inference.agent.tool_agent as _wta

_WM_HELPERS = @@@HELPERS@@@
_WM_REJECTS = 2
_WM_MINOBS  = 3

_WM_TAIL = (
    "\n# --- автопроверка харнесса: буфер уже вырос на сделанный ход ---\n"
    "# Страж: ошибка НАШЕГО кода не должна возвращаться модели трассировкой (урок v9).\n"
    "try:\n"
    "    try:\n"
    "        _wm_p = predict; _wm_s = state_of\n"
    "    except Exception:\n"          # NameError в песочнице НЕ определён, ловить только Exception
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

_WM_GATHER = (
    "[МОДЕЛЬ МИРА] Ход обязан содержать действие: буфер наблюдений растёт только от ходов.\n"
    "Наблюдений %d, для модели нужно хотя бы %d. Когда наберётся — в ТОМ ЖЕ ходу, вместе с действием,\n"
    "напиши `def state_of(frame)` (включи `wm_level(frame)`) и `def predict(state, action)`.\n"
    "Проверять вручную не надо: харнесс проверяет сам после твоих действий и присылает отчёт.\n"
    "Приём точный — принимается только программа, воспроизводящая КАЖДЫЙ записанный переход;\n"
    "правило пиши на тип объекта (`wm_objects(frame)`, поле `t`), а не на координаты.\n"
)
_WM_KEEP = (
    "[МОДЕЛЬ МИРА] Программа принята: воспроизводит все %d записанных переходов. Она УЖЕ подставлена\n"
    "в начало твоего кода — `state_of` и `predict` определены, переписывать их не нужно; проверка идёт\n"
    "сама после твоих действий. Ход обязан содержать действие. Награды ещё не было, цель неизвестна —\n"
    "ходи туда, где модель знает меньше всего: `wm_ontology()` печатает самые неопределённые строки.\n"
)
_WM_GOAL = (
    "[МОДЕЛЬ МИРА] Уровень взят — награда наблюдалась, теперь цель можно ЗАПИСАТЬ. Программа принята\n"
    "(%d переходов) и подставлена. В ТОМ ЖЕ ходу, вместе с действием, напиши предикат цели:\n"
    "  def goal(state): ...   # True ровно в том состоянии, которое взяло уровень\n"
    "Харнесс проверит его сам тем же точным повтором: goal(predict(до, ход_взявший_уровень)) обязана быть\n"
    "True, а на всех состояниях до взятия — False. Если predict не моделирует, ЧТО происходит при\n"
    "взятии (например, фишка встаёт на цель), допиши это в predict — иначе цель не пройдёт проверку.\n"
    "Отчёт WM_GOAL скажет: попадания и ложные срабатывания.\n"
)
_WM_PLAN = (
    "[МОДЕЛЬ МИРА] Программа принята (%d переходов) и ЦЕЛЬ ПРОВЕРЕНА. `state_of`, `predict`, `goal`\n"
    "подставлены. Планируй и исполняй в одном ходу:\n"
    "  p = wm_plan(predict, goal, state_of=state_of, max_depth=10)\n"
    "  if p['plan']: wm_execute(predict, p['plan'], state_of=state_of)\n"
    "Поиск идёт внутри модели и ходов не тратит; `wm_execute` делает настоящие ходы по одному и\n"
    "обрывает план при первом расхождении. Плана нет (лимит или пути нет) — всё равно сделай ход,\n"
    "по `wm_ontology()`, и подумай, не упущено ли в predict правило, без которого пути нет.\n"
)
_WM_REPAIR = (
    "[МОДЕЛЬ МИРА] Программа опровергнута на живом ходу. Контрпример:\n%s\n"
    "Почини определения В ЭТОМ ЖЕ ходу, вместе с действием: новая версия обязана воспроизвести и этот\n"
    "переход, и все прежние. Чинить, не играя, нельзя.\n"
)
_WM_REJ = (
    "[МОДЕЛЬ МИРА] Ход отклонён: в коде нет ни action(...), ни wm_execute(...). Ничего не исполнено.\n"
    "Разбор доски без хода не двигает игру и не растит буфер; модель проверяется сама. Сделай ход.\n"
    "Осталось попыток: %d."
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
                               "cex": 0, "synth": 0, "reuse": 0, "goal_try": 0, "goal_ok": 0,
                               "plans": 0, "plans_found": 0, "exec_ok": 0, "exec_bad": 0, "onto": 0,
                               "levels": 0, "tail_fail": 0}
    return st

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
        if getattr(self, "_wm_cex", None):
            head = _WM_REPAIR % (self._wm_cex,)
        elif not getattr(self, "_wm_prog", ""):
            head = _WM_GATHER % (int(getattr(self, "_wm_nobs", 0)), _WM_MINOBS)
        elif not getattr(self, "_wm_cleared", False):
            head = _WM_KEEP % n
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
        return _wta._ToolDispatchResult(content=_WM_REJ % left, step_executed=False)
    if acts: st["acted"] += 1
    if "wm_plan(" in code: st["plans"] += 1
    if "wm_ontology(" in code: st["onto"] += 1
    wrote = bool(_wre.search(r"def\s+predict\s*\(", code))
    wrote_goal = bool(_wre.search(r"def\s+goal\s*\(", code))
    if wrote: st["synth"] += 1
    if wrote_goal: st["goal_try"] += 1
    prog = getattr(self, "_wm_prog", "") or ""
    inject = prog if (prog and not wrote and len(prog) < 6000) else ""
    if inject and wrote_goal:
        # агент дописывает только goal: подставляем программу, его goal встанет после и перекроет старую
        pass
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
                merged = (inject + "\n" + code) if inject else code
                cand = _wm_defs(merged) if (wrote or wrote_goal) else ""
                if cand: self._wm_prog = cand          # хендофф: определения, включая goal
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
              "принятий %d, контрпримеров %d, синтезов %d, целей %d/%d, планов %d/%d, "
              "исполнений ок/сбой %d/%d, разведок %d, уровней %d, отклонений %d, СБОЕВ ХВОСТА %d"
              % (st["turns"], acts, bool(getattr(self, "_wm_prog", "")), bool(getattr(self, "_wm_goal_ok", False)),
                 getattr(self, "_wm_ntr", 0), st["acted"], st["auto"], st["admitted"], st["cex"], st["synth"],
                 st["goal_ok"], st["goal_try"], st["plans_found"], st["plans"], st["exec_ok"], st["exec_bad"],
                 st["onto"], st["levels"], st["rejected"], st["tail_fail"]), flush=True)
    except Exception as exc:
        print("[WM] учёт не сработал (игнорируем): %r" % (exc,), flush=True)
    return out
_wta.ToolAgent._run_python_tool = _wm_run
print("wm v10 installed: хвост чинён (except Exception вместо NameError) и под стражем; "
      "каждый ход действует, проверка бесплатна, цель после первой награды, план на уровнях 2+", flush=True)

# Проба на 30 минут: только оффлайн — бой сохраняет 7920 с.
if not TRUE_SUBMISSION:
    bm.solver.max_runtime_s_per_game = 1800.0
    print("WM10 PROBE: потолок 1800 с на игру (только оффлайн); точка сравнения — база-30 по той же "
          "формуле харнесса (потолок 115): RHAE 3.32, первый уровень в 13 играх из 25", flush=True)
'''
CELL = CELL.replace("@@@HELPERS@@@", "r" + Q3 + HELPERS + Q3)
open(os.path.join(S, os.pardir, "kernels", "notebooks_stockflash_wm10", "cell13.py"), "w", encoding="utf-8").write(CELL)
src = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))
nb = json.loads(json.dumps(src)); c = nb["cells"][13]
code = "".join(c["source"]) + "\n" + CELL
c["source"] = code.splitlines(keepends=True)
os.makedirs("kernels/notebooks_stockflash_wm10", exist_ok=True)
json.dump(nb, open("kernels/notebooks_stockflash_wm10/submission.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
m = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
m["id"] = "sergueimakarov/arc3-stock-flash-wm10"; m["title"] = "arc3 stock flash wm10"
json.dump(m, open("kernels/notebooks_stockflash_wm10/kernel-metadata.json", "w"), indent=2)
compile(code, "c13", "exec")
diff = [i for i in range(18) if "".join(nb["cells"][i]["source"]) != "".join(src["cells"][i]["source"])]
print("ok   изменена только ячейка 13:", diff == [13])
print("ok   компилируется, %d символов" % len(code))
print("ok   бой не затронут:", code.count("if not TRUE_SUBMISSION") == 1)

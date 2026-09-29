"""Функции модели живут между вызовами песочницы в пределах игры (nextfork, 27.09; вариант a4 на поде).

Песочница Tufa каждый вызов начинает с чистого листа, и модель переписывает одни и те же функции разбора доски
(24–27% строк кода — повтор внутри игры). Здесь верхнеуровневые функции, импорты (ровно белый список песочницы)
и константы-литералы из кода модели сохраняются и подставляются перед её кодом в следующих вызовах той же игры;
в сообщение добавляется строка со списком сохранённого.
Замер на поде 27.09 (обвязка Скотта, 30 мин, два прогона): повтор кода 14–17% против 24–27%, ответ модели −10–15%,
очередь −8–11%, баллы 2.37 / 2.66 при контролях 2.60 / 1.45; падений ~7% против 3–5% (функции ссылаются на рабочие
переменные прошлого кода — их хранить нельзя).
Ставится из inference/framework/solver.py после AGENTFIX Скотта; выключение: NEXTFORK_PERSIST=0.
"""

import ast as _pmast, sys as _pmsys
import inference.agent.tool_agent as _pmta
_PM_LIMIT = 8000
_PM_SAFE = {"bisect", "collections", "copy", "fractions", "functools", "heapq", "itertools", "json", "math", "operator", "random", "re",
            "statistics", "string"}   # ровно SAFE_MODULES песочницы: чужой импорт в префиксе ронял бы КАЖДЫЙ следующий вызов
_pm_orig_run = _pmta.ToolAgent._run_python_tool
def _pm_store(self, code):
    """Функции, импорты (белый список песочницы) и константы-литералы верхнего уровня — всё, на что опираются функции.
    27.09 v2: в первом прогоне a4 сохранялись только функции -> NameError на deque/Counter/collections и константах (8 из 26 падений)."""
    try:
        tree = _pmast.parse(code)
    except Exception:
        return
    st = self.__dict__.setdefault("_pm_defs", {})
    imp = self.__dict__.setdefault("_pm_imports", {})
    const = self.__dict__.setdefault("_pm_consts", {})
    for n in tree.body:
        src = _pmast.get_source_segment(code, n)
        if not src:
            continue
        if isinstance(n, _pmast.FunctionDef):          # классы песочница запрещает (__build_class__) — их не хранить
            st.pop(n.name, None); st[n.name] = src
        elif isinstance(n, (_pmast.Import, _pmast.ImportFrom)):
            mods = [a.name for a in n.names] if isinstance(n, _pmast.Import) else [n.module or ""]
            if all(m.split(".")[0] in _PM_SAFE for m in mods):
                imp[src] = True
        elif isinstance(n, _pmast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], _pmast.Name):
            try:
                _pmast.literal_eval(n.value)
                const.pop(n.targets[0].id, None); const[n.targets[0].id] = src
            except Exception:
                pass
    while st and sum(len(v) for v in st.values()) > _PM_LIMIT:
        st.pop(next(iter(st)))                                        # старейшее — вон
    while const and sum(len(v) for v in const.values()) > 3000:
        const.pop(next(iter(const)))
def _pm_run(self, state_path, arguments):
    try:
        key = str(getattr(state_path, "parent", state_path))
        if self.__dict__.get("_pm_game") != key:
            self._pm_game = key; self._pm_defs = {}; self._pm_imports = {}; self._pm_consts = {}
        code = str((arguments or {}).get("code", ""))
        parts = list(self.__dict__.get("_pm_imports", {})) + list(self.__dict__.get("_pm_consts", {}).values()) + \
                list(self.__dict__.get("_pm_defs", {}).values())
        prefix = ("\n".join(parts) + "\n\n") if parts else ""
        res = _pm_orig_run(self, state_path, dict(arguments or {}, code=prefix + code))
        _pm_store(self, code)
        return res
    except Exception as exc:
        print("PERSIST: сбой, вызов без слоя: %r" % (exc,), file=_pmsys.__stderr__, flush=True)
        return _pm_orig_run(self, state_path, arguments)
_pmta.ToolAgent._run_python_tool = _pm_run
_pm_orig_prompt = _pmta.ToolAgent._build_user_prompt
def _pm_prompt(self, action_num, **kw):
    text = _pm_orig_prompt(self, action_num, **kw)
    names = list(self.__dict__.get("_pm_defs", {}))
    return text + ("\nFunctions you define in `python` are kept for the rest of this game and are predefined in "
                   "every later call (redefine one to change it). Currently kept: %s." % (", ".join(names[-20:]) or "none"))
_pmta.ToolAgent._build_user_prompt = _pm_prompt
print("NEXTFORK PERSIST: определения модели сохраняются между вызовами (до %d знаков)" % _PM_LIMIT, flush=True)

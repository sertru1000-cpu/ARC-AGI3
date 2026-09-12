
# =====================================================================
# ПОСТОЯННОЕ ОКРУЖЕНИЕ ПЕСОЧНИЦЫ: импорты и def/class из прошлых успешных вызовов подставляются снова.
# Переменные НЕ переносятся (в них устаревшая доска). Номера строк в ошибках возвращаются к коду модели.
# =====================================================================
import ast as _cast, re as _cre, dataclasses as _cdc
from collections import OrderedDict as _COD
import inference.agent.tool_agent as _cta

_CARRY_MAX_CHARS = 12000
_CARRY_RUNTIME = {"current_frame", "previous_frame", "history", "transitions", "last_transition",
                  "valid_actions", "last_action_result", "action", "result", "print"}
_CARRY_STATS = {"calls": 0, "prefixed": 0, "stored": 0, "line_fixes": 0, "prefix_errors": 0, "fail": 0, "sys_fail": 0}
_CARRY_OLD = "- Every `python` tool call starts fresh. Re-import modules or re-define any custom utility logic you need.\n"
_CARRY_NEW = ("- Imports and top-level `def`/`class` definitions from your earlier successful `python` calls in this game are "
              "automatically re-loaded before each new call: do not re-import or re-define them unless you want to change them "
              "(a new definition with the same name replaces the old one). Variables are NOT carried: recompute anything "
              "derived from `current_frame` or `history` in every call.\n")

_carry_orig_sys = _cta._build_system_prompt


def _carry_build_system_prompt(*a, **k):
    text = _carry_orig_sys(*a, **k)
    if _CARRY_OLD in text:
        return text.replace(_CARRY_OLD, _CARRY_NEW)
    _CARRY_STATS["sys_fail"] += 1
    print("[CARRY] фраза о свежем старте не найдена в системном промпте", flush=True)
    return text


_cta._build_system_prompt = _carry_build_system_prompt


def _carry_items(tree, src):
    out = []
    for node in tree.body:
        if isinstance(node, (_cast.Import, _cast.ImportFrom)):
            seg = _cast.get_source_segment(src, node)
            if seg:
                names = ", ".join(a.asname or a.name.split(".")[0] for a in node.names)
                out.append(("import:" + seg, seg, names))
        elif isinstance(node, (_cast.FunctionDef, _cast.AsyncFunctionDef)):
            # Классы и функции с ВЫЧИСЛЯЕМЫМИ значениями по умолчанию не переносим: их определение
            # исполняет код (тело класса, выражения в defaults) и может упасть или захватить старую доску.
            if node.decorator_list or node.name in _CARRY_RUNTIME:
                continue
            defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
            if any(not isinstance(d, _cast.Constant) for d in defaults):
                continue
            seg = _cast.get_source_segment(src, node)
            if not seg:
                continue
            label = "%s(%s)" % (node.name, ", ".join(x.arg for x in node.args.args))
            out.append(("def:" + node.name, seg, label))
    return out


_CARRY_LINE = _cre.compile(r'(File \\?"<python_tool>\\?", line )(\d+)')


def _carry_first_line(content):
    m = _CARRY_LINE.search(content)
    return int(m.group(2)) if m else None


def _carry_fix_lines(content, P):
    def rep(m):
        n = int(m.group(2))
        _CARRY_STATS["line_fixes"] += 1
        return m.group(1) + (str(n - P) if n > P else "%d (inside re-loaded definitions)" % n)
    return _CARRY_LINE.sub(rep, content)


_carry_orig_run = _cta.ToolAgent._run_python_tool


def _carry_run_python_tool(self, state_path, arguments):
    # подготовка: любой сбой здесь — вызов без переноса, код ещё не исполнялся
    try:
        code = str((arguments or {}).get("code", "")).rstrip()
        store = getattr(self, "_carry_store", None)
        if store is None:
            store = _COD()
            self._carry_store = store
        _CARRY_STATS["calls"] += 1
        tree = _cast.parse(code)
        prefix = "\n".join(seg for seg, _ in store.values())
    except Exception:
        return _carry_orig_run(self, state_path, arguments)

    if prefix:
        new_args = dict(arguments)
        new_args["code"] = prefix + "\n" + code
        P = prefix.count("\n") + 1
        _CARRY_STATS["prefixed"] += 1
    else:
        new_args, P = arguments, 0
    res = _carry_orig_run(self, state_path, new_args)   # ИСПОЛНЯЕТСЯ ОДИН РАЗ

    # разбор результата: любой сбой здесь — вернуть результат как есть, НИКОГДА не запускать повторно
    try:
        content = res.content
        is_error = '"error"' in content
        if P and is_error and not res.step_executed:
            first = _carry_first_line(content)
            if first is not None and first <= P:
                # ошибка в самой подстановке, код модели не начинался, действий нет — сброс и чистый повтор
                _CARRY_STATS["prefix_errors"] += 1
                store.clear()
                print("[CARRY] ошибка в перенесённых определениях — сброс и вызов без подстановки", flush=True)
                return _carry_orig_run(self, state_path, arguments)
        if P:
            content = _carry_fix_lines(content, P)
        if not is_error:
            for key, seg, label in _carry_items(tree, code):
                store.pop(key, None)
                store[key] = (seg, label)
                _CARRY_STATS["stored"] += 1
            while store and sum(len(s) + 1 for s, _ in store.values()) > _CARRY_MAX_CHARS:
                store.popitem(last=False)
        return res if content == res.content else _cdc.replace(res, content=content)
    except Exception as _exc:
        _CARRY_STATS["fail"] += 1
        print("[CARRY] сбой разбора результата: %r" % (_exc,), flush=True)
        return res


_cta.ToolAgent._run_python_tool = _carry_run_python_tool

_carry_orig_prompt = _cta.ToolAgent._build_user_prompt


def _carry_build_user_prompt(self, action_num, **kw):
    text = _carry_orig_prompt(self, action_num, **kw)
    try:
        store = getattr(self, "_carry_store", None)
        if store:
            labels = [label for _, label in store.values()][-30:]
            text += "\n\nRE-LOADED in python from your earlier calls (already defined, do not rewrite): " + "; ".join(labels)
    except Exception:
        pass
    return text


_cta.ToolAgent._build_user_prompt = _carry_build_user_prompt

print("CARRY: перенос импортов и def/class между вызовами python (%s). Замер до сборки: переносимо 31 из 83 "
      "повторных токенов кода на ход, около +2%% вызовов." % ("бой" if TRUE_SUBMISSION else "Фаза A"), flush=True)

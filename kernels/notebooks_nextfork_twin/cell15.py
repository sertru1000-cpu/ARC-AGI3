
# =====================================================================
# TWIN: исполняемая модель мира с реплей-проверкой и планом (26.09). Ячейкой на датасет v3.
# =====================================================================
import ast as _twast, sys as _twsys
import inference.agent.tool_agent as _twta

_TW_HELPERS = '# --- Twin-помощники песочницы (26.09): исполняемая модель мира с реплей-проверкой и планом.\n# По статьям Twin (arXiv:2608.14490) и Tycho (arXiv:2607.28287): модель пишет step(grid, action) -> grid,\n# ход по плану делается только когда step воспроизводит ВСЕ наблюдённые переходы уровня; расхождения\n# возвращаются модели контрпримерами; поиск пути идёт внутри симулятора, бесплатно по ходам.\n# Только безопасные встроенные; ни одного импорта.\n\ndef parse_action(a):\n    """\'UP\' -> (\'UP\', None, None); \'MOUSE(row=3, col=5)\' -> (\'MOUSE\', 3, 5)."""\n    s = str(a).strip()\n    if s.startswith("MOUSE"):\n        nums = []\n        cur = ""\n        for ch in s:\n            if ch.isdigit():\n                cur += ch\n            elif cur:\n                nums.append(int(cur)); cur = ""\n        if cur:\n            nums.append(int(cur))\n        return ("MOUSE", nums[0] if nums else None, nums[1] if len(nums) > 1 else None)\n    return (s, None, None)\n\n\ndef _tw_grid(fr):\n    g = getattr(fr, "_grid", None)\n    return [list(r) for r in (g or [])]\n\n\ndef level_transitions(level=None):\n    """Наблюдённые переходы текущего уровня: список (before_grid, action_str, after_grid)."""\n    lv = current_frame.level if level is None else level\n    out = []\n    for t in transitions:\n        b, a = t.before_frame, t.after_frame\n        if b is None or a is None or b.level != lv or a.level != lv:\n            continue\n        out.append((_tw_grid(b), str(t.action), _tw_grid(a)))\n    return out\n\n\ndef validate(step, level=None, show=3):\n    """Реплей: прогнать step по всем переходам уровня. Печатает итог и первые контрпримеры\n    (ход, сколько клеток не сошлось, первые клетки r,c: предсказано -> наблюдено).\n    Возвращает число расхождений (0 = симулятор воспроизводит всё наблюдённое)."""\n    ts = level_transitions(level)\n    bad = 0\n    shown = 0\n    for i, (b, a, after) in enumerate(ts):\n        try:\n            pred = step([row[:] for row in b], a)\n        except Exception as e:\n            bad += 1\n            if shown < show:\n                print("  counterexample #%d %s: step raised %r" % (i, a, e)); shown += 1\n            continue\n        if pred != after:\n            bad += 1\n            if shown < show:\n                cells = []\n                for r in range(min(len(pred or []), len(after))):\n                    for c in range(min(len(pred[r]), len(after[r]))):\n                        if pred[r][c] != after[r][c]:\n                            cells.append((r, c, pred[r][c], after[r][c]))\n                print("  counterexample #%d %s: %d cells differ, e.g. %s (r,c,predicted,observed)"\n                      % (i, a, len(cells), cells[:6])); shown += 1\n    print("validate: %d/%d transitions reproduced" % (len(ts) - bad, len(ts)))\n    return bad\n\n\ndef plan(step, goal, actions=None, max_depth=60, max_nodes=60000):\n    """Поиск в ширину ВНУТРИ симулятора от текущей доски до goal(grid)==True.\n    actions — список строк ходов; по умолчанию клавиатурные из valid_actions (клики передавайте явно,\n    например [\'MOUSE(row=10, col=20)\', ...]). Возвращает список ходов или None."""\n    if actions is None:\n        actions = [a for a in valid_actions if not str(a).startswith("MOUSE")]\n    start = _tw_grid(current_frame)\n    key = lambda g: tuple(tuple(r) for r in g)\n    seen = {key(start)}\n    frontier = [(start, [])]\n    nodes = 0\n    for depth in range(max_depth):\n        nxt = []\n        for g, path in frontier:\n            for a in actions:\n                nodes += 1\n                if nodes > max_nodes:\n                    print("plan: node budget %d exhausted at depth %d" % (max_nodes, depth)); return None\n                try:\n                    g2 = step([row[:] for row in g], a)\n                except Exception:\n                    continue\n                k = key(g2)\n                if k in seen:\n                    continue\n                seen.add(k)\n                p2 = path + [a]\n                try:\n                    if goal(g2):\n                        print("plan: found %d moves, %d nodes" % (len(p2), nodes)); return p2\n                except Exception:\n                    pass\n                nxt.append((g2, p2))\n        frontier = nxt\n        if not frontier:\n            break\n    print("plan: no path within depth %d (%d nodes)" % (max_depth, nodes))\n    return None\n\n\ndef to_actions(path):\n    """Путь из plan() -> аргумент для action([...])."""\n    out = []\n    for a in path or []:\n        n, r, c = parse_action(a)\n        out.append({"action": "MOUSE", "row": r, "col": c} if n == "MOUSE" else {"action": n})\n    return out\n# --- конец Twin-помощников ---\n'

_TW_BLOCK = (
    "\n"
    "Executable world model (do this once the basic effects of actions are known):\n"
    "- Write `def step(grid, action):` returning the next grid (list of lists of ints) for an action string "
    "('UP', 'DOWN', 'LEFT', 'RIGHT', 'SPACE', or 'MOUSE(row=R, col=C)'). Use `parse_action(a)` for clicks.\n"
    "- Call `validate(step)`: it replays EVERY observed transition of this level through your step and prints "
    "counterexamples (cells predicted vs observed). Fix step until it prints 0 mismatches. Do not trust a plan "
    "from a step that does not reproduce all observations.\n"
    "- Write `def goal(grid):` -> True when the level would be cleared (your current goal hypothesis).\n"
    "- With 0 mismatches, call `path = plan(step, goal)` (breadth-first search inside your simulator; pass "
    "`actions=[...]` to include specific clicks) and execute the whole path at once: `action(to_actions(path))`.\n"
    "- `step`, `goal` and any helper functions you define are kept by the harness for later calls in this game: "
    "redefine them only to change them. Helpers available: validate, plan, to_actions, level_transitions, parse_action.\n"
    "- If the plan's moves do not produce the predicted result, the new transitions become counterexamples: "
    "revalidate and repair step, or revise goal.\n"
)
_tw_orig_prompt = _twta._build_system_prompt


def _tw_build_system_prompt(**kw):
    try:
        text = _tw_orig_prompt(**kw)
        return text if "Executable world model" in text else text + _TW_BLOCK
    except Exception as exc:
        print("TWIN: сбой промпта: %r" % (exc,), flush=True)
        return _tw_orig_prompt(**kw)


_tw_orig_run = _twta.ToolAgent._run_python_tool


def _tw_store(self, code):
    """Вынуть верхнеуровневые def из кода модели, если среди них есть step или goal."""
    try:
        tree = _twast.parse(code)
    except Exception:
        return
    defs = [n for n in tree.body if isinstance(n, _twast.FunctionDef)]
    if not any(n.name in ("step", "goal") for n in defs):
        return
    store = self.__dict__.setdefault("_tw_defs", {})
    for n in defs:
        src = _twast.get_source_segment(code, n)
        if src:
            store[n.name] = src


def _tw_run(self, state_path, arguments):
    try:
        key = str(getattr(state_path, "parent", state_path))
        if self.__dict__.get("_tw_game") != key:           # новая игра — забыть прежние функции
            self._tw_game = key; self._tw_defs = {}
        code = str((arguments or {}).get("code", ""))
        prefix = _TW_HELPERS + "\n" + "\n\n".join(self.__dict__.get("_tw_defs", {}).values()) + "\n"
        res = _tw_orig_run(self, state_path, dict(arguments or {}, code=prefix + code))
        _tw_store(self, code)
        if ("validate(" in code) or ("plan(" in code):
            print("TWIN: вызов с %s | сохранено функций: %s" % (
                ",".join(k for k in ("validate", "plan") if k + "(" in code),
                ",".join(sorted(self.__dict__.get("_tw_defs", {})))), file=_twsys.__stderr__, flush=True)
        return res
    except Exception as exc:
        print("TWIN: сбой слоя, вызов без него: %r" % (exc,), file=_twsys.__stderr__, flush=True)
        return _tw_orig_run(self, state_path, arguments)


_twta._build_system_prompt = _tw_build_system_prompt
_twta.ToolAgent._run_python_tool = _tw_run
print("TWIN: слой включён — помощники %d строк, блок +%d знаков, поверх датасета v3" % (
    len(_TW_HELPERS.splitlines()), len(_TW_BLOCK)), flush=True)

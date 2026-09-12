"""Проверка помощников v5 в НАСТОЯЩЕЙ песочнице бандла (отдельный процесс, SAFE_MODULES)."""
import importlib.util, sys
from pathlib import Path
ROOT = Path('/Users/sergeimakarov/Projects/ARC-AGI-3')
ARC3 = ROOT / "atlas_src" / "src" / "ARC3-Inference"
sys.path.insert(0, str(ARC3))
spec = importlib.util.spec_from_file_location("pts", ARC3 / "inference/agent/python_tool_sandbox.py")
pts = importlib.util.module_from_spec(spec); spec.loader.exec_module(pts)
run = pts.run_sandboxed_python
ns = {}; exec(open(Path(__file__).parent / "wm_helpers_v9.py").read(), ns); H = ns["WM_HELPERS"]

def grid(col, level_marker=0):
    g = [[0]*8 for _ in range(4)]
    g[1][col] = 9
    g[3][7] = level_marker
    return g

def fp(g, step, level=1):
    return {"ascii": "\n".join("".join(str(c) for c in r) for r in g), "step": step,
            "level": level, "shape": [len(g), len(g[0])], "grid": g}

def state(cols, level=1):
    hist = [{"action": "", "frame": fp(grid(cols[0]), 0, level)}]
    for i, c in enumerate(cols[1:], 1):
        hist.append({"action": "RIGHT", "frame": fp(grid(c), i, level)})
    return {"current_frame": fp(grid(cols[-1]), len(cols)-1, level), "history": hist,
            "valid_actions": ["RIGHT", "LEFT"], "last_action_result": {}}

ACTED = []
def go(code, st):
    ACTED.clear()
    def act(actions):
        ACTED.extend(actions); return {"ok": True}
    r = run(code=H + "\n" + code, timeout_seconds=30, initial_state=st, action_handler=act)
    return (r.get("stdout") or "") + str(r.get("error") or "")

ok = lambda b, m: print(("ok   " if b else "СБОЙ ") + m)
ST = state([1, 2, 3, 4])

# 1. ТОЧНЫЙ приём: верная модель принимается
out = go("""
def state_of(f):
    row = f.ascii.split('\\n')[1]
    return (wm_level(f), row.index('9') if '9' in row else -1)
def predict(s, a):
    if a == 'RIGHT': return (s[0], s[1] + 1)
    if a == 'LEFT':  return (s[0], s[1] - 1)
    return None
wm_check(predict, state_of=state_of)
""", ST)
ok("admitted=1" in out and "total=3" in out, "верная модель принята точным приёмом (все 3 перехода)")

# 2. частично верная модель НЕ принимается, и назван контрпример
out = go("""
def state_of(f):
    row = f.ascii.split('\\n')[1]
    return (wm_level(f), row.index('9') if '9' in row else -1)
def predict(s, a):
    if s[1] >= 3: return (s[0], s[1])       # ошибка ровно на последнем переходе
    return (s[0], s[1] + 1)
wm_check(predict, state_of=state_of)
""", ST)
ok("admitted=0" in out and "WM_COUNTEREXAMPLE" in out and "'action': 'RIGHT'" in out,
   "частичная модель отвергнута, контрпример назван конкретно")

# 3. модель со скрытым состоянием отвергается двойным прогоном
out = go("""
_n = [0]
def state_of(f):
    row = f.ascii.split('\\n')[1]
    return (wm_level(f), row.index('9') if '9' in row else -1)
def predict(s, a):
    _n[0] += 1
    return (s[0], s[1] + _n[0] % 2)
wm_check(predict, state_of=state_of)
""", ST)
ok("unstable=" in out and "WM_UNSTABLE" in out and "admitted=0" in out,
   "модель со скрытым состоянием отвергнута двойным прогоном")

# 4. непокрытое действие -> отказ с названием непокрытого
out = go("""
def state_of(f):
    row = f.ascii.split('\\n')[1]
    return (wm_level(f), row.index('9') if '9' in row else -1)
def predict(s, a):
    return None
wm_check(predict, state_of=state_of)
""", ST)
ok("admitted=0" in out and "not_covered=[('RIGHT', 3)]" in out, "непокрытые переходы названы и приём отклонён")

# 5. онтологическая ошибка считается и называет строки
out = go("wm_ontology()", ST)
ok("WM_ONTOLOGY eta=" in out and "неопределённость" in out, "онтологическая ошибка посчитана, строки названы")

# 6. планировщик находит путь и НЕ делает настоящих ходов
out = go("""
def state_of(f):
    row = f.ascii.split('\\n')[1]
    return (wm_level(f), row.index('9') if '9' in row else -1)
def predict(s, a):
    if a == 'RIGHT': return (s[0], s[1] + 1)
    if a == 'LEFT':  return (s[0], s[1] - 1)
p = wm_plan(predict, goal=lambda s: s[1] == 7, state_of=state_of)
print('шагов после планирования:', current_frame.step)
""", ST)
ok("WM_PLAN found len=3" in out and not ACTED,
   "план найден внутри модели, настоящих ходов не сделано (вызовов action: %d)" % len(ACTED))

# 7. оффлайновая проверка планировщика на уже взятом уровне
out = go("""
def state_of(f):
    row = f.ascii.split('\\n')[1]
    return (wm_level(f), row.index('9') if '9' in row else -1)
def predict(s, a):
    if a == 'RIGHT': return (s[0], s[1] + 1)
    if a == 'LEFT':  return (s[0], s[1] - 1)
wm_validate_plan(predict, goal=lambda s: s[1] == 6, state_of=state_of, level=1)
""", ST)
ok("WM_VALIDATE level=1 ok=1" in out, "планировщик проверен оффлайн из входного состояния взятого уровня")

# 8. буфер пуст -> честный отказ, а не пустая проверка
out = go("""
def state_of(f): return 0
def predict(s, a): return 0
wm_check(predict, state_of=state_of)
""", {"current_frame": fp(grid(1), 0), "history": [{"action": "", "frame": fp(grid(1), 0)}],
      "valid_actions": ["RIGHT"], "last_action_result": {}})
ok("total=0" in out and "admitted=0" in out and "переходов ещё нет" in out,
   "при пустом буфере проверка честно говорит, что наблюдений нет")

# --- 9: буфер отфильтрован по ТЕКУЩЕМУ уровню (правка 09.09) ----------------------
# Зачем: доска на новом уровне другая, и одна программа не может воспроизвести переходы
# двух досок. Без фильтра точный приём после взятого уровня недостижим в принципе.
def state_two_levels():
    """Три перехода на уровне 1, переход, взявший уровень, и два перехода на уровне 2."""
    hist = [{"action": "", "frame": fp(grid(1), 0, 1)}]
    for i, c in enumerate([2, 3, 4], 1):
        hist.append({"action": "RIGHT", "frame": fp(grid(c), i, 1)})
    hist.append({"action": "RIGHT", "frame": fp(grid(1), 4, 2)})      # уровень взят
    for i, c in enumerate([2, 3], 5):
        hist.append({"action": "RIGHT", "frame": fp(grid(c), i, 2)})
    return {"current_frame": hist[-1]["frame"], "history": hist,
            "valid_actions": ["RIGHT", "LEFT"], "last_action_result": {}}

ST2 = state_two_levels()
out = go("print('ВСЕГО ПЕРЕХОДОВ:', len(_wm_pairs(400)))\n"
         "print('УРОВНИ ДО:', [wm_level(b) for b, _a, _f in _wm_pairs(400)])", ST2)
ok("ВСЕГО ПЕРЕХОДОВ: 3" in out,
   "на уровне 2 в буфере только его переходы плюс взявший уровень (3 из 6): %s" %
   out.strip().splitlines()[0])
ok("УРОВНИ ДО: [1, 2, 2]" in out,
   "переход, ВОШЕДШИЙ в уровень, сохранён — на нём проверяется цель")

out = go("""
def state_of(f):
    row = f.ascii.split('\\n')[1]
    return (wm_level(f), row.index('9') if '9' in row else -1)
def predict(s, a):
    if a == 'RIGHT': return (s[0], s[1] + 1)
    if a == 'LEFT':  return (s[0], s[1] - 1)
def goal(s): return s[1] >= 4
wm_check(predict, state_of=state_of, goal=goal)
""", ST2)
ok("admitted=1" in out,
   "программа уровня 2 принимается: переходы прошлой доски приём больше не ломают")

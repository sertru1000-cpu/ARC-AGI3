"""Синтетическая проверка протокола разведки (solver._run_start_probe), 26.09.

Гоняется настоящий метод с подставным движком. Проверяем:
  1. каждое действие кроме мыши нажато по разу;
  2. молчавшие нажаты второй раз, сработавшие — нет;
  3. мышь (ACTION6) не трогается;
  4. взятие уровня останавливает протокол немедленно;
  5. засев Action model доезжает до анализатора и переживает _ensure_session;
  6. NEXTFORK_START_PROBE=0 отключает протокол целиком.
usage: .venv/bin/python scripts/test_start_probe.py
"""
from __future__ import annotations
import os, sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nextfork/src/ARC3-Inference"))
sys.path.insert(0, str(ROOT / "nextfork/src/tufa-arc-agi-framework/src"))
import inference.framework.solver as S              # noqa: E402
import inference.agent.tool_agent as TA             # noqa: E402
from inference.agent.runtime_state import RUNTIME_STATE_FILENAME  # noqa: E402

TMP = ROOT / ".tmp_startprobe"; TMP.mkdir(exist_ok=True)
STATE = TMP / RUNTIME_STATE_FILENAME; STATE.write_text("{}")

class RawState:  state = None
class CurState:
    def __init__(self, ids): self.available_actions = ids; self.raw = RawState()
class Game:
    def __init__(self, ids): self.current_state = CurState(ids)

class Engine:
    """Подставной движок: помнит нажатия; effects[name] = список board_changed по нажатиям;
    level_at = имя, нажатие которого завершает уровень."""
    def __init__(self, effects, level_at=None):
        self.effects, self.level_at, self.seen = effects, level_at, []
    def __call__(self, args):
        name = args["actions"][0]["action"]; self.seen.append(name)
        k = sum(1 for x in self.seen if x == name) - 1
        ch = self.effects.get(name, [False])[min(k, len(self.effects.get(name, [False])) - 1)]
        return {"executed": True, "board_changed": ch, "level_completed": name == self.level_at,
                "game_over": False, "run_complete": False}

def make_solver(engine, ids, analyzer):
    sv = S._HarnessGameSession.__new__(S._HarnessGameSession)   # класс с play() и step_env
    sv.game = Game(ids); sv.step_env = engine; sv.should_stop = lambda: False
    sv.analyzer = analyzer; sv.state_path = STATE
    return sv

def make_agent():
    a = TA.ToolAgent.__new__(TA.ToolAgent)
    a._session_runtime_dir = None; a._history_messages = []; a._session_total_tokens = 0
    a._session_generated_tokens = 0; a._last_step_summary = None; a._last_action_result = None
    a._summarized_knowledge = TA._empty_world_model(); a._noop_guard = TA.NoopGuard(); a._noop_guard_blocked = 0
    return a

ok = True
def check(label, cond):
    global ok; ok &= bool(cond); print("%s %s" % ("ОК  " if cond else "ПРОВАЛ", label))

os.environ["NEXTFORK_START_PROBE"] = "1"      # протокол выключен по умолчанию с 26.09
# 1-3: UP работает, DOWN молчит всегда, LEFT срабатывает со второго, мышь есть в списке
eng = Engine({"ACTION1": [True], "ACTION2": [False, False], "ACTION3": [False, True]})
sv = make_solver(eng, [1, 2, 3, 6], make_agent()); sv._run_start_probe()
check("1. каждое действие по разу: %s" % eng.seen[:3], eng.seen[:3] == ["ACTION1", "ACTION2", "ACTION3"])
check("2. молчавшие нажаты повторно, сработавшее нет: %s" % eng.seen[3:], eng.seen[3:] == ["ACTION2", "ACTION3"])
check("3. мышь не тронута", "ACTION6" not in eng.seen)
am = sv.analyzer._summarized_knowledge["action_model"]
check("5. засев доехал: %r" % am[:70], "Harness probe" in am and "LEFT: changed the board on press 2 of 2" in am
      and "DOWN: no visible change in 2" in am)
sv.analyzer._ensure_session(STATE)
check("5b. засев пережил _ensure_session", sv.analyzer._summarized_knowledge["action_model"] == am)

# 4: второе действие берёт уровень — протокол должен остановиться
eng = Engine({"ACTION1": [False], "ACTION2": [True], "ACTION3": [False]}, level_at="ACTION2")
sv = make_solver(eng, [1, 2, 3], make_agent()); sv._run_start_probe()
check("4. стоп при взятии уровня: %s" % eng.seen, eng.seen == ["ACTION1", "ACTION2"])

# 6: по умолчанию выключен
os.environ.pop("NEXTFORK_START_PROBE", None)
eng = Engine({"ACTION1": [True]}); sv = make_solver(eng, [1], make_agent()); sv._run_start_probe()
check("6. по умолчанию выключен: ходов %d (ожидалось 0)" % len(eng.seen), eng.seen == [])
print("\nИТОГ:", "все проверки пройдены" if ok else "ЕСТЬ ПРОВАЛЫ"); sys.exit(0 if ok else 1)

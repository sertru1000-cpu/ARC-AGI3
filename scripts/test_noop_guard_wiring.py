"""Проверка запрета повторного пустого хода в сборке nextfork (25.09).

Гоняем НАСТОЯЩИЙ путь обвязки — ToolAgent._run_python_tool -> _handle_action —
с подставным движком вместо игры. Правило: правил в промпте не выдумываем,
проверяем то, что реально исполнится в бою.

Четыре случая:
  1. пустой ход при той же доске повторно В ДВИЖОК НЕ УХОДИТ;
  2. та же доска, но ДРУГОЙ ход — уходит;
  3. доска изменилась — прежний запрет не действует;
  4. ход вернул несколько кадров (анимация) — пустым не считается, не блокируется.

usage: .venv/bin/python scripts/test_noop_guard_wiring.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nextfork/src/ARC3-Inference"))

import os as _os                                               # noqa: E402
if _os.environ.get("NOOP_TEST_AGENTFIX") == "1":              # 26.09: проверить запрет поверх AGENTFIX Скотта
    sys.path.insert(0, str(ROOT / "nextfork/src/tufa-arc-agi-framework/src"))
    _os.environ.setdefault("MULTIMODAL_UPSCALE", "4"); _os.environ.setdefault("MULTIMODAL_CONTEXT", "current_grid")
    import inference.framework.solver  # noqa: F401 — AGENTFIX ставится при импорте solver.py (с 26.09 модуль обвязки)
import inference.agent.tool_agent as ta                       # noqa: E402
from inference.agent.runtime_state import RUNTIME_STATE_FILENAME  # noqa: E402

TMP = ROOT / ".tmp_noopguard"
TMP.mkdir(exist_ok=True)
STATE = TMP / RUNTIME_STATE_FILENAME

BOARD_A = [[0] * 8 for _ in range(8)]
BOARD_B = [[1] * 8 for _ in range(8)]


def write_state(grid, level=1, step=0):
    STATE.write_text(json.dumps({
        "current_frame": {"grid": grid, "step": step, "level": level},
        "history": [],
    }), encoding="utf-8")


class Engine:
    """Подставной движок: помнит, какие ходы до него дошли."""

    def __init__(self):
        self.seen: list[str] = []
        self.next_board = BOARD_A
        self.next_changed = False
        self.next_frames = 1

    def __call__(self, request):
        name = request["actions"][0]["action"]
        self.seen.append(name)
        write_state(self.next_board)
        return {
            "executed": True, "action_num": len(self.seen), "level": 1, "score": 0,
            "reward": 0.0, "state": "NOT_FINISHED", "valid_actions": ["UP", "DOWN"],
            "board_changed": self.next_changed, "frame_count": self.next_frames,
            "done": False, "level_completed": False, "game_over": False,
            "run_complete": False, "action_display": name,
        }


def make_agent(engine):
    agent = ta.ToolAgent.__new__(ta.ToolAgent)          # без сети и конфигов
    agent._session_runtime_dir = TMP
    agent._history_messages = []
    agent._session_total_tokens = 0
    agent._session_generated_tokens = 0
    agent._last_step_summary = None
    agent._last_action_result = None
    agent._summarized_knowledge = ta._empty_world_model()
    agent._noop_guard = ta.NoopGuard(min_repeats=1)   # проверки 1-5 про механизм, не про порог
    agent._noop_guard_blocked = 0
    agent._step_env_callback = engine
    agent._current_valid_actions = ["UP", "DOWN"]
    agent._python_timeout = 30.0
    agent._tool_output_chars = 20000
    return agent


def run(agent, code):
    """Один вызов инструмента python.

    Любая ошибка песочницы — это провал теста, а не «ход не дошёл до движка».
    Без этой проверки тест зелёный по неверной причине: 25.09 отсутствие поля
    счётчика роняло запрет исключением, движок хода не видел, и проверка
    «повтор срезан» проходила, хотя срезал его не запрет, а падение.
    """
    res = agent._run_python_tool(STATE, {"code": code})
    payload = json.loads(res.content) if hasattr(res, "content") else res
    if payload.get("error"):
        raise AssertionError("песочница вернула ошибку: %s" % payload["error"])
    return payload


def main() -> None:
    ok = True

    # 1. повтор пустого хода при той же доске
    eng = Engine(); agent = make_agent(eng)
    write_state(BOARD_A)
    run(agent, "action([{'action': 'UP'}])")          # первый раз: уходит, пусто
    run(agent, "action([{'action': 'UP'}])")          # второй раз: должен быть срезан
    got = eng.seen
    print("1. повтор пустого при той же доске -> в движок ушло %s" % got)
    ok &= (got == ["UP"]); print("   %s" % ("ОК" if got == ["UP"] else "ПРОВАЛ: ход прошёл в движок"))

    # 2. та же доска, другой ход — пропускаем
    run(agent, "action([{'action': 'DOWN'}])")
    print("2. другой ход при той же доске -> %s" % eng.seen)
    ok &= (eng.seen == ["UP", "DOWN"]); print("   %s" % ("ОК" if eng.seen == ["UP", "DOWN"] else "ПРОВАЛ: лишний запрет"))

    # 3. доска изменилась — прежний запрет не действует
    eng = Engine(); agent = make_agent(eng)
    write_state(BOARD_A)
    run(agent, "action([{'action': 'UP'}])")          # пусто на доске A
    eng.next_board, eng.next_changed = BOARD_B, True
    run(agent, "action([{'action': 'DOWN'}])")        # доска стала B
    run(agent, "action([{'action': 'UP'}])")          # UP на доске B — запрета нет
    print("3. после смены доски -> %s" % eng.seen)
    exp = ["UP", "DOWN", "UP"]
    ok &= (eng.seen == exp); print("   %s" % ("ОК" if eng.seen == exp else "ПРОВАЛ: запрет пережил смену доски"))

    # 4. анимация (несколько кадров) пустым ходом не считается
    eng = Engine(); agent = make_agent(eng)
    eng.next_frames = 3
    write_state(BOARD_A)
    run(agent, "action([{'action': 'UP'}])")
    run(agent, "action([{'action': 'UP'}])")
    print("4. анимация при неизменной доске -> %s" % eng.seen)
    exp = ["UP", "UP"]
    ok &= (eng.seen == exp); print("   %s" % ("ОК" if eng.seen == exp else "ПРОВАЛ: срезали анимированный ход"))

    # 5. счётчик срезанных ходов — без него прогон не докажет, что слой работал
    eng = Engine(); agent = make_agent(eng)
    write_state(BOARD_A)
    for _ in range(3):
        run(agent, "action([{'action': 'UP'}])")
    print("5. счётчик срезанных ходов -> %d (ожидалось 2)" % agent._noop_guard_blocked)
    ok &= (agent._noop_guard_blocked == 2)
    print("   %s" % ("ОК" if agent._noop_guard_blocked == 2 else "ПРОВАЛ"))

    # 6. боевое значение min_repeats=3: механика «через раз» должна выживать
    eng = Engine(); agent = make_agent(eng)
    agent._noop_guard = ta.NoopGuard()            # боевой порог
    write_state(BOARD_A)
    for _ in range(5):
        run(agent, "action([{'action': 'UP'}])")
    print("6. боевой порог, 5 повторов -> в движок ушло %d (ожидалось 3)" % len(eng.seen))
    ok &= (len(eng.seen) == 3)
    print("   %s" % ("ОК" if len(eng.seen) == 3 else "ПРОВАЛ"))

    # 7. действие, срабатывающее со второго нажатия, не должно срезаться
    eng = Engine(); agent = make_agent(eng)
    agent._noop_guard = ta.NoopGuard()
    write_state(BOARD_A)
    run(agent, "action([{'action': 'UP'}])")      # первое нажатие: пусто, тумблер взведён
    eng.next_board, eng.next_changed = BOARD_B, True
    run(agent, "action([{'action': 'UP'}])")      # второе: срабатывает
    print("7. механика «через раз» -> в движок ушло %s" % eng.seen)
    ok &= (eng.seen == ["UP", "UP"])
    print("   %s" % ("ОК" if eng.seen == ["UP", "UP"] else "ПРОВАЛ: срезали рабочее действие"))

    print("\nИТОГ:", "все семь проверок пройдены" if ok else "ЕСТЬ ПРОВАЛЫ")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

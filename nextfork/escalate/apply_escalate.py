"""Эскалация застрявшего уровня (идея Kepler / Retrodict) в обвязке Франзена: при NEXTFORK_ESCALATE=1, когда на
одном уровне потрачено >= NEXTFORK_ESC_T1 ходов (по умолчанию 90), в каждое сообщение модели добавляется
обязательная инструкция; после NEXTFORK_ESC_T2 (180) — вторая ступень. Без флага поведение не меняется.

Пороги: у Kepler 3x / 6x человеческой нормы уровня. Нормы скрытых игр нам неизвестны; в контроле базы 03.10 (стенд
2 ч, 105 взятых уровней) копия Франзена тратит в медиане 30 ходов на взятый уровень, отсюда 3x = 90, 6x = 180.
При 90 инструкция включилась бы на 6% взятых уровней и на 33% застрявших в конце.
usage: python apply_escalate.py <src root с ARC3-Inference>
"""
import sys
from pathlib import Path

MARK = "NEXTFORK escalate"

HELPER = '''

# ---- NEXTFORK escalate (03.10): stuck-level directive, adapted from Kepler / Retrodict ----
_NF_ESC_T1_TEXT = (
    "ESCALATION (level {level}: {spent} actions on this level without completing it): this level is not "
    "yielding to live play. Binding directive until it completes:\\n"
    " 1. In your world model, list BOTH what the history leaves unexplained AND the places, objects or states "
    "you have never visited or touched. Unexplored territory outranks new mechanic hypotheses.\\n"
    " 2. Before acting, check your rules against the recorded history with python (`transitions`, "
    "`frame_diff`): a rule that does not reproduce what already happened is wrong.\\n"
    " 3. Infer the goal from what changed at the exact moments earlier levels ENDED, not from what changes "
    "while you play.\\n"
    " 4. Do not repeat moves whose outcome you can already predict from the history. Take a live move only to "
    "reach something unvisited or to separate two hypotheses."
)
_NF_ESC_T2_TEXT = (
    "ESCALATION 2: still stuck. Assume one of your rules is WRONG or a region is unvisited: prefer moves that "
    "reach never-before-seen board configurations, and re-derive any rule that says the goal is unreachable."
)


def _nf_escalation_lines(agent, level, step):
    if os.environ.get("NEXTFORK_ESCALATE", "0").strip().lower() not in ("1", "true", "on"):
        return []
    try:
        level, step = int(level), int(step)
    except (TypeError, ValueError):
        return []
    if getattr(agent, "_nf_esc_level", None) != level:
        agent._nf_esc_level = level
        agent._nf_esc_start = step
    spent = step - agent._nf_esc_start
    t1 = _get_env_int("NEXTFORK_ESC_T1", 90)
    t2 = _get_env_int("NEXTFORK_ESC_T2", 180)
    if spent < t1:
        return []
    out = [_NF_ESC_T1_TEXT.format(level=level, spent=spent)]
    if spent >= t2:
        out.append(_NF_ESC_T2_TEXT)
    return out
'''

ANCHOR_CALL = '''        lines.extend(
            [
                "Only tool: `python`. It receives `current_frame`'''
CALL = f'''        lines.extend(_nf_escalation_lines(self, current_level, current_step))  # {MARK}
'''


def main(root: Path):
    ta = root / "ARC3-Inference/inference/agent/tool_agent.py"
    t = ta.read_text()
    if MARK in t:
        print("уже применено:", ta.name)
        return
    assert t.count(ANCHOR_CALL) == 1, "якорь вызова не найден ровно один раз"
    assert "def _get_env_int(" in t and "\nimport os" in t, "нет _get_env_int или import os"
    t = t.replace(ANCHOR_CALL, CALL + ANCHOR_CALL)
    t = t.rstrip("\n") + "\n" + HELPER
    ta.write_text(t)
    print("escalate applied:", ta.name)


if __name__ == "__main__":
    main(Path(sys.argv[1]))

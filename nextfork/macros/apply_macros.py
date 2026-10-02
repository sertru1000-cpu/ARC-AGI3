"""Ставит макро-ходы (run / until / click_each) в обвязку Франзена поверх его патча.
usage: python apply_macros.py <src root, где лежит ARC3-Inference>
Включается переменной NEXTFORK_MACROS=1 (функции в песочнице есть всегда, строка в промпте — только при флаге)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = Path(sys.argv[1])
sb = root / "ARC3-Inference/inference/agent/python_tool_sandbox.py"
ta = root / "ARC3-Inference/inference/agent/tool_agent.py"
block = (HERE / "macros_block.py").read_text()
assert '"""' not in block, "блок не должен содержать тройные кавычки: он вставляется в r-строку"

s = sb.read_text()
anchor = '        runtime_globals["action"] = action\n'
assert s.count(anchor) == 1, "якорь в песочнице не найден ровно один раз"
if "NEXTFORK macros" not in s:
    s = s.replace(anchor, anchor + block)
    sb.write_text(s)

t = ta.read_text()
anchor2 = "        if undo_exposure_mode() == \"on\":\n            prompt += UNDO_INFO_ADDENDUM\n"
assert t.count(anchor2) == 1, "якорь в tool_agent не найден ровно один раз"
ADD = (
    "        if os.environ.get(\"NEXTFORK_MACROS\", \"0\").strip().lower() in (\"1\", \"true\", \"on\"):\n"
    "            prompt += (\n"
    "                \"\\nMacro moves (use them):\\n\"\n"
    "                \"- Once you know what the next moves should be, play them in ONE python call with a macro \"\n"
    "                \"instead of one move per call. Every extra call costs a full model turn; macros do not.\\n\"\n"
    "                \"- `run(\\\"U3 R2 S D* C12,30\\\")` plays moves one by one: U D L R = arrows, S = space, Z = undo, \"\n"
    "                \"a number repeats, `*` repeats until the board stops changing, C<row>,<col> clicks. It stops by itself \"\n"
    "                \"on a level change, game over, a move that changes nothing, or a guard, and prints a one-line report.\\n\"\n"
    "                \"- `until(\\\"R\\\", lambda f: <test on frame f>)` repeats one move until the test holds; \"\n"
    "                \"`click_each(\\\"r\\\")` clicks every object of that color.\\n\"\n"
    "                \"- Single moves are for probing an unknown effect. When executing a plan, navigating to a known place, \"\n"
    "                \"or repeating a move, use a macro.\\n\"\n"
    "            )\n"
)
if "NEXTFORK_MACROS" not in t:
    t = t.replace(anchor2, anchor2 + ADD)
    ta.write_text(t)
print("macros applied:", sb, ta)

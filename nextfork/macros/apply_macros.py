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
    "                \"- Macros in python for a sequence you already know: `run(\\\"U3 R2 S D* C12,30\\\")` \"\n"
    "                \"plays the moves one by one (U D L R S Z = up down left right space undo, a number repeats, \"\n"
    "                \"`*` repeats until the board stops changing, C<row>,<col> clicks) and stops early on a level \"\n"
    "                \"change, game over, a move that changes nothing, or a guard. `until(\\\"R\\\", lambda f: ...)` \"\n"
    "                \"repeats one move until a test on the current frame holds. `click_each(\\\"r\\\")` clicks every \"\n"
    "                \"object of that color. Each prints a one-line report. Probe single moves while the effect is unknown.\\n\"\n"
    "            )\n"
)
if "NEXTFORK_MACROS" not in t:
    t = t.replace(anchor2, anchor2 + ADD)
    ta.write_text(t)
print("macros applied:", sb, ta)

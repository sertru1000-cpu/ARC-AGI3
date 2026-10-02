"""Ставит в обвязку Франзена мини-библиотеку функций доски (NEXTFORK_LIB=1) и нотацию заметок (NEXTFORK_NOTES=1).
Функции в песочнице есть всегда; строки в промпте — только при флагах.
usage: python apply_extras.py <src root с ARC3-Inference>"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = Path(sys.argv[1])
sb = root / "ARC3-Inference/inference/agent/python_tool_sandbox.py"
ta = root / "ARC3-Inference/inference/agent/tool_agent.py"
block = (HERE / "lib_block.py").read_text()
assert '"""' not in block

s = sb.read_text()
anchor = '        runtime_globals["action"] = action\n'
assert s.count(anchor) == 1
if "NEXTFORK board helpers" not in s:
    s = s.replace(anchor, anchor + block)
    sb.write_text(s)

t = ta.read_text()
anchor2 = "        if undo_exposure_mode() == \"on\":\n            prompt += UNDO_INFO_ADDENDUM\n"
assert t.count(anchor2) == 1
ADD = '''        if os.environ.get("NEXTFORK_LIB", "0").strip().lower() in ("1", "true", "on"):
            prompt += (
                "- Board helpers are predefined in python: `crop(r0, c0, r1, c1)` prints a region with row/col "
                "headers, `objects(color=None, min_pixels=1)` lists (id, color, pixels, top-left, bottom-right), "
                "`bbox(color)` gives boxes of one color, `brief()` prints object counts per color. Use them instead "
                "of rewriting such utilities.\\n"
            )
        if os.environ.get("NEXTFORK_NOTES", "0").strip().lower() in ("1", "true", "on"):
            prompt += (
                "- Keep your working notes compact, one fact per line, in this notation instead of prose: "
                "`MECH: <move> -> <effect>` for a verified mechanic, `GOAL?: <hypothesis>`, "
                "`TRIED: <move>@<situation> -> <result>`, `PLAN: <macro string or move list>`. "
                "Update a line instead of re-explaining it.\\n"
            )
'''
if "NEXTFORK_LIB" not in t:
    t = t.replace(anchor2, anchor2 + ADD)
    ta.write_text(t)
print("extras applied:", sb.name, ta.name)

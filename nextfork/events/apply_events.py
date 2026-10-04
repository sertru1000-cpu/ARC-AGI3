"""Отчёт о событиях после каждого action() в обвязке Франзена (NEXTFORK_EVENTS=1) + строка в промпте.
Ответ критика 04.10: половина вызовов модели ничего не делает в игре, только смотрит на доску; короткая сводка сразу
после хода должна убрать часть таких вызовов. Без флага поведение не меняется (блок есть, но не оборачивает action).
usage: python apply_events.py <src root с ARC3-Inference>"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
root = Path(sys.argv[1])
sb = root / "ARC3-Inference/inference/agent/python_tool_sandbox.py"
ta = root / "ARC3-Inference/inference/agent/tool_agent.py"
block = (HERE / "events_block.py").read_text()
assert '"""' not in block, "блок не должен содержать тройные кавычки: он вставляется в r-строку"

s = sb.read_text()
anchor = '\n        runtime_globals["action"] = action\n'
if "NEXTFORK events" not in s:
    assert s.count(anchor) == 1, "якорь в песочнице не найден ровно один раз"
    s = s.replace(anchor, anchor + block)   # сразу после action: макро-ходы ниже увидят обёрнутый action
    # песочница запускается с урезанным окружением: пробросить флаг
    env_anchor = '        "PATH": os.environ.get("PATH", ""),\n    }\n'
    assert s.count(env_anchor) == 1, "якорь окружения песочницы не найден"
    s = s.replace(env_anchor, '        "PATH": os.environ.get("PATH", ""),\n'
                  '        "NEXTFORK_EVENTS": os.environ.get("NEXTFORK_EVENTS", "0"),\n    }\n')
    sb.write_text(s)

t = ta.read_text()
anchor2 = "        if undo_exposure_mode() == \"on\":\n            prompt += UNDO_INFO_ADDENDUM\n"
assert t.count(anchor2) == 1, "якорь в tool_agent не найден ровно один раз"
ADD = (
    "        if os.environ.get(\"NEXTFORK_EVENTS\", \"0\").strip().lower() in (\"1\", \"true\", \"on\"):\n"
    "            prompt += (\n"
    "                \"- After every `action([...])` the tool prints an `[events]` line: cells changed, objects moved \"\n"
    "                \"(color[row,col]->[row,col]), new or gone objects, level change, game over, and whether the resulting \"\n"
    "                \"board was seen before. Read it instead of spending a separate call just to look at what changed; when \"\n"
    "                \"the next moves are clear, play them in the same call as a list.\\n\"\n"
    "            )\n"
)
if "NEXTFORK_EVENTS" not in t:
    t = t.replace(anchor2, anchor2 + ADD)
    ta.write_text(t)
print("events applied:", sb.name, ta.name)

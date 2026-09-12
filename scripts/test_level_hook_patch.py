"""Проверка крючка на смене уровня на НАСТОЯЩЕМ ToolAgent из бандла Duck.

Исходное построение промпта и сам запрос к модели подменены предсказуемыми, чтобы мерить только
логику крючка; класс результата запроса — настоящий `_ChatCompletionResult` бандла, сигнатура
`_chat_completion` сверяется с настоящей.
"""
import inspect, sys, types
from pathlib import Path
B = Path("/tmp/duckbundle/src")
sys.path.insert(0, str(B / "ARC3-Inference")); sys.path.insert(0, str(B / "tufa-arc-agi-framework" / "src"))
import inference.agent.tool_agent as ta  # noqa: E402
ok = lambda b, m: print(("ok   " if b else "СБОЙ ") + m)
sig = inspect.signature(ta.ToolAgent._chat_completion)
ok("tools" in sig.parameters and sig.parameters["tools"].kind == inspect.Parameter.KEYWORD_ONLY, "сигнатура _chat_completion: tools — именованный")
CALLS = []
MODE = {"raise": False, "content": "Rule: push blocks onto targets.\nTest first: does UP still move the piece?", "reasoning": ""}
def fake_chat(self, messages, *, tools, request_timeout_seconds=None):
    CALLS.append((messages, tools))
    if MODE["raise"]:
        raise RuntimeError("сервер недоступен")
    return ta._ChatCompletionResult(message={"content": MODE["content"], "reasoning": MODE["reasoning"]}, finish_reason="stop", usage={"completion_tokens": 42})
ta.ToolAgent._chat_completion = fake_chat
ta.ToolAgent._build_user_prompt = lambda self, action_num, **kw: "БАЗОВЫЙ ПРОМПТ"
ns = {"TRUE_SUBMISSION": False, "bm": types.SimpleNamespace(), "print": lambda *a, **k: None}
exec(compile(Path("kernels/notebooks_stockflash_hook/cell15.py").read_text(encoding="utf-8"), "c", "exec"), ns)
ag = object.__new__(ta.ToolAgent)
F = lambda step, level: types.SimpleNamespace(step=step, level=level, ascii="x", grid=((0,),))
call = lambda f: ta.ToolAgent._build_user_prompt(ag, 0, valid_actions=["UP"], current_frame=f, history_entries=[], previous_step_summary={})
t1 = call(F(1, 1)); t2 = call(F(2, 1))
ok(t1 == t2 == "БАЗОВЫЙ ПРОМПТ" and not CALLS, "на первом уровне запросов нет, промпт не меняется")
t3 = call(F(3, 2))
ok(len(CALLS) == 1, "переход на уровень 2: ровно один запрос-обзор")
msgs, tools = CALLS[0]
ok(tools is None, "запрос без инструментов")
ok(msgs[0]["role"] == "system" and msgs[1]["content"].startswith("БАЗОВЫЙ ПРОМПТ") and "LEVEL REVIEW (asked once" in msgs[1]["content"], "в запросе промпт хода и просьба об обзоре, истории нет")
ok("LEVEL REVIEW NOTES" in t3 and "push blocks onto targets" in t3 and "completing level 1" in t3, "заметки дописаны в промпт хода")
t3r = call(F(3, 2))
ok(len(CALLS) == 1 and t3r == t3, "перезаход того же шага: повторного запроса нет, текст тот же")
t4 = call(F(4, 2))
ok(len(CALLS) == 1 and "push blocks onto targets" in t4, "следующий ход того же уровня: заметки на месте, запросов нет")
MODE.update(content="", reasoning="x" * 2000 + "ХВОСТ МЫШЛЕНИЯ")
t5 = call(F(5, 3))
ok(len(CALLS) == 2 and "ХВОСТ МЫШЛЕНИЯ" in t5 and len(t5) < 1700 + 200, "пустой текст ответа: взят хвост мышления, обрезан до 1500")
MODE.update(raise_=True); MODE["raise"] = True
t6 = call(F(6, 4))
ok(t6 == "БАЗОВЫЙ ПРОМПТ" and ns["_HOOK_STATS"]["fail"] == 1, "сбой запроса: ход не ломается, заметок нет, сбой посчитан")
ok(call(None) == "БАЗОВЫЙ ПРОМПТ", "кадра нет: молча пропускаем")
ok(ns["_HOOK_STATS"]["reviews"] == 2 and ns["_HOOK_STATS"]["reuse"] >= 1, "счётчики: обзоров %d, перезаходов %d" % (ns["_HOOK_STATS"]["reviews"], ns["_HOOK_STATS"]["reuse"]))

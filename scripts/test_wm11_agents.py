"""Проверки wm v11 на заглушке: чтение ответа синтезатора из ЧЕТЫРЁХ каналов, повтор на разрыв,
предел работников, машина состояний играющего агента.

Зачем именно эти проверки: два прогона подряд синтезатор молчал, и причина была в канале ответа
(читали только `content`, а он у Flash-Next пуст в 79% ответов) и в моём таймауте 300 с.
Проверки v7/v8 этого не ловили, потому что заглушка возвращала ответ ровно тем каналом, который
код и читал. Здесь каждый канал проверяется отдельно, включая случай «кода нет нигде».

usage:  .venv/bin/python scripts/test_wm11_agents.py
"""

import json
import sys
import types
from pathlib import Path

CELL = Path("kernels/notebooks_stockflash_wm11/cell13.py")


class _R:
    def __init__(self, content="", step_executed=True):
        self.content = content
        self.step_executed = step_executed


class _Res:
    def __init__(self, message):
        self.message = message


class ToolAgent:
    NEXT_OUT = ""
    SEEN = []
    REPLY = {}
    RAISE = []

    def __init__(self):
        self._python_timeout = 5
        self._current_valid_actions = ["RIGHT"]

    def _build_user_prompt(self, action_num, **kw):
        return "BASE PROMPT"

    def _run_python_tool(self, sp, args):
        ToolAgent.SEEN.append(args["code"])
        return _R(ToolAgent.NEXT_OUT)

    def _chat_completion(self, messages, tools=None, **kw):
        ToolAgent.LAST = {"messages": messages, "tools": tools, "kwargs": kw}
        if ToolAgent.RAISE:
            raise ToolAgent.RAISE.pop(0)
        return _Res(dict(ToolAgent.REPLY))


mod = types.ModuleType("inference.agent.tool_agent")
mod.ToolAgent = ToolAgent
mod._ToolDispatchResult = _R
mod.load_runtime_state = lambda p: (None, [])
mod._ascii_frame_view_payload = lambda f: None
mod._ascii_history_view_payload = lambda h: []
mod._normalize_message_content = lambda c: c if isinstance(c, str) else ""
mod._extract_reasoning_text = lambda m: m.get("reasoning") or m.get("reasoning_content") or ""


def _recover(*chunks):
    """Как в бандле (tool_agent.py:91): восстановленный вызов возвращается В ФОРМЕ OpenAI —
    {"id", "type", "function": {"name", "arguments": <json-строка>}}."""
    out = []
    for text in chunks:
        if not (text or "").strip():
            continue
        if "<tool_call>" in text:
            body = text.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
            try:
                parsed = json.loads(body)
            except Exception:
                continue
            out.append({"id": f"markup-call-{len(out) + 1}", "type": "function",
                        "function": {"name": parsed.get("name", ""),
                                     "arguments": json.dumps(parsed.get("arguments", {}))}})
    return out


mod._recover_tool_calls_from_markup = _recover

sb = types.ModuleType("inference.agent.python_tool_sandbox")
SANDBOX = {"digest": "ВАЛИДНЫЕ ДЕЙСТВИЯ: ['RIGHT']",
           "verify": "WM_CHECK admitted=1 correct=5 checked=5 total=5"}
CALLS = []


def _run(code, timeout_seconds, initial_state, action_handler):
    CALLS.append((code, action_handler))
    if code.rstrip().endswith("wm_check(predict, state_of=state_of)"):
        return {"stdout": SANDBOX["verify"]}
    return {"stdout": SANDBOX["digest"]}


sb.run_sandboxed_python = _run
sys.modules["inference"] = types.ModuleType("inference")
sys.modules["inference.agent"] = types.ModuleType("inference.agent")
sys.modules["inference.agent.tool_agent"] = mod
sys.modules["inference.agent.python_tool_sandbox"] = sb
TRUE_SUBMISSION = True
exec(CELL.read_text(encoding="utf-8"))

ok = lambda b, m: print(("ok   " if b else "СБОЙ ") + m)
PROG = "def state_of(f): return (wm_level(f),)\ndef predict(s,a): return s"
FENCE = "думаю вслух...\n```python\n" + PROG + "\n```"
MARKUP = '<tool_call>{"name": "python", "arguments": {"code": "' + PROG.replace("\n", "\\n") + '"}}</tool_call>'


class H:
    def __init__(self):
        self.action = "RIGHT"


def fresh(cleared=True):
    a = ToolAgent()
    a._wm_prog = ""
    a._wm_cex = None
    a._wm_nobs = 5
    a._wm_lastobs = -1
    a._wm_busy = False
    a._wm_cleared = cleared   # синтез разрешён только после взятого уровня
    return a


def synth(a, reply, raises=()):
    """Один синтез синхронно (без потока), чтобы проверять результат детерминированно."""
    ToolAgent.REPLY = reply
    ToolAgent.RAISE = list(raises)
    _wm_synth_body(a, "state.json")
    return a._wm_stats


# --- 1..5: четыре канала ответа и случай «кода нет» -------------------------------
a = fresh()
st = synth(a, {"tool_calls": [{"function": {"name": "python",
                                            "arguments": json.dumps({"code": PROG})}}]})
ok(st["ch_tool"] == 1 and st["synth_ok"] == 1 and a._wm_prog, "канал 1: программа взята из вызова инструмента и принята")

a = fresh()
st = synth(a, {"content": "", "reasoning": MARKUP})
ok(st["ch_markup"] == 1 and st["synth_ok"] == 1, "канал 2: разметка вызова внутри рассуждений разобрана")

a = fresh()
st = synth(a, {"content": FENCE})
ok(st["ch_content"] == 1 and st["synth_ok"] == 1, "канал 3: блок кода в content")

a = fresh()
st = synth(a, {"content": "", "reasoning_content": FENCE})
ok(st["ch_reason"] == 1 and st["synth_ok"] == 1,
   "канал 4: блок кода в КАНАЛЕ РАССУЖДЕНИЙ — ровно то, чего не умели v7 и v8")

a = fresh()
st = synth(a, {"content": "просто рассуждаю без кода", "reasoning": "и тут без кода"})
ok(st["ch_none"] == 1 and st["synth_bad"] == 1 and not a._wm_prog, "кода нет ни в одном канале: честный отказ, не падение")

# --- 6: в лог уходит канал и начало ответа ----------------------------------------
import io
import contextlib

a = fresh()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    synth(a, {"content": "мимо кассы, длинный ответ без программы"})
log = buf.getvalue()
ok("канал:" in log and "мимо кассы" in log, "в лог пишется канал ответа и его начало (в v7/v8 не писалось)")

# --- 7: запрос уходит с инструментом и БЕЗ моего таймаута 300 с -------------------
ok(ToolAgent.LAST["tools"] and ToolAgent.LAST["tools"][0]["function"]["name"] == "python",
   "синтезатор зовётся с тем же инструментом `python`, что и играющий агент")
ok("request_timeout_seconds" not in ToolAgent.LAST["kwargs"],
   "свой таймаут не навязывается — работает штатный (в v8 мои 300 с убили 20 попыток из 24)")

# --- 8: повтор на разрыв соединения -----------------------------------------------
a = fresh()
st = synth(a, {"tool_calls": [{"function": {"name": "python", "arguments": json.dumps({"code": PROG})}}]},
           raises=[ConnectionError("Connection aborted. RemoteDisconnected")])
ok(st["synth_ok"] == 1 and st["synth_err"] == 0, "разрыв соединения переживается одним повтором")

a = fresh()
st = synth(a, {}, raises=[ConnectionError("x"), ConnectionError("y")])
ok(st["synth_err"] == 1, "второй разрыв подряд — честная запись сбоя, игра продолжается")

# --- 9: синтезатору физически нельзя ходить ---------------------------------------
handler = [h for c, h in CALLS if h][-1]
try:
    handler([{"action": "RIGHT"}])
    acted = True
except RuntimeError:
    acted = False
ok(acted is False, "песочница синтезатора возбуждает исключение на любой ход")

# --- 10: в хендофф уходят только определения --------------------------------------
a = fresh()
synth(a, {"tool_calls": [{"function": {"name": "python",
                                       "arguments": json.dumps({"code": PROG + "\naction(['RIGHT'])"})}}]})
ok("action(" not in a._wm_prog and "def predict" in a._wm_prog, "вызовы вырезаны разбором синтаксиса")

# --- 11..14: машина состояний играющего агента ------------------------------------
a = fresh(cleared=False)
p = a._build_user_prompt(1, current_frame=None, history_entries=[H() for _ in range(4)])
ok("Первый уровень бери как обычно" in p and "после того, как уровень будет взят" in p,
   "до первого уровня: играй как сток, про модель не говорим (её и не строят)")
a._wm_cleared = True
p = a._build_user_prompt(2, current_frame=None, history_entries=[H() for _ in range(6)])
ok("Второй агент прямо сейчас строит модель" in p and "писать её не надо" in p,
   "уровень взят, модели ещё нет: играй, модель строит второй агент")
a._wm_prog = PROG


class F:
    def __init__(self, lv):
        self.level = lv


a._wm_lvl = 1
p = a._build_user_prompt(3, current_frame=F(2), history_entries=[H() for _ in range(7)])
ok("def goal(state)" in p and "Модель мира писать НЕ надо" in p,
   "после уровня просят ТОЛЬКО цель — дешёвая строка, а не программа")
a._wm_goal_ok = True
p = a._build_user_prompt(4, current_frame=F(2), history_entries=[H() for _ in range(8)])
ok("ЦЕЛЬ ПРОВЕРЕНА" in p and "wm_plan(predict, goal" in p, "с проверенной целью требуют план")

# --- 15: ход без действия отклоняется и будит синтезатор --------------------------
a = fresh()
a._wm_rej = 0
r = a._run_python_tool("state.json", {"code": "print(1)"})
ok(r.step_executed is False and "Сделай ход" in r.content, "ход без действия отклонён")

# --- 16: хвост дописан после кода, NameError в нём нет ----------------------------
a = fresh()
a._wm_rej = 0
ToolAgent.SEEN.clear()
ToolAgent.NEXT_OUT = "WM_CHECK admitted=1 correct=6 checked=6 total=6"
a._run_python_tool("state.json", {"code": "action(['RIGHT'])"})
sent = ToolAgent.SEEN[-1]
ok(sent.rstrip().endswith("print('WM_TAIL сбой харнесса: %r' % (_e,))") and "except NameError" not in sent,
   "хвост v10 на месте: под стражем и без NameError")

# --- 17: предел работников на весь прогон -----------------------------------------
import threading
import time

hold = threading.Event()
started = []


def slow_chat(self, messages, tools=None, **kw):
    started.append(1)
    hold.wait(5)
    return _Res({"tool_calls": [{"function": {"name": "python", "arguments": json.dumps({"code": PROG})}}]})


ToolAgent._chat_completion = slow_chat
agents = []
for _ in range(9):
    g = fresh(cleared=True)
    g._wm_rej = 0
    ToolAgent.NEXT_OUT = ""
    g._run_python_tool("state.json", {"code": "action(['RIGHT'])"})
    agents.append(g)
time.sleep(0.6)
ok(len(started) == 4, "одновременно работают ровно 4 синтезатора из 9 просивших (запущено %d)" % len(started))
ok(sum(g._wm_stats["sem_busy"] for g in agents) == 5, "остальные получили отказ по слоту, а не встали в очередь")
hold.set()
time.sleep(1.2)
ok(all(not getattr(g, "_wm_busy", False) for g in agents), "слоты освобождены после завершения")

# --- 18..19: НОВОЕ ПРАВИЛО — синтез только после взятого уровня --------------------
ToolAgent._chat_completion = ToolAgent.__dict__.get("_chat_completion", None) or ToolAgent._chat_completion
a = fresh(cleared=False)
a._wm_rej = 0
before = dict(a._wm_stats) if getattr(a, "_wm_stats", None) else {}
_wm_synth_maybe(a, "state.json")
ok(a._wm_stats["synth_run"] == 0 and not getattr(a, "_wm_busy", False),
   "до первого уровня синтезатор НЕ запускается: версия обязана быть неотличима от стока")

a = fresh(cleared=True)
a._wm_rej = 0
ToolAgent.REPLY = {"tool_calls": [{"function": {"name": "python", "arguments": json.dumps({"code": PROG})}}]}
ToolAgent.RAISE = []
_wm_synth_body(a, "state.json")
ok(a._wm_stats["synth_ok"] == 1, "после взятого уровня синтезатор работает как прежде")

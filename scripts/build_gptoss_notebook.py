"""Стоковый Duck на gpt-oss-120b (пункт 4 плана 13.09): единственная открытая модель, которая
влезает в 96 ГБ и не мерилась на этой обвязке.

ЧТО МЕНЯЕТСЯ относительно стокового ноутбука Flash-Next (три ячейки, всё остальное побайтово):
  * ячейка 7  -- входы: бандл Duck (keithtyser) остаётся; вместо рантайма и весов Flash-Next --
                модель danielhanchen/gpt-oss-120b и колёса vLLM 0.19.1 (philipvonderlind/vllm-deps),
                ровно как в официальном шаблоне ARC-AGI-3: GPT-OSS-120B;
  * ячейка 9  -- вместо setup_commands бандла (установка приколоченного vLLM и запуск Flash-Next)
                ставится vLLM 0.19.1 из колёс и поднимается gpt-oss-120b на 127.0.0.1:1234 (тот же адрес,
                что у стокового сервера) с флагами шаблона; окружение анализатора Duck выставляется вручную
                (тот же список, что пишет serving_setup.py), провайдер openrouter -- чистый OpenAI-запрос
                без top_k/chat_template_kwargs, которых gpt-oss не знает;
  * ячейка 15 -- сторож сервера (перезапуск Flash-Next при нездоровье) отключён; teardown -- остановка
                нашего процесса vLLM вместо serving_teardown.py.

ПОРОГИ (записаны до пуска), против базы runs/flash_v1_phaseA (10.25, 40 уровней), знаковым тестом:
польза -- победы−поражения >= +8 и медиана >= +8; вред -- <= −6. Механизм: сервер поднялся (models
отдаёт gpt-oss-120b), доля вызовов с ходом, вызовов на игру, генерация -- записать, порогов нет
(другая модель, другой темп).

usage:  .venv/bin/python scripts/build_gptoss_notebook.py
"""
import ast
import json
import os

SLUG = "sergueimakarov/arc3-duck-gptoss"
BUNDLE = "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1"
WHEELS_KERNEL = "philipvonderlind/vllm-deps"
MODEL_SOURCE = "danielhanchen/gpt-oss-120b/Transformers/default/1"
MODEL_PATH = "/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1"
SERVED_NAME = "gpt-oss-120b"
PORT = 1234
PROBE_CAP_S = 1800.0   # дымовая проба 30 мин вне боя (слово владельца 13.09; v3 -- 14.09 «ещё одна проба -- да»)

SETUP_CELL = r'''
# =====================================================================
# gpt-oss-120b вместо Flash-Next (13.09). Сервер -- по официальному шаблону ARC-AGI-3: GPT-OSS-120B
# (vLLM 0.19.1 из колёс, tool-call-parser openai, kv fp8, enforce-eager). Окружение анализатора Duck --
# тот же список, что пишет serving_setup.py бандла, провайдер openrouter (чистый OpenAI-запрос).
# =====================================================================
import importlib.util, shutil
from urllib.request import urlopen as _g_urlopen

_G_MODEL_PATH = %(model_path)r
_G_SERVED = %(served)r
_G_PORT = %(port)d
_G_BASE_URL = "http://127.0.0.1:%%d/v1" %% _G_PORT

def _g_find_wheel_dir():
    for root in (Path("/kaggle/input"), Path("/kaggle/usr/lib/notebooks"), Path("/kaggle/working")):
        if root.exists():
            for w in root.rglob("vllm*.whl"):
                return w.parent
    raise FileNotFoundError("колёса vLLM не найдены: нужен вход %(wheels)s")

def _g_find_tiktoken_dir():
    for root in (Path("/kaggle/input"), Path("/kaggle/usr/lib/notebooks"), Path("/kaggle/working")):
        if root.exists():
            for d in root.rglob("*"):
                if d.is_dir() and (d / "cl100k_base.tiktoken").exists() and (d / "o200k_base.tiktoken").exists():
                    return d
    return None

if importlib.util.find_spec("vllm") is None:
    wd = _g_find_wheel_dir()
    print("gpt-oss: ставлю vLLM 0.19.1 из", wd, flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(wd),
                           "vllm==0.19.1", "openai==2.24.0", "openai-harmony==0.0.8"])
import vllm as _g_vllm
print("gpt-oss: vllm", _g_vllm.__version__, flush=True)

_g_env = os.environ.copy()
_g_env["OPENAI_API_KEY"] = "EMPTY"
_g_env["LD_LIBRARY_PATH"] = os.pathsep.join([p for p in ("/usr/local/nvidia/lib64", "/usr/local/cuda/lib64", _g_env.get("LD_LIBRARY_PATH", "")) if p])
_g_env["VLLM_LOGGING_LEVEL"] = "DEBUG"
_g_env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
_tk = _g_find_tiktoken_dir()
if _tk is not None:
    _g_env["TIKTOKEN_ENCODINGS_BASE"] = str(_tk)
_g_cmd = [shutil.which("vllm") or sys.executable] + ([] if shutil.which("vllm") else ["-m", "vllm.entrypoints.openai.api_server", "--model"])
if shutil.which("vllm"):
    _g_cmd += ["serve"]
_g_cmd += [_G_MODEL_PATH, "--served-model-name", _G_SERVED, "Qwen/Qwen3.8-Flash-Next-NVFP4", "--host", "127.0.0.1", "--port", str(_G_PORT),
           "--enable-auto-tool-choice", "--tool-call-parser", "openai", "--max-num-seqs", "12",
           "--max-model-len", "40000", "--kv-cache-dtype", "fp8", "--tensor-parallel-size", "1", "--enforce-eager"]
_g_log = open(WORKING_DIR / "vllm-openai-server.log", "ab")
_g_proc = subprocess.Popen(_g_cmd, env=_g_env, stdout=_g_log, stderr=subprocess.STDOUT, cwd=str(WORKING_DIR))
print("gpt-oss: сервер запущен pid", _g_proc.pid, flush=True)
_g_deadline = time.time() + 1800
while True:
    if _g_proc.poll() is not None:
        raise RuntimeError("vLLM завершился с кодом %%s -- см. vllm-openai-server.log" %% _g_proc.returncode)
    try:
        with _g_urlopen(_G_BASE_URL + "/models", timeout=3) as r:
            if _G_SERVED in r.read().decode("utf-8", "replace"):
                break
    except Exception:
        pass
    if time.time() > _g_deadline:
        raise RuntimeError("vLLM не поднялся за 30 минут")
    time.sleep(5)
print("gpt-oss: сервер отвечает, %%s" %% _G_BASE_URL, flush=True)

_g_analyzer_env = {
    "LOCAL_ANALYZER_BASE_URL": _G_BASE_URL, "OPENAI_BASE_URL": _G_BASE_URL,
    "LOCAL_ANALYZER_PROVIDER": "openrouter", "OPENAI_PROVIDER": "openrouter",
    "LOCAL_ANALYZER_MODEL_ID": _G_SERVED, "INFERENCE_ANALYZER_MODEL": _G_SERVED,
    "OPENAI_API_KEY": "EMPTY", "LOCAL_ANALYZER_API_KEY": "EMPTY",
    "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": "32768", "LOCAL_ANALYZER_MAX_OUTPUT": "0",
    "LOCAL_ANALYZER_TOOL_STEPS": "0", "LOCAL_ANALYZER_TOOL_TIMEOUT": "30", "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
    "LOCAL_ANALYZER_YIELD_SECONDS": "60", "LOCAL_ANALYZER_TEMPERATURE": "0.6", "LOCAL_ANALYZER_TOP_P": "0.95",
    "LOCAL_ANALYZER_TOP_K": "20", "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    "MULTIMODAL_CONTEXT": "current_grid", "MULTIMODAL_UPSCALE": "4",
}
os.environ.update(_g_analyzer_env)
_g_persist = json.loads(SETUP_ENV_PATH.read_text())
_g_persist.update(_g_analyzer_env)
SETUP_ENV_PATH.write_text(json.dumps(_g_persist, indent=2, sort_keys=True) + "\n")
print("gpt-oss: окружение анализатора выставлено (%%d ключей)" %% len(_g_analyzer_env), flush=True)

# v3 (14.09): harmony-кодировщик gpt-oss падал на истории Duck: модель отдаёт по 10-20 вызовов инструмента
# за ответ, Duck исполняет первый и хранит все, а tool-результат один -- «Unexpected token 200012 while
# expecting start token 200006». Санитайзер: в assistant-сообщениях оставляем только вызовы с результатами,
# убираем reasoning, content -> строка; запрещаем параллельные вызовы. Импорт tool_agent -- после env.
import inference.agent.tool_agent as _wta
_g_orig_build_payload = _wta.build_chat_payload
def _g_sanitize_messages(messages):
    msgs = [dict(m) for m in (messages or []) if isinstance(m, dict)]
    answered = {str(m.get("tool_call_id", "")) for m in msgs if m.get("role") == "tool"}
    out = []
    for m in msgs:
        if m.get("role") == "assistant":
            m.pop("reasoning", None); m.pop("reasoning_content", None)
            if m.get("content") is None:
                m["content"] = ""
            calls = m.get("tool_calls") or []
            kept = [c for c in calls if str((c or {}).get("id", "")) in answered]
            if kept:
                m["tool_calls"] = kept
            else:
                m.pop("tool_calls", None)
                if not str(m.get("content") or "").strip():
                    m["content"] = "(tool call omitted)"
        elif m.get("role") == "tool":
            if m.get("content") is None:
                m["content"] = ""
        out.append(m)
    return out
def _g_build_payload(*a, **kw):
    if "messages" in kw:
        kw["messages"] = _g_sanitize_messages(kw["messages"])
    payload = _g_orig_build_payload(*a, **kw)
    if payload.get("tools"):
        payload["parallel_tool_calls"] = False
    return payload
_wta.build_chat_payload = _g_build_payload
print("gpt-oss: санитайзер истории под harmony установлен", flush=True)

# v4 (14.09 09:00, слово владельца «переписывай для gpt-oss»): три пробы показали, что chat/completions с harmony
# в vLLM 0.19.1 отдаёт 500 на большинстве запросов Duck (включая первый). Рекомендуемый путь gpt-oss -- Responses
# API. Адаптер: перехват requests.post в tool_agent -- запрос Duck переводится в /v1/responses, ответ -- обратно
# в форму chat/completions (choices[0].message с content/reasoning/tool_calls), которую Duck разбирает без правок.
# gpt-oss текстовая: картинки не шлём (MULTIMODAL_CONTEXT пуст, части image_url отбрасываются).
import types as _g_types, json as _g_json
_g_real_requests = _wta.requests

def _g_to_responses(payload):
    msgs = _g_sanitize_messages(payload.get("messages") or [])
    instructions = []; items = []
    for m in msgs:
        role = m.get("role"); content = m.get("content")
        if role == "system":
            instructions.append(content if isinstance(content, str) else " ".join(str(c.get("text", "")) for c in (content or []) if isinstance(c, dict)))
        elif role == "user":
            parts = []
            if isinstance(content, str):
                parts.append({"type": "input_text", "text": content})
            else:
                for c in (content or []):
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append({"type": "input_text", "text": str(c.get("text", ""))})
                    # image_url отбрасываем: gpt-oss текстовая
            if parts:
                items.append({"type": "message", "role": "user", "content": parts})
        elif role == "assistant":
            text = content if isinstance(content, str) else ""
            if text.strip():
                items.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
            for c in (m.get("tool_calls") or []):
                fn = (c or {}).get("function") or {}
                args = fn.get("arguments", "{}")
                items.append({"type": "function_call", "call_id": str(c.get("id", "")), "name": str(fn.get("name", "")),
                              "arguments": args if isinstance(args, str) else _g_json.dumps(args)})
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": str(m.get("tool_call_id", "")),
                          "output": content if isinstance(content, str) else _g_json.dumps(content)})
    tools = []
    for t in (payload.get("tools") or []):
        fn = (t or {}).get("function") or {}
        tools.append({"type": "function", "name": fn.get("name", ""), "description": fn.get("description", ""),
                      "parameters": fn.get("parameters") or {"type": "object", "properties": {}}})
    body = {"model": payload.get("model"), "input": items, "store": False, "parallel_tool_calls": False,
            "reasoning": {"effort": os.environ.get("GPTOSS_REASONING_EFFORT", "medium")}}
    if instructions:
        body["instructions"] = "\n\n".join(instructions)
    if tools:
        body["tools"] = tools; body["tool_choice"] = "auto"
    for k_src, k_dst in (("max_tokens", "max_output_tokens"), ("temperature", "temperature"), ("top_p", "top_p")):
        if payload.get(k_src) is not None:
            body[k_dst] = payload[k_src]
    return body

def _g_from_responses(resp):
    text = []; reasoning = []; calls = []
    for it in (resp.get("output") or []):
        t = it.get("type")
        if t == "message":
            for c in (it.get("content") or []):
                if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                    text.append(str(c.get("text", "")))
        elif t == "reasoning":
            for c in (it.get("summary") or []) + (it.get("content") or []):
                if isinstance(c, dict) and c.get("text"):
                    reasoning.append(str(c["text"]))
        elif t == "function_call":
            calls.append({"id": str(it.get("call_id") or it.get("id") or ""), "type": "function",
                          "function": {"name": str(it.get("name", "")), "arguments": it.get("arguments") if isinstance(it.get("arguments"), str) else _g_json.dumps(it.get("arguments") or {})}})
    message = {"role": "assistant", "content": "".join(text)}
    if reasoning:
        message["reasoning"] = "\n".join(reasoning)
    if calls:
        message["tool_calls"] = calls
    status = resp.get("status", "completed")
    finish = "tool_calls" if calls else ("length" if status == "incomplete" else "stop")
    u = resp.get("usage") or {}
    return {"choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": u.get("input_tokens", 0), "completion_tokens": u.get("output_tokens", 0),
                      "total_tokens": u.get("total_tokens", 0)}, "model": resp.get("model"), "id": resp.get("id")}

class _GResp:
    def __init__(self, status_code, data, text):
        self.status_code = status_code; self._data = data; self.text = text; self.headers = {}
    def json(self):
        return self._data
    def raise_for_status(self):
        if self.status_code >= 400:
            raise _g_real_requests.HTTPError("%%d Server Error" %% self.status_code, response=self)

_g_stats = {"requests": 0, "ok": 0, "errors": 0, "tool_calls": 0}
def _g_post(url, headers=None, json=None, timeout=None, **kw):
    if not str(url).endswith("/chat/completions") or not isinstance(json, dict) or "messages" not in json:
        return _g_real_requests.post(url, headers=headers, json=json, timeout=timeout, **kw)
    _g_stats["requests"] += 1
    body = _g_to_responses(json)
    r = _g_real_requests.post(str(url)[: -len("/chat/completions")] + "/responses", headers=headers, json=body, timeout=timeout, **kw)
    if r.status_code >= 400:
        _g_stats["errors"] += 1
        if _g_stats["errors"] <= 5:
            print("gpt-oss responses: HTTP %%d %%s" %% (r.status_code, r.text[:300]), flush=True)
        return _GResp(r.status_code, {}, r.text)
    try:
        data = _g_from_responses(r.json())
    except Exception as _e:
        _g_stats["errors"] += 1
        return _GResp(502, {}, "responses translate failed: %%r" %% (_e,))
    _g_stats["ok"] += 1; _g_stats["tool_calls"] += len(data["choices"][0]["message"].get("tool_calls") or [])
    if _g_stats["requests"] %% 50 == 0:
        print("gpt-oss responses stats:", _g_stats, flush=True)
    return _GResp(200, data, _g_json.dumps(data))

class _GRequests:
    def __getattr__(self, name):
        return getattr(_g_real_requests, name)
    post = staticmethod(_g_post)
_wta.requests = _GRequests()
os.environ["MULTIMODAL_CONTEXT"] = ""   # gpt-oss текстовая: без картинки
_g_persist = _g_json.loads(SETUP_ENV_PATH.read_text()); _g_persist["MULTIMODAL_CONTEXT"] = ""
SETUP_ENV_PATH.write_text(_g_json.dumps(_g_persist, indent=2, sort_keys=True) + "\n")
print("gpt-oss: адаптер Responses API установлен (chat/completions -> /v1/responses), картинки отключены", flush=True)
'''


def main() -> None:
    src = json.load(open("kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))
    nb = json.loads(json.dumps(src))

    # --- ячейка 5: колёса соревнования ищем по всему /kaggle/input (14.09: Kaggle монтирует их в
    # /kaggle/input/arc-prize-2026-arc-agi-3/, а стоковый путь с /competitions/ пуст -- explore v5 это показал)
    c5 = "".join(nb["cells"][5]["source"])
    guard = ("import time as _t5\n"
             "def _c5_find():\n"
             "    fixed = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels')\n"
             "    if fixed.exists():\n        return fixed\n"
             "    for root in (Path('/kaggle/input'),):\n"
             "        for d in root.rglob('arc_agi_3_wheels'):\n"
             "            if d.is_dir():\n                return d\n"
             "    return None\n"
             "COMP_WHEELS_DIR = None\n"
             "for _i5 in range(120):\n"
             "    COMP_WHEELS_DIR = _c5_find()\n"
             "    if COMP_WHEELS_DIR is not None:\n        break\n"
             "    print('taaf.kaggle: жду монтирования входов соревнования (%d)' % _i5, flush=True); _t5.sleep(5)\n"
             "print('taaf.kaggle: колёса соревнования:', COMP_WHEELS_DIR, flush=True)\n"
             "if COMP_WHEELS_DIR is None:\n    raise RuntimeError('входы соревнования не примонтированы')\n")
    lit = '"/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",'
    assert lit in c5
    nb["cells"][5]["source"] = (guard + c5.replace(lit, "str(COMP_WHEELS_DIR),")).splitlines(keepends=True)

    # --- ячейка 7: входы
    c7 = "".join(nb["cells"][7]["source"])
    old = 'DATASET_SOURCES = ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1", "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"]\nKERNEL_SOURCES = []'
    assert old in c7, "ячейка 7: не нашёл список входов"
    c7 = c7.replace(old, 'DATASET_SOURCES = [%r]\nKERNEL_SOURCES = [%r]' % (BUNDLE, WHEELS_KERNEL))
    nb["cells"][7]["source"] = c7.splitlines(keepends=True)

    # --- ячейка 9: вместо setup_commands бандла -- наш сервер
    c9 = "".join(nb["cells"][9]["source"])
    old = '''# Solver setup commands (wheels, vLLM server startup, ...) run before the benchmark loads.
env = _command_env()
for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):
    print(f"taaf.kaggle: setup command: {command}", flush=True)
    subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)
    # Re-read in case the command persisted new env keys.
    env = _command_env()
    os.environ.update(env)
'''
    assert old in c9, "ячейка 9: не нашёл цикл setup_commands"
    setup = SETUP_CELL % {"model_path": MODEL_PATH, "served": SERVED_NAME, "port": PORT, "wheels": WHEELS_KERNEL}
    c9 = c9.replace(old, "# setup_commands бандла (приколоченный vLLM + Flash-Next) НЕ выполняются: сервер ниже.\n" + setup + "\nenv = _command_env()\nos.environ.update(env)\n")
    nb["cells"][9]["source"] = c9.splitlines(keepends=True)

    # --- ячейка 15: сторож отключён, teardown -- наш процесс
    c15 = "".join(nb["cells"][15]["source"])
    lit15 = 'str(Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent / "environment_files")'
    assert lit15 in c15
    c15 = c15.replace(lit15, 'str(Path(COMP_WHEELS_DIR).parent / "environment_files")')
    a = c15.index("import vllm_server_watchdog as vllm_watchdog")
    b = c15.index("# Play the benchmark; watchdog stop and teardown run even if it raises.")
    c15 = c15[:a] + ("class vllm_watchdog:  # сторож Flash-Next отключён: сервер gpt-oss управляется ноутбуком\n"
                     "    @staticmethod\n    def stop_background(timeout_seconds=15.0):\n        return None\n"
                     "print('gpt-oss: сторож сервера отключён', flush=True)\n\n") + c15[b:]
    old = '(BUNDLE_DIR / "teardown_commands.json").read_text()'
    assert old in c15, "ячейка 15: не нашёл teardown"
    c15 = c15.replace(old, "'[\"pkill -f vllm.entrypoints || true\"]'")
    marker = "# Play the benchmark; watchdog stop and teardown run even if it raises."
    assert marker in c15
    c15 = c15.replace(marker, "if not TRUE_SUBMISSION:\n    bm.solver.max_runtime_s_per_game = %r    # дымовая проба вне боя\n\n" % PROBE_CAP_S + marker, 1)
    nb["cells"][15]["source"] = c15.splitlines(keepends=True)

    out = "kernels/notebooks_duck_gptoss"
    os.makedirs(out, exist_ok=True)
    json.dump(nb, open(os.path.join(out, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open("kernels/notebooks_stockflash/kernel-metadata.json"))
    meta["id"] = SLUG
    meta["title"] = "arc3 duck gptoss"
    meta["dataset_sources"] = [BUNDLE]
    meta["kernel_sources"] = [WHEELS_KERNEL]
    meta["model_sources"] = [MODEL_SOURCE]
    # v2 (13.09 14:55): приколоченный образ шаблона оказался БЕЗ CUDA (libcudart.so.12 не найден, «0 active drivers»,
    # vLLM «Failed to infer device type») -- образ по умолчанию, как у стокового кернела; LD_LIBRARY_PATH
    # с CUDA-библиотеками Kaggle выставляется серверу явно.
    meta.pop("docker_image", None)
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)

    for i in (7, 9, 15):
        compile("".join(nb["cells"][i]["source"]), "cell%d" % i, "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    diff = [i for i in range(18) if "".join(nb["cells"][i]["source"]) != "".join(src["cells"][i]["source"])]
    size = os.path.getsize(os.path.join(out, "submission.ipynb"))
    print("ok   изменены только ячейки 5, 7, 9, 15:", diff == [5, 7, 9, 15])
    print("ok   ячейки компилируются; входы: бандл %s, колёса %s, модель %s" % (BUNDLE, WHEELS_KERNEL, MODEL_SOURCE))
    print("ok   setup_commands бандла не исполняются:", "setup_commands.json\").read_text()" not in "".join(nb["cells"][9]["source"]))
    print("ok   сторож отключён, teardown свой:", "vllm_watchdog.start_background" not in c15 and "pkill -f vllm.entrypoints" in c15)
    print("ok   потолок пробы вне боя: %s с (в бою штатный):" % PROBE_CAP_S, "max_runtime_s_per_game = %r" % PROBE_CAP_S in c15)
    print("ok   слаг для пуша:", meta["id"], "| размер %.0f КБ" % (size / 1024))


if __name__ == "__main__":
    main()

"""Сухой прогон ячейки пакетного синтеза против поддельного vLLM на 127.0.0.1:1234 — до траты квоты.

Проверяется: ушло 420 запросов (21 игра x 20), в каждом инструмент python, системный промпт генератора,
сводка наблюдений, температура 0.9 и мышление; ответы разобраны, programs.json записан в формате
генератора, teardown вызван; затем локальная оценка `wm_eval_programs.py` проходит на этом файле.
"""
import json, os, subprocess, sys, tempfile, threading, types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

GOT = []
PROG = "def state_of(frame):\n    return (wm_level(frame),)\ndef predict(state, action):\n    return state"
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"data":[]}')
    def do_POST(self):
        n = int(self.headers["Content-Length"]); body = json.loads(self.rfile.read(n)); GOT.append(body)
        k = len(GOT) % 3
        if k == 0:   # вызов инструмента
            msg = {"content": None, "reasoning": "думаю", "tool_calls": [{"id": "x", "type": "function",
                   "function": {"name": "python", "arguments": json.dumps({"code": PROG})}}]}
        elif k == 1: # программа в ограде в тексте
            msg = {"content": "```python\n" + PROG + "\n```", "reasoning": ""}
        else:        # ничего полезного
            msg = {"content": "не знаю", "reasoning": "хм"}
        out = json.dumps({"choices": [{"finish_reason": "stop", "message": msg}], "usage": {"completion_tokens": 777}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

srv = ThreadingHTTPServer(("127.0.0.1", 1234), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
nb = json.load(open("kernels/notebooks_synth_batch/submission.ipynb", encoding="utf-8"))
cell = "".join(nb["cells"][-1]["source"])
tmp = Path(tempfile.mkdtemp())
(tmp / "teardown_commands.json").write_text(json.dumps(["echo teardown-called"]))
TD = []
ns = {"WORKING_DIR": tmp, "BUNDLE_DIR": tmp, "_command_env": lambda: {}, "subprocess": types.SimpleNamespace(run=lambda cmd, **kw: TD.append(cmd))}
exec(compile(cell, "synth", "exec"), ns)
srv.shutdown()
ok = lambda b, m: print(("ok   " if b else "СБОЙ ") + m)
res = json.loads((tmp / "programs.json").read_text())
ok(len(GOT) == 420, "запросов к серверу: %d (ожидалось 420)" % len(GOT))
ok(len(res) == 420, "записей в programs.json: %d" % len(res))
ok(all(b["tools"][0]["function"]["name"] == "python" and b["tool_choice"] == "auto" for b in GOT), "в каждом запросе инструмент python")
ok(all(b["temperature"] == 0.9 and b["chat_template_kwargs"]["enable_thinking"] is True and b["max_tokens"] == 8192 for b in GOT), "температура 0.9, мышление, потолок 8192")
ok(all(b["messages"][0]["content"].startswith("Ты — агент-синтезатор") and b["messages"][1]["content"].startswith("Наблюдения:\n") for b in GOT), "системный промпт генератора и сводка наблюдений")
chan = {}
for r in res: chan[r["channel"]] = chan.get(r["channel"], 0) + 1
ok(chan.get("вызов инструмента", 0) == 140 and chan.get("content", 0) == 140 and chan.get("кода нет", 0) == 140, "разбор по каналам: %s" % chan)
ok({r["game"] for r in res} == {g["game"] for g in json.load(open("docs/wm_digests.json", encoding="utf-8")) if g["pairs"] >= 10}, "игры — ровно те, у кого буфер >= 10 переходов")
ok(all(set(("game", "attempt", "level", "pairs", "code", "channel", "usage")) <= set(r) for r in res), "формат записей как у генератора")
ok(TD == ["echo teardown-called"], "teardown вызван")
# сквозная проверка: локальная оценка проходит на этом файле
out = tmp / "rows.json"
p = subprocess.run([sys.executable, "scripts/wm_eval_programs.py", str(tmp / "programs.json"), "--out", str(out)],
                   capture_output=True, text=True, timeout=1500)
ok(p.returncode == 0 and out.exists(), "локальная оценка wm_eval_programs.py отработала (код %d)" % p.returncode)
print("     " + "\n     ".join((p.stdout or p.stderr).strip().splitlines()[:4]))

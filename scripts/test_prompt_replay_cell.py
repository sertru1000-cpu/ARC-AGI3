"""Сухой прогон ячейки повтора против поддельного сервера — до траты квоты.

Урок 09.09: код, который исполняется только на Kaggle, проверять на синтетике до пуска.
Поднимаем локальный HTTP-сервер, притворяющийся vLLM (`/v1/models`, `/v1/chat/completions`),
на 127.0.0.1:1234, исполняем ячейку повтора из собранного ноутбука с подменёнными
WORKING_DIR / BUNDLE_DIR / _command_env / subprocess, и проверяем: все 100 запросов ушли,
в каждом есть tools, картинки и параметры семплирования, вариант Б отличается от A только
последним сообщением, результаты записаны, teardown вызван.
"""
import json, sys, tempfile, threading, types, subprocess as real_subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

GOT = []
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"data":[]}')
    def do_POST(self):
        n = int(self.headers["Content-Length"]); body = json.loads(self.rfile.read(n)); GOT.append(body)
        resp = {"choices": [{"finish_reason": "tool_calls", "message": {
            "reasoning": "думаю " * 10, "content": None,
            "tool_calls": [{"id": "x", "type": "function", "function": {"name": "python",
                            "arguments": json.dumps({"code": "action(['UP'])"})}}]}}],
            "usage": {"prompt_tokens": 20000, "completion_tokens": 1234}}
        out = json.dumps(resp).encode(); self.send_response(200)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(out)))
        self.end_headers(); self.wfile.write(out)

# бандл нужен ячейке для отрисовки картинок, как на Kaggle после ячейки 9
sys.path.insert(0, "/tmp/duckbundle/src/ARC3-Inference")
sys.path.insert(0, "/tmp/duckbundle/src/tufa-arc-agi-framework/src")

srv = ThreadingHTTPServer(("127.0.0.1", 1234), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()

nb = json.load(open("kernels/notebooks_prompt_replay/submission.ipynb", encoding="utf-8"))
cell = "".join(nb["cells"][-1]["source"])
tmp = Path(tempfile.mkdtemp())
(tmp / "teardown_commands.json").write_text(json.dumps(["echo teardown-called"]))
TD = []
fake_sub = types.SimpleNamespace(run=lambda cmd, **kw: TD.append(cmd))
ns = {"WORKING_DIR": tmp, "BUNDLE_DIR": tmp, "_command_env": lambda: {}, "subprocess": fake_sub}
exec(compile(cell, "replay", "exec"), ns)
srv.shutdown()

ok = lambda b, m: print(("ok   " if b else "СБОЙ ") + m)
res = json.loads((tmp / "replay_results.json").read_text())
ok(len(GOT) == 100, "запросов к серверу: %d (ожидалось 100)" % len(GOT))
ok(len(res) == 100 and all(r["ok"] for r in res), "результатов записано %d, все успешные" % len(res))
ok(all(b.get("tools") and b["tools"][0]["function"]["name"] == "python" for b in GOT), "в каждом запросе есть инструмент python")
ok(all(b.get("temperature") == 0.6 and b.get("top_p") == 0.95 and b.get("top_k") == 20 for b in GOT), "семплирование 0.6 / 0.95 / 20")
ok(all(b.get("chat_template_kwargs", {}).get("enable_thinking") is True for b in GOT), "мышление включено")
ok(all("max_tokens" not in b for b in GOT), "потолка токенов нет, как у анализатора")
imgs = [sum(1 for m in b["messages"] if m["role"] == "user" and isinstance(m["content"], list)
            and any(p.get("type") == "image_url" for p in m["content"])) for b in GOT]
users = [sum(1 for m in b["messages"] if m["role"] == "user") for b in GOT]
ok(imgs == users, "картинка есть в каждом сообщении пользователя")
ok(any(m.get("reasoning") for b in GOT for m in b["messages"] if m["role"] == "assistant"), "прошлое мышление передаётся")
ok(any(m.get("role") == "tool" and m.get("tool_call_id") for b in GOT for m in b["messages"]), "результаты инструмента с id передаются")
byk = {}
for b in GOT:
    last = b["messages"][-1]["content"][0]["text"]
    byk.setdefault("B" if "HARNESS DIFF" in last else "A", []).append(b)
ok(len(byk.get("A", [])) == 50 and len(byk.get("B", [])) == 50, "по 50 запросов каждого варианта")
a0 = next(b for b in byk["A"]); b0 = next(b for b in byk["B"] if len(b["messages"]) == len(a0["messages"]) and b["messages"][0] == a0["messages"][0] and b["messages"][-2] == a0["messages"][-2])
ok(a0["messages"][:-1] == b0["messages"][:-1], "Б отличается от A только последним сообщением")
ok(b0["messages"][-1]["content"][0]["text"].endswith("Current grid image:"), "в Б строка «Current grid image:» осталась последней")
ok(TD == ["echo teardown-called"], "teardown вызван")

import os
ok(os.path.getsize("kernels/notebooks_prompt_replay/submission.ipynb") < 1_000_000,
   "ноутбук меньше предела Kaggle 1 МБ: %.0f КБ" % (os.path.getsize("kernels/notebooks_prompt_replay/submission.ipynb") / 1024))
# картинка, нарисованная бандлом на «Kaggle», совпадает побайтно с локальной копией функции
sys.path.insert(0, "scripts")
from prompt_replay_payload import png as local_png
import lzma, base64 as b64
pl = json.loads(lzma.decompress(b64.b64decode(ns["PAYLOAD_B64"])))
m0 = next(m for m in pl["items"][0]["messages"] if m["role"] == "user")
grid = [[int(ch, 16) for ch in row] for row in m0["board"].split("/")]
# запросы идут по четыре разом, порядок прихода не совпадает с порядком игр — ищем по тексту
first_text = m0["text"] + "\n\nCurrent grid image:"
req = next(b for b in GOT if any(mm["role"] == "user" and mm["content"][0]["text"] == first_text for mm in b["messages"]))
sent = next(mm for mm in req["messages"] if mm["role"] == "user" and mm["content"][0]["text"] == first_text)["content"][1]["image_url"]["url"]
ok(sent == local_png(grid), "картинка от бандла совпадает с локальной отрисовкой побайтно")
ok(len(sent) < 20000, "картинка при увеличении 4, а не 16: %d знаков" % len(sent))

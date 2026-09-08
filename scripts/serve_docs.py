"""Локальный веб-сервер для документов проекта: промпты Gemini, план, разборы.

Заменяет публикацию артефактов: всё открывается в браузере с localhost и никуда не уходит.
Текстовые файлы (.txt, .md) отдаются с кнопкой «Скопировать» — это нужно для промптов Gemini.
HTML-страницы отдаются как есть.

usage:
    python scripts/serve_docs.py [--port 8765]
"""
from __future__ import annotations

import argparse
import html
import http.server
import socketserver
import urllib.parse
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

CSS = """
:root{--bg:#f6f4ee;--surface:#fffdf8;--ink:#1f1d18;--ink2:#5d5748;--rule:#dcd6c6;--accent:#2b5d8a;--soft:#e1ebf3}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#15140f;--surface:#1d1b15;--ink:#ece7da;--ink2:#aaa392;--rule:#332f26;--accent:#7fb0dc;--soft:#1a2a3a}}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font:15px/1.55 "Public Sans",-apple-system,"Segoe UI",sans-serif;margin:0}
main{max-width:960px;margin:0 auto;padding:28px 20px 60px}
h1{font:700 24px/1.2 system-ui;margin:0 0 4px}
p.lead{color:var(--ink2);margin:0 0 20px;font-size:14px}
h2{font:600 14px/1.3 system-ui;text-transform:uppercase;letter-spacing:.08em;color:var(--ink2);
   margin:26px 0 8px;padding-bottom:6px;border-bottom:1px solid var(--rule)}
ul{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:1px;background:var(--rule);
   border:1px solid var(--rule);border-radius:6px;overflow:hidden}
li{background:var(--surface);padding:0}
a.row{display:flex;justify-content:space-between;gap:16px;align-items:baseline;padding:9px 14px;
      text-decoration:none;color:var(--ink)}
a.row:hover{background:var(--soft)}
a.row .name{font-weight:500}
a.row .meta{color:var(--ink2);font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}
.bar{display:flex;gap:10px;align-items:center;margin:0 0 14px;flex-wrap:wrap}
button{font:600 13px system-ui;background:var(--accent);color:#fff;border:0;border-radius:6px;padding:8px 14px;cursor:pointer}
button:focus-visible{outline:3px solid var(--soft)}
.ok{color:var(--ink2);font-size:13px}
a.back{color:var(--accent);font-size:13px;text-decoration:none}
pre{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:16px 18px;
    white-space:pre-wrap;word-wrap:break-word;font:13px/1.5 "IBM Plex Mono",ui-monospace,Consolas,monospace;
    overflow-x:auto;margin:0}
"""

GROUPS = [
    ("Промпты для Gemini", lambda p: p.name.startswith("gemini_prompt")),
    ("Ответы Gemini", lambda p: p.name.startswith("gemini_round") and p.suffix == ".md"),
    ("План и беклог", lambda p: p.name.startswith(("plan_", "improvement_", "public_line"))),
    ("Разборы и дизайн", lambda p: True),
]


def _fmt(path: Path) -> str:
    kb = path.stat().st_size / 1024
    when = datetime.fromtimestamp(path.stat().st_mtime).strftime("%d.%m %H:%M")
    return f"{kb:,.0f} КБ · {when}"


def _index() -> bytes:
    files = sorted(
        [p for p in DOCS.iterdir() if p.is_file() and p.suffix in (".txt", ".md")],
        key=lambda p: -p.stat().st_mtime,
    )
    pages = sorted((DOCS / "artifacts").glob("*.html"), key=lambda p: -p.stat().st_mtime)
    seen: set[Path] = set()
    body = []
    for title, match in GROUPS:
        chosen = [p for p in files if p not in seen and match(p)]
        if not chosen:
            continue
        seen.update(chosen)
        body.append(f"<h2>{html.escape(title)}</h2><ul>")
        for p in chosen:
            body.append(
                f'<li><a class="row" href="/doc/{urllib.parse.quote(p.name)}">'
                f'<span class="name">{html.escape(p.name)}</span>'
                f'<span class="meta">{_fmt(p)}</span></a></li>'
            )
        body.append("</ul>")
    if pages:
        body.append("<h2>Страницы (HTML)</h2><ul>")
        for p in pages:
            body.append(
                f'<li><a class="row" href="/page/{urllib.parse.quote(p.name)}">'
                f'<span class="name">{html.escape(p.name)}</span>'
                f'<span class="meta">{_fmt(p)}</span></a></li>'
            )
        body.append("</ul>")
    return (
        f"<!doctype html><meta charset=utf-8><title>Документы ARC-AGI-3</title>"
        f'<meta name=viewport content="width=device-width,initial-scale=1"><style>{CSS}</style>'
        f"<main><h1>Документы ARC-AGI-3</h1>"
        f'<p class="lead">Локально из {html.escape(str(DOCS))}. Обновляется на лету — просто перезагрузите страницу.</p>'
        f'{"".join(body)}</main>'
    ).encode("utf-8")


def _doc(name: str) -> bytes:
    path = (DOCS / name).resolve()
    if DOCS.resolve() not in path.parents or not path.is_file():
        raise FileNotFoundError(name)
    text = path.read_text(encoding="utf-8", errors="replace")
    return (
        f"<!doctype html><meta charset=utf-8><title>{html.escape(name)}</title>"
        f'<meta name=viewport content="width=device-width,initial-scale=1"><style>{CSS}</style>'
        f'<main><div class="bar"><a class="back" href="/">← все документы</a>'
        f'<button id=c type=button>Скопировать</button><span class="ok" id=s></span></div>'
        f"<h1>{html.escape(name)}</h1>"
        f'<p class="lead">{_fmt(path)}</p><pre id=p>{html.escape(text)}</pre></main>'
        "<script>document.getElementById('c').addEventListener('click',async()=>{"
        "try{await navigator.clipboard.writeText(document.getElementById('p').textContent);"
        "document.getElementById('s').textContent='Скопировано';}"
        "catch(e){document.getElementById('s').textContent='Выделите текст и скопируйте вручную';}});</script>"
    ).encode("utf-8")


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):  # тишина в консоли
        pass

    def _send(self, payload: bytes, ctype: str = "text/html; charset=utf-8") -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        route = urllib.parse.unquote(self.path.split("?")[0])
        try:
            if route == "/":
                self._send(_index())
            elif route.startswith("/doc/"):
                self._send(_doc(route[len("/doc/"):]))
            elif route.startswith("/page/"):
                path = (DOCS / "artifacts" / route[len("/page/"):]).resolve()
                if (DOCS / "artifacts").resolve() not in path.parents or not path.is_file():
                    raise FileNotFoundError(route)
                self._send(path.read_bytes())
            else:
                self.send_error(404)   # кириллица в статусной строке роняет обработчик (latin-1)
        except FileNotFoundError:
            self.send_error(404)   # кириллица в статусной строке роняет обработчик (latin-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    port = ap.parse_args().port
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as srv:
        print(f"документы: http://127.0.0.1:{port}/", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()

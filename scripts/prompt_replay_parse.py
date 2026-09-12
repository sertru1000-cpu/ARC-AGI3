"""Разбор текстовых снимков запроса (runs/*/prompts/*.log) обратно в сообщения API.

Снимок — это отрисованный текстом последний вызов модели в игре: [SYSTEM], затем попеременно
[USER], [ASSISTANT] (текст), [REASONING], [ASSISTANT TOOL CALL: python] (id и разметка вызова),
[TOOL RESULT: id]; после [TURN TRANSCRIPT SO FAR] идёт дубль, он отбрасывается.
Проверка корректности: число восстановленных сообщений обязано совпасть с `message_count`
в заголовке снимка.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HEAD = re.compile(r"^\[(SYSTEM|USER|ASSISTANT|REASONING|ASSISTANT TOOL CALL: python|TOOL RESULT: [^\]]+|TURN TRANSCRIPT SO FAR)\]\s*$")


def parse(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    head = text[: text.find("[MODEL INPUT]")]
    meta = {k: v for k, v in re.findall(r"^(\w+): (.*)$", head, re.M)}
    body = text[text.find("[MODEL INPUT]") + len("[MODEL INPUT]"):]
    lines = body.split("\n")
    blocks, cur, buf = [], None, []
    for line in lines:
        m = HEAD.match(line)
        if m:
            if cur is not None:
                blocks.append((cur, "\n".join(buf).strip("\n")))
            cur, buf = m.group(1), []
            if cur == "TURN TRANSCRIPT SO FAR":
                break
            continue
        buf.append(line)
    if cur is not None and cur != "TURN TRANSCRIPT SO FAR":
        blocks.append((cur, "\n".join(buf).strip("\n")))

    msgs = []
    for kind, content in blocks:
        if kind == "SYSTEM":
            msgs.append({"role": "system", "content": content})
        elif kind == "USER":
            msgs.append({"role": "user", "content": content})
        elif kind == "ASSISTANT":
            msgs.append({"role": "assistant", "content": content})
        elif kind == "REASONING":
            if not msgs or msgs[-1]["role"] != "assistant":
                msgs.append({"role": "assistant", "content": ""})
            msgs[-1]["reasoning"] = content
        elif kind == "ASSISTANT TOOL CALL: python":
            cid = re.search(r"^id: (\S+)", content, re.M)
            code = re.search(r"<parameter=code>\n?(.*?)\n?</parameter>", content, re.S)
            if not msgs or msgs[-1]["role"] != "assistant":
                msgs.append({"role": "assistant", "content": ""})
            msgs[-1].setdefault("tool_calls", []).append({
                "id": cid.group(1) if cid else "call-%d" % len(msgs),
                "type": "function",
                "function": {"name": "python",
                             "arguments": json.dumps({"code": code.group(1) if code else ""})},
            })
        elif kind.startswith("TOOL RESULT: "):
            msgs.append({"role": "tool", "tool_call_id": kind[len("TOOL RESULT: "):].strip(),
                         "content": content})
    return {"meta": meta, "messages": msgs}


if __name__ == "__main__":
    import glob
    bad = 0
    for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "runs/flash_v1_phaseA/prompts/*.log")):
        r = parse(f)
        want = int(r["meta"].get("message_count", -1))
        got = len(r["messages"])
        roles = "".join({"system": "S", "user": "U", "assistant": "A", "tool": "T"}[m["role"]] for m in r["messages"])
        ok = want == got and roles.startswith("S") and roles.endswith("U")
        bad += not ok
        print("%s %-4s ожидалось %2d, разобрано %2d  %s  %s" % (
            Path(f).name[:4], "ок" if ok else "СБОЙ", want, got, roles[:40],
            "размышлений %d, вызовов %d" % (sum(1 for m in r["messages"] if m.get("reasoning")),
                                           sum(len(m.get("tool_calls", [])) for m in r["messages"]))))
    print("\nснимков с расхождением: %d" % bad)

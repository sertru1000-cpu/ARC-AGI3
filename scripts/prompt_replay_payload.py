"""Полезная нагрузка пробы промпта: 25 записанных запросов в двух вариантах, A (как было) и Б (с разбором перехода).

Из снимков восстанавливаются сообщения API (разбор проверен: число сообщений совпадает с
`message_count` во всех 25 снимках). К КАЖДОМУ сообщению пользователя возвращается картинка
доски, как в настоящем запросе: бандл кладёт картинку в сообщение при его создании и в истории
её не вырезает (7.4 картинки на запрос по прежнему замеру). Доска берётся из журнала по номеру
шага в тексте сообщения, отрисовка — копия `frame_to_png_data_url` бандла, увеличение 4
(`MULTIMODAL_UPSCALE` базового прогона). Параметры семплирования — как у анализатора:
temperature 0.6, top_p 0.95, top_k 20, мышление включено, потолка токенов нет, без сида.

Вариант Б отличается ровно одним: в текст ПОСЛЕДНЕГО сообщения пользователя перед строкой
«Current grid image:» вставлен структурный разбор последнего перехода с просьбой не пересказывать
его прозой (`scripts/prompt_replay_diff.py`).
"""
from __future__ import annotations

import base64
import glob
import gzip
import io
import json
import re
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_replay_diff import boards_for, diff  # noqa: E402
from prompt_replay_parse import parse  # noqa: E402

UPSCALE = 4
COLORS = {0: (255, 255, 255), 1: (204, 204, 204), 2: (153, 153, 153), 3: (102, 102, 102), 4: (51, 51, 51),
          5: (0, 0, 0), 6: (229, 58, 163), 7: (255, 123, 204), 8: (249, 60, 49), 9: (30, 147, 255),
          10: (136, 216, 241), 11: (255, 220, 0), 12: (255, 133, 27), 13: (146, 18, 49), 14: (79, 204, 48),
          15: (163, 86, 214)}
TOOLS = [{"type": "function", "function": {
    "name": "python",
    "description": None,  # подставляется из снимка
    "parameters": {"type": "object", "properties": {"code": {
        "type": "string",
        "description": "Python code to run. The snippet is ephemeral and is not saved across tool calls."}},
        "required": ["code"]}}}]


def png(board):
    rows, cols = len(board), len(board[0])
    im = Image.new("RGB", (cols, rows), COLORS[0]); px = im.load()
    for r in range(rows):
        for c in range(cols):
            px[c, r] = COLORS.get(int(board[r][c]), COLORS[0])
    im = im.resize((cols * UPSCALE, rows * UPSCALE), Image.Resampling.NEAREST)
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def board_at_step(ev, step):
    """Доска, которую агент видел в начале шага `step`: после действия step-1."""
    acts = {d["action_num"]: d for d in ev if d.get("type") == "action"}
    init = next(d for d in ev if d.get("type") == "initial")
    n = step - 1
    if n <= 0:
        return init["board"]
    while n > 0 and n not in acts:
        n -= 1
    return acts[n]["board"] if n > 0 else init["board"]


def with_image(msg_text, board):
    text = msg_text
    if text.rstrip().endswith("Current grid image:"):
        text = text.rstrip()[: -len("Current grid image:")].rstrip()
    return {"role": "user", "content": [
        {"type": "text", "text": f"{text}\n\nCurrent grid image:"},
        {"type": "image_url", "image_url": {"url": png(board)}}]}


def pack_board(board):
    """64x64 в строку шестнадцатеричных цифр, строки через «/» — xz сжимает её до долей КБ."""
    return "/".join("".join("0123456789abcdef"[int(v)] for v in row) for row in board)


def main():
    import lzma
    items, images = [], 0
    for f in sorted(glob.glob("runs/flash_v1_phaseA/prompts/*.log")):
        gid = Path(f).name[:4]
        snap, before, after, executed, lvl = boards_for(f)
        run = Path(f).parent.parent
        ev = [json.loads(l) for l in open(glob.glob(str(run / "artifacts" / f"{gid}*_events.jsonl"))[0], encoding="utf-8")]
        acts = {d["action_num"]: d for d in ev if d.get("type") == "action"}
        tool_desc = re.search(r"\[AVAILABLE TOOLS\]\n- python: (.*?)\n\n\[MODEL INPUT\]",
                              Path(f).read_text(encoding="utf-8", errors="replace"), re.S).group(1).strip()
        msgs = []
        for m in snap["messages"]:
            if m["role"] == "user":
                st = re.search(r"Current state: step (\d+)", m["content"])
                step = int(st.group(1)) if st else 0
                b = board_at_step(ev, step) if st else after
                lv = re.search(r"level (\d+)", m["content"])
                text = m["content"].rstrip()
                if text.endswith("Current grid image:"):
                    text = text[: -len("Current grid image:")].rstrip()
                msgs.append({"role": "user", "text": text, "board": pack_board(b), "step": step,
                             "level": int(lv.group(1)) if lv else 1})
                images += 1
            elif m["role"] == "assistant":
                mm = {"role": "assistant", "content": m["content"] or None}
                if m.get("reasoning"): mm["reasoning"] = m["reasoning"]
                if m.get("tool_calls"): mm["tool_calls"] = m["tool_calls"]
                msgs.append(mm)
            else:
                msgs.append(dict(m))
        tools = json.loads(json.dumps(TOOLS)); tools[0]["function"]["description"] = tool_desc
        items.append({"game": gid, "analysis_step": snap["meta"].get("analysis_step"),
                      "messages": msgs, "tools": tools, "diff": diff(before, after, executed, lvl)})
    payload = {"model": "Qwen/Qwen3.8-Flash-Next-NVFP4", "upscale": UPSCALE,
               "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                            "chat_template_kwargs": {"enable_thinking": True}},
               "items": items}
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    xz = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
    out = Path("kernels/notebooks_prompt_replay"); out.mkdir(parents=True, exist_ok=True)
    (out / "payload.json.xz").write_bytes(xz)
    (out / "payload.json.gz").unlink(missing_ok=True)
    print("снимков %d, досок %d, сырой объём %.0f КБ, xz %.0f КБ, base64 %.0f КБ"
          % (len(items), images, len(raw) / 1024, len(xz) / 1024, len(xz) * 4 / 3 / 1024))


if __name__ == "__main__":
    main()

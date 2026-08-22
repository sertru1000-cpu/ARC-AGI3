"""Acceptance gate for a freshly trained VL student — run on the pod, costs nothing.

Student v6 failed on Kaggle for a measurable reason, not a mysterious one: it
emitted replies 41% as long as its training targets, a third of their action()
density, and a WORLD_MODEL block 40% of the time against 74% in the data.
Those three numbers are visible WITHOUT playing a single game, so there is no
excuse for spending Kaggle quota on a checkpoint that misses them.

This script generates on held-out valid prompts and compares the student's
output against the targets in that same file:

    reply length      >= LEN_MIN   (v6: 535 chars, data: ~1300)
    WORLD_MODEL rate  >= WM_MIN    (v6: 40%,       data: ~74%)
    action() / reply  >= ACT_MIN   (v6: 0.5,       data: ~1.5)

Exit code 0 = PASS (worth deploying), 1 = FAIL (train longer / raise rank).
Comparing against the file's own targets rather than hardcoded constants keeps
the gate honest if the dataset changes.

Usage (on the pod, after run_full_vl.sh):
    BASE_MODEL=Qwen/Qwen3.6-27B ADAPTER=/workspace/out_vl27/adapter_final \
    DATA=./data_vision/valid.jsonl N=16 python check_student.py
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import statistics
import sys

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE = os.getenv("BASE_MODEL", "Qwen/Qwen3.6-27B")
ADAPTER = os.getenv("ADAPTER", "/workspace/out_vl27/adapter_final")
DATA = os.getenv("DATA", "./data_vision/valid.jsonl")
N = int(os.getenv("N", "16"))
MAX_NEW = int(os.getenv("MAX_NEW_TOKENS", "1200"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))  # runtime setting, not 0
IMAGE_SCALE = int(os.getenv("IMAGE_SCALE", "8"))
# Gates: fraction of what the data itself teaches.
LEN_FRAC = float(os.getenv("LEN_FRAC", "0.70"))
WM_FRAC = float(os.getenv("WM_FRAC", "0.80"))
ACT_FRAC = float(os.getenv("ACT_FRAC", "0.70"))

_ACT_RE = re.compile(r"\baction\s*\(")


def metrics(texts: list[str]) -> dict:
    return {
        "n": len(texts),
        "len": statistics.median(len(t) for t in texts),
        "wm": sum("WORLD_MODEL" in t for t in texts) / len(texts),
        "act": sum(len(_ACT_RE.findall(t)) for t in texts) / len(texts),
        "unclosed": sum(t.count("```") % 2 for t in texts) / len(texts),
    }


def image_caption(h: int, w: int, scale: int) -> str:
    """Byte-identical to train_vl.py and agent/harness/vision.py."""
    return (
        f"[image] The PNG above is the CURRENT board ({h}x{w} cells, each cell "
        f"drawn as a {scale}x{scale} pixel block, same 16-color palette as the "
        "ascii symbols 0-F in current_frame.ascii; pixel (px,py) -> cell "
        f"(x=px//{scale}, y=py//{scale})). Use it to see shapes, zones and "
        "layout at a glance; use the grid/ascii/segmentation in code for exact "
        "coordinates."
    )


def with_image(messages: list[dict], img: Image.Image) -> list[dict]:
    msgs = [dict(m) for m in messages]
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] == "user":
            h, w = img.height // IMAGE_SCALE, img.width // IMAGE_SCALE
            msgs[i]["content"] = [
                {"type": "image"},
                {"type": "text",
                 "text": image_caption(h, w, IMAGE_SCALE) + "\n\n" + msgs[i]["content"]},
            ]
            break
    return msgs


def main() -> None:
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("image_b64")][:N]
    if not rows:
        sys.exit(f"no examples with image_b64 in {DATA}")

    print(f"base={BASE}\nadapter={ADAPTER}\nexamples={len(rows)} temp={TEMPERATURE}", flush=True)
    proc = AutoProcessor.from_pretrained(BASE)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()

    gen: list[str] = []
    for i, r in enumerate(rows, 1):
        img = Image.open(io.BytesIO(base64.b64decode(r["image_b64"]))).convert("RGB")
        msgs = with_image(r["messages"][:-1], img)  # drop the target reply
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=TEMPERATURE > 0,
                                 temperature=TEMPERATURE or None, top_p=0.95)
        reply = proc.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True).strip()
        gen.append(reply)
        print(f"  [{i}/{len(rows)}] {len(reply)} chars, "
              f"{len(_ACT_RE.findall(reply))} action(), "
              f"WM={'yes' if 'WORLD_MODEL' in reply else 'NO'}", flush=True)

    ref = metrics([r["messages"][-1]["content"] for r in rows])
    got = metrics(gen)
    print("\n| metric | target (data) | student | gate |")
    print("|---|---|---|---|")
    checks = [
        ("reply length", ref["len"], got["len"], ref["len"] * LEN_FRAC),
        ("WORLD_MODEL rate", ref["wm"], got["wm"], ref["wm"] * WM_FRAC),
        ("action() per reply", ref["act"], got["act"], ref["act"] * ACT_FRAC),
    ]
    ok = True
    for name, want, have, gate in checks:
        passed = have >= gate
        ok &= passed
        fmt = (lambda v: f"{v:.0%}") if "rate" in name else (lambda v: f"{v:.1f}")
        print(f"| {name} | {fmt(want)} | {fmt(have)} | >= {fmt(gate)} {'PASS' if passed else 'FAIL'} |")
    if got["unclosed"]:
        print(f"\nWARNING: {got['unclosed']:.0%} of replies have an unclosed code fence "
              "(truncation -- raise MAX_NEW_TOKENS before trusting the numbers)")

    print("\nSAMPLE GENERATION:\n" + "-" * 60 + f"\n{gen[0][:1500]}\n" + "-" * 60)
    print("\nVERDICT:", "PASS -- worth deploying to Kaggle" if ok else
          "FAIL -- more epochs / higher rank; do NOT spend Kaggle quota")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""Autopsy of a 'lean' run (pod or Kaggle Phase A): did the model use the harness
aids, and what did it cost?

Per game and overall, from transcripts/*.txt (raw_tool_calls carry the code)
and artifacts/*_events.jsonl (levels, actions):
  * levels reached, actions, model calls, actions per action-bearing call;
  * output tokens per call (Qwen3 tokenizer if available, else chars/3.6);
  * helper use: calls whose code mentions find_path / probe / move_until /
    objects / cells / frame_diff; PROTOCOL headers; rejections;
  * responses with no tool call (thinking-off failure signature);
  * [DIFF]/[STATE]/[SHORT]/[NOTICE] presence in prompts (sanity: wrappers live).

usage:
    python scripts/lean_autopsy.py runs/pod_lean_25pub_04_09
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

HDR = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d:\d\d:\d\d) \|", re.M)
ARGS = re.compile(r'"arguments": "(\{.*?\})"\s*\n\s*\}', re.S)
HELPERS = ("find_path(", "probe(", "move_until(", "objects(", "cells(", "frame_diff(")


def tokenizer():
    try:
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download
        env = {}
        p = Path(".env")
        if p.is_file():
            env = dict(l.strip().split("=", 1) for l in p.read_text(encoding="utf-8").splitlines() if "=" in l and not l.startswith("#"))
        return Tokenizer.from_file(hf_hub_download("Qwen/Qwen3-30B-A3B", "tokenizer.json", token=env.get("HF_TOKEN")))
    except Exception:
        return None


def parse_transcript(text: str, ntok) -> list[dict]:
    turns = []
    pos = list(HDR.finditer(text))
    for i, m in enumerate(pos):
        seg = text[m.start(): pos[i + 1].start() if i + 1 < len(pos) else len(text)]
        th = re.search(r"\[THINKING\]\n(.*?)\n\[(?:ASSISTANT|TOOL CALL|TOOL RESULT|ANALYZER STATUS)", seg, re.S)
        asst = re.search(r"\[ASSISTANT\]\n(.*?)\n\[(?:TOOL CALL|TOOL RESULT|ANALYZER STATUS)", seg, re.S)
        codes = []
        for c in ARGS.findall(seg):
            try:
                codes.append(json.loads(json.loads('"' + c + '"')).get("code", ""))
            except Exception:
                codes.append(c)
        code = "\n".join(codes)
        tc = re.search(r"tool_call_count: (\d+)", seg)
        turns.append(dict(
            step=int(m.group(1)), time=m.group(3),
            think=ntok(th.group(1) if th else ""), text=ntok(asst.group(1) if asst else ""), code=ntok(code),
            tool_calls=int(tc.group(1)) if tc else len(codes),
            helpers=[h[:-1] for h in HELPERS if h in code],
            protocol=bool(re.search(r"^\s*(?:#[^\n]*\n|\s*\n)*PROTOCOL\s*=\s*\{", code, re.S)),
            rejected="Rejected before execution" in seg,
            has_action="action(" in code,
            prompt_state="[STATE]" in seg, prompt_diff="[DIFF]" in seg, prompt_short="[SHORT]" in seg, prompt_notice="[NOTICE]" in seg, prompt_think="[THINK]" in seg,
        ))
    return turns


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("--game", action="append")
    args = ap.parse_args()
    root = Path(args.run)
    tok = tokenizer()
    ntok = (lambda s: len(tok.encode(s).ids)) if tok else (lambda s: int(len(s) / 3.6))
    print("tokens:", "Qwen3 tokenizer" if tok else "estimate chars/3.6")
    files = sorted(glob.glob(str(root / "transcripts" / "*.txt")))
    if not files:
        print("no transcripts under", root)
        return 1
    tot = Counter(); all_turns = []; per_call_actions = []
    print(f"{'игра':6}{'lvl':>4}{'дейст':>6}{'вызов':>6}{'д/выз':>6}{'ток/отв':>8}{'мышл':>6}{'helper%':>8}{'PROTO':>6}{'откз':>5}{'noTC':>5}  хелперы")
    for f in files:
        gid = Path(f).name.split("-")[0]
        if args.game and gid not in args.game:
            continue
        turns = parse_transcript(open(f, encoding="utf-8", errors="replace").read(), ntok)
        ev_files = glob.glob(str(root / "artifacts" / f"{gid}*_events.jsonl"))
        acts = []; level = 1
        for ef in ev_files:
            for line in open(ef, encoding="utf-8"):
                e = json.loads(line)
                if e.get("type") == "action":
                    acts.append(e); level = max(level, int(e.get("level") or 1))
        by_step = Counter(e.get("analysis_step") for e in acts)
        per_call_actions += list(by_step.values())
        n = len(turns) or 1
        helper_calls = sum(1 for t in turns if t["helpers"])
        no_tc = sum(1 for t in turns if t["tool_calls"] == 0)
        rej = sum(1 for t in turns if t["rejected"])
        proto = sum(1 for t in turns if t["protocol"])
        toks = [t["think"] + t["text"] + t["code"] for t in turns]
        hc = Counter(h for t in turns for h in t["helpers"])
        print(f"{gid:6}{level:>4}{len(acts):>6}{len(turns):>6}{(len(acts)/max(1,len(by_step))):>6.1f}{st.median(toks) if toks else 0:>8.0f}"
              f"{st.median([t['think'] for t in turns]) if turns else 0:>6.0f}{100*helper_calls/n:>7.0f}%{proto:>6}{rej:>5}{no_tc:>5}  {dict(hc)}")
        tot.update(dict(games=1, levels=level - 1 if level > 1 else 0, l1=1 if level >= 2 else 0, l2=1 if level >= 3 else 0,
                        calls=len(turns), helper=helper_calls, proto=proto, rej=rej, no_tc=no_tc, actions=len(acts)))
        all_turns += turns
    c = tot; toks = [t["think"] + t["text"] + t["code"] for t in all_turns]
    print(f"\nигр {c['games']}; с L1 (уровень 2 открыт) {c['l1']}; с L2 (уровень 3 открыт) {c['l2']}; уровней пройдено всего {c['levels']}")
    print(f"вызовов {c['calls']} ({c['calls']/max(1,c['games']):.1f} на игру); действий {c['actions']} ({c['actions']/max(1,c['games']):.1f} на игру); "
          f"действий на вызов с действиями: медиана {st.median(per_call_actions) if per_call_actions else 0}, доля ≥5: {sum(1 for x in per_call_actions if x>=5)/max(1,len(per_call_actions)):.0%}")
    if toks:
        print(f"токенов на ответ: медиана {st.median(toks):.0f}, среднее {st.mean(toks):.0f}, p90 {sorted(toks)[int(len(toks)*0.9)]}; "
              f"мышление медиана {st.median([t['think'] for t in all_turns]):.0f}")
    print(f"хелперы в {c['helper']} вызовах ({c['helper']/max(1,c['calls']):.0%}); PROTOCOL в {c['proto']}; отказов {c['rej']} ({c['rej']/max(1,c['calls']):.0%}); "
          f"ответов без tool-call {c['no_tc']} ({c['no_tc']/max(1,c['calls']):.0%})")
    live = Counter(k for t in all_turns for k in ("prompt_state", "prompt_diff", "prompt_short", "prompt_notice", "prompt_think") if t[k])
    print("обёртки в промптах:", dict(live))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

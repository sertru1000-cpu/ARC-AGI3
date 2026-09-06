"""Did the model follow the PROTOCOL? Read a stock-lab run's transcripts.

For the 'protocol' hint (Gemini round 17, 03.09): every python call must start
with `PROTOCOL = {...}`; the harness rejects code without it and appends a
[NOTICE] after zero-diff calls. This reports, per game and overall:
  * python calls, how many carried the PROTOCOL header, how many were rejected;
  * how often [NOTICE] appeared and whether the next call's `diff`/`rule` text
    acknowledged it (mentions zero / no change / animation / reach);
  * a sample of PROTOCOL values (hypothesis, rule) from the deepest level.

Sources: `transcripts/*.txt` (raw_tool_calls carry the code) and
`artifacts/*_events.jsonl` (levels). Never summary.txt.

usage:
    python scripts/protocol_autopsy.py runs/stocklab_v9_protocol
    python scripts/protocol_autopsy.py runs/stocklab_v9_protocol --game lm01 --samples 6
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

CALL_RE = re.compile(r'"arguments":\s*"(\{.*?\})"\s*\n\s*\}', re.S)
TURN_RE = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \|", re.M)
PROTO_RE = re.compile(r"PROTOCOL\s*=\s*\{(.*?)\n\s*\}", re.S)
FIELD_RE = re.compile(r"['\"](inventory|diff|rule|budget|hypothesis|next)['\"]\s*:\s*(['\"])(.*?)\2\s*,?", re.S)
ACK_WORDS = ("zero", "no change", "nothing changed", "unchanged", "animation", "animating", "reach", "out of range", "no effect")


def calls_from_transcript(text: str) -> list[dict]:
    """One entry per analysis turn: code (decoded), notice flag, rejection flag."""
    turns = []
    positions = [m.start() for m in TURN_RE.finditer(text)] + [len(text)]
    for a, b in zip(positions, positions[1:]):
        seg = text[a:b]
        code = None
        m = CALL_RE.search(seg)
        if m:
            try:
                code = json.loads('"' + m.group(1).replace('"', '\\"').replace('\\\\"', '\\"') + '"')
                code = json.loads(code).get("code") if code.strip().startswith("{") else code
            except Exception:
                try:
                    code = json.loads(json.loads(f'"{m.group(1)}"')).get("code")
                except Exception:
                    code = m.group(1)
        turns.append(dict(
            code=code or "",
            notice="[NOTICE] The previous call executed actions that changed NOTHING" in seg,
            rejected="Rejected before execution" in seg,
            reminder="[PROTOCOL] Your python code must START" in seg,
        ))
    return turns


def fields(code: str) -> dict[str, str]:
    m = PROTO_RE.search(code or "")
    if not m:
        return {}
    return {k: v.strip() for k, _q, v in FIELD_RE.findall(m.group(1))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("--game", action="append")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()
    files = sorted(glob.glob(str(Path(args.run) / "transcripts" / "*.txt")))
    if not files:
        print("no transcripts under", args.run)
        return 1
    tot = Counter()
    print(f"{'игра':6}{'вызовов':>8}{'с PROTOCOL':>11}{'отказов':>8}{'NOTICE':>7}{'учтён':>6}  гипотеза (последняя)")
    samples: list[tuple[str, dict]] = []
    for f in files:
        gid = Path(f).name.split("-")[0]
        if args.game and gid not in args.game:
            continue
        text = open(f, encoding="utf-8", errors="replace").read()
        turns = calls_from_transcript(text)
        n = len(turns)
        with_proto = sum(1 for t in turns if PROTO_RE.search(t["code"]))
        rejected = sum(1 for t in turns if t["rejected"])
        notices = sum(1 for t in turns if t["notice"])
        acked = 0
        for t in turns:
            if t["notice"]:
                fl = fields(t["code"])
                blob = (fl.get("diff", "") + " " + fl.get("rule", "")).lower()
                if any(w in blob for w in ACK_WORDS):
                    acked += 1
        last = next((fields(t["code"]) for t in reversed(turns) if PROTO_RE.search(t["code"])), {})
        tot.update(dict(calls=n, proto=with_proto, rejected=rejected, notices=notices, acked=acked))
        print(f"{gid:6}{n:>8}{with_proto:>11}{rejected:>8}{notices:>7}{acked:>6}  {last.get('hypothesis', '')[:80]}")
        for t in turns:
            fl = fields(t["code"])
            if fl and len([s for s in samples if s[0] == gid]) < args.samples:
                samples.append((gid, fl))
    c = tot
    print(f"\nвсего: вызовов {c['calls']}, с PROTOCOL {c['proto']} ({c['proto']/max(1,c['calls']):.0%}), "
          f"отказов {c['rejected']}, NOTICE {c['notices']}, из них учтено в diff/rule {c['acked']}")
    print("\n=== образцы PROTOCOL ===")
    for gid, fl in samples:
        print(f"--- {gid}: " + " | ".join(f"{k}: {fl.get(k, '')[:110]}" for k in ("diff", "rule", "budget", "hypothesis")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

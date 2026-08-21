"""Convert teacher traces into an SFT dataset (chat-format JSONL).

Each training example = the dialogue prefix the student would see at some
turn, with the teacher's actual reply as the target. We reconstruct the
message history exactly as llm_policy built it (system prompt, initial user
message, per-turn feedback), so the student trains on the same distribution
it will face at inference.

Selection rules:
  - only games where the teacher completed at least one level;
  - only turns up to (and including) the turn that completed the last level
    reached — the aimless tail after the final level-up teaches nothing;
  - drop turns whose reply had no code block (degenerate);
  - drop turns whose code raised a sandbox error (don't teach mistakes).

Usage:
    .venv/Scripts/python.exe scripts/build_sft_dataset.py data/teacher --out data/sft/train.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.harness.prompts import SYSTEM_PROMPT  # noqa: E402

# ── retro prior-line injection ───────────────────────────────────────────────
# Traces collected before the `prior:` line joined the WORLD_MODEL format lack
# it; training the student on a prompt that demands the line while targets
# omit it would teach him to ignore the format. Derive the line from each
# reply's OWN goal line (keyword mapping) so every injected prior is coherent
# with what the teacher believed in that very reply.
import re  # noqa: E402

_PRIOR_RULES: list[tuple[str, str]] = [
    (r"pattern|legend|reference|reproduce|copy|template|as shown", "REPRODUCE-REFERENCE"),
    (r"program|queue|sequence.{0,30}(execut|run)|play button|one cycle", "PROGRAM-THEN-RUN"),
    (r"chase|chaser|escape|outrun|survive|opponent|enemy|avoid.{0,20}(death|trap)", "ESCAPE-AGENT"),
    (r"flow|pour|liquid|lava.{0,30}(reach|direct|target)", "ROUTE-FLOW"),
    (r"fit|cutout|slot|socket|dock|plug|assemb", "COMPLEMENTARITY"),
    (r"match|pair|same colou?r|same shape|align|overlap|meet|bring.{0,20}together"
     r"|onto.{0,20}(target|zone)|reach.{0,20}(target|exit|goal|zone)", "LIKE-TO-LIKE"),
]

_WM_RE = re.compile(r"(WORLD_MODEL:\s*\ncontrols:[^\n]*\n)", re.IGNORECASE)


def inject_prior(reply: str) -> str:
    """Insert a derived `prior:` line after `controls:` if absent."""
    if not reply or re.search(r"^prior:", reply, re.MULTILINE | re.IGNORECASE):
        return reply
    m = re.search(r"^goal:\s*(.+)$", reply, re.MULTILINE | re.IGNORECASE)
    goal = (m.group(1) if m else "").lower()
    label = next((lab for pat, lab in _PRIOR_RULES if re.search(pat, goal)),
                 "unknown yet - narrowing down")
    return _WM_RE.sub(lambda w: w.group(1) + f"prior: {label}\n", reply, count=1)


_HINT_PARA_RE = re.compile(r"\n*\[human hint\][^\n]*(\n(?!\[)[^\n]+)*", re.IGNORECASE)
_HINT_MENTION_RE = re.compile(r"\bhint(ed|s)?\b|as (the )?(human|user) (said|suggested|told)",
                              re.IGNORECASE)


CHARS_PER_TOKEN = 2.5  # measured on our grid-heavy data (MLX rehearsal, 18.08)


def _approx_tokens(msgs: list[dict]) -> int:
    return int(sum(len(m["content"]) for m in msgs) / CHARS_PER_TOKEN)


def fit_to_budget(head: list[dict], tail: list[dict], reply: dict, max_tokens: int) -> list[dict] | None:
    """Drop the OLDEST context pairs until the example fits max_tokens (the
    trainer's max_length -- anything longer gets truncated mid-reply, i.e. the
    target is lost). Keeps at least the last pair; returns None if even that
    doesn't fit (skip the example rather than train on a cut-off target)."""
    while True:
        msgs = head + tail + [reply]
        if max_tokens <= 0 or _approx_tokens(msgs) <= max_tokens:
            return msgs
        if len(tail) <= 2:
            return None
        tail = tail[2:]


def episode_examples(recs: list[dict], max_context_pairs: int = 8,
                     strip_hints: bool = False, max_tokens: int = 0) -> list[dict]:
    # Lost turns (API failures, 21.08 soft harness) carry no reply/code --
    # they are debugging records, not dialogue turns.
    turns = [r for r in recs if r.get("turn", 0) > 0 and "turn_failed" not in r]
    if not turns:
        return []
    max_level = max(r.get("level", 0) for r in turns)
    if max_level == 0:
        return []
    # Cut after the turn that reached max_level for the first time.
    last_useful = next(i for i, r in enumerate(turns) if r.get("level", 0) == max_level)
    turns = turns[: last_useful + 1]

    probe = next((r.get("probe_summary") for r in recs if r.get("turn") == 0), None)
    cross = next((r.get("cross_note") for r in recs if r.get("turn") == 0), None)
    first_user = "New unknown game. Levels 0/?. The board is in `current_frame`."
    if probe:
        first_user += "\n\n" + probe
    if cross:
        if strip_hints:
            # The student plays without human hints: teach the conclusion
            # the teacher reached, not the crutch it was handed.
            cross = _HINT_PARA_RE.sub("", cross).strip()
        if cross:
            first_user += "\n\n" + cross

    examples: list[dict] = []
    dialogue: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": first_user},
    ]
    for r in turns:
        reply = inject_prior(r.get("reply") or "")
        good = bool(r.get("code")) and not r.get("sandbox_error")
        if good and strip_hints and _HINT_MENTION_RE.search(reply):
            good = False  # reply leans on the hint explicitly -- skip as a target
        if good:
            # Trim context the same way the runtime does.
            head, tail = dialogue[:2], dialogue[2:]
            tail = tail[-max_context_pairs * 2:]
            msgs = fit_to_budget(head, tail, {"role": "assistant", "content": reply}, max_tokens)
            if msgs is not None:
                examples.append({"turn": r.get("turn"), "messages": msgs})
        dialogue.append({"role": "assistant", "content": reply})
        fb = "[python output]\n" + (r.get("sandbox_output") or "(empty)")
        if r.get("sandbox_error"):
            fb = "[python error]\n" + r["sandbox_error"] + "\n\n" + fb
        fb += (f"\n\n[status] level {r.get('level', 0)}, budget left "
               f"{r.get('budget_left', '?')}, valid actions {r.get('valid_actions', [])}. Continue.")
        dialogue.append({"role": "user", "content": fb})
    return examples


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", help="teacher data root (searched recursively)")
    p.add_argument("--out", default="data/sft/train.jsonl")
    p.add_argument("--max-context-pairs", type=int, default=8,
                   help="dialogue history pairs per example (8 = runtime "
                        "distribution; lower for memory-constrained rehearsal)")
    p.add_argument("--strip-hints", action="store_true",
                   help="remove [human hint] paragraphs from the prompt and drop "
                        "replies that explicitly lean on the hint")
    p.add_argument("--holdout", default="",
                   help="comma-separated game ids excluded from BOTH splits (eval only)")
    p.add_argument("--upweight", default="",
                   help="comma-separated game ids whose examples are duplicated "
                        "(e.g. full-win trajectories)")
    p.add_argument("--upweight-factor", type=int, default=2)
    p.add_argument("--valid-episodes", type=int, default=0,
                   help="hold this many random episodes (seeded) out as valid.jsonl "
                        "next to --out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=16000,
                   help="per-example token budget (~chars/2.5); older context pairs "
                        "are dropped to fit, examples that can't fit are skipped. "
                        "Match the trainer's MAX_LENGTH. 0 = off")
    p.add_argument("--since", default="",
                   help="only trace dirs whose name contains a timestamp >= this "
                        "(YYYYMMDD_HHMMSS), e.g. 20260821_000000 for the vision era")
    args = p.parse_args()

    import random
    holdout = {g.strip() for g in args.holdout.split(",") if g.strip()}
    upweight = {g.strip() for g in args.upweight.split(",") if g.strip()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    episodes: list[tuple[str, str, list[dict]]] = []  # (game, episode_id, examples)
    n_files = n_skipped_holdout = 0
    for path in sorted(Path(args.src).rglob("*.jsonl")):
        if path.name.startswith("_"):
            continue
        if args.since:
            m = re.search(r"(\d{8}_\d{6})", path.parent.name)
            if not m or m.group(1) < args.since:
                continue
        game = path.stem.split("-")[0]
        if game in holdout:
            n_skipped_holdout += 1
            continue
        n_files += 1
        recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        exs = episode_examples(recs, max_context_pairs=args.max_context_pairs,
                               strip_hints=args.strip_hints, max_tokens=args.max_tokens)
        if not exs:
            continue
        eid = f"{path.parent.name}/{path.stem}"
        for ex in exs:
            ex["source"] = path.stem
            ex["episode_id"] = eid
        episodes.append((game, eid, exs))

    rng = random.Random(args.seed)
    valid_ids: set[str] = set()
    # Upweighted (full-win) episodes are too valuable to spend on validation.
    candidates = [e for e in episodes if e[0] not in upweight]
    if args.valid_episodes > 0 and len(candidates) > args.valid_episodes:
        valid_ids = {eid for _, eid, _ in rng.sample(candidates, args.valid_episodes)}

    n_train = n_valid = n_dup = 0
    valid_path = out.parent / "valid.jsonl"
    with out.open("w", encoding="utf-8") as ftr, valid_path.open("w", encoding="utf-8") as fva:
        for game, eid, exs in episodes:
            if eid in valid_ids:
                for ex in exs:
                    fva.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    n_valid += 1
                continue
            reps = args.upweight_factor if game in upweight else 1
            for _ in range(reps):
                for ex in exs:
                    ftr.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    n_train += 1
            if reps > 1:
                n_dup += len(exs) * (reps - 1)

    per_game: dict[str, int] = {}
    for game, eid, exs in episodes:
        if eid not in valid_ids:
            per_game[game] = per_game.get(game, 0) + 1
    print(f"scanned {n_files} traces (+{n_skipped_holdout} held out: {sorted(holdout)}) "
          f"-> {len(episodes)} successful episodes -> train {n_train} examples "
          f"(incl. {n_dup} upweight duplicates), valid {n_valid} examples "
          f"from {len(valid_ids)} episodes")
    print("episodes per game (train):", dict(sorted(per_game.items())))
    print(f"dataset: {out}  valid: {valid_path if n_valid else '(none)'}")


if __name__ == "__main__":
    main()

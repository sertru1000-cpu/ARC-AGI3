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

# ── reformatting: drop fruitless inspections + gate-dodge predicts ─────────
# See docs/student_vs_teacher_analysis.md #5.2: the student learned "modal
# turn = look-only" because inspection turns and the dummy-predict gate-dodge
# ritual outnumber decisive turns in the SFT target distribution.

_TRIVIAL_RETURN_RE = re.compile(r"return\s+(grid|g)(\.copy\(\))?\s*$")


def _predict_bodies(code: str) -> list[str]:
    """Extract the body of every top-level `def predict(...)` in the code
    (indentation-delimited, since the sandbox executes plain Python)."""
    lines = code.splitlines()
    bodies: list[str] = []
    i = 0
    while i < len(lines):
        m = re.match(r"^(\s*)def predict\(", lines[i])
        if m:
            indent = len(m.group(1))
            body: list[str] = []
            i += 1
            while i < len(lines):
                line = lines[i]
                if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                    break
                body.append(line)
                i += 1
            bodies.append("\n".join(body))
        else:
            i += 1
    return bodies


def is_dummy_predict(code: str) -> bool:
    """A `predict(grid, action, data)` whose entire body is a no-op return --
    the gate-dodge used to fake a passing verify_theory accuracy without a
    real transition theory (analysis doc #4: `def predict(...): return
    grid.copy()`, called just to unlock long action batches)."""
    for body in _predict_bodies(code):
        stmts = [ln.strip() for ln in body.splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        if stmts and all(_TRIVIAL_RETURN_RE.fullmatch(s) for s in stmts):
            return True
    return False


def is_fruitless_inspection(turns: list[dict], i: int) -> bool:
    """A zero-action turn that doesn't precede a productive (acting) turn --
    i.e. it's part of a stalled inspection run, not the normal one-turn look
    before an action. The teacher's real productive pattern is diff-then-act;
    a turn that neither acts nor is immediately followed by acting teaches
    the student nothing about that pattern."""
    r = turns[i]
    if r.get("actions_executed", 0) > 0:
        return False
    nxt = turns[i + 1] if i + 1 < len(turns) else None
    return not (nxt and nxt.get("actions_executed", 0) > 0)


def is_decisive(turns: list[dict], i: int) -> bool:
    """A turn worth upweighting: it batches multiple actions in one turn, or
    a level-up follows within 2 turns."""
    r = turns[i]
    if r.get("actions_executed", 0) >= 2:
        return True
    level = r.get("level", 0)
    for j in range(i + 1, min(i + 3, len(turns))):
        if turns[j].get("level", 0) > level:
            return True
    return False


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


_ARC_CHARS = "0123456789ABCDEF"


def frame_ascii_to_png_b64(ascii_grid: str, scale: int = 8) -> str:
    """Rebuild the exact PNG the teacher saw from the trace's frame_seen."""
    import base64
    import numpy as np
    from agent.harness.vision import grid_to_png_bytes
    rows = [[_ARC_CHARS.index(ch) if ch in _ARC_CHARS else 0 for ch in line]
            for line in ascii_grid.splitlines() if line]
    grid = np.array(rows, dtype=np.int64)
    return base64.b64encode(grid_to_png_bytes(grid, scale)).decode("ascii")


def episode_examples(recs: list[dict], max_context_pairs: int = 8,
                     strip_hints: bool = False, max_tokens: int = 0,
                     vision: bool = False, drop_fruitless: bool = False,
                     drop_dummy_predict: bool = False,
                     upweight_decisive_factor: int = 1,
                     stats: dict | None = None) -> list[dict]:
    if stats is None:
        stats = {}
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
    for i, r in enumerate(turns):
        reply = inject_prior(r.get("reply") or "")
        good = bool(r.get("code")) and not r.get("sandbox_error")
        if good and strip_hints and _HINT_MENTION_RE.search(reply):
            good = False  # reply leans on the hint explicitly -- skip as a target
        stats["turns_considered"] = stats.get("turns_considered", 0) + 1
        if good and drop_fruitless and is_fruitless_inspection(turns, i):
            good = False
            stats["fruitless_dropped"] = stats.get("fruitless_dropped", 0) + 1
        if good and drop_dummy_predict and is_dummy_predict(r.get("code") or ""):
            good = False
            stats["dummy_predict_dropped"] = stats.get("dummy_predict_dropped", 0) + 1
        if good:
            # Trim context the same way the runtime does.
            head, tail = dialogue[:2], dialogue[2:]
            tail = tail[-max_context_pairs * 2:]
            msgs = fit_to_budget(head, tail, {"role": "assistant", "content": reply}, max_tokens)
            if msgs is not None:
                ex = {"turn": r.get("turn"), "messages": msgs}
                if vision:
                    # The board the teacher LOOKED AT for this reply; the VL
                    # trainer attaches it to the last user message exactly
                    # like VisionLLM does at runtime. Traces predating
                    # frame_seen (round 1) yield no vision example.
                    if not r.get("frame_seen"):
                        ex = None
                    else:
                        ex["image_b64"] = frame_ascii_to_png_b64(r["frame_seen"])
                if ex is not None:
                    decisive = is_decisive(turns, i)
                    stats["kept"] = stats.get("kept", 0) + 1
                    if decisive:
                        stats["decisive"] = stats.get("decisive", 0) + 1
                    reps = upweight_decisive_factor if decisive else 1
                    for _ in range(reps):
                        examples.append(dict(ex))
                    stats["decisive_dup"] = stats.get("decisive_dup", 0) + (reps - 1)
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
    p.add_argument("--vision", action="store_true",
                   help="add image_b64 (PNG of the board the teacher saw, from frame_seen) "
                        "to every example; examples without frame_seen are skipped")
    p.add_argument("--max-tokens", type=int, default=16000,
                   help="per-example token budget (~chars/2.5); older context pairs "
                        "are dropped to fit, examples that can't fit are skipped. "
                        "Match the trainer's MAX_LENGTH. 0 = off")
    p.add_argument("--since", default="",
                   help="only trace dirs whose name contains a timestamp >= this "
                        "(YYYYMMDD_HHMMSS), e.g. 20260821_000000 for the vision era")
    p.add_argument("--drop-fruitless-inspection", action="store_true",
                   help="drop zero-action turns that don't immediately precede an "
                        "acting turn (stalled-inspection runs, not the normal "
                        "look-before-you-act pattern)")
    p.add_argument("--drop-dummy-predict", action="store_true",
                   help="drop turns whose code defines a no-op predict() used only "
                        "to fake a passing verify_theory gate check")
    p.add_argument("--upweight-decisive-factor", type=int, default=1,
                   help="duplicate 'decisive' examples this many times: turns that "
                        "batch >=2 actions, or are followed by a level-up within 2 turns")
    args = p.parse_args()

    import random
    holdout = {g.strip() for g in args.holdout.split(",") if g.strip()}
    upweight = {g.strip() for g in args.upweight.split(",") if g.strip()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    episodes: list[tuple[str, str, list[dict]]] = []  # (game, episode_id, examples)
    n_files = n_skipped_holdout = 0
    agg_stats: dict[str, int] = {}
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
        ep_stats: dict[str, int] = {}
        exs = episode_examples(recs, max_context_pairs=args.max_context_pairs,
                               strip_hints=args.strip_hints, max_tokens=args.max_tokens,
                               vision=args.vision,
                               drop_fruitless=args.drop_fruitless_inspection,
                               drop_dummy_predict=args.drop_dummy_predict,
                               upweight_decisive_factor=args.upweight_decisive_factor,
                               stats=ep_stats)
        for k, v in ep_stats.items():
            agg_stats[k] = agg_stats.get(k, 0) + v
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
    if args.drop_fruitless_inspection or args.drop_dummy_predict or args.upweight_decisive_factor > 1:
        considered = agg_stats.get("turns_considered", 0)
        fruitless = agg_stats.get("fruitless_dropped", 0)
        dummy = agg_stats.get("dummy_predict_dropped", 0)
        kept = agg_stats.get("kept", 0)
        decisive = agg_stats.get("decisive", 0)
        decisive_dup = agg_stats.get("decisive_dup", 0)
        frac = decisive / kept if kept else 0.0
        print(f"reformat: {considered} candidate turns -> dropped {fruitless} fruitless-inspection, "
              f"{dummy} dummy-predict -> {kept} kept ({kept + decisive_dup} after decisive upweight); "
              f"decisive {decisive}/{kept} ({frac:.1%})")
    print(f"dataset: {out}  valid: {valid_path if n_valid else '(none)'}")


if __name__ == "__main__":
    main()

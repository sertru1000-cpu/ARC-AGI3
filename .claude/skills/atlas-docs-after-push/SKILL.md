---
name: atlas-docs-after-push
description: "After ANY Kaggle push for this project (arc3-atlas-src dataset version, or an arc3-atlas kernel push) — sync docs/plan_top10_by_3009.md and the 'Atlas Version History' artifact so they reflect what actually shipped. Run this immediately, in the same turn as the push, without waiting for the user to ask."
metadata:
  author: user
  version: "1.0.0"
---

# Atlas docs-after-push sync

Scope: this project only (ARC-AGI-3 / Kaggle ARC Prize 2026, the `atlas_src` substrate). Triggers on a successful `kaggle datasets version -p atlas_src ...` or `kaggle kernels push -p ...` for `sergueimakarov/arc3-atlas-src` / `sergueimakarov/arc3-atlas`.

**This skill does not authorize a push.** Pushing still requires the user's separate, per-artifact consent ([[feedback-kaggle-push-consent]] in memory — never push on your own initiative). This skill only covers what happens right after a push the user already approved: keeping the two living documents in sync so the user never has to remind you.

## Do this immediately after every push, same turn

1. **Update `docs/plan_top10_by_3009.md`.**
   - Find the backlog item(s) describing what was just shipped.
   - Replace any stale "не запушено" / "ждёт согласия" wording with what actually happened: dataset pushed, kernel not rebuilt yet (say so explicitly — a dataset push alone does not change the running kernel); or kernel pushed as version N.
   - If the change isn't in the backlog yet at all (e.g. a same-day fix that was implemented and pushed in one go), add it before marking it pushed.

2. **Update the "Atlas Version History" artifact.**
   - Find its URL via `Artifact action: list` if you don't already have it in context (it's also referenced from `arc-agi-3-next-steps` memory). Read it with `action: read` before editing.
   - Edit the **local scratchpad source file** that was used to publish it last (search the scratchpad dir for `atlas_versions.html` or similar — do not hand-reconstruct from the fetched frame-wrapped copy, which has extra runtime chrome around the real content).
   - **A kernel push** (new version number): add a full timeline entry — version number, date, badge (`phase-a` calibration vs `submission` Phase B vs `pending`/cancelled), what changed (`<ul class="changes">`, mark real bug fixes with `class="fix"`), and results IF ALREADY CONFIRMED (see honesty rule below).
   - **A dataset-only push** (no kernel rebuild): add or update the special dashed "src" entry (`entry pending`, badge text "ждёт кернела", no invented version number) listing what changed. Do not fold it into a kernel version entry that hasn't actually been rebuilt with this content.
   - Update the stat-strip at the top (current version number, latest confirmed score) if it's now stale.
   - Republish to the **same URL** (pass `url:` to `Artifact`) so it updates in place rather than creating a duplicate.

3. **Update the `arc-agi-3-next-steps` memory** with a short paragraph, same as the rest of this session's continuity log — this is the mechanism that carries context into the next session, not optional bookkeeping.

## Honesty rule (do not skip)

Never write a score, "confirmed clean," or a win/loss count into either document unless you've actually pulled and read the corresponding log/transcript/`kaggle competitions submissions` output in this session. A kernel push that finished (`kernels status` = COMPLETE) is not the same as "results analyzed" — if you haven't looked, write "Phase A завершена, не разобрана" / "results not yet analyzed," not a fabricated or guessed number. This bit the version-history artifact once already (stats stayed at v12's numbers for two kernel versions because nobody had gone back to update them) — don't repeat it.

## Why this exists

User (24.08.2026): "После push datset обязательно обновляй всю документацию - беклог, справочник версий и т.д. Чтобы я каждый раз не напоминал." Both documents had drifted behind actual pushes multiple times this session (extract= shipped and pushed, but the backlog still said "не запушено"; v13/v14 existed but the version-history artifact stopped at v12) — this skill exists so that drift stops happening silently.

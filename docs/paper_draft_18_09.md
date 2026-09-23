# Scaffolding Multiplies Ability It Cannot Create: Twenty Measured Layers, a Score Budget, and the Engineering of Changing the Core on ARC-AGI-3

**Code:** Kaggle kernels `sergueimakarov/arc3-stock-flash` (baseline) and `arc3-stock-flash-predhint` (goal-predicate layer).

## Abstract

We ran a five-week ablation programme on ARC-AGI-3 with a fixed open-weights agent (Duck harness + Qwen3.8-Flash-Next-NVFP4 on one 96 GB card). Twenty harness layers produced no score change distinguishable from run-to-run spread. We localised the failure — the model induces correct mechanics but not goals — and formalised the level goal as a machine-checkable predicate. Under an honest control (a random non-goal state declared the goal, parameters re-fitted by the same search) the vocabulary separates true completion states on 83% of levels against 51% for pseudo-goals. Leave-one-game-out prediction of the next level's goal succeeds in 27–35% of 26 level pairs against 0–2% for two nulls. The decisive measurement is a goal-conditioned progress function: it separates winning from sibling actions at AUC 0.63 given the true completion frame and at chance (0.47) given the goal transferred from the previous level — the only goal available in play.

Two further contributions are aimed at anyone choosing where to spend effort on this benchmark. First, a **score budget**: 64% of the levels our agent completes are already at the scorer's cap, so all efficiency work is bounded at +24%, clearing a first level in every game is worth +32%, and one additional level per game is worth +137% — the only lever that exceeds the current score, and the one no harness layer moved. Two popular ideas die in this arithmetic: two short passes with best-of-runs scoring lose 19% against one long pass, and reallocating time between games is bounded near +4%. Second, changing the core model is measured rather than assumed: a stronger model crushed to 2–3 bits to fit the card is *worse* than the baseline (1 level against 3–5), an uncompressed stronger model through the same harness is far better (10 levels in 14 minutes against 4/5/3), and we document, with the exact constraints, two working ways to deliver modified weights into the sealed competition kernel — a runtime adapter that costs four serving concessions, and a byte-level weight overlay that costs nothing at all.

## 1. Setup

All measurements use the 25 public games, the Duck harness (Milestone 1 winner) and Qwen3.8-Flash-Next-NVFP4 on one RTX Pro 6000 — the configuration the public leaderboard converged on. A deterministic local replica reproduces recorded trajectories exactly (561/561 actions; per-level counts matched official counters on 121/121 levels), making counterfactual analysis free.

**Baseline spread is the unit of measurement.** Three runs of the *identical* configuration score 1.67 / 2.73 / 3.32 at a 30-minute per-game cap and 5.46 / 6.32 / 7.23 at 80 minutes; the same build scored 1.85–4.09 across ten competition submissions. Any claim ignoring this spread is unfalsifiable, so every probe was pre-registered against three truncations of base runs.


## 2. Twenty layers, no signal

Layers, each with pre-registered thresholds and a paired bootstrap: goal gate with refutation, stuck reset, scripted explorer, level-fact injection, macro-actions, prefix caching, best-of-N voting, ten executable-world-model variants, history truncation, oracle rule injection, doubled call budget, world-model carry, three search layers. No delta fell outside baseline spread; every bootstrap interval covered zero. Three findings run against intuition:

* Doubling calls per game (56 → 112) moved 40 → 41 levels: the extra budget went into revisiting states (returns ×2.5).
* Removing reasoning tokens destroys the agent (2.91 vs 9.43), while *adding* input information — oracle rules, world-model digests, level facts — changes almost nothing: the model restated an injected rule in 27 of 27 answers and still did not act on it.
* History is working memory: truncating it halved the action rate (44% → 22%).

## 3. Mechanics without goals

The same model that gains no score writes *correct dynamics*: its synthesised transition programs predicted unseen transitions in 86% of changed states (89/103), and on one game a counterfactual check was correct in 200 of 200 out-of-distribution clicks. Asked explicitly for the level goal, it was right in 1 of 6 games, and a learned goal detector lost to the untrained heuristic "largest board change" in all 5 held-out games. Six transfer measurements (imitation and cross-game policies, few-shot, online context, shuffled labels, learned detector) returned zero transfer to held-out games at 0.64–0.68 accuracy on training games — memorisation, not rules.

## 4. Goals as verifiable predicates

We define a level goal as a predicate over board state drawn from a 21-template vocabulary (colour presence and counts; component counts; single component; rectangle; symmetry; shape match; containment; region-pattern equality; match to a hole — the "complementary object"; match to a shape seen earlier; and four templates comparing the state to the level's start). Induction keeps predicates true at the completion frame and false at every state seen on that level.

* **Separation, honestly controlled.** The true completion state is separated from every other state of its level on 44 of 53 levels (83%). When a random non-goal state is declared the goal and the same search re-fits its parameters, it is separated in 536 of 1060 trials (51%). Our first control (8%) re-used parameters fitted to the true goal and so tested the wrong hypothesis: the honest gap is 83 against 51, not 90 against 8.
* **Cross-level transfer, leave-one-game-out** (26 pairs; per-kind rates estimated on the other 14 games): Coverage@1 = 27% for all three parameter-transfer rules, Coverage@3 = 35% for the monotone rule ("fewer than before"). Nulls: the same transfer from a *pseudo-goal* covers 1–2%, and from another game's goal 0–1% — so what transfers is the goal, but weakly.
* **Per-kind precision** curates the vocabulary: "colour c is gone" transfers 6/6, "exactly m components of colour c" 4/34, shape matches 11–12%. Keeping kinds above 40% raises precision from 31% to 70% and halves coverage (23 of 53 levels vs 38).

## 5. Search guided by goal hypotheses

A best-first search over the real engine, ordered by a dense residual (assignment-problem matching between object sets: shape mismatch plus centroid distance) toward parameter-free candidates from the same vocabulary, solves **14 of 25** first levels — as many as blind BFS under identical budgets — using **64,092 moves against 127,911** on the 13 games both solve. One design fact was measured, not assumed: step-function residuals carry no signal (median Spearman ρ = −0.02) while dense matching residuals do (ρ = 0.20–0.93 where the goal is in the vocabulary).

Search also bounds the popular "learn a simulator, plan inside it, execute the plan" recipe. Granted a *perfect* simulator — search on the real engine with state snapshots — and 420 s and 20,000 states per game, breadth-first search finishes **1 of 25 games** (tu93: 9 levels in 185 moves against a 462-move baseline, a perfect level score, after 8,334 search moves), clears at least one level in 15, and none in 10. What limits planning here is the size of the space, not the fidelity of the model of it.

## 6. Why search does not pay, and what the goal hint did

A level scores `(baseline_actions / actions_used)²`, and search actions are counted: a level reached after thousands of search moves is worth ≈0, and a level the model would have solved itself is destroyed if search touched it. Counterfactual replay of three base runs quantified this before compute was spent — search after 30 minutes of no progress costs −0.2 to −1.1, before the model's first call −2.2 to −2.3 (30–42% of the score) — and a probe confirmed sign and magnitude (−0.58 to −2.78).

Checked in the live bundle rather than assumed: the harness sets `ONLY_RESET_LEVELS=true` process-wide, so the engine never performs the full reset that would open a fresh scored run, and the scorecard admits one run per game id. The "win, then replay optimally" strategy that works offline is unavailable in play, so exploration cost is never recovered — which is why a training-free graph-exploration agent ranked 3rd by *levels solved* would score ≈0 here.

The remaining zero-cost use is to hand the model the verified goal of the level it just completed. Two probes: 0.25 vs 0.35 baseline with a noisy vocabulary, 0.38 vs 0.28 with the curated one (p = 0.51). The model reads the hint (112 of 379 responses), but the effect is indistinguishable at this sample size, and whole-run score cannot measure it: half the score comes from first levels the layer never sees.



**The decisive test is about progress, not recognition.** For every transition on a known solution path we computed an object-matching distance to the goal and asked whether it falls more often for the winning action than for its siblings. With the *true* completion frame as the goal, it does: 0.69 against 0.49, AUC 0.629 over 89 transitions in 10 games. With the goal *transferred from the previous level* it does not: 0.17 against 0.09, AUC 0.473. A pre-registered criterion (AUC ≥ 0.60 and a ≥ 0.10 gap) therefore closes this line as a score lever while leaving it intact as a measurement instrument.

## 7. The score budget: where points can come from at all

Before asking which layer to build, we computed what the scorer can pay for. Using the official rule (the ratio is clipped at 1.0 *before* squaring, so a level is worth at most 100) on three full runs of the base:

| Counterfactual | Score | Change |
|---|---|---|
| As played | 8.11 | — |
| Every completed level played perfectly | 10.06 | +24% |
| At least one level cleared in every game (16% of game-runs clear none) | 10.68 | +32% |
| **One more level cleared per game** | **19.21** | **+137%** |

The first row of that table is the important one: **64% of the 121 completed levels are already at the cap** (median ratio baseline/actions = 1.16), and the levels below the cap average 47 of 100. Efficiency work is therefore bounded at about a quarter, while depth is worth more than the entire current score. Every harness lever we measured acts on the bounded half.

Two further counterfactuals close popular ideas. First, the competition scores a game by the *best* run, and our run-to-run spread is large (`ar25` scored 24.5 / 8.3 / 22.2 across identical runs), which suggests two short passes instead of one long one. Replaying the three runs truncated to half the per-game cap and taking the per-game maximum of two gives **6.56 against 8.11** — worse, because halving the time costs 38% and the maximum recovers only part of it. Second, the marginal value of time falls steeply: the second half of a 132-minute game produces 38% of its score, while a further 108 minutes (a 4-hour cap probe) adds 24% more levels but only **+3.8%** score. Reallocating time between games is therefore bounded by roughly the same +4%.

## 8. Is the ceiling the model?

Twenty negative layers make the claim *scaffolding cannot create ability* plausible but not tested: we never varied the ability. We did so twice, holding the harness, the engine and the games fixed and changing only the core.

**A stronger model, crushed by quantisation, is worse.** DeepSeek-V4-Flash is a larger and stronger model than our baseline, but the competition GPU (96 GB) only fits it at 2 or 3 bits. On the same four games at a 30-minute cap it cleared **1 level against 3–5 for the baseline**, with and without thinking, at both quantisations (4 configurations, 1 level each). Server timings attribute the loss to generation, not to prompt reading: 18.7–27.5 tokens/s per request, with prompt processing 9–13% of wall time. Capacity that does not fit the card is not capacity.

**An uncompressed stronger model is much better.** Gemini-3.6-Flash through the same harness on the same local engine cleared **10 levels in ~14 minutes on four games**, against 4 / 5 / 3 for three baseline runs at 30 minutes. Per-level move efficiency (human baseline ÷ agent moves) had median 0.82, with 8 of 10 levels inside 2× the human baseline; the baseline's own median is 2.16 on the levels it does clear, so the teacher trades efficiency for depth. Repeating one game at the *battle* context window (32 768 rather than the 131 072 we first used by mistake) reproduced the depth exactly — 4 of 9 levels on `tu93`, RHAE 17.9 against 15.1 — showing the advantage is not an artefact of extra memory.

**What the stronger model does differently per turn.** On the same four games, counted from transcripts: the teacher emits 45 lines of tool code per harness turn against the baseline's 14, executes 6.2 engine moves per turn against 2.1, and leaves only 10% of turns without any move against the baseline's 50%. Both consult the frame segmentation at the same rate (0.48 vs 0.57), so the difference is not tool access but what is done with one look. Note that this does *not* identify spare turns as the binding constraint: doubling the call budget in §2 moved 40 levels to 41. The teacher is not taking more turns, it is finishing more per turn.

Together with §2 these bound the problem: on this task the scaffolding ceiling is set by the core model, and the core model that fits the hardware budget is the constraint. A harness change that would matter must therefore either buy ability (a better model that fits, or distillation from a better one) or exploit a structure the model already has — which §§4–6 measured and did not find.

**Three cheap tests of the remaining structure, all negative.** Ranking the winning move against its siblings by displacement toward the *true* goal reaches top-1 0.44 (chance 0.19); with the goal transferred from the previous level — the only goal available in play — it is 0.19, exactly chance, and the measure is flat across all moves in 53% of states. Choosing moves to discriminate among surviving goal hypotheses (median 79 per state) reaches 0.23; averaging progress over all live hypotheses reaches 0.34. Transferring *verified mechanics* between levels has a hard score ceiling of +2.4% (112 of 2568 moves on cleared levels are first uses of a simple action, and 78–80 of them had no stable effect on the previous level anyway). Finally, stalled games are not looping: 72% of their moves reach states never seen before, so anti-loop machinery has nothing to catch — what repeats is the *pattern* of moves (64% of click triples), not the position.

**Two measurement pitfalls worth stating.** First, the OpenAI-compatible endpoint of our teacher reports `completion_tokens` without thinking tokens while billing them as output: on a fixed prompt it reported 528 completion tokens with 8 076 total, hiding 7 473 — a token-based cost model understates spend twofold. Second, a serving parameter transplanted between stacks silently changed the experiment: `131072` was the *server-wide* context split across four slots in one setting and the *per-conversation* window in another, giving the agent four times the memory it has in competition. Both were caught only by comparing against an external ledger (the account balance, the recorded competition environment), which we now do by default.

## 9. Delivering a modified model to the competition kernel

If the ceiling is the core model, a team's next step is to change it — which raises an engineering question we could not find documented anywhere: can a modified model actually be served inside the competition kernel, where there is no internet, the checkpoint is pinned by hash, and the card is already 89% full? We measured both routes on the real kernel. This section is the part of our work most directly reusable by others.

The checkpoint is a 125B mixture-of-experts model in 419 files: **68 GB of routed experts in NVFP4, 16 GB of bf16 modules** (full attention, linear attention, routers, shared expert, embeddings) and 51 GB of per-layer embeddings that live in host RAM. Only the bf16 part is trainable in practice, and a rank-16 adapter over all 48 layers is 21.9M parameters — 0.018% of the model, 44 MB on disk.

**Route A — serve the adapter at runtime.** vLLM does apply LoRA to this architecture, and — contrary to a risk we had recorded from an earlier experiment — it applies it both to the 12 full-attention layers and to the 36 linear-attention layers; the only modules it silently ignores are the vision tower's. Getting the server to start, however, required four concessions, each found by a separate failed run: the KV cache had to drop from 5 GiB to 3 GiB (it cannot go below 1.55 GiB, the floor for a single 32,768-token window); the batch had to drop from 8,192 to 2,048 tokens, because with LoRA enabled vLLM materialises a logits buffer over the whole batch (8,192 × 248,320 vocabulary × 2 bytes = 3.79 GiB); CUDA graphs had to be disabled; and only one adapter may be active per batch. All four cost throughput, which on this task is the scarce resource.

**Route B — merge into the weights and overlay the files.** Because the trainable part is confined to four bf16 shards (0.32 / 1.27 / 3.68 / 10.73 GB), the trained delta can be written into those shards and the remaining 119 GB left untouched. The kernel builds a directory in which 419 model files are **symbolic links** to the mounted read-only checkpoint and only the modified shards are real files, rewrites the manifest, and points the loader at it. Weights must be patched **in place at the byte level** — re-serialising a safetensors file changes its size and fails the harness's byte-total check. Verified on the kernel: the harness accepted the overlay (`files=419 bytes=135,253,622,894`), the server started on the **stock profile** (5 GiB cache, concurrency 28, CUDA graphs on), and zeroing two matrices in 2 of 48 layers visibly degraded play — the model answered a fixed prompt with `1, 2, 3,` where the stock model answers `1, 2, 3, 4,`, and 25 games produced 73 moves instead of thousands. Route B therefore delivers modified weights **at no runtime cost**, and a team never needs to upload the 135 GB checkpoint: a 44 MB adapter travels as a dataset and is merged inside the kernel.

## 10. What a distillation attempt would cost, measured rather than guessed

For teams considering the obvious next step — distil a stronger teacher into the served model — we measured the inputs. A teacher (Gemini-3.6-Flash) on our harness produces one training example per *harness turn*, not per model call: at the battle context window a turn takes 1.74 calls, and 69 turns cost $2.78, i.e. **$0.040 per turn**. Only turns inside levels the teacher actually cleared are usable (52%), giving **$0.077 per usable example**; the median example is 26,830 tokens with the history reconstructed as the harness sends it, so an epoch over 2,500 examples is 67M tokens. Training the bf16 modules only — the fast regime, since adapters over fused expert tensors ran 12× slower in our earlier work — puts a two-epoch run in the low hundreds of dollars on rented B200s rather than the four figures we first estimated.

One accounting trap is worth publishing: the OpenAI-compatible endpoint of the teacher reports `completion_tokens` **without** thinking tokens while billing them as output. On a fixed prompt it reported 528 completion tokens against 8,076 total, hiding 7,473. A token-based cost model built on that field understates spend twofold; only the account balance is authoritative. Reducing the thinking budget does not help either: at `reasoning_effort: low` the per-call price fell 2.3× but the model compensated with 3.7 tool iterations per turn instead of 1.7, leaving the price **per turn** unchanged ($0.039 vs $0.040) and the play visibly worse (first level cleared in 12× the human baseline).

## 11. Reproducibility

Every number above comes from a recorded run or a script in the project repository, and the counterfactual analyses (score budget, truncations, search bounds, goal induction) run offline on a deterministic replica of the engine, so they can be re-executed without a GPU. The Kaggle kernels behind the live measurements are `arc3-stock-flash` (baseline), `arc3-stock-flash-predhint` (goal-predicate layer), `arc3-lora-canary-j` (runtime adapter) and `arc3-merge-canary` (weight overlay).

## 12. What we contribute

1. **A negative result with a mechanism:** twenty harness layers multiply an ability that is absent, the missing ability is not dynamics, and swapping the core model — up or down — moves the score where no layer did.
2. **A measurement discipline:** baseline spread from ≥3 identical runs; pre-registered thresholds; counterfactual evaluation on a deterministic replica, which predicted a layer's sign before compute was spent; and controls that re-fit parameters to a pseudo-goal, overturning two of our own headline numbers.
3. **A goal vocabulary with measured per-kind transfer,** turning "the agent lacks goals" into numbers a successor can build on.
4. **A score budget** that tells a successor which levers can matter before any of them is built, and two negative counterfactuals (best-of-two short passes, time reallocation) derived from recorded runs at zero compute cost.
5. **The engineering of changing the core inside a sealed kernel:** the exact serving constraints a runtime adapter imposes, and a weight-overlay route that imposes none — with the byte-level patching rule that makes a pinned-hash checkpoint modifiable without re-uploading 135 GB.

The natural extension to *histories* (ever / never / count / before / within / action-run) covers all 53 levels and fails its control completely: a random trajectory prefix, where no level was cleared, is "separated" in 986 of 986 trials. Coverage without a control is worthless.

**What would change our mind:** a progress measure that separates winning from sibling actions at AUC ≥ 0.6 using only a *transferred* goal, or evidence that the next-level rate rises on ≥60 first-level games. Absent either, our reading is that on a fixed small-for-the-task model the binding constraint is turning a known goal into a plan, not naming the goal.

**Limitations.** All live measurements are on the 25 public games; the hidden set is roughly four times larger and our public-to-hidden transfer ratio is about 0.35, so absolute numbers do not carry over even though the counterfactual comparisons do. The score budget is computed on three base runs and inherits their spread. The delivery experiments prove that modified weights *load and take effect*; they do not prove that a *trained* adapter helps, which remains the open question and the one we would spend money on next — first one epoch on the cheapest surface, then a measurement, and only then the rest. Finally, our own attempt at that surface in an earlier experiment under-fitted badly (0 levels of 16 with the experts frozen), which is the honest prior against the idea we are recommending others evaluate.

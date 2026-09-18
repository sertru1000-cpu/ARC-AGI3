# Scaffolding Multiplies Ability It Cannot Create: Twenty Measured Layers and a Goal-Induction Study on ARC-AGI-3

**Code submission:** Kaggle kernels `sergueimakarov/arc3-stock-flash` (baseline) and `arc3-stock-flash-predhint` (goal-predicate layer).

## Abstract

We ran a 26-day, single-model ablation programme on ARC-AGI-3 with a fixed open-weights agent (Duck harness + Qwen3.8-Flash-Next-NVFP4). Twenty harness layers — goal gates, stuck-detection resets, scripted explorers, macro-actions, prefix caching, best-of-N voting, ten world-model variants, move-history transfer, and three search layers — produced no score change distinguishable from run-to-run spread. We then localised the failure: the model induces correct *mechanics* but not *goals*. We formalise the level goal as a machine-checkable predicate. Under an honest control — a random non-goal state declared the goal, with parameters re-fitted by the same search — the vocabulary separates the true completion state on 83% of levels against 51% for the pseudo-goal, so most of the apparent separation is the vocabulary's flexibility, not a property of goals. Leave-one-game-out prediction of the next level's goal from the previous level succeeds in 27% (top-1) to 35% (top-3) of 26 level pairs, against 1-2% when the same transfer is attempted from a pseudo-goal of that level and 0-1% from another game. A goal-directed search using this vocabulary solves the same 14 of 25 first levels as blind breadth-first search at half the move cost. Supplying the verified goal to the model did not raise the rate at which it clears the next level (0.38 vs 0.28 baseline, p = 0.51). A goal-conditioned progress measure separates winning actions from sibling actions at AUC 0.63 when the true completion frame is known, and at AUC 0.47 — chance — when the goal is the one transferred from the previous level, which is the only goal available in play. We also report the arithmetic that makes in-episode search net-negative under this benchmark's scoring, and a counterfactual evaluation method that predicted a layer's sign and magnitude before compute was spent.

## 1. Setup

All measurements use the 25 public ARC-AGI-3 games, the Duck harness (ARC-AGI-3 Milestone 1 winner) and Qwen3.8-Flash-Next-NVFP4 served by vLLM on one RTX Pro 6000, the configuration the public leaderboard line converged on. Scores are RHAE. A local deterministic replica of the engine reproduces recorded trajectories exactly (561 of 561 actions; reconstructed per-level action counts matched the official counters in 121 of 121 levels), which makes counterfactual analysis free.

**Baseline spread is the unit of measurement.** Three runs of the *identical* configuration score 1.67 / 2.73 / 3.32 at a 30-minute per-game cap, 3.91 / 5.40 / 6.11 at 60 minutes, and 5.46 / 6.32 / 7.23 at 80 minutes. In competition submissions the same build scored 1.85 to 4.09 over ten days. Any claim about a layer that ignores this spread is unfalsifiable; we pre-registered thresholds against three truncations of base runs before every probe.

## 2. Twenty layers, no signal

Layers tested (each with pre-registered thresholds, sign test on paired per-game deltas, and paired bootstrap): goal gate with refutation; stuck reset preserving memory; scripted explorer; level-completion fact injection; macro-actions; prefix caching; best-of-N voting; ten executable-world-model variants (ports of published approaches); amnesia (history truncation); oracle rule injection; doubled call budget; world-model carry across levels; three search layers. None produced a delta outside baseline spread; the paired bootstrap interval covered zero in every case.

Three results are worth stating individually because they are counter-intuitive:

* **More calls do not buy levels past a point.** Doubling calls per game (56 → 112) moved 40 → 41 levels; the extra budget went into revisiting states (returns ×2.5).
* **Removing reasoning tokens destroys the agent** (2.91 vs 9.43), while *adding* input information — oracle rules, world-model digests, level facts — changes almost nothing. In a controlled stand test, the model restated an injected rule in 27 of 27 answers and still did not act on it.
* **History is working memory, not clutter.** Truncating it lowered the action rate from 44% to 22%.

## 3. Mechanics without goals

The same model that never gains score writes *correct dynamics*. Its synthesised transition programs predicted unseen transitions in 86% of changed states (89/103); on one game a counterfactual check of its world model was correct in 200 of 200 out-of-distribution clicks. Yet in a stand run where a "strategist" phase asked explicitly for the level goal, it was right in 1 of 6 games, and a learned goal/progress detector lost to the untrained heuristic "the move with the largest board change" in all 5 held-out games (0.35–0.86 vs 0.44–0.97 top-1).

Six independent transfer measurements (imitation policies on generated level variants, cross-game policy transfer with two architectures, few-shot with 3 seeds, online context, shuffled-label control, learned goal detector) returned zero transfer to held-out games while reaching 0.64–0.68 accuracy on training games — i.e. memorisation, not rules.

## 4. Goals as verifiable predicates

We define a level goal as a predicate over board state drawn from a 21-template vocabulary (colour absent; colour count; component counts; single component; rectangle; symmetry; shape match; containment; region-pattern equality; shape matches a hole — the "complementary object"; shape matches a shape seen at any earlier time; and four templates comparing the state to the level's start). Induction keeps predicates true at the completion frame and false at every state seen on that level.

* **Separation, honestly controlled.** The true completion state is separated from every other state of its level on 44 of 53 levels (83%). When a random non-goal state is declared the goal and the same dictionary search re-fits its parameters, it is separated in 536 of 1060 trials (51%). Our first control (8%) re-used parameters fitted to the true goal and therefore tested the wrong hypothesis; the honest gap is 83 against 51, not 90 against 8.
* **Cross-level transfer, leave-one-game-out** (26 pairs; per-kind transfer rates estimated on the other 14 games): Coverage@1 = 27% for all three parameter-transfer rules, Coverage@3 = 35% for the monotone rule ("fewer than before"), 31% for the point estimate. The first prediction is false in 46-62% of pairs. Nulls: the same transfer from a *pseudo-goal* of the same level covers 1-2%, and from another game's goal 0-1% — so what transfers is the goal, but weakly.
* **Per-kind precision** drives a curated vocabulary. "Colour c is gone" and "colour c present at level start is now gone" transfer 6/6 each; "all of colour c joined into one group" 3/4; "exactly m components of colour c" only 4/34 (12%), and shape-match templates 11–12%. Restricting the shown vocabulary to kinds above 40% raises transfer precision from 31% to 70% at the cost of coverage (23 of 53 levels vs 38).

## 5. Search guided by goal hypotheses

Using parameter-free candidates from the same vocabulary plus a dense residual (assignment-problem matching between object sets: shape mismatch plus centroid distance), a best-first search over the real engine solves **14 of 25** first levels — the same as blind BFS under identical budgets — using **64,092 moves against 127,911** on the 13 games both solve (up to 8.3× cheaper on individual games). Two design facts were measured, not assumed: step-function residuals carry no signal (median Spearman ρ = −0.02 between residual and remaining distance), dense matching residuals do (ρ = 0.20–0.93 where the goal is in the vocabulary); and interleaving guided and blind expansions *through one queue* is worse than either arm, because blind children flood the guided frontier. Two independent searches run in sequence give 15 of 25 with at most a 2× budget.

## 6. Why search does not pay in the competition, and what the goal hint did

RHAE scores a level as `(baseline_actions / actions_used)²`. Search actions are counted, so a level reached after thousands of search moves is worth ≈0, and a level the model would have solved itself is destroyed if search touched it. Counterfactual replay of three base runs quantified this before we spent compute: a search triggered after 30 minutes of no progress costs −0.2 to −1.1 RHAE; a search run before the model's first call costs −2.2 to −2.3 (30–42% of the score). A probe confirmed the sign and magnitude (measured −0.58 to −2.78).

A stronger constraint, checked in the live competition bundle rather than assumed: the harness sets `ONLY_RESET_LEVELS=true` process-wide, so the engine never performs the full reset that would start a fresh scored run, and the competition scorecard admits only one run per game id. The "win, then reset and replay optimally" strategy that works offline is therefore unavailable in play, and exploration cost can never be recovered. This is why a training-free graph-exploration agent that ranks 3rd by *levels solved* would score ≈0 here: the metric, not the capability, decides which approach looks strong.

The remaining zero-cost use is to give the model the verified goal of the level it just completed, since the kind recurs. Two probes: with a noisy vocabulary the next-level rate was 0.25 vs 0.35 baseline; with the curated vocabulary 0.38 vs 0.28 (Fisher p = 0.51). The model reads the hint — it referenced it in 112 of 379 responses across 18 games — but the effect is indistinguishable at this sample size, and whole-run score cannot measure it at all: half the score comes from first levels, which the layer cannot influence.

**The decisive test is about progress, not recognition.** For every transition on a known solution path we computed an object-matching distance to the goal and asked whether it falls more often for the winning action than for its siblings. With the *true* completion frame as the goal, it does: 0.69 against 0.49, AUC 0.629 over 89 transitions in 10 games. With the goal *transferred from the previous level* — the only goal an agent has in play — it does not: 0.17 against 0.09, AUC 0.473. A pre-registered criterion (AUC ≥ 0.60 and a ≥ 0.10 gap) therefore closes this line as a score lever while leaving it intact as a measurement instrument.

## 7. What we contribute

1. **A negative result with mechanism:** on a fixed small model, twenty harness layers multiply an ability that is absent; the missing ability is goal induction, not dynamics.
2. **A measurement discipline:** baseline spread from ≥3 identical runs; pre-registered thresholds; counterfactual evaluation on a deterministic local replica that predicted a layer's sign before compute was spent; and the observation that whole-run score is the wrong metric for any layer acting after level 1.
3. **A goal vocabulary with measured transfer,** including per-kind precision, which converts "the agent lacks goals" from a slogan into numbers a successor system can build on.

We also tested the natural extension to *histories* (ever / never / count / before / within / action-run predicates, with a description-length penalty and the requirement that the predicate first becomes true at the completing step). It covers all 53 levels — and fails its control completely: a random prefix of a trajectory, where no level was cleared, is "separated" in 986 of 986 trials. An expressive temporal language describes anything, so coverage without a control is worthless.

**What would change our mind:** a goal-conditioned progress measure that separates winning from sibling actions at AUC ≥ 0.6 using only a *transferred* goal; or evidence that the next-level rate rises on ≥60 first-level games. Absent either, our reading is that on a small fixed model the binding constraint is turning a known goal into a plan, not naming the goal.

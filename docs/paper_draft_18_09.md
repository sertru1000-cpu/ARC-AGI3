# Scaffolding Multiplies Ability It Cannot Create: Twenty Measured Layers and a Goal-Induction Study on ARC-AGI-3

**Code submission:** Kaggle kernels `sergueimakarov/arc3-stock-flash` (baseline) and `arc3-stock-flash-predhint` (goal-predicate layer).

## Abstract

We ran a 26-day ablation programme on ARC-AGI-3 with a fixed open-weights agent (Duck harness + Qwen3.8-Flash-Next-NVFP4). Twenty harness layers produced no score change distinguishable from run-to-run spread. We localised the failure — the model induces correct mechanics but not goals — and formalised the level goal as a machine-checkable predicate. Under an honest control (a random non-goal state declared the goal, parameters re-fitted by the same search) the vocabulary separates true completion states on 83% of levels against 51% for pseudo-goals: most apparent separation is the vocabulary's flexibility. Leave-one-game-out prediction of the next level's goal succeeds in 27–35% of 26 level pairs, against 0–2% for two nulls. Goal-directed search solves the same 14 of 25 first levels as blind search at half the move cost, but supplying the verified goal to the model did not raise its next-level rate (0.38 vs 0.28, p = 0.51). The decisive measurement is a goal-conditioned progress function: it separates winning from sibling actions at AUC 0.63 given the true completion frame and at AUC 0.47 — chance — given the goal transferred from the previous level, the only goal available in play. We also report why in-episode search cannot pay under this scoring, and a counterfactual method that predicted a layer's sign before compute was spent.

## 1. Setup

All measurements use the 25 public ARC-AGI-3 games, the Duck harness (ARC-AGI-3 Milestone 1 winner) and Qwen3.8-Flash-Next-NVFP4 served by vLLM on one RTX Pro 6000, the configuration the public leaderboard line converged on. Scores are RHAE. A local deterministic replica of the engine reproduces recorded trajectories exactly (561 of 561 actions; reconstructed per-level action counts matched the official counters in 121 of 121 levels), which makes counterfactual analysis free.

**Baseline spread is the unit of measurement.** Three runs of the *identical* configuration score 1.67 / 2.73 / 3.32 at a 30-minute per-game cap, 3.91 / 5.40 / 6.11 at 60 minutes, and 5.46 / 6.32 / 7.23 at 80 minutes. In competition submissions the same build scored 1.85 to 4.09 over ten days. Any claim about a layer that ignores this spread is unfalsifiable; we pre-registered thresholds against three truncations of base runs before every probe.

## 2. Twenty layers, no signal

Layers (each with pre-registered thresholds, a sign test on paired per-game deltas and a paired bootstrap): goal gate with refutation, stuck reset, scripted explorer, level-fact injection, macro-actions, prefix caching, best-of-N voting, ten executable-world-model variants, history truncation, oracle rule injection, doubled call budget, world-model carry, three search layers. No delta fell outside baseline spread; every bootstrap interval covered zero. Three findings are counter-intuitive:

* **More calls do not buy levels past a point.** Doubling calls per game (56 → 112) moved 40 → 41 levels; the extra budget went into revisiting states (returns ×2.5).
* **Removing reasoning tokens destroys the agent** (2.91 vs 9.43), while *adding* input information — oracle rules, world-model digests, level facts — changes almost nothing. In a controlled stand test, the model restated an injected rule in 27 of 27 answers and still did not act on it.
* **History is working memory, not clutter.** Truncating it lowered the action rate from 44% to 22%.

## 3. Mechanics without goals

The same model that gains no score writes *correct dynamics*: its synthesised transition programs predicted unseen transitions in 86% of changed states (89/103), and on one game a counterfactual check was correct in 200 of 200 out-of-distribution clicks. Asked explicitly for the level goal, it was right in 1 of 6 games, and a learned goal detector lost to the untrained heuristic "largest board change" in all 5 held-out games. Six independent transfer measurements (imitation policies, cross-game policy transfer with two architectures, few-shot with three seeds, online context, shuffled labels, learned goal detector) returned zero transfer to held-out games at 0.64–0.68 accuracy on training games — memorisation, not rules.

## 4. Goals as verifiable predicates

We define a level goal as a predicate over board state drawn from a 21-template vocabulary (colour absent; colour count; component counts; single component; rectangle; symmetry; shape match; containment; region-pattern equality; shape matches a hole — the "complementary object"; shape matches a shape seen at any earlier time; and four templates comparing the state to the level's start). Induction keeps predicates true at the completion frame and false at every state seen on that level.

* **Separation, honestly controlled.** The true completion state is separated from every other state of its level on 44 of 53 levels (83%). When a random non-goal state is declared the goal and the same dictionary search re-fits its parameters, it is separated in 536 of 1060 trials (51%). Our first control (8%) re-used parameters fitted to the true goal and therefore tested the wrong hypothesis; the honest gap is 83 against 51, not 90 against 8.
* **Cross-level transfer, leave-one-game-out** (26 pairs; per-kind transfer rates estimated on the other 14 games): Coverage@1 = 27% for all three parameter-transfer rules, Coverage@3 = 35% for the monotone rule ("fewer than before"), 31% for the point estimate. The first prediction is false in 46-62% of pairs. Nulls: the same transfer from a *pseudo-goal* of the same level covers 1-2%, and from another game's goal 0-1% — so what transfers is the goal, but weakly.
* **Per-kind precision** drives a curated vocabulary. "Colour c is gone" and "colour c present at level start is now gone" transfer 6/6 each; "all of colour c joined into one group" 3/4; "exactly m components of colour c" only 4/34 (12%), and shape-match templates 11–12%. Restricting the shown vocabulary to kinds above 40% raises transfer precision from 31% to 70% at the cost of coverage (23 of 53 levels vs 38).

## 5. Search guided by goal hypotheses

Using parameter-free candidates from the same vocabulary plus a dense residual (assignment-problem matching between object sets: shape mismatch plus centroid distance), a best-first search over the real engine solves **14 of 25** first levels — the same as blind BFS under identical budgets — using **64,092 moves against 127,911** on the 13 games both solve (up to 8.3× cheaper on individual games). Two design facts were measured, not assumed: step-function residuals carry no signal (median Spearman ρ = −0.02), dense matching residuals do (ρ = 0.20–0.93 where the goal is in the vocabulary); and interleaving guided and blind expansions through one queue is worse than either arm, because blind children flood the guided frontier — two independent searches in sequence give 15 of 25 at a 2× budget.

## 6. Why search does not pay in the competition, and what the goal hint did

A level scores `(baseline_actions / actions_used)²`, and search actions are counted: a level reached after thousands of search moves is worth ≈0, and a level the model would have solved itself is destroyed if search touched it. Counterfactual replay of three base runs quantified this before compute was spent — search after 30 minutes of no progress costs −0.2 to −1.1, search before the model's first call −2.2 to −2.3 (30–42% of the score) — and a probe confirmed sign and magnitude (−0.58 to −2.78).

A stronger constraint, checked in the live competition bundle rather than assumed: the harness sets `ONLY_RESET_LEVELS=true` process-wide, so the engine never performs the full reset that would start a fresh scored run, and the competition scorecard admits only one run per game id. The "win, then reset and replay optimally" strategy that works offline is therefore unavailable in play, and exploration cost can never be recovered. This is why a training-free graph-exploration agent that ranks 3rd by *levels solved* would score ≈0 here: the metric, not the capability, decides which approach looks strong.

The remaining zero-cost use is to give the model the verified goal of the level it just completed, since the kind recurs. Two probes: with a noisy vocabulary the next-level rate was 0.25 vs 0.35 baseline; with the curated vocabulary 0.38 vs 0.28 (Fisher p = 0.51). The model reads the hint — it referenced it in 112 of 379 responses across 18 games — but the effect is indistinguishable at this sample size, and whole-run score cannot measure it at all: half the score comes from first levels, which the layer cannot influence.

**The decisive test is about progress, not recognition.** For every transition on a known solution path we computed an object-matching distance to the goal and asked whether it falls more often for the winning action than for its siblings. With the *true* completion frame as the goal, it does: 0.69 against 0.49, AUC 0.629 over 89 transitions in 10 games. With the goal *transferred from the previous level* — the only goal an agent has in play — it does not: 0.17 against 0.09, AUC 0.473. A pre-registered criterion (AUC ≥ 0.60 and a ≥ 0.10 gap) therefore closes this line as a score lever while leaving it intact as a measurement instrument.

## 7. What we contribute

1. **A negative result with mechanism:** twenty harness layers multiply an ability that is absent, and the missing ability is not dynamics.
2. **A measurement discipline:** baseline spread from ≥3 identical runs; pre-registered thresholds; counterfactual evaluation on a deterministic replica that predicted a layer's sign before compute was spent; controls that re-fit parameters to a pseudo-goal, which overturned two of our own headline numbers; and the observation that whole-run score cannot measure a layer acting after level 1.
3. **A goal vocabulary with measured per-kind transfer,** turning "the agent lacks goals" into numbers a successor system can build on.

We also tested the natural extension to *histories* (ever / never / count / before / within / action-run predicates, with a description-length penalty and the requirement that the predicate first becomes true at the completing step). It covers all 53 levels — and fails its control completely: a random prefix of a trajectory, where no level was cleared, is "separated" in 986 of 986 trials. An expressive temporal language describes anything, so coverage without a control is worthless.

**What would change our mind:** a goal-conditioned progress measure that separates winning from sibling actions at AUC ≥ 0.6 using only a *transferred* goal; or evidence that the next-level rate rises on ≥60 first-level games. Absent either, our reading is that on a small fixed model the binding constraint is turning a known goal into a plan, not naming the goal.

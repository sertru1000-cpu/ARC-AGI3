# Scaffolding Multiplies Ability It Cannot Create: Twenty Measured Layers and a Goal-Induction Study on ARC-AGI-3

**Code submission:** Kaggle kernels `sergueimakarov/arc3-stock-flash` (baseline) and `arc3-stock-flash-predhint` (goal-predicate layer).

## Abstract

We ran a 26-day, single-model ablation programme on ARC-AGI-3 with a fixed open-weights agent (Duck harness + Qwen3.8-Flash-Next-NVFP4). Twenty harness layers — goal gates, stuck-detection resets, scripted explorers, macro-actions, prefix caching, best-of-N voting, ten world-model variants, move-history transfer, and three search layers — produced no score change distinguishable from run-to-run spread. We then localised the failure: the model induces correct *mechanics* but not *goals*. We formalise the level goal as a machine-checkable predicate, show that such predicates separate the completion state from every other state seen on that level in 9 of 10 games (control: 8%), and that the *kind* of goal transfers across levels of a game in 24 of 30 adjacent level pairs (80%), while the exact parameters transfer in only 31%. A goal-directed search using this vocabulary solves the same 14 of 25 first levels as blind breadth-first search at half the move cost. Supplying the verified goal to the model, however, did not raise the rate at which it clears the next level (0.38 vs 0.28 baseline, p = 0.51). We report the arithmetic that makes in-episode search net-negative under RHAE, and a counterfactual evaluation method that predicted one layer's sign and magnitude before spending compute.

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

We define a level goal as a predicate over board state drawn from a 17-template vocabulary (colour absent; colour count; component counts; single component; rectangle; symmetry; shape match; containment; region-pattern equality; shape matches a hole — the "complementary object"; shape matches a shape seen at any earlier time; and four templates comparing the state to the level's start). Induction keeps predicates true at the completion frame and false at every state seen on that level.

* **Separation.** In 9 of 10 games with a searched level-1 solution, at least one predicate separates the goal from 119–137 non-goal states. Control: a randomly chosen ordinary state is separated in 4 of 50 trials (8%).
* **Cross-level transfer** (30 adjacent level pairs from 15 games, extracted by replaying recorded trajectories): the *kind* of goal recurs in 24 pairs (80%); a *concrete* predicate still separates the next level's goal in 10 pairs (33%); control 2 pairs.
* **Per-kind precision** drives a curated vocabulary. "Colour c is gone" and "colour c present at level start is now gone" transfer 6/6 each; "all of colour c joined into one group" 3/4; "exactly m components of colour c" only 4/34 (12%), and shape-match templates 11–12%. Restricting the shown vocabulary to kinds above 40% raises transfer precision from 31% to 70% at the cost of coverage (23 of 53 levels vs 38).

## 5. Search guided by goal hypotheses

Using parameter-free candidates from the same vocabulary plus a dense residual (assignment-problem matching between object sets: shape mismatch plus centroid distance), a best-first search over the real engine solves **14 of 25** first levels — the same as blind BFS under identical budgets — using **64,092 moves against 127,911** on the 13 games both solve (up to 8.3× cheaper on individual games). Two design facts were measured, not assumed: step-function residuals carry no signal (median Spearman ρ = −0.02 between residual and remaining distance), dense matching residuals do (ρ = 0.20–0.93 where the goal is in the vocabulary); and interleaving guided and blind expansions *through one queue* is worse than either arm, because blind children flood the guided frontier. Two independent searches run in sequence give 15 of 25 with at most a 2× budget.

## 6. Why search does not pay in the competition, and what the goal hint did

RHAE scores a level as `(baseline_actions / actions_used)²`. Search actions are counted, so a level reached after thousands of search moves is worth ≈0, and a level the model would have solved itself is destroyed if search touched it. Counterfactual replay of three base runs quantified this before we spent compute: a search triggered after 30 minutes of no progress costs −0.2 to −1.1 RHAE; a search run before the model's first call costs −2.2 to −2.3 (30–42% of the score). A probe confirmed the sign and magnitude (measured −0.58 to −2.78).

The remaining zero-cost use is to give the model the verified goal of the level it just completed, since the kind recurs. Two probes: with a noisy vocabulary the next-level rate was 0.25 vs 0.35 baseline; with the curated vocabulary 0.38 vs 0.28 (Fisher p = 0.51). The model does read the hint — it referenced it in 112 of 379 responses across 18 games — but at this sample size the effect is indistinguishable. Separating it requires ~60–80 games with a first level, because the overall score is dominated by first levels the layer cannot influence.

## 7. What we contribute

1. **A negative result with mechanism:** on a fixed small model, twenty harness layers multiply an ability that is absent; the missing ability is goal induction, not dynamics.
2. **A measurement discipline:** baseline spread from ≥3 identical runs; pre-registered thresholds; counterfactual evaluation on a deterministic local replica that predicted a layer's sign before compute was spent; and the observation that whole-run score is the wrong metric for any layer acting after level 1.
3. **A goal vocabulary with measured transfer,** including per-kind precision, which converts "the agent lacks goals" from a slogan into numbers a successor system can build on.

**What would change our mind:** a demonstration that the same curated goal predicates raise the next-level rate on ≥60 first-level games, or that predicate induction over *action histories* (rather than single frames) covers the three games where our vocabulary is silent at every level.

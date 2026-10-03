# The Server Was the Ceiling: What Twenty Null Layers, a Score Budget and a 4-bit KV Cache Taught Us on ARC-AGI-3

*ARC Prize 2026, ARC-AGI-3 track. An open-weights agent on one 96 GB card: why our harness work did not move the score, what did, and the serving measurements a team should make first.*

## 1. Setup and unit of measurement

All measurements use the 25 public games, the Duck harness lineage and Qwen3.8-Flash-Next on one RTX Pro 6000. A deterministic local replica of the engine reproduces recorded trajectories exactly (561/561 actions), which makes counterfactual analysis free.

The unit of measurement is run-to-run spread. Identical builds scored 1.67 / 2.73 / 3.32 on a 30-minute bench and 1.85–4.09 across ten competition submissions. On the current stack, two identical 25-game bench runs scored 46.06 and 48.73, but single games flipped between 4 and 100, so the difference between a variant and two controls has a standard error of about 6 points. Seventy-seven teams submitting near-copies of one open notebook on the same day spread with a standard deviation of 3.1 on the leaderboard. A single run resolves only large effects. Every claim below is stated against that spread.

## 2. Twenty harness layers, no signal

For a month we measured twenty harness layers with pre-registered thresholds and paired bootstraps. They included a goal gate, a scripted explorer, fact and rule injection, best-of-N voting, ten executable-world-model variants, search layers and a doubled call budget. None moved the score outside baseline spread. Three findings mattered later:

- Removing reasoning tokens destroys the agent (2.91 vs 9.43), while adding information to the prompt changes almost nothing. The model restated an injected rule in 27 of 27 answers and still did not act on it.
- Truncating history halved the action rate (44% → 22%). We tested a shorter window and never a longer one.
- The model writes correct local dynamics (86% of unseen changed transitions predicted) but rarely names the goal. An exact executable copy of a whole game came out in 6 of 24 games, so the executable-world-model recipe of frontier systems does not transfer to this model.

## 3. A score budget before building anything

The score is `(human_actions / agent_actions)²` per level, capped. On three recorded base runs:

| Counterfactual | Change |
|---|---|
| Every completed level played perfectly | +24% |
| One level cleared in every game | +32% |
| **One more level cleared per game** | **+137%** |

64% of completed levels were already at the cap. Efficiency work is bounded at about a quarter of the score, and depth is worth more than the whole score. Two shortcuts also lose when replayed from recorded runs: two half-length passes with best-of-runs scoring (−19%), and giving a stalled game more time (+4% for 82% more time).

## 4. The correction: memory, not instructions

We read twenty null layers and two model swaps as proof that the model was the ceiling. A 2–3-bit DeepSeek was worse than ours; an uncompressed Gemini-3.6-Flash cleared 10 levels in 14 minutes against our 3–5. Milestone 2 refuted that reading. Four disclosed solutions scored 20.00–27.89 against our 4.30 with the same model family, card and harness lineage. All kept two to four times more history (70–131k tokens against our 32k), paid for with an fp8 or int4 KV cache, prefix caching and trimming in large blocks. One author moved from 14.49 to 22.53 by changing only the cache format and spending the freed memory on history. Our unmodified copy of the best solution scored 24.25 and 26.78 in two submissions.

The signals had been in our own data. The server queue showed `Running: 3, Waiting: 22` for a month: a 5 GiB KV cache served three requests out of sixteen slots, and we read it as slow generation. The teacher comparison said the same thing. At equal move counts the stronger model cleared the same number of levels as ours, and its whole advantage was tempo, 2–3 s per move against our 57–146 s. We attributed that tempo to its dedicated API instead of asking what serving stack we could run ourselves. In a fixed-hardware competition the order of work is model packaging, serving engine and version, cache and what it is spent on, time across games, and only then the harness. We worked in the reverse order.

## 5. Serving experiments on the disclosed stack

We then audited the adopted stack the way we should have audited ours, with two instruments of our own.

**A server bench without games.** We record every request the harness sends during a run (prompts with images, tool schemas, template arguments). We then replay these requests against a server configuration, one sequential stream per game, with each answer forced to its recorded length. The load is identical across configurations, and repeats agree to 0.3%.

**A KV-quality probe.** On 20 recorded contexts of 34–70k tokens we measure the likelihood of the model's own recorded answer with the full history and with a truncated one. This compares cache formats by how well they preserve what the model uses.

Findings:

1. **At the battle configuration the cache does not bind.** With ten games at a 128k window the 1.01M-token pool peaks at 67% full and 93% of prompt tokens come from the prefix cache.
2. **A hidden concurrency cap.** The server reserves five Mamba-state slots per request, so the default 60 slots silently cap concurrency at 12.
3. **Twenty games break the cache, and a host-RAM tier repairs it.** At twenty streams the pool overflows after about nine minutes, cache hits fall from 93% to 46%, and throughput drops 32% against ten games. With a 30 GB host-RAM cache tier, evicted history is restored instead of recomputed: 86% hits and **+7%** throughput, at half the generation per game.
4. **fp8 KV costs nothing measurable** against BF16 (likelihood difference +0.003, within ±0.01–0.02).
5. **A 4-bit KV cache now works on this stack, with our patch.** NVFP4 KV did not start on the disclosed server. The cause was not the attention backend: the model's full-attention layers always run through a sparse-attention path, which copies the selected top-k tokens into a scratch buffer and has no notion of packed 4-bit values or their block scales. We wrote a Triton kernel that dequantises only the selected tokens. We also fixed a CUDA-graph capture failure in the cache write and dropped an unused full-pool workspace. The kernel was verified bit-exact against the library's dequantiser. Measured against fp8 on one Kaggle machine: **pool ×1.50** (2.11M vs 1.41M tokens), likelihood **+0.0015 (95% CI −0.005…+0.008)**, equal throughput. It buys nothing at ten games (finding 1). It is the cheap alternative to a host-RAM tier wherever the pool binds.
6. **Newer is not automatically better.** SGLang 0.5.21 with the same patches allocated an 18% smaller pool.

## 6. What we contribute

1. **A corrected conclusion with its evidence.** On this benchmark and hardware the binding constraint was retained history, set by the serving stack, not the model. We show where each signal was available in our own logs and misread.
2. **A score budget** that shows which levers can matter before any is built.
3. **A server bench and a cache-quality probe** that compare serving configurations without spending games. They located where the cache starts to bind, a hidden concurrency cap, and what repairs the overflow.
4. **An NVFP4 KV cache for Qwen3.8-Flash-Next's sparse attention in SGLang,** a ×1.5 larger pool with no measurable loss.
5. **Measurement discipline:** spread from identical runs, pre-registered thresholds, counterfactual replay on a deterministic engine, and controls that overturned two of our own headline numbers.
6. **Engineering for sealed kernels:** two verified ways to deliver modified weights without re-uploading 135 GB (a runtime adapter with four serving concessions, and a byte-level weight overlay with none).

## Limitations

Harness verdicts from the first month were measured at a 32k window and are not repeated on the long-history stack. The server bench measures serving, not play, and whether twenty games at half speed beat ten is open. The NVFP4 result is one probe and one throughput check. Our macro-move layer, the one component of our current build that is ours, scored 51.61 against controls of 46.06 and 48.73, inside the spread. All live numbers are on public games.

*Full draft with all measurements and code references: attached project link.*

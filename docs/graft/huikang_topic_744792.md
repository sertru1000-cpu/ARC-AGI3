Congratulations to the winners (tentative, to be confirmed by the organizers) of the Milestone 2 notebooks!

All three run the Tufa ARC-AGI Framework (TAAF) harness, serve Qwen3.8-Flash-Next, and
target a single RTX PRO 6000 (96 GB).

Here is a comparison of the notebooks (I am still looking through the code and logs, I will update)

| | dfranzen | lordhansolo | sirikilohit |
|---|---|---|---|
| Public LB score | 27.89 | 23.84 | 22.53 |
| Server | SGLang Pennyroyal v2.5.3 | vLLM nightly | SGLang Pennyroyal v2.5.0 |
| Server context | 139,264 | 147,072 | 69,632 |
| Harness window | 131,072 | 127,488 | 69,632 |
| Reply reserve | 12,288 | 512 | 512 + 4,096 |
| Trim rule | drain 58k blocks, 150-turn cap | soft target 81,536 | watermarks 57k → 45k |
| Max sequences | 10 | 14 | 16 |
| Games admitted | 110, gated to 10 active streams | 14 | 28 over 16 slots |
| KV dtype | fp8_e4m3 | fp8_e4m3 | fp8_e4m3 |
| Pool sizing | mem fraction 0.96 | gpu util 0.98, profiled | mem fraction 0.97 + 48 GB host tier |
| Speculative | NEXTN 3 steps + FR-Spec 64k map | MTP 3 tokens + 32k draft vocab | NEXTN 3 steps, NVFP4 draft |
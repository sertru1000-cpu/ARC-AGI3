# Еженедельный скан публичной линии (правило владельца, 06.09)

Снимок 06.09 (источники: kaggle kernels/datasets/models list, kernels output):

| что | последнее | балл публичных 25 |
|---|---|---|
| лидерборд топ | mostik.ai 7.51, Daniel Franzen 6.66, Third Intelligence 6.43; порог топ-10 4.74; Tufa Labs 4.71 (04.09) | — |
| бандл Duck | jakobbrggen/taaf-kaggle-source 01.09 (ветка experiment/avo-v2, AvoAgent, Qwen3.8-27B-FP8) | не опубликован |
| кернел | keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp 01.09 (Duck 30.06 + Qwen3.8-Flash-Next-NVFP4, MTP-3) | 6.76 |
| кернел | jakobbrggen/duck-qwen3-8-anim-base 01.09 (копия предыдущего) | 9.00 |
| модели Kaggle | keithtyser/qwen3-8-flash-next-nvfp4 (RadixArk NVFP4, 135 ГБ); woochangsim/qwen38-flash-next-w4a16-autoround | — |
| наша база | Qwen3.8-27B-FP8 + бандл 07.08 | 3.02 |

Следующие проверки: 13.09, 20.09, 27.09. Порядок: скан → таблица дельт → новая база в очередь раньше любых надстроек.

## Скан 17.09 (вместо 20.09, по просьбе владельца)

Источники: `kaggle competitions leaderboard --show` (API снова отвечает, 12.09 был 403), `kaggle kernels list --competition ... --sort-by dateRun|voteCount`, `kaggle datasets list --user keithtyser|jakobbrggen|mirzamilanfarabi`, `-s taaf-kaggle`, `kaggle models list -s Qwen3.8|Flash-Next|Qwen4|Gemma|DeepSeek|nvfp4`, `kaggle models get`; веб: vLLM recipes, Artificial Analysis, HF.

| что | 17.09 | было |
|---|---|---|
| лидерборд | Tufa Labs **18.81** (16.09), Daniel Franzen 11.59, Lord Han Solo 8.84, Tong Hui Kang 8.72, Ebi 8.68, NVARC3 8.40, Third Intelligence 8.21, mostik.ai 7.51, Mark Slavin 7.29, Kyutai 7.19 (10-е место), …, Fususu 6.91 (12-е) | 06.09: mostik.ai 7.51 первый, порог топ-10 4.74; 07.09 Tufa 11.04 |
| публичный бандл Duck/Tufa | jakobbrggen/taaf-kaggle-source 01.09, keithtyser smoke-v1 31.08 — без изменений | то же |
| **новая модель в соревновании** | `phuongncn/arc3-dsv4-flash-weights` (автор Fususu, 12.09): DeepSeek-V4-Flash-0731, 284B MoE / 13B активных (у Flash-Next 6B), FP4 эксперты + FP8, 155 ГиБ, «for v162 in-kernel serve». Официальный рецепт vLLM: минимум 8× RTX Pro 6000; на одной 96 ГБ не заявлен; у SM120 (Blackwell Pro) известный сбой DeepGEMM. Публичного кернела с ней нет; связь с 6.91 Fususu не подтверждена | — |
| прочие модели | `dineshkumar0705/qwen3-8-flash-next-fp8` (15.09, 186 ГБ — та же модель, крупнее); `impactganyu/qwen38-27b-*-nvfp4-dflash2` (15.09, 27B + черновик для ускорения); `davidaylward/whittle-next-27b-a3b` (13.09, дистиллят 27B/3B, CPU-демо); `mirzamilanfarabi/kimi-linear-48b` (12.09, датасеты) | 12.09: ничего свежее Flash-Next |
| кернелы | на Flash-Next: juliancamilovilla nvfp4 carry*/batch/moe-long (15–16.09), yuriimuktarov Duck Qwen38 public25 r9–r12, woguoat keith-repro, wuliao0 anim-base (обновлён 16.09); другой модели в публичных кернелах нет | — |
| вне Kaggle | Qwen3.8-Max открыт (2.4T/A95B) — на одну карту не помещается; DeepSeek-V4.1-Flash на HF (4× RTX Pro 6000) | — |
| наша база | Flash-Next стоковый Duck, бой максимум 4.09 | 3.97 |

Не сделано: выход кернелов juliancamilovilla carry* не скачивался; обсуждения Kaggle не читались; литературный поиск — один общий запрос, новых работ по ARC-AGI-3 не найдено.

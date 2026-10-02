# Под: стенд серверов и памяти на записанной нагрузке (02.10)

Под: 1× RTX PRO 6000 (96 ГБ), ОЗУ ≥ 250 ГБ, драйвер ≥ 580 (колёса Pennyroyal под CUDA 13), шаблон «RunPod Pytorch 2.8.0»,
container disk ≥ 400 ГБ (без сетевого тома — нечему остаться в счёте после Terminate).

1. С мака: `scp -P <port> -r pod_franzen root@<ip>:/workspace/` ; `scp -P <port> ~/.kaggle/access_token root@<ip>:/root/.kaggle/access_token` ;
   `scp -P <port> -r runs/kaggle_graft_p10_v1/*_p0_requests.jsonl root@<ip>:/workspace/replay/` ; `scp -P <port> scripts/replay_load.py root@<ip>:/workspace/`
2. На поде: `nohup bash /workspace/pod_franzen/setup.sh > /workspace/setup.log 2>&1 &` — паспорт машины, проверка драйвера, колёса, модель (~200 ГБ).
3. Сервер: `PY=$(uv python find 3.12); MODEL_DIR=/workspace/models/intel DRAFT_MODEL_DIR=/workspace/models/albucino WHEELHOUSE_DIR=/workspace/wheels/pennyroyal nohup $PY /workspace/pod_franzen/serve.py > /workspace/serve_run.log 2>&1 &`
   варианты переменными: POD_KVDTYPE=nvfp4|bf16, POD_MAXREQ=8|14, POD_CTX_K=258, POD_HICACHE_GB=40, POD_SPEC=0
4. Нагрузка: `python3 /workspace/replay_load.py --run /workspace/replay --streams <N> --out /workspace/run/<вариант>.json`
5. Конец: Terminate (не Stop), проверить сетевые тома.

Матрица (каждая строка — перезапуск сервера ~3 мин + нагрузка ~5–10 мин):
| вариант | переменные | streams |
|---|---|---|
| база | — | 10 |
| 8 мест | POD_MAXREQ=8 | 8 |
| 14 мест | POD_MAXREQ=14 | 14 |
| fp4 | POD_KVDTYPE=nvfp4 | 10 |
| fp4 + 14 | POD_KVDTYPE=nvfp4 POD_MAXREQ=14 | 14 |
| HiCache | POD_HICACHE_GB=40 | 10 |
| окно 250k | POD_CTX_K=258 POD_MAXREQ=4 POD_HICACHE_GB=40 | 4 |
| bf16 (эталон качества) | POD_KVDTYPE=bf16 | 10 |
Плюс проба качества кэша: kernels/graft_kvprobe (та же логика, локальным скриптом).

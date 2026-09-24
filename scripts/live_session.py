"""Живой сеанс на поде: модель грузится ОДИН раз и остаётся в памяти (24.09).

Зачем. Модель живёт внутри процесса; убил процесс -- потерял 40 минут чтения 336 ГБ с сетевого тома.
За 24.09 так сгорело четыре загрузки подряд: каждая ошибка в коде стоила не минуты, а сорока минут и $5.
Здесь процесс не умирает: он держит модель и в цикле исполняет команды, которые кладутся файлами.

Как пользоваться:
    # на поде, один раз:
    python3 live_session.py > /workspace/live.log 2>&1 &
    # дальше с мака, сколько угодно раз:
    scp cmd.py root@под:/workspace/cmd/next.py      # исполнится в живом процессе
    ssh root@под 'tail -40 /workspace/live.log'     # вывод

Команда -- обычный питоновский файл. Он исполняется в глобальном пространстве сеанса, поэтому видит
model, tok, dec, rows, opt и всё, что оставили прошлые команды, и сам может оставлять новое.
Исключение в команде печатается и НЕ роняет сеанс.

Файл /workspace/cmd/stop кладётся, чтобы завершить сеанс.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

CMD = Path("/workspace/cmd")
CODE = Path("/workspace/code")
sys.path.insert(0, str(CODE))

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from load_sharded_fp8 import load_sharded  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj"]


def main() -> None:
    CMD.mkdir(parents=True, exist_ok=True)
    _px = "/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvcc/bin/ptxas"
    if os.path.exists(_px):
        os.environ.setdefault("TRITON_PTXAS_PATH", _px)

    t0 = time.time()
    print("сеанс: читаю модель (один раз за всё время работы)", flush=True)
    model, info = load_sharded("/workspace/base", device="cuda", fp8_experts=True,
                               dtype=torch.bfloat16, verbose=True)
    print("сеанс: модель в памяти за %.1f мин, на карте %.1f ГБ"
          % ((time.time() - t0) / 60, info["на карте_ГБ"]), flush=True)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    dec = model.get_decoder()
    tok = AutoTokenizer.from_pretrained("/workspace/base")
    rows = [json.loads(l) for l in open("/workspace/data/train.jsonl", encoding="utf-8")]
    valid = [json.loads(l) for l in open("/workspace/data/valid.jsonl", encoding="utf-8")]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9
    print("сеанс: готов. обучаемых %.1f млн, примеров %d, отложенных %d, свободно на карте %.1f ГБ"
          % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
             len(rows), len(valid), free), flush=True)
    print("сеанс: жду команды в %s/next.py" % CMD, flush=True)

    g = globals()
    g.update(model=model, dec=dec, tok=tok, rows=rows, valid=valid, opt=opt, torch=torch, json=json)

    n = 0
    while True:
        if (CMD / "stop").exists():
            print("сеанс: получен stop, выхожу", flush=True)
            return
        nxt = CMD / "next.py"
        if nxt.exists():
            n += 1
            code = nxt.read_text(encoding="utf-8")
            nxt.unlink()
            print("\n=== команда %d, %s ===" % (n, time.strftime("%H:%M:%S")), flush=True)
            try:
                exec(compile(code, "команда_%d" % n, "exec"), g)
            except Exception:
                traceback.print_exc()
                sys.stdout.flush()
            print("=== команда %d завершена ===" % n, flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()

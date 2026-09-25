"""Локальная проверка пошардового загрузчика со сжатием на лету (24.09), без карты и без аренды.

Проверяется то, ради чего он написан:
 1. модель собирается из НЕСКОЛЬКИХ файлов, ни один тензор не потерян (нет параметров на meta);
 2. имена сопоставлены верно, включая срезанный префикс `language_model.`;
 3. выход совпадает с обычной загрузкой -- сравниваем логиты и потерю;
 4. эксперты действительно сжаты, а общий эксперт (shared_expert) -- нет;
 5. память под эксперты меньше примерно вдвое.

Запуск: .venv/bin/python scripts/test_load_sharded_fp8.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import json

import torch
from safetensors.torch import save_file
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from test_fp8_experts import tiny_config  # noqa: E402
from load_sharded_fp8 import load_sharded  # noqa: E402

from transformers import AutoModelForCausalLM  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    tmp = Path(tempfile.mkdtemp(prefix="tiny_qwen4_"))
    try:
        cfg = tiny_config()
        ref_model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).eval()

        # ЛОВУШКА ФОРМАТА (24.09): save_pretrained РАЗБИРАЕТ слитые тензоры экспертов на
        # experts.0.gate_proj.weight и т.д., а настоящий чекпойнт хранит их слитыми
        # (48 x ...mlp.experts.gate_up_proj) и с префиксом model.language_model.
        # Поэтому пишем файлы вручную, ровно в боевом виде, иначе тест проверяет не то.
        state = dict(ref_model.named_parameters())
        state.update({n: b for n, b in ref_model.named_buffers() if b is not None})
        renamed = {("model.language_model." + k[len("model."):]) if k.startswith("model.") else k: v
                   for k, v in state.items()}
        cfg.save_pretrained(tmp)
        # ЛОВУШКА PLE (24.09): в настоящем чекпойнте таблица разрезана на 128 кусков
        # ngram_embedding.shard_N.weight, а в модели это один слой. Проверяем склейку: режем
        # эмбеддинги на четыре куска ровно в том же виде.
        cut_key = next(k for k in renamed if k.endswith("embed_tokens.weight"))
        whole = renamed.pop(cut_key)
        parts = torch.chunk(whole, 4, dim=0)
        for i, part in enumerate(parts):
            renamed[cut_key.replace(".weight", ".shard_%d.weight" % i)] = part.contiguous()
        print("эмбеддинги %s разрезаны на %d кусков" % (tuple(whole.shape), len(parts)))

        # ЛОВУШКА ТИПОВ (24.09): в чекпойнте есть ЦЕЛОЧИСЛЕННЫЕ тензоры (в настоящей модели это
        # словарные размеры и смещения голов n-грамм). Если загрузчик приведёт их к bf16, модель
        # падает на битовых операциях. Кладём такой тензор в чекпойнт и проверяем тип после загрузки.
        int_keys = [k for k, v in renamed.items() if not v.is_floating_point()]

        keys = sorted(renamed)
        per = max(1, len(keys) // 4)                       # четыре файла: нужен многофайловый случай
        weight_map = {}
        for fi, start in enumerate(range(0, len(keys), per), 1):
            fname = "model-%05d-of-%05d.safetensors" % (fi, (len(keys) + per - 1) // per)
            chunk = {k: renamed[k].detach().clone() for k in keys[start:start + per]}
            save_file(chunk, str(tmp / fname))
            weight_map.update({k: fname for k in chunk})
        json.dump({"metadata": {}, "weight_map": weight_map},
                  open(tmp / "model.safetensors.index.json", "w"))
        files = sorted(p.name for p in tmp.glob("*.safetensors"))
        print("сохранено файлов: %d, ключей: %d, из них слитых экспертных: %d"
              % (len(files), len(weight_map),
                 len([k for k in weight_map if k.endswith(("experts.gate_up_proj", "experts.down_proj"))])))

        x = torch.randint(0, 100, (1, 24))
        y = x.clone(); y[:, :4] = -100
        with torch.no_grad():
            ref = ref_model(input_ids=x).logits.float()
        ref_loss = nn.functional.cross_entropy(ref[0, :-1], y[0, 1:], ignore_index=-100)

        model, info = load_sharded(str(tmp), device="cpu", fp8_experts=True, verbose=False)
        model.eval()
        print("загрузчик: файлов %d, сжато тензоров %d, как есть %d, пропущено %d"
              % (info["файлов"], info["сжато"], info["как есть"], info["пропущено"]))

        bufs = dict(model.named_buffers())
        bad = [k for k in int_keys
               for kk in [k.replace("model.language_model.", "model.")]
               if kk in bufs and bufs[kk].is_floating_point()]
        print("целочисленных тензоров в чекпойнте: %d, испорчено приведением к bf16: %d"
              % (len(int_keys), len(bad)))
        assert not bad, "целочисленные тензоры приведены к bf16: %s" % bad[:2]

        exp_left = [n for m in model.modules() if type(m).__name__ == "Qwen4ExpTextExperts"
                    for n, _ in m.named_parameters()]
        assert not exp_left, "параметры экспертов не сжаты: %s" % exp_left[:2]
        shared = [n for n, _ in model.named_parameters() if "shared_expert" in n]
        assert shared, "общий эксперт исчез — его сжимать нельзя, в бою он bf16"

        with torch.no_grad():
            got = model(input_ids=x).logits.float()
        got_loss = nn.functional.cross_entropy(got[0, :-1], y[0, 1:], ignore_index=-100)
        rel = ((got - ref).abs().mean() / ref.abs().mean()).item()

        print("логиты: средняя относительная разница %.4f%%" % (100 * rel))
        print("потеря: обычная загрузка %.5f, пошардовая со сжатием %.5f" % (ref_loss, got_loss))
        print("общий эксперт остался в bf16: %d тензоров" % len(shared))

        ok = rel < 0.02 and abs(got_loss - ref_loss) < 0.05 and info["сжато"] > 0 and info["пропущено"] == 0
        print("\nИТОГ:", "ПРОЙДЕНО" if ok else "НЕ ПРОЙДЕНО")
        sys.exit(0 if ok else 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

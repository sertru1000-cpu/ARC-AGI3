"""Локальная проверка fp8-сжатия экспертов на крошечной модели той же архитектуры (24.09).

Проверяется ровно то, ради чего всё делается:
 1. сжатие вообще применяется (модули найдены, параметры заменены буферами);
 2. выход модели не уезжает — сравниваем логиты до и после на одном входе;
 3. потеря на тех же метках совпадает в пределах допуска;
 4. память под эксперты падает примерно вдвое;
 5. обратный проход жив — градиент доходит до обучаемого слоя внимания (эксперты заморожены).

Запуск без карты и без аренды: python3 scripts/test_fp8_experts.py
"""
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from fp8_experts import compress_experts  # noqa: E402

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402


REAL_CONFIG = ("Qwen/Qwen3.8-Flash-Next", "config.json")   # настоящий конфиг, ужатый до игрушечных размеров


def tiny_config():
    """Берём НАСТОЯЩИЙ конфиг модели и ужимаем размеры. Свой конфиг с нуля не годится: часть полей
    (indexer_budget, indexer_compress_ratio и прочие) не имеет значений по умолчанию, и модель падает
    ещё на сборке — проверять сжатие надо на той же архитектуре, а не на похожей."""
    import json
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig

    c = json.load(open(hf_hub_download(*REAL_CONFIG)))
    t = c["text_config"]
    t.update(hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
             num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
             num_experts=8, num_experts_per_tok=2, vocab_size=512,
             indexer_head_dim=16, indexer_n_heads=2, indexer_kv_heads=1, indexer_budget=64,
             shared_expert_intermediate_size=32, ple_layer_ids=[])
    if isinstance(t.get("layer_types"), list):
        t["layer_types"] = t["layer_types"][:4]
    c["tie_word_embeddings"] = False
    c.pop("model_type", None)
    return AutoConfig.for_model("qwen4_exp", **c)


def main() -> None:
    torch.manual_seed(0)
    cfg = tiny_config()
    model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).eval()
    n_exp = sum(p.numel() for m in model.modules() if type(m).__name__ == "Qwen4ExpTextExperts"
                for p in m.parameters())
    print("крошечная модель: %.2f млн параметров, из них эксперты %.2f млн"
          % (sum(p.numel() for p in model.parameters()) / 1e6, n_exp / 1e6))

    x = torch.randint(0, 100, (1, 24))
    y = x.clone(); y[:, :4] = -100

    with torch.no_grad():
        ref = model(input_ids=x).logits.float()
    ref_loss = nn.functional.cross_entropy(ref[0, :-1], y[0, 1:], ignore_index=-100)

    summary = compress_experts(model)

    # только внутри Qwen4ExpTextExperts: общий эксперт (shared_expert) мы НЕ сжимаем намеренно -
    # в боевом чекпойнте он тоже лежит в bf16
    left = [n for m in model.modules() if type(m).__name__ == "Qwen4ExpTextExperts"
            for n, _ in m.named_parameters()]
    assert not left, "параметры экспертов остались: %s" % left[:3]

    with torch.no_grad():
        got = model(input_ids=x).logits.float()
    got_loss = nn.functional.cross_entropy(got[0, :-1], y[0, 1:], ignore_index=-100)

    rel = ((got - ref).abs().mean() / ref.abs().mean()).item()
    print("логиты: средняя относительная разница %.4f%%" % (100 * rel))
    print("потеря: было %.5f, стало %.5f, разница %.5f" % (ref_loss, got_loss, abs(got_loss - ref_loss)))
    print("память экспертов: %.6f -> %.6f ГБ" % (summary["было_ГБ"], summary["стало_ГБ"]))

    # обратный проход: эксперты заморожены, градиент должен дойти до внимания
    for p in model.parameters():
        p.requires_grad_(False)
    attn = dict(model.named_parameters())
    name = next(n for n in attn if "q_proj.weight" in n)
    attn[name].requires_grad_(True)
    out = model(input_ids=x).logits.float()
    loss = nn.functional.cross_entropy(out[0, :-1], y[0, 1:], ignore_index=-100)
    loss.backward()
    g = attn[name].grad
    print("обратный проход: градиент на %s -> норма %.6f" % (name.split(".")[-2], g.norm().item()))

    ok = (rel < 0.02 and abs(got_loss - ref_loss) < 0.05
          and summary["стало_ГБ"] < 0.6 * summary["было_ГБ"] and g.norm().item() > 0)
    print("\nИТОГ:", "ПРОЙДЕНО" if ok else "НЕ ПРОЙДЕНО")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

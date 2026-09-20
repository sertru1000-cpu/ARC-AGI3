"""Пустой адаптер LoRA к боевой модели -- проверка ПУТИ ДОСТАВКИ обученной модели (20.09).

Зачем. Прежде чем платить за обучение ($535-1425, docs/artifacts/training_cost_20_09.html), надо убедиться,
что обученную модель вообще можно довезти до боя. Ядро прибито к RadixArk/Qwen3.8-Flash-Next-NVFP4 тремя
проверками в serving_setup.py, а vLLM умеет молча пропустить модули адаптера -- так уже было со студентом 27B
в августе. Адаптер из НУЛЕЙ (матрица B нулевая) математически не меняет ответ модели: если балл не изменился,
а в логе видно загруженный адаптер -- путь доставки жив; если vLLM падает или не видит модулей -- линия обучения
мертва, и мы узнаём это бесплатно.

Цели: q_proj/k_proj/v_proj/o_proj в 12 слоях ПОЛНОГО внимания (остальные 36 слоёв -- линейное внимание,
их in_proj_*/out_proj vLLM для LoRA не берёт; именно на них в августе адаптер молча пропустили).
Имена модулей взяты из описи весов модели (model.safetensors.index.json), формы -- из config.json
(hidden 2560, 24 головы по 256, 2 головы ключей).

usage:  .venv/bin/python scripts/build_empty_lora.py [--rank 16] [--out data/empty_lora]
"""
import argparse, json
from pathlib import Path
import torch
from safetensors.torch import save_file

HID, HEADS, KV, HEAD_DIM, LAYERS = 2560, 24, 2, 256, 48
REPO = "RadixArk/Qwen3.8-Flash-Next-NVFP4"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", default="data/empty_lora")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    full = [i for i in range(LAYERS) if (i + 1) % 4 == 0]      # 3, 7, ... 47 -- слои полного внимания
    shapes = {"q_proj": (HEADS * HEAD_DIM, HID), "k_proj": (KV * HEAD_DIM, HID),
              "v_proj": (KV * HEAD_DIM, HID), "o_proj": (HID, HEADS * HEAD_DIM)}
    g = torch.Generator().manual_seed(0)
    tensors = {}
    for i in full:
        for name, (o_f, i_f) in shapes.items():
            base = "base_model.model.model.language_model.layers.%d.self_attn.%s" % (i, name)
            # A -- обычная инициализация Кайминга, B -- НУЛИ: произведение B@A = 0, ответ модели не меняется
            tensors[base + ".lora_A.weight"] = (torch.randn(a.rank, i_f, generator=g) * (1.0 / i_f ** 0.5)).to(torch.bfloat16)
            tensors[base + ".lora_B.weight"] = torch.zeros(o_f, a.rank, dtype=torch.bfloat16)
    save_file(tensors, str(out / "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": REPO,
           "r": a.rank, "lora_alpha": 2 * a.rank, "lora_dropout": 0.0, "bias": "none",
           "fan_in_fan_out": False, "inference_mode": True,
           "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
    json.dump(cfg, open(out / "adapter_config.json", "w"), indent=1)
    size = sum(t.numel() * t.element_size() for t in tensors.values())
    print("ok   слоёв полного внимания %d, тензоров %d, ранг %d, размер %.1f МБ -> %s"
          % (len(full), len(tensors), a.rank, size / 1e6, out))
    print("     B нулевая => ответ модели не меняется; любое изменение балла -- шум или накладные расходы vLLM")


if __name__ == "__main__":
    main()

"""Загрузка модели по одному файлу со сжатием экспертов на лету (24.09).

Зачем. Обычный `from_pretrained` держит весь чекпойнт в оперативной памяти, а потом уже раскладывает.
Для нашей модели это 360 ГБ, и на подах B300 в EU-NL-1 столько памяти нет: там 251 ГБ и 32 ядра
(прежний под с 2 ТБ был исключением и достался однажды). Поэтому читаем 131 файл по одному и каждый
тензор сразу кладём на карту — эксперты по дороге сжимаем в fp8, остальное как есть.

Что это даёт: модель 360 -> 243 ГБ, помещается на карту ЦЕЛИКОМ вместе с таблицей PLE. Именно PLE
в оперативной памяти был причиной того, что карта простаивала на две трети (измерено 24.09: загрузка
карты 31-38%, скорость 70 токенов/с).

Две ловушки имён, обе проверены:
 1. в чекпойнте модули зовутся `model.language_model.*`, в загруженной CausalLM префикс `language_model.`
    срезан -- имена сопоставляем обе стороны, а не угадываем;
 2. общий эксперт (`shared_expert`) -- обычный MLP, его НЕ сжимаем: в боевом чекпойнте он тоже bf16.
    Сжимаем только то, что лежит внутри `Qwen4ExpTextExperts`.

usage:
    from load_sharded_fp8 import load_sharded
    model, info = load_sharded("/workspace/base", device="cuda", fp8_experts=True)
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from fp8_experts import _forward_fp8, _quantize

EXPERT_PARAMS = ("gate_up_proj", "down_proj")


def _expert_module_names(model) -> set[str]:
    return {n for n, m in model.named_modules() if type(m).__name__ == "Qwen4ExpTextExperts"}


def _resolve(name: str, known: set[str]) -> str | None:
    """Имя тензора из файла -> имя параметра в модели. Пробуем как есть и без `language_model.`."""
    if name in known:
        return name
    stripped = name.replace("model.language_model.", "model.")
    if stripped in known:
        return stripped
    if name.startswith("model.") and name[6:] in known:
        return name[6:]
    return None


def _set(model, full_name: str, tensor: torch.Tensor, device) -> None:
    mod_name, _, attr = full_name.rpartition(".")
    mod = model.get_submodule(mod_name) if mod_name else model
    tensor = tensor.to(device, non_blocking=False)
    if attr in dict(mod.named_parameters(recurse=False)):
        mod._parameters[attr] = torch.nn.Parameter(tensor, requires_grad=False)
    else:
        mod._buffers[attr] = tensor


def load_sharded(path: str, device: str = "cuda", fp8_experts: bool = True,
                 dtype=torch.bfloat16, verbose: bool = True, probe_layers: int = 0):
    """probe_layers > 0 -- собрать модель ИЗ ЭТОГО ЖЕ чекпойнта, но только с таким числом слоёв.
    Полная загрузка читает 336 ГБ и занимает 42 минуты; ошибки типов, устройств и битовых операций
    ловятся на двух слоях за семь минут (24.09: три полные загрузки подряд ушли на такие ошибки)."""
    t0 = time.time()
    path = Path(path)
    cfg = AutoConfig.from_pretrained(path)
    if probe_layers:
        tc = cfg.get_text_config()
        tc.num_hidden_layers = probe_layers
        if isinstance(getattr(tc, "layer_types", None), list):
            tc.layer_types = tc.layer_types[:probe_layers]
        if verbose:
            print("ПРОБА: собираю модель из %d слоёв вместо полного набора" % probe_layers, flush=True)
    with init_empty_weights():                      # буферы при этом создаются настоящие
        model = AutoModelForCausalLM.from_config(cfg)
    model = model.to(dtype)          # затрагивает только вещественные тензоры

    known_p = {n for n, _ in model.named_parameters()}
    known_b = {n for n, _ in model.named_buffers()}
    known = known_p | known_b
    experts = _expert_module_names(model)

    index = path / "model.safetensors.index.json"
    files = (sorted({v for v in json.load(open(index))["weight_map"].values()})
             if index.exists() else sorted(p.name for p in path.glob("*.safetensors")))

    gb_gpu = gb_saved = 0.0
    n_q = n_plain = n_skip = 0
    pending: dict[str, dict[str, torch.Tensor]] = {}
    # ЛОВУШКА (24.09): таблица PLE лежит в чекпойнте 128 кусками ngram_embedding.shard_0.weight ...
    # shard_127.weight, а в модели это ОДИН слой ngram_embedding.weight -- transformers их склеивает,
    # и загрузчик обязан делать то же, иначе 102 ГБ молча пропадают.
    shards: dict[str, dict[int, torch.Tensor]] = {}
    SHARD_RE = re.compile(r"\.shard_(\d+)\.")

    for i, fname in enumerate(files, 1):
        with safe_open(path / fname, framework="pt") as f:
            for key in f.keys():
                m = SHARD_RE.search(key)
                if m is not None:
                    base = SHARD_RE.sub(".", key)
                    tgt = _resolve(base, known)
                    if tgt is not None:
                        _t = f.get_tensor(key)
                        shards.setdefault(tgt, {})[int(m.group(1))] = (
                            _t.to(dtype) if _t.is_floating_point() else _t)
                        continue
                target = _resolve(key, known)
                if target is None:
                    n_skip += 1
                    continue
                mod_name, _, attr = target.rpartition(".")
                w = f.get_tensor(key)
                # ЛОВУШКА ТИПОВ (24.09): приводить к bf16 ВСЁ подряд нельзя -- таблица PLE строит
                # индексы n-грамм битовыми операциями, а они на bf16 не определены
                # ("bitwise_xor_cuda not implemented for BFloat16"). Тип меняем только у вещественных.
                if fp8_experts and mod_name in experts and attr in EXPERT_PARAMS:
                    q, s = _quantize(w.to(dtype))
                    gb_saved += w.numel() * w.element_size() / 1e9
                    pending.setdefault(mod_name, {})[attr] = None     # пометка: модуль сжат
                    mod = model.get_submodule(mod_name)
                    if attr in mod._parameters:
                        del mod._parameters[attr]
                    pre = "gate_up" if attr.startswith("gate_up") else "down"
                    mod.register_buffer(pre + "_q", q.to(device), persistent=False)
                    mod.register_buffer(pre + "_scale", s.to(device), persistent=False)
                    gb_gpu += (q.numel() * q.element_size() + s.numel() * s.element_size()) / 1e9
                    n_q += 1
                else:
                    _set(model, target, w.to(dtype) if w.is_floating_point() else w, device)
                    gb_gpu += w.numel() * 2 / 1e9
                    n_plain += 1
        if verbose and (i % 10 == 0 or i == len(files)):
            print("  файл %d/%d | на карте %.1f ГБ | сжато тензоров %d"
                  % (i, len(files), gb_gpu, n_q), flush=True)

    for tgt, parts in shards.items():          # склейка кусков в порядке номеров
        whole = torch.cat([parts[i] for i in sorted(parts)], dim=0)
        _set(model, tgt, whole, device)
        gb_gpu += whole.numel() * whole.element_size() / 1e9
        n_plain += 1
        if verbose:
            print("  склеено %d кусков -> %s %s" % (len(parts), tgt, tuple(whole.shape)), flush=True)
        del whole, parts
    shards.clear()

    for mod_name in pending:                         # патчим счёт только у сжатых модулей
        mod = model.get_submodule(mod_name)
        mod.forward = _forward_fp8.__get__(mod, type(mod))

    # ЛОВУШКА (24.09): буферы, которых НЕТ в чекпойнте, модель создаёт сама при сборке — они
    # остаются в оперативной памяти, и первый же прямой проход падает на «tensors on different
    # devices» (rotary inv_freq). Переносим всё, что осталось не на карте.
    dev = torch.device(device)
    moved = 0
    for name, buf in list(model.named_buffers()):
        if buf is not None and not buf.is_meta and buf.device != dev:
            _set(model, name, buf, device); moved += 1
    for name, prm in list(model.named_parameters()):
        if not prm.is_meta and prm.device != dev:
            _set(model, name, prm.data, device); moved += 1
    if verbose and moved:
        print("  доперенесено на карту: %d буферов и параметров" % moved, flush=True)

    left = [n for n, p in model.named_parameters() if p.is_meta]
    if left:
        raise SystemExit("не заполнено %d параметров, первый: %s" % (len(left), left[0]))

    info = {"файлов": len(files), "сжато": n_q, "как есть": n_plain, "пропущено": n_skip,
            "на карте_ГБ": gb_gpu, "сэкономлено_ГБ": gb_saved - (gb_gpu if n_q == 0 else 0),
            "минут": (time.time() - t0) / 60}
    if verbose:
        print("ЗАГРУЖЕНА за %.1f мин | на карте %.1f ГБ | сжато тензоров %d, как есть %d, пропущено %d"
              % (info["минут"], info["на карте_ГБ"], n_q, n_plain, n_skip), flush=True)
    return model, info

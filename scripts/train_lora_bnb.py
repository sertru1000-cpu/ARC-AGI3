"""Обучение адаптера на ИСХОДНОЙ модели bf16 со сжатием ТОЛЬКО экспертов (24.09).

Почему не боевой чекпойнт. Боевой файл сжат инструментом NVIDIA в формат `modelopt`; transformers его не
читает вовсе, а встроенный `nvfp4` умеет только сжимать сам («Loading pre-quantized NVFP4 checkpoints is not
supported yet»). Поэтому берём исходные веса Qwen/Qwen3.8-Flash-Next (360 ГБ, bf16) и сжимаем экспертов сами,
библиотекой bitsandbytes, которую transformers поддерживает.

Ключевая деталь, ради которой всё сходится: обучаемые модули (внимание, линейное внимание, маршрутизаторы,
общий эксперт) мы НЕ сжимаем -- они остаются bf16. В боевом чекпойнте эти же модули тоже лежат в bf16 и
НИКОГДА не сжимались, то есть это побитово те же веса. Значит адаптер, обученный здесь, ляжет в боевую
модель точно, без приближения, через уже проверенную подмену bf16-файлов.

Память на карте: эксперты в четырёх битах ~60 ГБ + bf16-часть ~16 ГБ + активации. Влезает в H200 (141 ГБ).

usage:
  python3 train_lora_bnb.py --smoke            # загрузка + 2 шага, проверка памяти
  python3 train_lora_bnb.py --epochs 1 --out /workspace/out/lora_v1
"""
import argparse, json, math, os, sys, time
from pathlib import Path

# Модули, которые НЕ сжимаем: их мы обучаем, и они же лежат в bf16 в боевом чекпойнте.
KEEP_BF16 = ["self_attn", "linear_attn", "mlp.gate", "shared_expert", "hyper_connection",
             "embed_tokens", "lm_head", "mtp", "visual", "ple"]
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/base")
    ap.add_argument("--gpu-mem", default="240GiB",
                    help="сколько отдать карте. ЛОВУШКА (измерено 24.09): загрузчик тянет на карту ВСЁ, включая "
                         "послойные эмбеддинги (51 ГБ) и зрительную башню, которые в бою лежат в оперативной "
                         "памяти. Итого ~360 ГБ вместо 258, и B300 (288) переполняется на 74% загрузки")
    ap.add_argument("--cpu-mem", default="1200GiB", help="сколько разрешить выгрузить в оперативную память")
    ap.add_argument("--no-quant", action="store_true",
                    help="без bitsandbytes: он всё равно не сжимает слитые тензоры экспертов, а с выгрузкой в "
                         "оперативную память конфликтует. Модель 258 ГБ влезает в B300 (275 ГБ) как есть")
    ap.add_argument("--devices", default="auto", help="auto -- разложить слои по всем картам (нужно при 2xH200)")
    ap.add_argument("--data", default="/workspace/data/train.jsonl")
    ap.add_argument("--valid", default="/workspace/data/valid.jsonl")
    ap.add_argument("--out", default="/workspace/out/lora_v1")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=10000,
                    help="ЛОВУШКА ПАМЯТИ: буфер логитов = длина x словарь 248320 x 2 байта, на 26 тыс. токенов\n"
                         "это 13 ГБ (и вдвое больше при потере в fp32). На двух H200 свободно всего 24 ГБ,\n"
                         "поэтому режем длину и считаем потерю кусками (см. chunked_loss ниже)")
    ap.add_argument("--loss-chunk", type=int, default=1024, help="по сколько токенов считать потерю за раз")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--no-preflight", action="store_true", help="пропустить пробу компилятора")
    ap.add_argument("--valid-n", type=int, default=60,
                    help="сколько отложенных примеров считать после каждой эпохи (0 - не считать)")
    ap.add_argument("--probe-layers", type=int, default=0,
                    help="ПРОБА: собрать модель из N слоёв (обычно 2) из того же чекпойнта, прогнать\n"
                         "один настоящий пример с обратным проходом и выйти. Семь минут вместо сорока\n"
                         "двух: 24.09 три полные загрузки подряд ушли на ошибки типов и устройств")
    ap.add_argument("--fp8-experts", action="store_true",
                    help="читать чекпойнт по одному файлу и сжимать экспертов в fp8 на лету: модель\n"
                         "занимает 243 ГБ вместо 360 и влезает на карту ЦЕЛИКОМ вместе с таблицей PLE.\n"
                         "Проверено локально (scripts/test_load_sharded_fp8.py): потеря та же до 5-го знака")
    ap.add_argument("--ple-cpu", action="store_true",
                    help="таблицу PLE (102 ГБ) держать в оперативной памяти, всё остальное на карте")
    ap.add_argument("--smoke", action="store_true", help="загрузка + 2 шага: проверка памяти и скорости")
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # ПРОБА ПЕРЕД ЗАГРУЗКОЙ (24.09): на B300 (sm_103) ptxas из CUDA 12.8 не знает архитектуру -
    # "Value 'sm_103a' is not defined". Падало это внутри torch.compile у линейного внимания, то есть
    # ПОСЛЕ 19 минут загрузки весов и $2.5. Лечение - ptxas 12.9 (pip nvidia-cuda-nvcc-cu12).
    # Проверяем компилятор на крошечном ядре ДО загрузки: секунды вместо двадцати минут.
    _px = "/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvcc/bin/ptxas"
    if os.path.exists(_px) and not os.environ.get("TRITON_PTXAS_PATH"):
        os.environ["TRITON_PTXAS_PATH"] = _px
    if not a.no_preflight:
        import triton, triton.language as tl

        @triton.jit
        def _add1(xp, yp, n, BLOCK: tl.constexpr):
            o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            m = o < n
            tl.store(yp + o, tl.load(xp + o, mask=m) + 1.0, mask=m)

        _x = torch.arange(1024, device="cuda", dtype=torch.float32); _y = torch.empty_like(_x)
        _add1[(1,)](_x, _y, 1024, BLOCK=1024)
        assert (_y - _x - 1).abs().max().item() == 0
        print("проба: ядра Triton компилируются (ptxas %s)" % os.environ.get("TRITON_PTXAS_PATH", "штатный"),
              flush=True)

    t0 = time.time()
    qc = None if a.no_quant else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                            llm_int8_skip_modules=KEEP_BF16)
    print("загружаю модель (%s)" % ("чистый bf16, всё на карте" if a.no_quant else "эксперты -> 4 бита"), flush=True)
    # ЛОВУШКА ПАМЯТИ (измерено 24.09): bf16-репозиторий 360 ГБ = эксперты 247 + PLE 102.5 +
    # внимание 7 + словарь 2.6 + зрение 0.9. Наши расчётные 258 ГБ считали всё, КРОМЕ PLE:
    # в боевом профиле PLE лежит в fp8 в памяти хоста и на карту не попадает. При обучении он
    # часть обычного чекпойнта и вдвое тяжелее. PLE - таблица поиска по n-граммам, матричных
    # умножений в ней нет, поэтому её место на CPU: через шину идёт срез на токены, а не веса.
    # Без этого accelerate выносит на CPU куски экспертов и шаг считается минутами.
    if a.fp8_experts:
        # Пошардовая загрузка со сжатием: оперативная память пода (251 ГБ) меньше чекпойнта (360),
        # поэтому "загрузить всё, потом сжать" не проходит - читаем по файлу и сразу кладём на карту.
        sys.path.insert(0, str(Path(__file__).parent))
        from load_sharded_fp8 import load_sharded
        model, _info = load_sharded(a.model, device="cuda", fp8_experts=True, dtype=torch.bfloat16,
                                    probe_layers=a.probe_layers)
    elif a.ple_cpu:
        # ЛОВУШКА ACCELERATE (измерено 24.09): пометить модуль "cpu" в device_map НЕДОСТАТОЧНО.
        # accelerate вешает на него AlignDevicesHook с execution_device=0 и при инициализации
        # хука тащит веса на карту - падение "Tried to allocate 95.37 GiB" в самом конце
        # загрузки, то есть через 25 минут и $3. Поэтому раскладку делаем руками: грузим в
        # оперативную память, переносим на карту всё, кроме таблицы PLE, и вешаем на неё два
        # хука - вход на процессор, результат обратно на карту.
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
        ple_names = [n for n, _ in model.named_modules() if n.endswith("ple.ple_embedding")]
        if not ple_names:
            raise SystemExit("модуль PLE не найден - раскладка не построена, запуск отменён")
        print("PLE остаётся в оперативной памяти: %s" % ", ".join(ple_names), flush=True)
        keep = tuple(n + "." for n in ple_names)
        on_gpu = on_cpu = 0
        for n, prm in model.named_parameters():
            if n.startswith(keep):
                on_cpu += prm.numel() * prm.element_size(); continue
            prm.data = prm.data.to("cuda"); on_gpu += prm.numel() * prm.element_size()
        for n, buf in model.named_buffers():
            if not n.startswith(keep):
                buf.data = buf.data.to("cuda")

        def _in_to_cpu(mod, args, kwargs):
            f = lambda t: t.to("cpu") if torch.is_tensor(t) else t
            return tuple(f(x) for x in args), dict((k, f(v)) for k, v in kwargs.items())

        def _out_to_gpu(mod, args, out):
            f = lambda t: t.to("cuda") if torch.is_tensor(t) else t
            if torch.is_tensor(out):
                return f(out)
            if isinstance(out, (tuple, list)):
                return type(out)(f(x) for x in out)
            return out

        for n in ple_names:
            m = model.get_submodule(n)
            m.register_forward_pre_hook(_in_to_cpu, with_kwargs=True)
            m.register_forward_hook(_out_to_gpu)
        print("на карте %.1f ГБ, в оперативной памяти %.1f ГБ" % (on_gpu/1e9, on_cpu/1e9), flush=True)
    else:
        kw = {"dtype": torch.bfloat16, "device_map": (a.devices if a.devices == "auto" else {"": 0})}
        if a.devices == "auto":
            kw["max_memory"] = {0: a.gpu_mem, "cpu": a.cpu_mem}
        if qc is not None:
            kw["quantization_config"] = qc
        model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    print("ЗАГРУЖЕНА за %.1f мин | на карте %.1f ГБ" % ((time.time()-t0)/60, torch.cuda.memory_allocated()/1e9), flush=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2*a.rank, lora_dropout=0.0,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("обучаемых параметров %.1f млн" % (trainable/1e6), flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    print("примеров %d" % len(rows), flush=True)

    def render(messages):
        ids, labels = [], []
        for m in messages:
            txt = m.get("content") or ""
            if m.get("tool_calls"):
                txt = (txt + "\n" + json.dumps(m["tool_calls"], ensure_ascii=False)).strip()
            if m["role"] == "assistant":
                r = (m.get("reasoning_content") or "").strip()
                txt = "<think>\n%s\n</think>\n\n%s" % (r, txt) if r else "<think>\n\n</think>\n\n" + txt
            role = m["role"] if m["role"] != "tool" else "user"
            piece = "<|im_start|>%s\n%s<|im_end|>\n" % (role, txt)
            t = tok(piece, add_special_tokens=False)["input_ids"]
            ids += t
            labels += t if m["role"] == "assistant" else [-100]*len(t)
        return ids, labels

    dec = model.get_decoder()
    import gc; gc.collect(); torch.cuda.empty_cache()
    print("свободно на карте перед счётом %.1f ГБ"
          % ((torch.cuda.get_device_properties(0).total_memory
              - torch.cuda.memory_reserved()) / 1e9), flush=True)
    def valid_loss() -> float:
        """Потеря на ОТЛОЖЕННЫХ играх: они в обучении не участвуют, поэтому только это число
        отличает обобщение от заучивания 495 примеров."""
        if not (a.valid_n and Path(a.valid).exists()):
            return float("nan")
        vrows = [json.loads(l) for l in open(a.valid, encoding="utf-8")][:a.valid_n]
        model.eval(); vs = 0.0; vn = 0
        with torch.no_grad():
            for vr in vrows:
                vi, vl = render(vr["messages"])
                if len(vi) > a.max_len:
                    vi, vl = vi[-a.max_len:], vl[-a.max_len:]
                if not any(t != -100 for t in vl):
                    continue
                vx = torch.tensor([vi]).cuda(); vy = torch.tensor([vl]).cuda()
                vh = dec(input_ids=vx, use_cache=False).last_hidden_state
                hd = model.get_output_embeddings()
                for c0 in range(0, vh.shape[1] - 1, a.loss_chunk):
                    c1 = min(c0 + a.loss_chunk, vh.shape[1] - 1)
                    tg = vy[:, c0 + 1:c1 + 1]; mk = tg != -100
                    if mk.any():
                        lgv = hd(vh[:, c0:c1]).float()
                        vs += float(torch.nn.functional.cross_entropy(
                            lgv[mk], tg[mk], reduction="sum")); vn += int(mk.sum())
                        del lgv
        model.train()
        return vs / max(1, vn)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    model.train()
    base_valid = float("nan") if a.probe_layers else valid_loss()   # НУЛЕВАЯ ТОЧКА
    if not a.probe_layers:
        print("ДО ОБУЧЕНИЯ: потеря на отложенных играх %.3f" % base_valid, flush=True)
    step = seen = 0; losses = []; t1 = time.time()
    max_steps = 2 if a.smoke else 0
    if a.probe_layers:                      # проба: один пример, один шаг, и выходим
        a.accum = 1; max_steps = 1
    for epoch in range(math.ceil(a.epochs)):
        print("=== ЭПОХА %d из %d ===" % (epoch + 1, math.ceil(a.epochs)), flush=True)
        for i, r in enumerate(rows):
            ids, labels = render(r["messages"])
            if len(ids) > a.max_len:
                ids, labels = ids[-a.max_len:], labels[-a.max_len:]
            if not any(x != -100 for x in labels):
                continue
            # ПОТЕРЯ КУСКАМИ: просим модель отдать скрытые состояния и считаем логиты порциями, иначе
            # на длинном примере логиты на всю длину не влезают в память (см. --max-len).
            t_ex = time.time()
            x = torch.tensor([ids]).cuda(); y = torch.tensor([labels]).cuda()
            # ЛОВУШКА ПАМЯТИ (OOM 24.09): output_hidden_states=True хранит ВСЕ 49 слоёв - при 14 тыс.
            # токенов это 3.4 ГБ впустую, нам нужен только последний. Зовём декодер напрямую; PLE
            # при этом отработает правильно - при ple_input_ids=None он берёт их из input_ids
            # (modeling_qwen4_exp.py:1425).
            hs = dec(input_ids=x, use_cache=False).last_hidden_state
            head = model.get_output_embeddings()
            loss_sum = torch.zeros((), device=hs.device, dtype=torch.float32); n_tok = 0
            for c0 in range(0, hs.shape[1] - 1, a.loss_chunk):
                c1 = min(c0 + a.loss_chunk, hs.shape[1] - 1)
                lg = head(hs[:, c0:c1]).float()
                tgt = y[:, c0 + 1:c1 + 1]
                m = tgt != -100
                if m.any():
                    loss_sum = loss_sum + torch.nn.functional.cross_entropy(
                        lg[m], tgt[m], reduction="sum")
                    n_tok += int(m.sum())
                del lg
            out_loss = loss_sum / max(1, n_tok)
            (out_loss / a.accum).backward()
            class _O: pass
            out = _O(); out.loss = out_loss
            losses.append(float(out.loss.detach())); seen += len(ids)
            # ПЕЧАТЬ ПОСЛЕ КАЖДОГО ПРИМЕРА (24.09). Без неё прогон слеп: между шагами проходит
            # восемь примеров по 32 тыс. токенов, то есть полчаса молчания, и понять, считает он
            # или встал, невозможно -- на этом мы потеряли вечер.
            _dt = time.time() - t_ex
            print("  пример %d/%d | %d токенов | потеря %.3f | %.0f ток/с (этот) | %.1f мин на пример"
                  " | память %.0f ГБ"
                  % (len(losses), len(rows), len(ids), losses[-1], len(ids) / max(_dt, 1e-9),
                     _dt / 60, torch.cuda.max_memory_allocated() / 1e9), flush=True)
            if len(losses) % a.accum == 0:
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                dt = time.time()-t1
                print("шаг %d | потеря %.3f | %.0f ток/с | память %.0f ГБ | прошло %.1f мин"
                      % (step, sum(losses[-a.accum:])/a.accum, seen/dt, torch.cuda.max_memory_allocated()/1e9, dt/60), flush=True)
                if max_steps and step >= max_steps:
                    print("ДЫМОВОЙ ПРОГОН ПРОЙДЕН"); return
        if a.valid_n and Path(a.valid).exists():
            print("ПРОВЕРКА после эпохи %d: потеря на отложенных играх %.3f (было до обучения %.3f)"
                  % (epoch + 1, valid_loss(), base_valid if base_valid else float("nan")), flush=True)
        ep_dir = "%s_epoch%d" % (a.out, epoch + 1)      # адаптер после каждой эпохи отдельно
        Path(ep_dir).mkdir(parents=True, exist_ok=True); model.save_pretrained(ep_dir)
        print("ЭПОХА %d СОХРАНЕНА в %s | средняя потеря %.3f"
              % (epoch + 1, ep_dir, sum(losses[-len(rows):]) / max(1, len(losses[-len(rows):]))), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    print("ГОТОВО: шагов %d, средняя потеря %.3f, %.0f ток/с, адаптер в %s"
          % (step, sum(losses)/max(1,len(losses)), seen/(time.time()-t1), a.out), flush=True)


if __name__ == "__main__":
    main()

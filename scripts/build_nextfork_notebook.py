"""Сборка кернела на нашем форке боевой базы (25.09).

С 25.09 все изменения обвязки живут в nextfork/ — это распакованный боевой
бандл keithtyser плюс наши правки. Кернел отличается от стокового ровно одним:
вместо чужого датасета с исходниками подключён наш. Рантайм vLLM, веса модели
и настройки прогона остаются чужими и нетронутыми.

Отдельно вшита проверка подлинности бандла. 06.09 мы уже потеряли прогон на
том, что кернел молча подхватил не ту сборку (старый форк 27B) — результат
оказался бессмысленным, а узнали мы об этом через четыре часа. Теперь кернел
на старте требует в бандле файл NEXTFORK_VERSION.txt и печатает его: если
подключён чужой датасет, прогон падает сразу, а не тратит квоту.

Скрипт ничего не пушит.

Режим --smoke собирает получасовую пробу: потолок игры 1800 с вместо 7920.
Смысл не в балле, а в том, чтобы за час проверить всю цепочку — смонтировался
ли наш датасет, прошла ли проверка бандла, поднялся ли сервер, сработал ли
запрет (искать в логе строки NOOPGUARD). Точка сравнения для балла, если он
всё-таки понадобится: docs/base30_flash_v1_h115.json, 3.32.

usage: .venv/bin/python scripts/build_nextfork_notebook.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_NB = ROOT / "kernels/notebooks_stockflash/submission.ipynb"

SMOKE_CAP_S = 1800.0        # получасовая проба; полный прогон — 7920
FULL_CAP_LINE = "bm.solver.max_runtime_s_per_game = 7920.0"

UPSTREAM_SRC = "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1"
OUR_SRC = "sergueimakarov/arc3-nextfork"

OLD_SOURCES = f'DATASET_SOURCES = ["{UPSTREAM_SRC}", "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"]'
NEW_SOURCES = f'DATASET_SOURCES = ["{OUR_SRC}", "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"]'

ANCHOR = 'print(f"taaf.kaggle: source bundle = {BUNDLE_DIR}")'
CHECK = ANCHOR + '''

# Подключён ли ИМЕННО наш форк. Штамп кладёт scripts/build_nextfork_dataset.py;
# в чужом бандле его нет, поэтому подмена источника обнаруживается на первых
# секундах прогона, а не по бессмысленному результату через четыре часа.
_stamp = BUNDLE_DIR / "NEXTFORK_VERSION.txt"
if not _stamp.is_file():
    raise RuntimeError(
        f"NEXTFORK_VERSION.txt не найден в {BUNDLE_DIR}: подключён не наш форк. "
        f"Ожидался датасет {DATASET_SOURCES[0]}."
    )
print("nextfork: " + _stamp.read_text().strip().replace("\\n", " | "))
# 27.09: штамп должен содержать всё, на что рассчитан этот ноутбук, — иначе в бой уйдёт старая версия датасета,
# а выключатели в ячейке 3 молча ничего не сделают.
_need = __NEED__
_miss = [m for m in _need if m not in _stamp.read_text()]
if _miss:
    raise RuntimeError(f"в датасете {DATASET_SOURCES[0]} нет частей {_miss}: опубликуйте свежий nextfork")'''

SMOKE_BLOCK_T = """
# Получасовая проба: цель — проверить цепочку, а не взять балл. В настоящем
# сабмите потолок не трогаем.
if not TRUE_SUBMISSION:
    bm.solver.max_runtime_s_per_game = %.1f
    print("NEXTFORK SMOKE: потолок игры %%.0f с, сравнение docs/base30_flash_v1_h115.json = 3.32"
          %% bm.solver.max_runtime_s_per_game, flush=True)
"""
SMOKE_BLOCK = SMOKE_BLOCK_T % SMOKE_CAP_S

METADATA = {
    "id": "sergueimakarov/arc3-nextfork",
    "title": "arc3 nextfork",
    "code_file": "submission.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "machine_shape": "NvidiaRtxPro6000",
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": [],
    "dataset_sources": [OUR_SRC, "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"],
    "kernel_sources": [],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "model_sources": ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
}


NOPROBE_LINE = 'os.environ["MPLBACKEND"] = "Agg"'
NOCONS_BLOCK = NOPROBE_LINE + """
# Плечо «форк БЕЗ согласования двух ответов» для парного замера 26.09. Датасет тот же.
os.environ["NEXTFORK_CONSENSUS"] = "0"
print("NEXTFORK: согласование двух ответов ВЫКЛЮЧЕНО (плечо сравнения)", flush=True)"""
NOPROBE_BLOCK = NOPROBE_LINE + """
# Плечо «форк БЕЗ протокола разведки» для парного замера 26.09. Датасет тот же,
# что и у плеча с протоколом; отличие ровно в одной переменной окружения.
os.environ["NEXTFORK_START_PROBE"] = "0"
print("NEXTFORK: протокол разведки на старте ВЫКЛЮЧЕН (плечо сравнения)", flush=True)"""


# 26.09: AGENTFIX Скотта теперь МОДУЛЬ нашей обвязки (inference/agent/agentfix_scott.py) и ставится сам при импорте
# solver.py, включён по умолчанию; окно контекста — переменной NEXTFORK_CONTEXT_WINDOW, которую читает tool_agent.
# Ноутбук остаётся стоковым: сборщик только дописывает переменные окружения в ячейку 3 (до всякого импорта).
SEQS_OLD = '"TAAF_VLLM_MAX_NUM_SEQS": "8",'
KV_OLD = '"TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "5368709120",'
KVDT_OLD = '"TAAF_VLLM_KV_CACHE_DTYPE": "auto",'
# 29.09: сборка vLLM (NEXTFORK_RUNTIME в serving_setup.py): keith — боевая дев-сборка 26.08; v030 — vLLM 0.30.0
# (публичный рантайм foysal); nightly — наш рантайм из ночного образа vLLM (fp8-кэш QSA). Меняется датасет рантайма.
KEITH_RUNTIME = "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"
RUNTIME_DATASETS = {"keith": KEITH_RUNTIME, "v030": "foysalemonshanto/vllm-0-30-0-runtime",
                    "nightly": "sergueimakarov/vllm-nightly-36768d1b-runtime"}


def build(smoke: bool = False, noprobe: bool = False, nocons: bool = False, agentfix: bool = False,
          seqs: int = 0, ctx: int = 0, kv_gib: int = 0, cap: float = 0.0, name: str = "", persist: bool = True,
          observe: bool = False, zoom: bool = False, noopguard: bool = True, v3prompts: bool = True, model: str = "keith", buildwm: bool = False, kv_fp8: bool = False, effort: str = "", compact: int = 0, runtime: str = "keith", phase_a_cap: float = 0.0) -> None:
    suffix = "_smoke" if smoke else "_noprobe" if noprobe else "_nocons" if nocons else ""
    if name:
        suffix = "_" + name
    out_dir = ROOT / ("kernels/notebooks_nextfork" + suffix)
    out_nb = out_dir / "submission.ipynb"
    metadata = dict(METADATA)
    if model == "swift":                     # 28.09: модель Swift 1.5 — зеркало lordhansolo (186.5 ГБ, экземпляр PyTorch/hf-nvfp4 v1)
        metadata["model_sources"] = ["lordhansolo/swift-1-5-qwen3-8-flash-next-nvfp4/PyTorch/hf-nvfp4/1"]
    if model == "reap448":                   # 29.09: REAP 448 экспертов — зеркало boristown (124.2 ГБ, PyTorch/nvfp4-reap-k448 v2)
        metadata["model_sources"] = ["boristown/qwen3-8-flash-next-nvfp4-reap-448e/PyTorch/nvfp4-reap-k448/2"]
    if runtime != "keith":
        metadata["dataset_sources"] = [OUR_SRC, RUNTIME_DATASETS[runtime]]
    if smoke:
        metadata["id"] = "sergueimakarov/arc3-nextfork-smoke"
        metadata["title"] = "arc3 nextfork smoke"
    if noprobe:
        metadata["id"] = "sergueimakarov/arc3-nextfork-noprobe"
        metadata["title"] = "arc3 nextfork noprobe"
    if nocons:
        metadata["id"] = "sergueimakarov/arc3-nextfork-nocons"
        metadata["title"] = "arc3 nextfork nocons"
    if name:
        metadata["id"] = "sergueimakarov/arc3-nextfork-" + name.replace("_", "-")
        metadata["title"] = "arc3 nextfork " + name.replace("_", " ")

    nb = json.loads(BASE_NB.read_text(encoding="utf-8"))
    replaced_sources = replaced_check = replaced_cap = replaced_noprobe = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if OLD_SOURCES in src:
            src = src.replace(OLD_SOURCES, NEW_SOURCES.replace(KEITH_RUNTIME, RUNTIME_DATASETS[runtime]))
            replaced_sources += 1
        if ANCHOR in src and "NEXTFORK_VERSION" not in src:
            need = (["AGENTFIX"] if agentfix else []) + (["persist_defs"] if persist else []) + \
                   (["выключатели частей v3"] if not (noopguard and v3prompts) else []) + \
                   ([f"NEXTFORK_MODEL={model}"] if model != "keith" else []) + \
                   (["NEXTFORK_BUILDWM"] if buildwm else []) + \
                   (["NEXTFORK_COMPACT"] if compact else []) + \
                   ([f"NEXTFORK_RUNTIME={runtime}"] if runtime != "keith" else [])
            src = src.replace(ANCHOR, CHECK.replace("__NEED__", repr(need)), 1)
            replaced_check += 1
        if smoke and FULL_CAP_LINE in src:
            # 29.09: --phase-a-cap — короткая фаза A (проверка, что сервер встал на Kaggle и игры пошли); фаза B не трогается
            block = SMOKE_BLOCK_T % phase_a_cap if phase_a_cap else SMOKE_BLOCK
            src = src.replace(FULL_CAP_LINE, FULL_CAP_LINE + "\n" + block.strip(), 1)
            replaced_cap += 1
        if noprobe and NOPROBE_LINE in src and "NEXTFORK_START_PROBE" not in src:
            src = src.replace(NOPROBE_LINE, NOPROBE_BLOCK, 1)
            replaced_noprobe += 1
        if nocons and NOPROBE_LINE in src and "NEXTFORK_CONSENSUS" not in src:
            src = src.replace(NOPROBE_LINE, NOCONS_BLOCK, 1)
            replaced_noprobe += 1
        if seqs and SEQS_OLD in src:
            src = src.replace(SEQS_OLD, '"TAAF_VLLM_MAX_NUM_SEQS": "%d",' % seqs, 1)
        if kv_fp8 and KVDT_OLD in src:           # 28.09: кэш внимания в fp8 — вдвое больше токенов в том же объёме
            src = src.replace(KVDT_OLD, '"TAAF_VLLM_KV_CACHE_DTYPE": "fp8",', 1)
        if kv_gib and KV_OLD in src:
            src = src.replace(KV_OLD, '"TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "%d",' % (kv_gib * 1024 ** 3), 1)
        if cap and FULL_CAP_LINE in src:
            src = src.replace(FULL_CAP_LINE, "bm.solver.max_runtime_s_per_game = %.1f   # проба; сравнивать с обрезкой до того же потолка" % cap, 1)
        cell["source"] = src.splitlines(keepends=True)

    env_lines = ([] if agentfix else ['os.environ["NEXTFORK_AGENTFIX"] = "0"   # замер без AGENTFIX Скотта']) + \
                ([] if persist else ['os.environ["NEXTFORK_PERSIST"] = "0"   # замер без памяти функций']) + \
                (['os.environ["NEXTFORK_OBSERVE"] = "1"   # наблюдение в сообщении v2'] if observe else []) + \
                (['os.environ["NEXTFORK_ZOOM"] = "1"   # крупная картинка вокруг игрока'] if zoom else []) + \
                (['os.environ["NEXTFORK_BUILDWM"] = "1"   # достройка симулятора по ходу игры (a8c)'] if buildwm else []) + \
                ([f'os.environ["NEXTFORK_REASONING_EFFORT"] = "{effort}"   # уровень рассуждения шаблона'] if effort else []) + \
                (['os.environ["NEXTFORK_COMPACT"] = "1"   # сжатие истории вместо обрезки (OpenAI)',
                  f'os.environ["NEXTFORK_COMPACT_TOKENS"] = "{compact}"'] if compact else []) + \
                ([] if noopguard else ['os.environ["NEXTFORK_NOOPGUARD"] = "0"   # без запрета пустого хода (v3)']) + \
                ([] if v3prompts else ['os.environ["NEXTFORK_V3PROMPTS"] = "0"   # стоковый промпт Tufa, без трёх правок v3']) + \
                ([f'os.environ["NEXTFORK_MODEL"] = "{model}"   # модель: {model}'] if model != "keith" else []) + \
                ([f'os.environ["NEXTFORK_RUNTIME"] = "{runtime}"   # сборка vLLM: {RUNTIME_DATASETS[runtime]}'] if runtime != "keith" else []) + \
                ([f'os.environ["NEXTFORK_CONTEXT_WINDOW"] = "{ctx}"   # настоящее окно обвязки (tool_agent читает при импорте)'] if ctx else [])
    if env_lines:
        c3 = next(c for c in nb["cells"] if c["cell_type"] == "code" and NOPROBE_LINE in "".join(c["source"]))
        src3 = "".join(c3["source"]).replace(NOPROBE_LINE, NOPROBE_LINE + "\n" + "\n".join(env_lines), 1)
        c3["source"] = src3.splitlines(keepends=True)
    if seqs and SEQS_OLD not in "".join("".join(c["source"]) for c in nb["cells"]) and '"TAAF_VLLM_MAX_NUM_SEQS": "%d"' % seqs not in "".join("".join(c["source"]) for c in nb["cells"]):
        raise SystemExit("не нашёл MAX_NUM_SEQS в профиле")

    if replaced_sources != 1:
        raise SystemExit(f"строка DATASET_SOURCES заменена {replaced_sources} раз, ожидался ровно один")
    if replaced_check != 1:
        raise SystemExit(f"проверка бандла вставлена {replaced_check} раз, ожидался ровно один")
    if smoke and replaced_cap != 1:
        raise SystemExit(f"потолок пробы вставлен {replaced_cap} раз, ожидался ровно один")
    if (noprobe or nocons) and replaced_noprobe != 1:
        raise SystemExit(f"выключатель протокола вставлен {replaced_noprobe} раз, ожидался ровно один")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    import subprocess, sys
    subprocess.run([sys.executable, str(ROOT / "scripts/patch_wheels_wait.py"), str(out_dir)], check=True)   # оба пути данных соревнования

    # Ни одного упоминания чужого источника не должно остаться.
    text = out_nb.read_text(encoding="utf-8")
    stale = text.count(UPSTREAM_SRC)
    print("записан:      %s" % out_nb)
    print("метаданные:   %s" % (out_dir / "kernel-metadata.json"))
    print("режим:        %s%s" % (("фаза A: потолок игры %.0f с, фаза B 7920 с" % (phase_a_cap or SMOKE_CAP_S)) if smoke else "полный прогон (7920 с)",
                                    ", протокол разведки ВЫКЛ" if noprobe else ""))
    print("источник:     %s" % OUR_SRC)
    print("упоминаний чужого бандла осталось: %d %s" % (stale, "(ок)" if stale == 0 else "(ПРОВЕРИТЬ)"))
    if stale:
        raise SystemExit("в ноутбуке остался чужой источник исходников")
    print()
    print("НЕ запущено. Запуск — с разрешения владельца:")
    print("  .venv/bin/kaggle kernels push -p %s" % out_dir.relative_to(ROOT))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="получасовая проба цепочки вместо полного прогона")
    ap.add_argument("--noprobe", action="store_true", help="плечо без протокола разведки (NEXTFORK_START_PROBE=0)")
    ap.add_argument("--nocons", action="store_true", help="плечо без согласования двух ответов (NEXTFORK_CONSENSUS=0)")
    ap.add_argument("--no-agentfix", dest="agentfix", action="store_false", help="замер БЕЗ AGENTFIX Скотта (по умолчанию он включён в коде)")
    ap.add_argument("--no-persist", dest="persist", action="store_false", help="замер БЕЗ памяти функций (по умолчанию включена в коде)")
    ap.add_argument("--observe", action="store_true", help="включить наблюдение в сообщении v2 (NEXTFORK_OBSERVE=1)")
    ap.add_argument("--zoom", action="store_true", help="включить крупную картинку вокруг игрока (NEXTFORK_ZOOM=1)")
    ap.add_argument("--no-noopguard", dest="noopguard", action="store_false", help="без запрета пустого хода (NEXTFORK_NOOPGUARD=0)")
    ap.add_argument("--no-v3prompts", dest="v3prompts", action="store_false", help="стоковый промпт без трёх правок v3 (NEXTFORK_V3PROMPTS=0)")
    ap.add_argument("--no-v3", action="store_true", help="v4-lite: обе части v3 выключены")
    ap.add_argument("--compact", type=int, default=0, help="сжатие истории: порог оценки истории в токенах (например 12000); 0 — выкл")
    ap.add_argument("--effort", default="", choices=["", "xhigh", "medium", "low"], help="reasoning_effort шаблона Qwen3.8 (NEXTFORK_REASONING_EFFORT)")
    ap.add_argument("--kv-fp8", action="store_true", help="кэш внимания vLLM в fp8 (TAAF_VLLM_KV_CACHE_DTYPE=fp8)")
    ap.add_argument("--buildwm", action="store_true", help="достройка симулятора по ходу игры (NEXTFORK_BUILDWM=1, вариант a8c)")
    ap.add_argument("--model", default="keith", choices=["keith", "swift", "reap448"], help="swift — Swift 1.5; reap448 — REAP-обрезка экспертов 512->448 (boristown)")
    ap.add_argument("--runtime", default="keith", choices=["keith", "v030", "nightly"], help="сборка vLLM: v030 — 0.30.0 (foysal), nightly — ночной образ (fp8-кэш)")
    ap.add_argument("--seqs", type=int, default=0, help="TAAF_VLLM_MAX_NUM_SEQS (у Скотта 16)")
    ap.add_argument("--ctx", type=int, default=0, help="настоящее окно контекста обвязки, например 16384")
    ap.add_argument("--kv-gib", type=int, default=0, help="кэш внимания vLLM, ГиБ (боевой 5)")
    ap.add_argument("--cap", type=float, default=0.0, help="потолок на игру, с (проба)")
    ap.add_argument("--phase-a-cap", type=float, default=0.0, help="с --smoke: потолок игры в фазе A, с (по умолчанию 1800); фаза B — 7920")
    ap.add_argument("--name", default="", help="суффикс папки и id кернела")
    a = ap.parse_args()
    if a.no_v3:
        a.noopguard = a.v3prompts = False
    build(a.smoke, a.noprobe, a.nocons, a.agentfix, a.seqs, a.ctx, a.kv_gib, a.cap, a.name, a.persist, a.observe, a.zoom,
          a.noopguard, a.v3prompts, a.model, a.buildwm, a.kv_fp8, a.effort, a.compact, a.runtime, a.phase_a_cap)

"""Сборка `arc3-wmprobe`: зонд модели мира v3 (цикл CEGIS из статьи Twin) на Kaggle вместо пода (26.09).

Под недоступен, кредиты HF кончились — модель есть только в Kaggle. Кернел поднимает боевой vLLM теми же
ячейками 3/5/7/9, что и бой (датасет nextfork v3 + рантайм keithtyser), но игр НЕ играет: гоняет
scripts/wm_cegis_probe.py против своего сервера на 127.0.0.1:1234.

Два плеча, по очереди, на тех же 24 играх (переходы первого уровня прогона flash_v1_phaseA, 6 на обучение,
до 3 кругов ремонта, маска индикатора у края, нулевой базлайн «ничего не меняй»):
  A — рассуждение включено, потолок ответа 12k, при упоре — вызов «только код»;
  B — рассуждение выключено (enable_thinking=false), сразу код.
Кэш внимания 8 ГиБ вместо боевых 5 (больше длинных запросов одновременно); если сервер с ним не поднимется —
повтор с боевыми 5 ГиБ. Результат: /kaggle/working/wm_A.json, wm_B.json (+ коды и тексты ответов модели).
usage: .venv/bin/python scripts/build_wmprobe_notebook.py
"""
import ast, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_NB = os.path.join(ROOT, "kernels/notebooks_nextfork/submission.ipynb")
OUT = os.path.join(ROOT, "kernels/notebooks_wmprobe")

SETUP_TAIL_OLD = '''env = _command_env()
for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):
    print(f"taaf.kaggle: setup command: {command}", flush=True)
    subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)
    # Re-read in case the command persisted new env keys.
    env = _command_env()
    os.environ.update(env)
'''
SETUP_TAIL_NEW = '''def _wm_setup():
    env = _command_env()
    for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):
        print(f"taaf.kaggle: setup command: {command}", flush=True)
        subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)
        env = _command_env()
        os.environ.update(env)


# WMPROBE: кэш внимания 8 ГиБ; если сервер не поднялся — боевые 5 ГиБ.
os.environ["TAAF_VLLM_KV_CACHE_MEMORY_BYTES"] = str(8 * 1024 ** 3)
try:
    _wm_setup()
    print("WMPROBE: сервер поднят с кэшем 8 ГиБ", flush=True)
except Exception as _exc:
    print("WMPROBE: с кэшем 8 ГиБ не поднялся (%r) — повтор с 5 ГиБ" % (_exc,), flush=True)
    os.environ["TAAF_VLLM_KV_CACHE_MEMORY_BYTES"] = "5368709120"
    _wm_setup()
    print("WMPROBE: сервер поднят с кэшем 5 ГиБ", flush=True)
'''

RUN_CELL = r'''
# =====================================================================
# WMPROBE: зонд модели мира v3 против своего vLLM (два плеча по очереди)
# =====================================================================
import json, os, subprocess, sys, time
from pathlib import Path
_W = Path("/kaggle/working/wm"); (_W / "scripts").mkdir(parents=True, exist_ok=True)
(_W / "runs/flash_v1_phaseA").mkdir(parents=True, exist_ok=True)
(_W / "scripts/wm_cegis_probe.py").write_text(__PROBE__)
(_W / "runs/flash_v1_phaseA/benchmark.json").write_text(__BENCH__)
_env_dir = None
for _p in Path("/kaggle/input").rglob("environment_files"):
    if _p.is_dir():
        _env_dir = str(_p); break
print("WMPROBE: environment_files =", _env_dir, flush=True)
_base = dict(os.environ, WM_URL="http://127.0.0.1:1234/v1/chat/completions", WM_MODEL="Qwen/Qwen3.8-Flash-Next-NVFP4",
             WM_MAX_TOKENS="12000", WM_MASK_HUD="1", WM_MODE="OFFLINE", WM_ENV_DIR=_env_dir or "",
             WM_INFER_SRC=str(BUNDLE_DIR / "src" / "ARC3-Inference"))
for _arm, _think in (("A", "1"), ("B", "0")):
    _t0 = time.time()
    _env = dict(_base, WM_THINK=_think, WM_OUT="/kaggle/working/wm_%s.json" % _arm)
    print("WMPROBE: плечо %s (рассуждение %s) старт" % (_arm, "вкл" if _think == "1" else "выкл"), flush=True)
    _r = subprocess.run([sys.executable, str(_W / "scripts/wm_cegis_probe.py"), "--games", "all", "--par", "8", "--train", "6"],
                        env=_env, cwd=str(_W), capture_output=True, text=True)
    Path("/kaggle/working/wm_%s.log" % _arm).write_text(_r.stdout + "\n--- stderr ---\n" + _r.stderr[-20000:])
    print(_r.stdout[-6000:], flush=True)
    if _r.returncode:
        print("WMPROBE: плечо %s упало: %s" % (_arm, _r.stderr[-3000:]), flush=True)
    print("WMPROBE: плечо %s готово за %.0f мин" % (_arm, (time.time() - _t0) / 60), flush=True)
'''


def main() -> None:
    nb = json.load(open(SRC_NB, encoding="utf-8"))
    cells = nb["cells"]
    c9 = "".join(cells[9]["source"])
    if SETUP_TAIL_OLD not in c9:
        raise SystemExit("не нашёл цикл setup-команд в ячейке 9 — сборка остановлена")
    cells[9]["source"] = c9.replace(SETUP_TAIL_OLD, SETUP_TAIL_NEW).splitlines(keepends=True)
    probe = open(os.path.join(ROOT, "scripts/wm_cegis_probe.py"), encoding="utf-8").read()
    bench = json.load(open(os.path.join(ROOT, "runs/flash_v1_phaseA/benchmark.json")))["game_runs"]
    compact = {"game_runs": [{"game_id": g["game_id"], "history": [{"action": h.get("action")} for h in (g.get("history") or [])]} for g in bench]}
    run = RUN_CELL.replace("__PROBE__", repr(probe)).replace("__BENCH__", repr(json.dumps(compact)))
    new = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": run.splitlines(keepends=True)}
    keep = [cells[i] for i in (3, 5, 7, 9)]
    for c in keep:
        if c["cell_type"] == "code":
            c["outputs"] = []; c["execution_count"] = None
    nb["cells"] = keep + [new]
    for c in nb["cells"]:
        compile("".join(c["source"]), "cell", "exec")
    os.makedirs(OUT, exist_ok=True)
    json.dump(nb, open(os.path.join(OUT, "submission.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open(os.path.join(ROOT, "kernels/notebooks_nextfork/kernel-metadata.json")))
    meta["id"] = "sergueimakarov/arc3-wmprobe"; meta["title"] = "arc3 wmprobe"
    json.dump(meta, open(os.path.join(OUT, "kernel-metadata.json"), "w"), indent=2)
    print("собрано:", OUT, "| ячеек", len(nb["cells"]), "| размер %.1f МБ" % (os.path.getsize(os.path.join(OUT, "submission.ipynb")) / 1e6))


if __name__ == "__main__":
    main()

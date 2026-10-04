"""Сборка kernels/graft_tttprobe — проба «дообучение во время игры», вариант A. Основа — kernels/graft_kvprobe
(сервер Франзена, черновик выключен). Правки: путь датасета пробы; в ячейке сервера перед запуском — патч выгрузки
скрытых состояний (nextfork/ttt/apply_dump.py) и NF_DUMP_DIR; ячейка 14 — nextfork/ttt/probe_cell.py.
usage: .venv/bin/python scripts/build_ttt_probe.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "kernels/graft_kvprobe", ROOT / "kernels/graft_tttprobe"
APPLY = (ROOT / "nextfork/ttt/apply_dump.py").read_text()
CELL = (ROOT / "nextfork/ttt/probe_cell.py").read_text()
LAUNCH = "# ---- launch detached and wait for health ----\n"
PATCH = (
    "# NEXTFORK 04.10: выгрузка скрытых состояний перед lm_head (проба дообучения во время игры)\n"
    "import glob as _tg\n"
    "Path('/tmp/nf_ttt').mkdir(exist_ok=True); Path('/tmp/nf_dump').mkdir(exist_ok=True)\n"
    f"Path('/tmp/nf_ttt/apply_dump.py').write_text({APPLY!r})\n"
    "_r = subprocess.run([sys.executable, '/tmp/nf_ttt/apply_dump.py', sorted(_tg.glob(f'{VENV}/lib/python3*/site-packages/sglang'))[0]], capture_output=True, text=True)\n"
    "print(_r.stdout, _r.stderr[-2000:]); assert _r.returncode == 0, 'патч выгрузки не применился'\n"
    "os.environ['NF_DUMP_DIR'] = '/tmp/nf_dump'; env['NF_DUMP_DIR'] = '/tmp/nf_dump'\n"
)


def main():
    nb = json.loads((SRC / "submission.ipynb").read_text())
    c4 = "".join(nb["cells"][4]["source"])
    a = "PROBE_DIR         = '/kaggle/input/datasets/sergueimakarov/arc3-kvprobe-prompts'"
    assert c4.count(a) == 1
    nb["cells"][4]["source"] = c4.replace(a, "PROBE_DIR         = '/kaggle/input/datasets/sergueimakarov/arc3-ttt-probe'").splitlines(keepends=True)
    c12 = "".join(nb["cells"][12]["source"])
    assert c12.count(LAUNCH) == 1 and "SPEC=False" in c12 and "env = dict(os.environ)" in c12
    assert c12.count("MEMFRAC=0.96,") == 1
    c12 = c12.replace("MEMFRAC=0.96,", "MEMFRAC=0.88,   # проба: место под логиты по словарю 248k")
    nb["cells"][12]["source"] = c12.replace(LAUNCH, PATCH + LAUNCH).splitlines(keepends=True)
    nb["cells"][13]["source"] = ["## Проба: дообучение во время игры (вариант A)\n"]
    nb["cells"][14]["source"] = CELL.splitlines(keepends=True)
    compile(CELL, "cell14", "exec")
    DST.mkdir(exist_ok=True)
    (DST / "submission.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    meta.update(id="sergueimakarov/arc3-graft-tttprobe", title="arc3 graft tttprobe")
    meta["dataset_sources"] = [d if d != "sergueimakarov/arc3-kvprobe-prompts" else "sergueimakarov/arc3-ttt-probe"
                               for d in meta["dataset_sources"]]
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print("собрано:", DST, meta["dataset_sources"])


if __name__ == "__main__":
    main()

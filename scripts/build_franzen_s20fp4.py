"""Сборка kernels/graft_dfranzen_m2_s20fp4: 20 активных игр на кэше nvfp4 с патчем QSA (nextfork/fp4qsa), без HiCache.
Основа — kernels/graft_dfranzen_m2_s20hc (20 игр, слотов Mamba 100). Проба fp4 на Kaggle 03.10: пул x1.50 к fp8,
качество в пределах шума (NLL +0.0015, 95% ДИ [-0.005, +0.008]), скорость та же.

Правки ячейки сервера: KVDTYPE nvfp4; кэш черновика fp8 (nvfp4 для черновика не поддерживается); HiCache убран;
перед первым запуском сервера — патч fp4qsa в установленный sglang и NEXTFORK_FP4_QSA=1, NEXTFORK_FP4_NOWS=1.
usage: .venv/bin/python scripts/build_franzen_s20fp4.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "kernels/graft_dfranzen_m2_s20hc", ROOT / "kernels/graft_dfranzen_m2_s20fp4"
NF = (ROOT / "nextfork/fp4qsa/nf_fp4.py").read_text()
APPLY = (ROOT / "nextfork/fp4qsa/apply_fp4qsa.py").read_text()

POPEN = "# ---- launch detached and wait for health ----\n"   # вставка перед запуском (сам Popen внутри with)
PATCH = (
    "# NEXTFORK 03.10: кэш nvfp4 — патч разреженного внимания QSA (nextfork/fp4qsa) до первого запуска сервера\n"
    "import glob as _fglob\n"
    "_FP4_DIR = Path('/kaggle/working/nextfork_fp4qsa'); _FP4_DIR.mkdir(parents=True, exist_ok=True)\n"
    f"(_FP4_DIR / 'nf_fp4.py').write_text({NF!r})\n"
    f"(_FP4_DIR / 'apply_fp4qsa.py').write_text({APPLY!r})\n"
    "_SGL = sorted(_fglob.glob(f'{VENV}/lib/python3*/site-packages/sglang'))[0]\n"
    "_r = subprocess.run([sys.executable, str(_FP4_DIR / 'apply_fp4qsa.py'), _SGL], capture_output=True, text=True)\n"
    "print(_r.stdout, _r.stderr[-3000:]); assert _r.returncode == 0, 'патч fp4qsa не применился'\n"
    "env.update(NEXTFORK_FP4_QSA='1', NEXTFORK_FP4_NOWS='1')\n"
)


def main():
    nb = json.loads((SRC / "submission.ipynb").read_text())
    c = nb["cells"][12]
    s = "".join(c["source"])
    reps = [
        ('KVDTYPE="fp8_e4m3",', 'KVDTYPE="nvfp4",   # NEXTFORK 03.10: кэш KV в 4 битах (патч QSA ниже)'),
        ('"--speculative-draft-kv-cache-dtype", CFG["KVDTYPE"],', '"--speculative-draft-kv-cache-dtype", "fp8_e4m3",   # nvfp4 для черновика не поддерживается'),
    ]
    for a, b in reps:
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    # HiCache: убрать вставку из основы s20hc целиком
    i = s.find("HICACHE_GB = 40")
    assert i > 0
    head = s.rfind("\n# NEXTFORK 02.10: второй ярус", 0, i)
    tail = s.find('"--hicache-write-policy", "write_through", "--hicache-io-backend", "kernel"]', i)
    assert head > 0 and tail > i
    tail = s.find("\n", tail) + 1
    s = s[:head + 1] + s[tail:]
    assert "hicache" not in s.lower(), "HiCache остался"
    assert s.count(POPEN) == 1 and "VENV" in s and "sys." in s
    s = s.replace(POPEN, PATCH + POPEN)
    c["source"] = s.splitlines(keepends=True)
    for cell in nb["cells"]:
        t = "".join(cell["source"])
        if "'ARC3_MAX_ACTIVE_STREAMS': 20," in t:
            t = t.replace("'ARC3_MAX_ACTIVE_STREAMS': 20,   # вариант: 20 активных игр + HiCache (под 02.10: 650 против 608 т/с)",
                          "'ARC3_MAX_ACTIVE_STREAMS': 20,   # вариант: 20 активных игр на кэше nvfp4 (пул x1.5)")
            cell["source"] = t.splitlines(keepends=True)
    DST.mkdir(exist_ok=True)
    (DST / "submission.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    meta.update(id="sergueimakarov/arc3-graft-dfranzen-m2-s20fp4", title="arc3 graft dfranzen m2 s20fp4")
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print("собрано:", DST, meta["id"])


if __name__ == "__main__":
    main()

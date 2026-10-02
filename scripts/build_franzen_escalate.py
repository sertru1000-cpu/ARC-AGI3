"""Сборка kernels/graft_dfranzen_m2_escalate: стенд Франзена + эскалация застрявшего уровня (nextfork/escalate,
идея Kepler / Retrodict). Основа — kernels/graft_dfranzen_m2_stand; вставка по образцу сборки макро-ходов.
usage: .venv/bin/python scripts/build_franzen_escalate.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "kernels/graft_dfranzen_m2_stand", ROOT / "kernels/graft_dfranzen_m2_escalate"
APPLY = (ROOT / "nextfork/escalate/apply_escalate.py").read_text()

GIT = 'subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/harness-changes.patch"], cwd=f"{BUNDLE_DIR}/src", check=True)\n'
BLOCK = (
    "# NEXTFORK 03.10: эскалация застрявшего уровня (nextfork/escalate, идея Kepler): после 60 ходов на уровне — обязательная инструкция\n"
    "_ESC_DIR = Path('/kaggle/working/nextfork_escalate'); _ESC_DIR.mkdir(parents=True, exist_ok=True)\n"
    f"(_ESC_DIR / 'apply_escalate.py').write_text({APPLY!r})\n"
    "subprocess.run([sys.executable, str(_ESC_DIR / 'apply_escalate.py'), str(BUNDLE_DIR / 'src')], check=True)\n"
    "os.environ['NEXTFORK_ESCALATE'] = '1'\n"
)
ENV_ANCHOR = "    'EXPOSE_UNDO': 'on',\n"
ENV_ADD = "    'NEXTFORK_ESCALATE': '1',   # эскалация застрявшего уровня (пороги NEXTFORK_ESC_T1/T2: 60/120 ходов)\n"


def main():
    nb = json.loads((SRC / "submission.ipynb").read_text())
    done = {"git": 0, "env": 0}
    for c in nb["cells"]:
        s = "".join(c["source"])
        if GIT in s:
            assert s.count(GIT) == 1
            s = s.replace(GIT, GIT + BLOCK); done["git"] += 1
        if ENV_ANCHOR in s:
            assert s.count(ENV_ANCHOR) == 1
            s = s.replace(ENV_ANCHOR, ENV_ANCHOR + ENV_ADD); done["env"] += 1
        c["source"] = s.splitlines(keepends=True)
    assert done == {"git": 1, "env": 1}, done
    DST.mkdir(exist_ok=True)
    (DST / "submission.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    meta.update(id="sergueimakarov/arc3-graft-dfranzen-m2-escalate", title="arc3 graft dfranzen m2 escalate")
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print("собрано:", DST, meta["id"])


if __name__ == "__main__":
    main()

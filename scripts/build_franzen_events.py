"""Сборка kernels/graft_dfranzen_m2_events: база 4 + отчёт о событиях после каждого action() (nextfork/events).
Вставка — после блока макро-ходов в ячейке 4 (как у макро-ходов); флаг NEXTFORK_EVENTS=1 в окружении и в setup_env.
usage: .venv/bin/python scripts/build_franzen_events.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "kernels/graft_dfranzen_m2_base4", ROOT / "kernels/graft_dfranzen_m2_events"
BLOCK_SRC = (ROOT / "nextfork/events/events_block.py").read_text()
APPLY = (ROOT / "nextfork/events/apply_events.py").read_text()
AFTER = "subprocess.run([sys.executable, str(_MACRO_DIR / 'apply_macros.py'), str(BUNDLE_DIR / 'src')], check=True)\n"
INS = (
    "# NEXTFORK 04.10: отчёт о событиях после каждого action() (nextfork/events), ответ критика про вызовы без ходов\n"
    "_EV_DIR = Path('/kaggle/working/nextfork_events'); _EV_DIR.mkdir(parents=True, exist_ok=True)\n"
    f"(_EV_DIR / 'events_block.py').write_text({BLOCK_SRC!r})\n"
    f"(_EV_DIR / 'apply_events.py').write_text({APPLY!r})\n"
    "subprocess.run([sys.executable, str(_EV_DIR / 'apply_events.py'), str(BUNDLE_DIR / 'src')], check=True)\n"
    "os.environ['NEXTFORK_EVENTS'] = '1'\n"
)
ENV_ANCHOR = "    'EXPOSE_UNDO': 'on',\n"
ENV_ADD = "    'NEXTFORK_EVENTS': '1',   # отчёт о событиях после каждого хода\n"


def main():
    nb = json.loads((SRC / "submission.ipynb").read_text())
    done = {"after": 0, "env": 0}
    for c in nb["cells"]:
        s = "".join(c["source"])
        if AFTER in s:
            s = s.replace(AFTER, AFTER + INS); done["after"] += 1
        if ENV_ANCHOR in s and ENV_ADD not in s:
            s = s.replace(ENV_ANCHOR, ENV_ANCHOR + ENV_ADD); done["env"] += 1
        c["source"] = s.splitlines(keepends=True)
    assert done == {"after": 1, "env": 1}, done
    DST.mkdir(exist_ok=True)
    (DST / "submission.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    meta.update(id="sergueimakarov/arc3-graft-dfranzen-m2-events", title="arc3 graft dfranzen m2 events")
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print("собрано:", DST)


if __name__ == "__main__":
    main()

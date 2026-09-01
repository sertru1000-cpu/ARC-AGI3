"""Resumable driver for the calibration + coercion-arm series.

WHY THIS EXISTS. The owner powers the VM down between sessions, and a
background shell (or a Claude session) dies with it -- that is exactly how
the 01.09 morning sequencer was lost mid-series. This script keeps the whole
plan in a state FILE instead, so any invocation picks up wherever the series
actually is. Run it from a scheduled task at boot and every 15 minutes, and
a shutdown merely pauses the series.

SAFETY. Every push spends the owner's weekly GPU quota, and consent was given
for a FIXED set of runs (three calibration, three arm). So:
  * the step list is written ONCE and this script never appends to it -- it
    physically cannot push a run nobody approved;
  * exactly one run may be in flight; nothing is pushed while another is
    running, and nothing is pushed before the previous run's output has been
    downloaded (a push overwrites the previous version's output on Kaggle);
  * every rebuild passes the FULL knob env explicitly. Letting the builder
    fall back to its defaults once already overwrote the live notebook.

usage:
    python scripts/series.py          # advance the series by one step
    python scripts/series.py --status # print state, touch nothing
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "runs" / "series_state.json"
LOG = ROOT / "runs" / "series.log"
KAGGLE = ROOT / ".venv/Scripts/kaggle.exe"
PY = ROOT / ".venv/Scripts/python.exe"
KERNEL = "sergueimakarov/arc3-atlas"

# The knobs every run in this series shares. One wave: 25 games at
# concurrency 28. Two-hour cap (V33 showed 2h converts like 4h).
COMMON = {
    "ATLAS_CONCURRENCY_BUILD": "28",
    "ATLAS_CALIBRATION_CAP_S": "7200",
    "ATLAS_GAME_CAP_CEILING_BUILD": "14400",
    "ATLAS_DRAWS_BUILD": "0",
    "ATLAS_PASSES_BUILD": "1",
    "ATLAS_DYNAMIC_BUDGET_BUILD": "0",
}

# `rebuild: false` means "push the notebook exactly as it sits on disk".
# calib_3 uses it so the three baseline runs stay byte-identical -- calib_1
# and calib_2 were pushed before the coercion block existed.
PLAN = [
    {"name": "calib_2", "arm": "base", "rebuild": False, "status": "pushed"},
    {"name": "calib_3", "arm": "base", "rebuild": False, "status": "pending"},
    {"name": "arm_1", "arm": "no-coercion", "rebuild": True,
     "env": {"ATLAS_CHECKPOINTS_BUILD": "0"}, "status": "pending"},
    {"name": "arm_2", "arm": "no-coercion", "rebuild": True,
     "env": {"ATLAS_CHECKPOINTS_BUILD": "0"}, "status": "pending"},
    {"name": "arm_3", "arm": "no-coercion", "rebuild": True,
     "env": {"ATLAS_CHECKPOINTS_BUILD": "0"}, "status": "pending"},
]

GUARDS = [
    "test_atlas_notebook_placeholders.py",
    "test_atlas_coercion_arm.py",
    "test_atlas_fit_game_cap.py",
]


def now() -> str:
    return datetime.now(timezone.utc).strftime("%d.%m %H:%M UTC")


def log(msg: str) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def save(steps: list[dict]) -> None:
    STATE.write_text(
        json.dumps({"updated": now(), "steps": steps}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def load() -> list[dict]:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))["steps"]
    STATE.parent.mkdir(parents=True, exist_ok=True)
    save(PLAN)
    log(f"состояние создано: {len(PLAN)} шагов, больше их не станет")
    return PLAN


def kernel_status() -> str:
    out = subprocess.run([str(KAGGLE), "kernels", "status", KERNEL],
                         capture_output=True, text=True).stdout or ""
    for token in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if token in out:
            return token
    return "UNKNOWN"


def conversion(run: Path) -> str:
    r = subprocess.run([str(PY), str(ROOT / "scripts/conversion.py"), str(run)],
                       capture_output=True, text=True, encoding="utf-8")
    return (r.stdout or r.stderr or "").strip()


def advance() -> None:
    steps = load()

    # --- 1. Is a run in flight? Collect it if it has landed. --------------
    flying = next((s for s in steps if s["status"] == "pushed"), None)
    if flying is not None:
        st = kernel_status()
        if st in ("QUEUED", "RUNNING"):
            log(f"{flying['name']}: {st}, ждём")
            return
        if st != "COMPLETE":
            flying["status"] = "failed"
            save(steps)
            log(f"{flying['name']}: ОТКАЗ ({st}). Серия остановлена, нужен человек.")
            return
        # Download BEFORE anything else can push over it.
        dest = ROOT / "runs" / flying["name"]
        subprocess.run([str(KAGGLE), "kernels", "output", KERNEL, "-p", str(dest), "-o"],
                       capture_output=True, text=True)
        flying["status"] = "done"
        save(steps)
        log(f"{flying['name']}: скачан в runs/{flying['name']}")
        log(conversion(dest))

    # --- 2. Anything failed? Stop and wait for a human. -------------------
    if any(s["status"] == "failed" for s in steps):
        log("в серии есть отказавший шаг — дальше не иду")
        return

    # --- 3. Next pending step. -------------------------------------------
    nxt = next((s for s in steps if s["status"] == "pending"), None)
    if nxt is None:
        log("СЕРИЯ ЗАВЕРШЕНА")
        finish(steps)
        return

    if nxt["rebuild"]:
        env = dict(os.environ)
        env.update(COMMON)
        env.update(nxt.get("env", {}))
        r = subprocess.run([str(PY), str(ROOT / "scripts/build_atlas_notebook.py")],
                           capture_output=True, text=True, cwd=str(ROOT), env=env)
        if r.returncode != 0:
            log(f"{nxt['name']}: СБОРКА УПАЛА\n{(r.stdout or '')[-1500:]}\n{(r.stderr or '')[-1500:]}")
            return
        knobs = " ".join(f"{k}={v}" for k, v in nxt.get("env", {}).items())
        log(f"{nxt['name']}: собран ({knobs})")

    for guard in GUARDS:
        g = ROOT / "scripts" / guard
        if not g.exists():
            continue
        r = subprocess.run([str(PY), str(g)], capture_output=True, text=True, cwd=str(ROOT))
        if r.returncode != 0:
            log(f"{nxt['name']}: СТРАЖ {guard} НЕ ПРОШЁЛ — не пушу\n{(r.stdout or '')[-1200:]}")
            return
    log(f"{nxt['name']}: стражи пройдены")

    r = subprocess.run([str(KAGGLE), "kernels", "push", "-p", str(ROOT / "notebooks_atlas")],
                       capture_output=True, text=True)
    if "successfully pushed" not in (r.stdout or ""):
        log(f"{nxt['name']}: ПУШ НЕ ПРОШЁЛ: {(r.stdout or '').strip()} {(r.stderr or '').strip()}")
        return
    nxt["status"] = "pushed"
    save(steps)
    log(f"{nxt['name']}: запушен ({r.stdout.strip().splitlines()[-1]})")


def finish(steps: list[dict]) -> None:
    """Both arms are in: write the paired comparison next to the runs."""
    base = [ROOT / "runs" / n for n in ("calib_1", "calib_2", "calib_3")]
    arm = [ROOT / "runs" / s["name"] for s in steps if s["arm"] == "no-coercion"]
    if not all(p.is_dir() for p in base + arm):
        log("не все каталоги прогонов на месте — сравнение не считаю")
        return
    r = subprocess.run(
        [str(PY), str(ROOT / "scripts/conversion_ab.py"),
         "--a", *[str(p) for p in base], "--b", *[str(p) for p in arm],
         "--label-a", "принуждения", "--label-b", "без них"],
        capture_output=True, text=True, encoding="utf-8")
    out = r.stdout or r.stderr or ""
    (ROOT / "runs" / "ab_result.txt").write_text(out, encoding="utf-8")
    log("СРАВНЕНИЕ ГОТОВО (runs/ab_result.txt):\n" + out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--status", action="store_true",
                    help="показать состояние, ничего не делать")
    args = ap.parse_args()
    steps = load()
    if args.status:
        print(f"{'шаг':10} {'рукав':13} {'состояние':10}")
        for s in steps:
            print(f"{s['name']:10} {s['arm']:13} {s['status']:10}")
        print(f"\nкернел сейчас: {kernel_status()}")
        return
    advance()


if __name__ == "__main__":
    main()

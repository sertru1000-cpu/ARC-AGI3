"""Очередь Kaggle недели 03.10 (слово владельца 02.10: «все прогоны Kaggle — начало в 3.05 ночи по Москве,
два параллельных»). С 03:05 МСК держит два прогона: как только один завершился (COMPLETE / ERROR / CANCEL),
пушит следующий по списку. Сабмиты в бой не делает.

Стоп: touch runs/.kaggle_queue_stop (новые пуши прекращаются, идущие прогоны не трогаются).
Журнал: runs/kaggle_queue_0310.log, состояние: runs/kaggle_queue_0310.json.
Запуск (переживает сессию): nohup .venv/bin/python scripts/kaggle_queue_0310.py > /dev/null 2>&1 &
"""
import json, os, subprocess, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "runs/kaggle_queue_0310.log"
STATE = ROOT / "runs/kaggle_queue_0310.json"
STOP = ROOT / "runs/.kaggle_queue_stop"
KAGGLE = str(ROOT / ".venv/bin/kaggle")
MSK = timezone(timedelta(hours=3))
START = datetime(2026, 10, 3, 3, 5, tzinfo=MSK)
SLOTS = 2

# порядок — как в очереди недели (docs/queue_week_0310.md); №14 и №18–20 условные — не здесь
QUEUE = [
    ("проба fp4", "graft_fp4probe"),
    ("№1 контроль базы", "graft_dfranzen_m2_stand"),
    ("№3 макро-ходы", "graft_dfranzen_m2_macros"),
    ("№2 контроль базы (повтор)", "graft_dfranzen_m2_stand2"),
    ("s20hc 20 игр + HiCache", "graft_dfranzen_m2_s20hc"),
    ("№8в окно 250k + HiCache", "graft_dfranzen_m2_w250"),
    ("s20fp4 20 игр на кэше fp4", "graft_dfranzen_m2_s20fp4"),
    ("№6 14 игр, окно 72k", "graft_dfranzen_m2_s14w72"),
    ("№11 планировщик nodecay", "graft_dfranzen_m2_sched_nodecay"),
    ("№12 планировщик patient", "graft_dfranzen_m2_sched_patient"),
    ("№13 планировщик impatient", "graft_dfranzen_m2_sched_impatient"),
    ("№15 картинка x4", "graft_dfranzen_m2_img4"),
    ("№16 NVFP4-модель", "graft_dfranzen_m2_nvfp4"),
    ("№17 модель huikang", "graft_dfranzen_m2_huikang"),
    ("№21 мини-библиотека", "graft_dfranzen_m2_lib"),
    ("№22 нотация", "graft_dfranzen_m2_notes"),
    ("№8 vLLM LHS", "graft_dfranzen_m2_vllm_lhs"),
    ("эскалация застрявшего уровня (Kepler)", "graft_dfranzen_m2_escalate"),
]


def now():
    return datetime.now(MSK)


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{now():%d.%m %H:%M:%S} МСК  {msg}\n")


def kenv():
    return dict(os.environ, KAGGLE_API_TOKEN=(Path.home() / ".kaggle/access_token").read_text().strip())


def slug(d):
    return json.loads((ROOT / "kernels" / d / "kernel-metadata.json").read_text())["id"]


def run(args, timeout=300):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=kenv())
        return (p.stdout + p.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"исключение: {e}"


def push(name, d):
    for attempt in (1, 2):
        out = run([KAGGLE, "kernels", "push", "-p", str(ROOT / "kernels" / d)], timeout=600)
        tail = out.splitlines()[-1] if out else ""
        log(f"ПУШ {name} ({d}), попытка {attempt}: {tail[:300]}")
        if "successfully pushed" in out:
            return True
        time.sleep(90)
    return False


def status(s):
    out = run([KAGGLE, "kernels", "status", s], timeout=120)
    for key in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if key in out:
            return key
    return "НЕИЗВЕСТНО: " + out[-160:]


def save(state):
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1))


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {"pending": [d for _, d in QUEUE], "running": {}, "done": {}}
    names = dict((d, n) for n, d in QUEUE)
    log(f"очередь запущена, старт пушей {START:%d.%m %H:%M} МСК, в списке {len(state['pending'])}")
    while now() < START:
        time.sleep(30)
    while state["pending"] or state["running"]:
        for d, info in list(state["running"].items()):
            st = status(info["slug"])
            if st != info.get("last"):
                log(f"статус {names[d]}: {st}")
                info["last"] = st
            if st in ("RUNNING", "QUEUED"):
                info["seen_active"] = True
            # сразу после пуша Kaggle может отдавать статус ПРОШЛОЙ версии (COMPLETE) — не верим ему,
            # пока не видели прогон в работе или не прошло 20 минут
            fresh = now() - datetime.fromisoformat(info["pushed"]) < timedelta(minutes=20)
            if st in ("COMPLETE", "ERROR", "CANCEL") and (info.get("seen_active") or not fresh):
                state["done"][d] = dict(info, end=f"{now():%d.%m %H:%M}", result=st)
                del state["running"][d]
            elif st == "QUEUED" and now() - datetime.fromisoformat(info["pushed"]) > timedelta(minutes=75):
                if not info.get("warned"):
                    log(f"ВНИМАНИЕ: {names[d]} в очереди Kaggle дольше 75 мин (застрявшая очередь?)")
                    info["warned"] = True
        while state["pending"] and len(state["running"]) < SLOTS and not STOP.exists():
            d = state["pending"].pop(0)
            if push(names[d], d):
                state["running"][d] = {"slug": slug(d), "pushed": now().isoformat(), "last": "PUSHED"}
            else:
                state["done"][d] = {"slug": slug(d), "result": "ПУШ НЕ УДАЛСЯ", "end": f"{now():%d.%m %H:%M}"}
            save(state)
            time.sleep(20)
        if STOP.exists() and not state["running"]:
            log("стоп-файл: новых пушей нет, идущих прогонов нет — выхожу")
            break
        save(state)
        time.sleep(120)
    log("очередь пуста — выхожу")


if __name__ == "__main__":
    main()

"""Сабмит base2 в бой 04.10 в 03:05 МСК (слово владельца 03.10 19:25: «Сабмит base2 в бой … — да»).

Защиты: ровно одна попытка (метка runs/.submit_base2_0410.done ставится до отправки); окно 04.10 03:05–23:59 МСК;
сабмитится только версия 1 кернела arc3-graft-dfranzen-m2-base2 и только если она COMPLETE; файл ноутбука в репозитории
обязан совпасть по sha256 с отпечатком, снятым при сборке. Журнал: runs/submit_base2_0410.log.
Запуск: nohup caffeinate -i .venv/bin/python scripts/submit_base2_0410.py > /dev/null 2>&1 &
"""
import hashlib, os, subprocess, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG, DONE = ROOT / "runs/submit_base2_0410.log", ROOT / "runs/.submit_base2_0410.done"
KAGGLE = str(ROOT / ".venv/bin/kaggle")
MSK = timezone(timedelta(hours=3))
START, END = datetime(2026, 10, 4, 3, 5, tzinfo=MSK), datetime(2026, 10, 4, 23, 59, tzinfo=MSK)
SLUG = "sergueimakarov/arc3-graft-dfranzen-m2-base2"
NB = ROOT / "kernels/graft_dfranzen_m2_base2/submission.ipynb"
SHA = os.environ.get("BASE2_SHA", "dcca2afd0e3b1cfdaa3b6c3f4a8f8287a9c0128188cfa8810bb9ac1ba0ccd338")


def log(m):
    with open(LOG, "a") as f:
        f.write(f"{datetime.now(MSK):%d.%m %H:%M:%S} МСК  {m}\n")


def run(args):
    env = dict(os.environ, KAGGLE_API_TOKEN=(Path.home() / ".kaggle/access_token").read_text().strip())
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=300, env=env)
        return (p.stdout + p.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"исключение: {e}"


def main():
    log("запущен, жду 04.10 03:05 МСК")
    while datetime.now(MSK) < START:
        time.sleep(30)
    if DONE.exists():
        log("метка уже стоит — не сабмичу"); return
    if datetime.now(MSK) > END:
        log("окно прошло — не сабмичу"); return
    got = hashlib.sha256(NB.read_bytes()).hexdigest()
    if got != SHA:
        log(f"ОТПЕЧАТОК НЕ СОВПАЛ ({got[:12]} против {SHA[:12]}) — не сабмичу"); return
    for attempt in range(1, 31):   # до 2.5 ч ждать, пока контроль base2 завершится
        st = run([KAGGLE, "kernels", "status", SLUG])
        log(f"статус base2: {st[-80:]}")
        if "COMPLETE" in st:
            break
        if "ERROR" in st or "CANCEL" in st:
            log("версия base2 упала — не сабмичу"); return
        time.sleep(300)
    else:
        log("base2 так и не завершилась — не сабмичу"); return
    DONE.touch()
    out = run([KAGGLE, "competitions", "submit", "arc-prize-2026-arc-agi-3", "-k", SLUG, "-v", "1",
               "-f", "submission.parquet", "-m", "04.10: base2 = Franzen copy + macro moves + impatient scheduler"])
    log("сабмит: " + out[-300:])
    time.sleep(30)
    log("список: " + run([KAGGLE, "competitions", "submissions", "arc-prize-2026-arc-agi-3"]).splitlines()[2][:170])


if __name__ == "__main__":
    main()

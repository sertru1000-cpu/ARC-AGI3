"""Форк боевой обвязки Duck: исходники в нашем репозитории, упаковка в приватный датасет, проверка тождества (19.09).

Зачем. До сих пор мы меняли обвязку заплатками поверх чужого бандла (`keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1`).
Слово владельца 19.09: «готовь» форк, чтобы менять её изнутри. Рецепт проверен ещё 22.08 на прежней версии Duck (atlas).

Как устроен бандл (проверено по коду, а не по памяти):
  * ноутбук кладёт в путь импорта `src/*/src` и `src/*` -- это обычные файлы Python, никакой установки пакетов;
  * упакованные объекты `benchmark_initial.pkl` / `deploy_target.pkl` ссылаются на классы по ИМЕНАМ модулей,
    при правке кода имена не меняются -- объекты грузятся;
  * `serving_setup.py` сверяет СВОЙ хеш с `SOURCE_IDENTITY.json` (serving_setup_sha256) и падает при расхождении.
    Код агента (ARC3-Inference, taaf) этой проверкой НЕ закрыт. Отсюда правило: `serving_setup.py` не править,
    а если придётся -- обновить хеш в SOURCE_IDENTITY.json (команда `verify` предупреждает об этом).
  * `src/sglang-rtxpro6000` (66 МБ, 4327 файлов) боевым путём vLLM не используется, но лежит в бандле. В git его
    не держим: при упаковке он берётся из нетронутой копии эталона.

Раскладка:
  harness/duck/                      -- наш рабочий код (в git), всё кроме sglang
  harness/duck/UPSTREAM.json         -- откуда взят эталон и хеш КАЖДОГО его файла (включая sglang)
  runs/peer_kernels/duck_upstream/   -- нетронутая копия эталона с Kaggle (не в git; команда `init` её скачивает)

Команды:
  init     скачать эталон с Kaggle, разложить в harness/duck, записать UPSTREAM.json (только если harness/duck пуст)
  pack     собрать папку датасета: harness/duck + sglang из эталона + dataset-metadata.json (приватный)
  verify   сравнить собранную папку с эталоном по хешам: тождество / список правок; ловит правку serving_setup.py
  publish  отправить папку в Kaggle как приватный датасет -- ТОЛЬКО с ключом --owner-approved (слово владельца)

usage:
  .venv/bin/python scripts/duck_fork.py init
  .venv/bin/python scripts/duck_fork.py pack --out /tmp/duckfork
  .venv/bin/python scripts/duck_fork.py verify /tmp/duckfork
  .venv/bin/python scripts/duck_fork.py publish /tmp/duckfork --owner-approved
"""
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REF = "keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1"
FORK = ROOT / "harness" / "duck"
PRISTINE = ROOT / "runs" / "peer_kernels" / "duck_upstream"
VENDORED = "src/sglang-rtxpro6000"          # в git не держим, берём из эталона
DATASET_ID = "sergueimakarov/arc3-duck-fork"
SKIP_PARTS = {"__pycache__", ".DS_Store"}


def kaggle(*args):
    env = dict(os.environ)
    tok = ROOT / ".kaggle/access_token"
    if "KAGGLE_API_TOKEN" not in env and tok.exists():
        env["KAGGLE_API_TOKEN"] = tok.read_text().strip()
    return subprocess.run([str(ROOT / ".venv/bin/kaggle"), *args], capture_output=True, text=True, env=env)


def files(root: Path) -> list:
    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and not (set(rel.parts) & SKIP_PARTS):
            out.append(str(rel))
    return out


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def manifest(root: Path) -> dict:
    return {rel: sha(root / rel) for rel in files(root)}


def cmd_init(a) -> int:
    if FORK.exists() and any(FORK.iterdir()):
        print("harness/duck уже не пуст -- init не перезаписывает рабочий код. Удалите папку сами, если нужно заново.")
        return 1
    if not PRISTINE.exists() or not any(PRISTINE.iterdir()):
        PRISTINE.mkdir(parents=True, exist_ok=True)
        r = kaggle("datasets", "download", UPSTREAM_REF, "-p", str(PRISTINE), "--unzip")
        if r.returncode != 0:
            print("скачать эталон не вышло:", (r.stderr or r.stdout)[-400:]); return 1
    meta = kaggle("datasets", "list", "-s", UPSTREAM_REF.split("/")[1], "--user", UPSTREAM_REF.split("/")[0], "-v")
    updated = ""
    import csv, io
    for row in csv.DictReader(io.StringIO(meta.stdout or "")):
        if row.get("ref") == UPSTREAM_REF:
            updated = row.get("lastUpdated", "")
    man = manifest(PRISTINE)
    FORK.mkdir(parents=True, exist_ok=True)
    copied = 0
    for rel in man:
        if rel.startswith(VENDORED + "/"):
            continue
        dst = FORK / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PRISTINE / rel, dst); copied += 1
    (FORK / "UPSTREAM.json").write_text(json.dumps({
        "dataset": UPSTREAM_REF, "upstream_last_updated": updated,
        "vendored_not_in_git": VENDORED, "file_count": len(man),
        "total_bytes": sum((PRISTINE / r).stat().st_size for r in man),
        "files": man}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("эталон: %d файлов; в harness/duck скопировано %d (без %s); UPSTREAM.json записан" % (len(man), copied, VENDORED))
    return 0


def cmd_pack(a) -> int:
    up = json.loads((FORK / "UPSTREAM.json").read_text(encoding="utf-8"))
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    n = 0
    for rel in files(FORK):
        if rel == "UPSTREAM.json":
            continue
        dst = out / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(FORK / rel, dst); n += 1
    vend = [r for r in up["files"] if r.startswith(up["vendored_not_in_git"] + "/")]
    missing = [r for r in vend if not (PRISTINE / r).exists()]
    if missing:
        print("в копии эталона нет %d файлов %s -- запустите init (он скачает эталон)" % (len(missing), up["vendored_not_in_git"])); return 1
    for rel in vend:
        dst = out / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(PRISTINE / rel, dst)
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "arc3 duck fork", "id": DATASET_ID, "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
    print("собрано в %s: наших файлов %d + из эталона %d; метаданные датасета %s (приватный по умолчанию)"
          % (out, n, len(vend), DATASET_ID))
    return 0


def cmd_verify(a) -> int:
    up = json.loads((FORK / "UPSTREAM.json").read_text(encoding="utf-8"))["files"]
    d = Path(a.dir)
    got = {rel: sha(d / rel) for rel in files(d) if rel != "dataset-metadata.json"}
    same = [r for r in up if got.get(r) == up[r]]
    changed = sorted(r for r in up if r in got and got[r] != up[r])
    removed = sorted(r for r in up if r not in got)
    added = sorted(r for r in got if r not in up)
    print("эталон %d файлов: совпало %d, изменено %d, удалено %d, добавлено %d"
          % (len(up), len(same), len(changed), len(removed), len(added)))
    for title, lst in (("изменено", changed), ("удалено", removed), ("добавлено", added)):
        for r in lst[:30]:
            print("  %s: %s" % (title, r))
    if "serving_setup.py" in changed:
        print("ВНИМАНИЕ: serving_setup.py изменён -- в ядре он сверит свой хеш с SOURCE_IDENTITY.json и упадёт, "
              "если там не обновлён serving_setup_sha256")
    ident = not (changed or removed or added)
    print("ИТОГ: %s" % ("ПОБАЙТОВО СОВПАДАЕТ С БОЕВЫМ БАНДЛОМ" if ident else "форк отличается от эталона (см. список выше)"))
    return 0 if (ident or a.allow_changes) else 2


def cmd_publish(a) -> int:
    if not a.owner_approved:
        print("ОТКАЗ: публикация датасета -- только по отдельному слову владельца (ключ --owner-approved).")
        return 3
    d = Path(a.dir)
    exists = kaggle("datasets", "files", DATASET_ID).returncode == 0
    r = (kaggle("datasets", "version", "-p", str(d), "-m", a.message, "-r", "tar") if exists
         else kaggle("datasets", "create", "-p", str(d), "-r", "tar"))
    print((r.stdout or r.stderr)[-600:])
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    p = sub.add_parser("pack"); p.add_argument("--out", required=True)
    v = sub.add_parser("verify"); v.add_argument("dir"); v.add_argument("--allow-changes", action="store_true")
    q = sub.add_parser("publish"); q.add_argument("dir"); q.add_argument("--owner-approved", action="store_true")
    q.add_argument("-m", "--message", default="fork update")
    a = ap.parse_args()
    return {"init": cmd_init, "pack": cmd_pack, "verify": cmd_verify, "publish": cmd_publish}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

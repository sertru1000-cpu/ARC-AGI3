"""ОБЯЗАТЕЛЬНАЯ проверка перед КАЖДЫМ пушем в Kaggle (19.09, после двух ошибок за сутки).

Ошибки, ради которых это написано:
  * 18.09: слой тестировался на бандле duck_base, а стоковый ноутбук ставит ДРУГОЙ (duck_smoke_live) -- проба
    1.6 ч квоты сгорела на отсутствующей функции;
  * 19.09: проба ушла без `competition_sources`, Kaggle выдал T4 x2 вместо RTX Pro 6000, и модель на 86 ГБ
    не могла влезть в принципе.
Обе ошибки -- не в коде ячейки, а в УСЛОВИЯХ запуска. Проверка ловит именно их.

Что проверяется (всё до пуша, бесплатно):
  1. machine_shape и competition_sources совпадают с рабочим кернелом (иначе мощной карты не будет);
  2. enable_gpu=true, enable_internet=false (интернет в ядрах соревнования всё равно выключен);
  3. каждый dataset_source и model_source СУЩЕСТВУЕТ и доступен (ловит опечатки в слагах);
  4. title и id согласованы: Kaggle делает адрес кернела ИЗ ЗАГОЛОВКА и молча переименовывает при расхождении;
  5. все ячейки ноутбука компилируются;
  6. если рядом лежит тест слоя (scripts/test_<имя>_patch.py) -- он запускается на боевом бандле duck_smoke_live;
  7. потолок пробы: если в ячейках есть max_runtime_s_per_game, он стоит ТОЛЬКО под `if not TRUE_SUBMISSION`.

usage:  .venv/bin/python scripts/preflight_push.py kernels/notebooks_dsv4_smoke [--test scripts/test_x.py]
        (код возврата 0 -- можно пушить; иначе пуш ЗАПРЕЩЁН)
"""
import argparse, json, os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "kernels/notebooks_stockflash/kernel-metadata.json"
fails, warns = [], []


def check(ok, msg):
    print(("ok   " if ok else "СБОЙ ") + msg)
    if not ok:
        fails.append(msg)


def warn(ok, msg):
    print(("ok   " if ok else "!    ") + msg)
    if not ok:
        warns.append(msg)


def kaggle(*args):
    env = dict(os.environ)
    if "KAGGLE_API_TOKEN" not in env:
        tok = ROOT / ".kaggle/access_token"
        if tok.exists():
            env["KAGGLE_API_TOKEN"] = tok.read_text().strip()
    r = subprocess.run([str(ROOT / ".venv/bin/kaggle"), *args], capture_output=True, text=True, env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("kernel_dir"); ap.add_argument("--test", default=None)
    ap.add_argument("--expect-probe", action="store_true", help="требовать потолок пробы под if not TRUE_SUBMISSION")
    a = ap.parse_args()
    d = Path(a.kernel_dir)
    meta = json.load(open(d / "kernel-metadata.json", encoding="utf-8"))
    ref = json.load(open(REF, encoding="utf-8"))

    # 1-2. условия запуска
    check(meta.get("machine_shape") == ref.get("machine_shape"),
          "machine_shape = %r (у рабочего кернела %r)" % (meta.get("machine_shape"), ref.get("machine_shape")))
    check(list(meta.get("competition_sources") or []) == list(ref.get("competition_sources") or []),
          "competition_sources = %r -- без него Kaggle даёт T4 x2 вместо RTX Pro 6000" % (meta.get("competition_sources"),))
    check(bool(meta.get("enable_gpu")), "enable_gpu = %r" % meta.get("enable_gpu"))
    warn(meta.get("enable_internet") is False, "enable_internet = %r (в ядрах соревнования интернета нет)" % meta.get("enable_internet"))

    # 3. существование входов
    for ds in meta.get("dataset_sources") or []:
        code, out = kaggle("datasets", "files", ds)
        check(code == 0 and "404" not in out, "датасет доступен: %s" % ds)
    for ms in meta.get("model_sources") or []:
        base = "/".join(ms.split("/")[:2])
        code, out = kaggle("models", "get", base)
        check(code == 0 and "404" not in out, "модель доступна: %s" % ms)

    # 4. слаг из заголовка
    slug = re.sub(r"[^a-z0-9]+", "-", str(meta.get("title", "")).lower()).strip("-")
    check(meta.get("id", "").split("/")[-1] == slug,
          "id и title согласованы: id=%r, слаг из заголовка=%r" % (meta.get("id"), slug))

    # 5. компиляция ячеек
    nb = json.load(open(d / "submission.ipynb", encoding="utf-8"))
    bad = []
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c["source"])
        try:
            compile(src, "cell%d" % i, "exec", 0x2000)
        except SyntaxError as exc:
            bad.append((i, str(exc)[:80]))
    check(not bad, "все ячейки компилируются%s" % ("" if not bad else ": " + str(bad)))

    # 6. тест слоя на боевом бандле
    if a.test:
        bundle = ROOT / "runs/peer_kernels/duck_smoke_live"
        check(bundle.exists(), "боевой бандл на месте: %s" % bundle)
        r = subprocess.run([str(ROOT / ".venv/bin/python"), a.test, "--bundle", str(bundle)],
                           capture_output=True, text=True, cwd=str(ROOT))
        tail = (r.stdout or "").strip().splitlines()[-1:] or ["(нет вывода)"]
        check(r.returncode == 0, "тест слоя на боевом бандле: %s" % tail[0])

    # 7. потолок пробы
    all_src = "\n".join("".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code")
    caps = re.findall(r"max_runtime_s_per_game\s*=\s*([0-9.]+)", all_src)
    if caps or a.expect_probe:
        guarded = "if not TRUE_SUBMISSION:" in all_src and all_src.index("if not TRUE_SUBMISSION:") < all_src.rindex("max_runtime_s_per_game") if caps else False
        check(bool(caps), "потолок пробы задан: %s" % caps)
        check(guarded, "потолок пробы стоит под if not TRUE_SUBMISSION (в бою не действует)")

    print("\nИТОГ: %s" % ("ПУШ РАЗРЕШЁН" if not fails else "ПУШ ЗАПРЕЩЁН, сбоев %d" % len(fails)))
    if warns:
        print("предупреждения: %d" % len(warns))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

"""Копия опубликованного решения под наш аккаунт — готовая к пушу, но НЕ пушится (26.09, к этапу 30.09).

Что делает:
  1. скачивает ноутбук и метаданные чужого кернела (kaggle kernels pull -m) в kernels/pub_<slug>/;
  2. ставит наш id (sergueimakarov/arc3-pub-<slug>), источники (модели, датасеты, соревнование) оставляет их;
  3. по флагам вставляет наши ячейки-слои перед первым `await bm.run(` (если ноутбук — Duck/TAAF) и
     меняет потолок на игру (--cap, для пробы на 1 ч).
Пуш — отдельно и только по слову владельца: .venv/bin/kaggle kernels push -p kernels/pub_<slug>
Слои: cell15.py из kernels/notebooks_nextfork_<слой>1h (open, fresh, short, lean, vision, cons) и
      scripts/patch_wheels_wait.py-логика НЕ применяется (их ячейка установки другая).
usage: .venv/bin/python scripts/fork_published.py huikang/some-kernel [--layers short,open] [--cap 3600] [--name x]
"""
from __future__ import annotations
import argparse, ast, json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K = str(ROOT / ".venv/bin/kaggle")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("kernel"); ap.add_argument("--layers", default="")
    ap.add_argument("--cap", type=float, default=0); ap.add_argument("--name", default="")
    ap.add_argument("--swap", default="", help="их_датасет=наш_датасет: подменить бандл исходников (после graft_bundle.py)")
    a = ap.parse_args()
    slug = a.name or a.kernel.split("/", 1)[1][:30]
    out = ROOT / "kernels" / ("pub_" + slug + ("_" + a.layers.replace(",", "_") if a.layers else "") + ("_cap%d" % a.cap if a.cap else ""))
    out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([K, "kernels", "pull", a.kernel, "-p", str(out), "-m"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("не скачался: " + r.stderr[-400:])
    meta = json.loads((out / "kernel-metadata.json").read_text())
    nb_path = out / meta["code_file"]
    print("их источники: модели %s | датасеты %s | соревнования %s | машина %s" % (
        meta.get("model_sources"), meta.get("dataset_sources"), meta.get("competition_sources"), meta.get("machine_shape")))
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    if a.cap:
        hit = 0
        for c in cells:
            s = "".join(c["source"])
            s2, n = re.subn(r"(max_runtime_s_per_game\s*=\s*)[0-9.]+", r"\g<1>%.1f" % a.cap, s)
            if n:
                c["source"] = s2.splitlines(keepends=True); hit += n
        print("потолок на игру -> %.0f с: замен %d%s" % (a.cap, hit, "" if hit else "  (НЕ НАЙДЕН — проверить руками)"))
    for layer in [x for x in a.layers.split(",") if x]:
        cell = (ROOT / f"kernels/notebooks_nextfork_{layer}1h/cell15.py").read_text(encoding="utf-8")
        for c in cells:
            s = "".join(c["source"])
            at = s.find("await bm.run(")
            if c["cell_type"] == "code" and at >= 0:
                # вставка перед ВЕРХНЕУРОВНЕВЫМ оператором, внутри которого стоит запуск (часто это `try:`)
                line0 = s.rfind("\n", 0, at) + 1
                while line0 > 0 and (s[line0:line0 + 1] in (" ", "\t", "\n", "#") or s[line0:].startswith(("except", "finally", "else", "elif"))):
                    line0 = s.rfind("\n", 0, line0 - 1) + 1
                c["source"] = (s[:line0] + cell + "\n" + s[line0:]).splitlines(keepends=True)
                print("слой %s вставлен перед await bm.run(" % layer); break
        else:
            print("слой %s: НЕ НАЙДЕН await bm.run( — ноутбук не Duck/TAAF, вставлять руками" % layer)
    for c in cells:
        if c["cell_type"] == "code":
            try:
                compile("".join(c["source"]), "c", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            except SyntaxError as exc:
                print("ВНИМАНИЕ: ячейка не компилируется (магия jupyter?): %s" % exc)
    if a.swap:
        theirs, ours = a.swap.split("=", 1)
        meta["dataset_sources"] = [ours if d == theirs else d for d in meta.get("dataset_sources") or []]
        n = 0
        for c in cells:
            s = "".join(c["source"])
            if theirs in s:
                c["source"] = s.replace(theirs, ours).splitlines(keepends=True); n += 1
        print("бандл %s -> %s: в метаданных %s, в ячейках %d замен" % (theirs, ours, "да" if ours in meta["dataset_sources"] else "НЕТ", n))
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    meta["id"] = "sergueimakarov/arc3-pub-" + re.sub(r"[^a-z0-9-]", "-", out.name[4:].lower())[:45]
    meta["title"] = meta["id"].split("/", 1)[1].replace("-", " ")
    meta["is_private"] = True
    for k in ("id_no",):
        meta.pop(k, None)
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("готово к пушу (НЕ отправлено):", out, "| id", meta["id"])


if __name__ == "__main__":
    main()

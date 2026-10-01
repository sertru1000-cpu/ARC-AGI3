"""Перепаковка колеса sglang 0.5.21 с перенесёнными патчами Франзена: версия 0.5.21+arc3fz."""
import base64, csv, hashlib, io, sys, zipfile
from pathlib import Path
src, tree, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
LOCAL = "0.5.21+arc3fz"
zin = zipfile.ZipFile(src)
names = zin.namelist()
old_di = next(n.split('/')[0] for n in names if n.endswith('.dist-info/METADATA'))
new_di = f"sglang-{LOCAL}.dist-info"
files = {}
for n in names:
    data = zin.read(n)
    if n.startswith('sglang/') and (tree / n).is_file():
        data = (tree / n).read_bytes()
    files[n.replace(old_di, new_di)] = data
# новые файлы из дерева (которых не было в колесе)
added = []
for p in (tree / 'sglang').rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts:
        rel = p.relative_to(tree).as_posix()
        if rel not in files:
            files[rel] = p.read_bytes(); added.append(rel)
md = files[f"{new_di}/METADATA"].decode()
md = md.replace("Version: 0.5.21\n", f"Version: {LOCAL}\n", 1)
assert f"Version: {LOCAL}" in md
files[f"{new_di}/METADATA"] = md.encode()
rec = io.StringIO(); w = csv.writer(rec, lineterminator='\n')
for n, d in files.items():
    if n == f"{new_di}/RECORD": continue
    h = base64.urlsafe_b64encode(hashlib.sha256(d).digest()).rstrip(b'=').decode()
    w.writerow([n, f"sha256={h}", len(d)])
w.writerow([f"{new_di}/RECORD", "", ""])
files[f"{new_di}/RECORD"] = rec.getvalue().encode()
out = out_dir / f"sglang-{LOCAL}-cp312-cp312-manylinux_2_34_x86_64.whl"
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for n, d in files.items():
        z.writestr(n, d)
changed = sum(1 for n in names if n.startswith('sglang/') and (tree / n).is_file() and zin.read(n) != (tree / n).read_bytes())
print(out.name, f"{out.stat().st_size/1e6:.1f} МБ; изменено файлов {changed}, добавлено {len(added)}: {added}")

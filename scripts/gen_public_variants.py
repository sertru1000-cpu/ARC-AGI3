"""Вариации публичных игр без разбора правил (15.09, слово владельца «разобрать 25 публичных игр… и написать
генератор вариаций каждой — да»). Уровень в каждой из 25 игр — это `Level(sprites=[... .set_position(x, y) ...],
grid_size=(w, h), data={...})`; правила — в step(). Вариант = тот же файл игры, где список levels заменён ОДНИМ
уровнем с переставленными спрайтами (случайный сдвиг части подвижных спрайтов в пределах сетки; фон в (-1,-1) и
спрайты, стоящие в (0,0), не трогаем). Решаемость и разметка — отдельно, поиском в движке (scripts/engine_bfs.py).

usage: gen_public_variants.py --games sp80,cn04 --level 1 --n 20 --out <dir> [--jitter 3] [--frac 0.5] [--seed 1]
"""
import argparse, hashlib, json, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POS_RE = re.compile(r"(sprites\[\"[^\"]+\"\]\.clone\(\)(?:\.[a-z_]+\([^)]*\))*?\.set_position\()(-?\d+),\s*(-?\d+)(\))")


def find_block(src: str, start: int, open_ch: str, close_ch: str) -> int:
    """Индекс закрывающей скобки для скобки в src[start]."""
    depth = 0
    for i in range(start, len(src)):
        if src[i] == open_ch:
            depth += 1
        elif src[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("unbalanced")


def split_levels(src: str):
    m = re.search(r"^levels(?:\s*:\s*[^=]+)?\s*=\s*\[", src, re.M)
    if not m:
        raise ValueError("levels list not found")
    lb = src.index("[", m.start()); rb = find_block(src, lb, "[", "]")
    body = src[lb + 1: rb]
    blocks = []; i = 0
    while True:
        j = body.find("Level(", i)
        if j < 0:
            break
        k = find_block(body, j + len("Level"), "(", ")")
        blocks.append(body[j:k + 1]); i = k + 1
    return m.start(), lb, rb, blocks


SPR_RE = re.compile(r"\"([^\"]+)\":\s*Sprite\(\s*pixels=\[")


def sprite_sizes(src: str) -> dict:
    """{имя: (width, height)} по литералам pixels=[[...],...] в словаре sprites."""
    sizes = {}
    for m in SPR_RE.finditer(src):
        lb = m.end() - 1
        try:
            rb = find_block(src, lb, "[", "]")
        except ValueError:
            continue
        body = src[lb + 1: rb]
        rows = re.findall(r"\[([^\[\]]*)\]", body)
        if rows:
            sizes[m.group(1)] = (len([v for v in rows[0].split(",") if v.strip()]), len(rows))
    return sizes


def mirror(block: str, sizes: dict, axis: str) -> str:
    """Зеркало уровня: позиции пересчитаны по размерам спрайтов, к каждому спрайту добавлен set_mirror_lr/ud."""
    w, h = grid_size(block)
    out = []; last = 0
    for m in POS_RE.finditer(block):
        out.append(block[last:m.start()])
        name = re.search(r"sprites\[\"([^\"]+)\"\]", m.group(1)).group(1)
        sw, sh = sizes.get(name, (1, 1))
        x, y = int(m.group(2)), int(m.group(3))
        if x >= 0 and y >= 0:
            if axis == "lr":
                x = max(0, w - sw - x)
            else:
                y = max(0, h - sh - y)
        setter = ".set_mirror_lr(True)" if axis == "lr" else ".set_mirror_ud(True)"
        out.append(f"{m.group(1)}{x}, {y}{m.group(4)}{setter}")
        last = m.end()
    out.append(block[last:])
    return "".join(out)


def recolor(src: str, rng: random.Random, n_swaps: int = 2) -> str:
    """Перестановка цветов в пикселях спрайтов (весь файл): n_swaps случайных пар цветов 1..15 (0 и -1 не трогаем)."""
    m0 = re.search(r"^sprites\s*=\s*\{", src, re.M)
    if not m0:
        return src
    lb = src.index("{", m0.start()); rb = find_block(src, lb, "{", "}")
    body = src[lb:rb + 1]
    perm = {c: c for c in range(16)}
    cols = list(range(1, 16))
    for _ in range(n_swaps):
        a_, b_ = rng.sample(cols, 2); perm[a_], perm[b_] = perm[b_], perm[a_]
    def sub(mm):
        v = int(mm.group(0)); return str(perm.get(v, v)) if 0 <= v <= 15 else mm.group(0)
    body2 = re.sub(r"(?<![\w.-])-?\d+(?![\w.])", sub, body)
    return src[:lb] + body2 + src[rb + 1:]


def grid_size(block: str):
    g = re.search(r"grid_size=\((\d+),\s*(\d+)\)", block)
    return (int(g.group(1)), int(g.group(2))) if g else (64, 64)


def vary(block: str, rng: random.Random, jitter: int, frac: float) -> str:
    w, h = grid_size(block)
    items = list(POS_RE.finditer(block))
    movable = [m for m in items if not (int(m.group(2)) < 0 or int(m.group(3)) < 0) and not (int(m.group(2)) == 0 and int(m.group(3)) == 0)]
    if not movable:
        return block
    k = max(1, int(round(len(movable) * frac)))
    chosen = set(id(m) for m in rng.sample(movable, k))
    out = []; last = 0
    for m in items:
        out.append(block[last:m.start()])
        x, y = int(m.group(2)), int(m.group(3))
        if id(m) in chosen:
            x = min(w - 1, max(0, x + rng.randint(-jitter, jitter)))
            y = min(h - 1, max(0, y + rng.randint(-jitter, jitter)))
        out.append(f"{m.group(1)}{x}, {y}{m.group(4)}")
        last = m.end()
    out.append(block[last:])
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True); ap.add_argument("--level", type=int, default=1); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True); ap.add_argument("--jitter", type=int, default=3); ap.add_argument("--frac", type=float, default=0.5); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--keep-original", action="store_true", help="также записать оригинальный уровень как вариант 000")
    ap.add_argument("--levels", default=None, help="'all' -- каждый уровень игры как источник (иначе --level)")
    ap.add_argument("--mirror", action="store_true", help="добавить зеркала lr/ud каждого варианта")
    ap.add_argument("--recolor", type=int, default=0, help="доля вариантов с перестановкой цветов, в процентах")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); ids = []
    for g in a.games.split(","):
        src_path = next((ROOT / "environment_files" / g).glob("*/*.py"))
        src0 = src_path.read_text(encoding="utf-8")
        lstart, lb, rb, blocks = split_levels(src0)
        sizes = sprite_sizes(src0)
        levels = list(range(1, len(blocks) + 1)) if a.levels == "all" else [a.level]
        rng = random.Random(a.seed * 7919 + sum(map(ord, g)))
        vi = 0
        for lv in levels:
            if lv > len(blocks):
                continue
            base = blocks[lv - 1]
            variants = ([base] if a.keep_original else []) + [vary(base, rng, a.jitter, a.frac) for _ in range(a.n)]
            if a.mirror:
                variants += [mirror(v, sizes, "lr") for v in variants[: max(1, len(variants) // 2)]] + [mirror(v, sizes, "ud") for v in variants[: max(1, len(variants) // 2)]]
            for vb in variants:
                src = recolor(src0, rng) if (a.recolor and rng.random() < a.recolor / 100.0) else src0
                lstart2, lb2, rb2, _ = split_levels(src)
                body = src[:lb2 + 1] + "\n    " + vb + ",\n" + src[rb2:]
                ver = hashlib.md5(body.encode()).hexdigest()[:8]
                gid = f"{g}v{vi:03d}"; vi += 1
                d = out / gid / ver; d.mkdir(parents=True, exist_ok=True)
                (d / src_path.name).write_text(body, encoding="utf-8")
                meta = {"game_id": f"{gid}-{ver}", "source_game": g, "source_level": lv, "title": f"{g} L{lv} variant"}
                (d / "metadata.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
                ids.append(f"{gid}-{ver}")
        print(f"{g}: уровней-источников {len(levels)}, вариантов {vi}")
    (out / "game_ids.txt").write_text(",".join(ids)); print("всего", len(ids), "->", out)
    return
    for g in []:
        vi = 0
        for vb in []:
            body = ""
            ver = hashlib.md5(body.encode()).hexdigest()[:8]
            gid = f"{g}v{vi:03d}"
            d = out / gid / ver; d.mkdir(parents=True, exist_ok=True)
            (d / src_path.name).write_text(body, encoding="utf-8")
            meta = json.loads(next(src_path.parent.glob("metadata.json")).read_text()) if (src_path.parent / "metadata.json").exists() else {}
            meta.update({"game_id": f"{gid}-{ver}", "title": f"{g} variant {vi} of level {a.level}", "source_game": g, "source_level": a.level, "original": bool(a.keep_original and vi == 0)})
            (d / "metadata.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
            ids.append(f"{gid}-{ver}")
        print(f"{g}: уровень {a.level}, спрайтов с позицией {len(POS_RE.findall(base))}, вариантов {len(variants)}")
    (out / "game_ids.txt").write_text(",".join(ids)); print("всего", len(ids), "->", out)


if __name__ == "__main__":
    main()

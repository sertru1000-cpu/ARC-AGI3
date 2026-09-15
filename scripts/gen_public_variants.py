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
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); ids = []
    for g in a.games.split(","):
        src_path = next((ROOT / "environment_files" / g).glob("*/*.py"))
        src = src_path.read_text(encoding="utf-8")
        lstart, lb, rb, blocks = split_levels(src)
        if a.level > len(blocks):
            print(g, "уровней", len(blocks), "-- пропуск"); continue
        base = blocks[a.level - 1]
        rng = random.Random(a.seed * 7919 + sum(map(ord, g)))
        variants = ([base] if a.keep_original else []) + [vary(base, rng, a.jitter, a.frac) for _ in range(a.n)]
        for vi, vb in enumerate(variants):
            body = src[:lb + 1] + "\n    " + vb + ",\n" + src[rb:]
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

"""Кто и на основании чего вызывает action() в песочнице: разбор кода модели по транскриптам (27.09).
Каждый вызов python с ходом классифицируется по аргументам action(...):
  литерал — ход записан готовым (решён в рассуждении): action([{"action": "RIGHT"}]), action("UP"), action([...все литералы...]);
  вычислен — аргумент — переменная/выражение (путь из поиска, результат функции, элемент цикла);
  в цикле — action вызван внутри for/while (управляющий цикл «сделай — посмотри — сделай»);
  с условием — action под if (реакция на результат/доску).
usage: .venv/bin/python scripts/action_origin.py runs/night_nextfork-b1 [...]
"""
import ast, re, sys
from collections import Counter
from pathlib import Path
SEC = re.compile(r"^\[([A-Z][A-Z :_a-z]*)\]\s*$", re.M)
CODE = re.compile(r"<parameter=code>\n?(.*?)</parameter>", re.S)

def codes(run):
    for f in sorted(Path(run, "transcripts").glob("*.txt")):
        parts = SEC.split(f.read_text(errors="ignore"))
        for i in range(1, len(parts) - 1, 2):
            if parts[i].startswith("TOOL CALL: python"):
                m = CODE.search(parts[i + 1])
                if m:
                    yield m.group(1)

def classify(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    calls = []
    def walk(node, loop=False, cond=False):
        for ch in ast.iter_child_nodes(node):
            l = loop or isinstance(node, (ast.For, ast.While, ast.comprehension))
            c = cond or isinstance(node, ast.If)
            if isinstance(ch, ast.Call) and getattr(ch.func, "id", None) == "action":
                lit = all(isinstance(a, ast.Constant) or (isinstance(a, (ast.List, ast.Dict, ast.Tuple)) and all(
                    isinstance(x, (ast.Constant, ast.Dict)) and (not isinstance(x, ast.Dict) or all(isinstance(v, ast.Constant) for v in x.values))
                    for x in (a.elts if hasattr(a, "elts") else a.values))) for a in ch.args)
                calls.append((lit, l, c))
            walk(ch, l, c)
    walk(tree)
    return calls

for run in sys.argv[1:]:
    n = 0; k = Counter(); per = []
    for code in codes(run):
        cl = classify(code)
        if not cl:
            continue
        n += 1; per.append(len(cl))
        k["литерал"] += all(lit for lit, _, _ in cl)
        k["вычислен"] += any(not lit for lit, _, _ in cl)
        k["в цикле"] += any(l for _, l, _ in cl)
        k["с условием"] += any(c for _, _, c in cl)
        k["несколько action в одном коде"] += len(cl) > 1
    print("%s: вызовов python с ходом %d | " % (run, n) + " | ".join("%s %.0f%%" % (key, 100 * v / max(n, 1)) for key, v in k.items()))

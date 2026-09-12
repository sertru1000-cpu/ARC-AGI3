"""Помогла бы принятая модель мира взять уровень? Офлайн, без единого запроса к модели.

ЗАЧЕМ. Двенадцать версий линии строили дорогу к планированию и ни разу до него не дошли:
план по замыслу включается после взятого уровня и принятой программы, а это совпадение
не случилось ни разу в бою. Вопрос остался открытым: если бы программа была на руках,
нашёл бы поиск внутри неё путь к взятию уровня?

Ответ можно получить на записях. У нас есть 33 программы, принятые ТОЧНЫМ приёмом в прогоне
v5 (каждая воспроизводит все переходы своей игры на момент приёма), и полные записи ходов
с досками (artifacts/*_events.jsonl). Берём игру, где уровень БЫЛ взят, и спрашиваем:
из состояния входа в уровень доходит ли обход в ширину ВНУТРИ программы до того состояния,
из которого агент сделал победный ход?

Что значат исходы:
  * путь найден и короче фактического — метод жив, мешала только доставка модели в игру;
  * путь не найден — программа предсказывает переходы, но для планирования не годится,
    и это отдельный результат, который стоит записать;
  * уровень в игре не взят — оценивать не на чем, такие игры считаются отдельно.

Ограничения, названные заранее: действия берутся ровно те, что встречались в этой игре
(в кликовых играх это конкретные клетки, а не все 4096); поиск ограничен узлами и глубиной;
цель — состояние ПЕРЕД победным ходом, а не «уровень взят», потому что кадр после победы
показывает уже новую доску и программой не описан.

Расширение (09.09): программы можно брать из ОДНИХ прогонов, а записи ходов — из ДРУГОГО.
Программа моделирует правила игры, а не траекторию, поэтому её законно проверять на записи
того же названия из другого прогона. Так выборка растёт: уровень взят в 4 играх из 25 у нас
и в 21 из 25 у базы, а программы приняты в 23 играх.

usage:
    .venv/bin/python scripts/wm_plan_offline.py runs/flash_wm_v5
    .venv/bin/python scripts/wm_plan_offline.py runs/flash_v1_phaseA --programs runs/flash_wm_v5 runs/flash_wm_v10
"""

from __future__ import annotations

import ast
import glob
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARC3 = ROOT / "atlas_src" / "src" / "ARC3-Inference"
sys.path.insert(0, str(ARC3))
spec = importlib.util.spec_from_file_location("pts", ARC3 / "inference/agent/python_tool_sandbox.py")
pts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pts)

HELPERS_NS: dict = {}
exec(open(ROOT / "scripts" / "wm_helpers_v9.py", encoding="utf-8").read(), HELPERS_NS)
HELPERS = HELPERS_NS["WM_HELPERS"]

DEPTH = int(os.environ.get('WM_PLAN_DEPTH', 14))
NODES = int(os.environ.get('WM_PLAN_NODES', 20000))

CODE_RE = re.compile(r'"arguments":\s*"(.*?)"\s*\n\s*\}', re.S)
ADMIT_RE = re.compile(r"WM_CHECK admitted=1 correct=\d+ checked=\d+ total=(\d+)")


def accepted_programs(run: Path) -> list[tuple[str, int, str]]:
    """Программы, принятые ТОЧНЫМ приёмом: код хода плюс total=N из отчёта проверки."""
    out = []
    for f in sorted((run / "transcripts").glob("*.txt")):
        game = f.name[:4]
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in CODE_RE.finditer(text):
            try:
                code = json.loads(json.loads('"' + m.group(1) + '"')).get("code", "")
            except Exception:
                continue
            if "def predict" not in code or "def state_of" not in code:
                continue
            nxt = text.find("[TOOL RESULT: python]", m.end())
            if nxt < 0:
                continue
            verdict = ADMIT_RE.search(text[nxt:nxt + 2000])
            if verdict:
                out.append((game, int(verdict.group(1)), code))
    return out


def definitions_only(code: str) -> str:
    """Только определения — как делает харнесс перед хендоффом: вызовы action(...) в офлайне
    исполнить нельзя, а сама программа для поиска нужна целиком."""
    try:
        tree = ast.parse(code)
    except Exception:
        return code
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            seg = ast.get_source_segment(code, node)
            if seg and "action(" not in seg:
                keep.append(seg)
    src = "\n".join(keep)
    return src if ("def predict" in src and "def state_of" in src) else code


def game_state(events_path: str) -> dict | None:
    """Вся запись игры как состояние песочницы: кадры с сеткой, ascii и уровнем."""
    hist = []
    for line in open(events_path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") not in ("initial", "action") or not e.get("board"):
            continue
        hist.append({"action": str(e.get("action_display") or "") if e.get("type") == "action" else "",
                     "frame": {"ascii": e["board_ascii"], "step": int(e.get("action_num") or 0),
                               "level": int(e.get("level") or 0), "shape": [64, 64], "grid": e["board"]}})
    if not hist:
        return None
    return {"current_frame": hist[-1]["frame"], "history": hist,
            "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT", "SPACE"], "last_action_result": {}}


# Исполняется ВНУТРИ песочницы, рядом с программой модели.
PLAN = r'''
_all = _wm_pairs(10000, level=None if False else _LEVEL)
_win = None
for _i, (_b, _a, _f) in enumerate(_all):
    if wm_level(_f) != wm_level(_b):
        _win = _i; break
if _win is None:
    print("PLAN skip=уровень в этой игре не взят")
else:
    _entry = state_of(_all[0][0])
    _target = _wm_norm(state_of(_all[_win][0]))
    _acts = []
    for _b, _a, _f in _all:
        if _a not in _acts: _acts.append(_a)
    _acc_ok = 0; _acc_n = 0
    for _b, _a, _f in _all[:_win]:
        try: _pp = predict(state_of(_b), _a)
        except Exception: _pp = None
        if _pp is None: continue
        _acc_n += 1
        if _wm_norm(_pp) == _wm_norm(state_of(_f)): _acc_ok += 1
    _seen = {_wm_norm(_entry)}
    _queue = _wm_deque([(_entry, [])])
    _found = None; _nodes = 0; _dead = 0
    while _queue and _nodes < _NODES:
        _s, _path = _queue.popleft()
        if len(_path) >= _DEPTH: continue
        for _a in _acts:
            _nodes += 1
            try: _n = predict(_s, _a)
            except Exception: _n = None
            if _n is None: _dead += 1; continue
            _k = _wm_norm(_n)
            if _k == _target:
                _found = _path + [_a]; break
            if _k in _seen: continue
            _seen.add(_k); _queue.append((_n, _path + [_a]))
        if _found: break
    print("PLAN actions=%d nodes=%d dead=%d actual=%d found=%s len=%s accok=%d accn=%d" %
          (len(_acts), _nodes, _dead, _win, "1" if _found else "0", len(_found) if _found else "-",
           _acc_ok, _acc_n))
'''


def main(run_dir: str, program_runs: list[str] | None = None) -> int:
    run = Path(run_dir)
    progs = []
    for src in (program_runs or [run_dir]):
        progs += accepted_programs(Path(src))
    events = {os.path.basename(f).split("_p0")[0][:4]: f
              for f in glob.glob(str(run / "artifacts" / "*_events.jsonl"))}
    print(f"принятых программ: {len(progs)} в {len({g for g, _, _ in progs})} играх\n")
    print("%-6s %-8s %-8s %-9s %-8s %-10s %s" % ("игра", "действий", "узлов", "тупиков", "факт", "план", "верных"))
    found = skipped = failed = acc_ok_tot = acc_n_tot = 0
    wins = []
    done = set()
    for game, _n, code in progs:
        if game in done or game not in events:
            continue
        done.add(game)
        state = game_state(events[game])
        if state is None:
            continue
        level = state["history"][0]["frame"]["level"]
        out = pts.run_sandboxed_python(
            code=HELPERS + "\n_LEVEL = %d\n_DEPTH = %d\n_NODES = %d\n" % (level, DEPTH, NODES) + definitions_only(code) + "\n" + PLAN,
            timeout_seconds=120, initial_state=state,
            action_handler=lambda a: (_ for _ in ()).throw(RuntimeError("ходить нельзя")))
        so = str(out.get("stdout", "") or "") + str(out.get("error", "") or "")
        if "PLAN skip" in so:
            skipped += 1
            print("%-6s %-8s %-8s %-9s %-8s %-10s %s" % (game, "-", "-", "-", "уровень не взят", "-", "-"))
            continue
        m = re.search(r"PLAN actions=(\d+) nodes=(\d+) dead=(\d+) actual=(\d+) found=(\d) len=(\S+) accok=(\d+) accn=(\d+)", so)
        if not m:
            failed += 1
            print("%-6s %-8s %s" % (game, "-", "сбой: " + so.strip().splitlines()[-1][:80] if so.strip() else "пусто"))
            continue
        acts, nodes, dead, actual, ok, ln = m.group(1), m.group(2), m.group(3), int(m.group(4)), m.group(5) == "1", m.group(6)
        accok, accn = int(m.group(7)), int(m.group(8))
        found += ok
        acc_ok_tot += accok; acc_n_tot += accn
        if ok:
            wins.append((game, int(ln), actual))
        print("%-6s %-8s %-8s %-9s %-8s %-10s %s" % (game, acts, nodes, dead, actual,
              ln if ok else "не найден", f"{accok}/{accn}" if accn else "-"))

    n = len(done) - skipped - failed
    print(f"\nигр с взятым уровнем и принятой программой: {n}")
    print(f"поиск внутри программы нашёл путь: {found} из {n}")
    if wins:
        print("где нашёл — длина плана против фактических действий:")
        for g, ln, actual in wins:
            print(f"  {g}: план {ln}, факт {actual}" + ("  (короче)" if ln < actual else "  (не короче)"))
    if skipped:
        print(f"игр без взятого уровня (оценивать не на чем): {skipped}")
    if failed:
        print(f"сбоев исполнения: {failed}")
    if acc_n_tot:
        print(f"попутно: на этих траекториях программа предсказала верно {acc_ok_tot} из {acc_n_tot} "
              f"переходов ({100*acc_ok_tot/acc_n_tot:.0f}%)")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    rec = args[0] if args else "runs/flash_wm_v5"
    srcs = args[args.index("--programs") + 1:] if "--programs" in args else None
    raise SystemExit(main(rec, srcs))

"""Обобщают ли программы, написанные МОДЕЛЬЮ, на ходы, которых при приёме ещё не было?

Проспективный тест без переобучения: программа была принята точным приёмом на первых N
переходах игры (N известен из WM_CHECK total=N). Проверяем ту же программу в НАСТОЯЩЕЙ
песочнице бандла на переходах N+1..конец из записи той же игры (artifacts/*_events.jsonl).
Ни одна из этих проверок не тратит ни запроса к модели, ни квоты Kaggle.

Запуск: .venv/bin/python scripts/llm_program_generalization.py /tmp/v5_admitted_programs.json runs/flash_wm_v5 <helpers.py>
"""
import importlib.util, json, sys, glob, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARC3 = ROOT / "atlas_src" / "src" / "ARC3-Inference"
sys.path.insert(0, str(ARC3))
spec = importlib.util.spec_from_file_location("pts", ARC3 / "inference/agent/python_tool_sandbox.py")
pts = importlib.util.module_from_spec(spec); spec.loader.exec_module(pts)


def game_state(events_path):
    """История игры целиком: кадры с сеткой, ascii, уровнем; действие — в живом формате."""
    hist = []
    for line in open(events_path):
        try: e = json.loads(line)
        except Exception: continue
        if e.get("type") not in ("initial", "action") or not e.get("board"): continue
        fr = {"ascii": e["board_ascii"], "step": int(e.get("action_num") or 0),
              "level": int(e.get("level") or 0), "shape": [64, 64], "grid": e["board"]}
        act = str(e.get("action_display") or "") if e.get("type") == "action" else ""
        hist.append({"action": act, "frame": fr})
    if not hist: return None
    return {"current_frame": hist[-1]["frame"], "history": hist,
            "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT", "SPACE"], "last_action_result": {}}


CHECK = r'''
_pairs = _wm_pairs(10000)
_new = _pairs[%d:]
_st = state_of
_ok = 0; _chk = 0; _skip = 0
for _b, _a, _af in _new:
    try:
        _sb = _st(_b); _sa = _st(_af); _p = predict(_sb, _a)
    except Exception:
        _skip += 1; continue
    if _p is None: _skip += 1; continue
    _chk += 1
    if _wm_norm(_p) == _wm_norm(_sa): _ok += 1
print("GEN new=%%d checked=%%d correct=%%d skipped=%%d" %% (len(_new), _chk, _ok, _skip))
'''


def main(progs_path, run_dir, helpers_path):
    ns = {}; exec(open(helpers_path).read(), ns); H = ns["WM_HELPERS"]
    progs = json.load(open(progs_path))
    events = {os.path.basename(f).split("_p0")[0]: f for f in glob.glob(os.path.join(run_dir, "artifacts", "*_events.jsonl"))}
    tot_new = tot_chk = tot_ok = 0; per_game = {}
    for g, n, code in progs:
        st = game_state(events.get(g, ""))
        if st is None: continue
        out = pts.run_sandboxed_python(code=H + "\n" + code + "\n" + CHECK % n, timeout_seconds=60,
                                       initial_state=st, action_handler=lambda a: (_ for _ in ()).throw(RuntimeError("no actions")))
        so = str(out.get("stdout", "") or "")
        import re
        m = re.search(r"GEN new=(\d+) checked=(\d+) correct=(\d+) skipped=(\d+)", so)
        if not m:
            per_game.setdefault(g, []).append(("сбой", 0, 0, 0)); continue
        new, chk, ok, sk = map(int, m.groups())
        tot_new += new; tot_chk += chk; tot_ok += ok
        per_game.setdefault(g, []).append((n, new, chk, ok))
    print("программ проверено: %d в %d играх" % (len(progs), len(per_game)))
    print("новых переходов всего %d; из них программа взялась предсказать %d; верно %d" % (tot_new, tot_chk, tot_ok))
    if tot_new: print("ВЕРНО ИЗ ВСЕХ НОВЫХ: %.0f%%   |   верно из тех, что взялась: %.0f%%"
                      % (100 * tot_ok / tot_new, 100 * tot_ok / tot_chk if tot_chk else 0))
    print("\n%-16s %-9s %-7s %-9s %-6s" % ("игра", "буфер N", "новых", "взялась", "верно"))
    for g, lst in sorted(per_game.items()):
        for n, new, chk, ok in lst:
            print("%-16s %-9s %-7s %-9s %-6s" % (g, n, new, chk, ok))


if __name__ == "__main__":
    main(*sys.argv[1:4])

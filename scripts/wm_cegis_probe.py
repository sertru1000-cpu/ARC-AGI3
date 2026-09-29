"""Зонд: пишет ли НАША модель рабочий симулятор игры — цикл Twin без квоты Kaggle (26.09).

Модель та же (Qwen3.8-Flash-Next через HF Inference, поставщик featherless-ai), рассуждение включено.
Для каждой игры: переходы первого уровня из записанного прогона (runs/flash_v1_phaseA), воспроизведённые на
локальном движке. Первые TRAIN переходов даются модели (начальная доска целиком + по каждому ходу изменившиеся
клетки); модель пишет `def step(grid, action)`. Проверка — в настоящей песочнице v3 (тот же белый список, что в бою).
Если не все переходы воспроизведены — контрпримеры и до REPAIRS кругов ремонта (цикл CEGIS из статьи Twin).
Затем обобщение: отложенные переходы того же уровня, которых модель не видела.

Метрики: игр с точным воспроизведением обучающих переходов; доля точно предсказанных отложенных переходов.
Порог, записанный ДО запуска: если точное воспроизведение меньше чем в трети игр — цикл Twin нашей модели не по силам,
и прогон arc3-nextfork-twin читать только как проверку, а не как надежду.
usage: .venv/bin/python scripts/wm_cegis_probe.py [--games all] [--train 16] [--repairs 3]
"""
from __future__ import annotations
import argparse, json, logging, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor/ARC-AGI-3-Agents"))
sys.path.insert(0, os.environ.get("WM_INFER_SRC", "/tmp/nf_kaggle/src/ARC3-Inference"))
logging.disable(logging.ERROR)
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction
from inference.agent.python_tool_sandbox import run_sandboxed_python

# свой сервер на поде: WM_URL=http://127.0.0.1:1234/v1/chat/completions WM_MODEL=Qwen/Qwen3.8-Flash-Next-NVFP4
URL = os.environ.get("WM_URL", "https://router.huggingface.co/v1/chat/completions")
MODEL = os.environ.get("WM_MODEL", "Qwen/Qwen3.8-Flash-Next:featherless-ai")
MAX_TOKENS = int(os.environ.get("WM_MAX_TOKENS", "32000"))
HEX = "0123456789abcdef"
NAMES = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT", "ACTION5": "SPACE", "ACTION7": "ACTION7"}


def env_token():
    if os.environ.get("WM_URL"):
        return "local"
    for line in open(ROOT / ".env", encoding="utf-8"):
        if line.startswith("HF_INFERENCE_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("нет HF_INFERENCE_TOKEN в .env")


def act_str(a):
    if a["id"] == "ACTION6":
        d = a.get("data") or {}
        return "MOUSE(row=%d, col=%d)" % (d.get("y"), d.get("x"))
    return NAMES.get(a["id"], a["id"])


def level1_transitions(game_id, history):
    arc = arc_agi.Arcade(operation_mode=getattr(OperationMode, os.environ.get("WM_MODE", "NORMAL")), environments_dir=os.environ.get("WM_ENV_DIR", str(ROOT / "environment_files")))
    env = arc.make(game_id); fr = env.reset()
    g = np.asarray(fr.frame[-1]).tolist(); lv = getattr(fr, "levels_completed", 0) or 0
    out = []
    for rec in history:
        a = rec.get("action") or {}
        if not a.get("id") or a["id"] == "RESET":
            continue
        fr2 = env.step(GameAction[a["id"]], data=a.get("data"))
        if fr2 is None or not fr2.frame:
            break
        lv2 = getattr(fr2, "levels_completed", lv) or lv
        if lv2 != lv:
            break
        g2 = np.asarray(fr2.frame[-1]).tolist()
        if len(g2) != len(g):
            break
        out.append((g, act_str(a), g2)); g = g2
    return out


def hud_lines(train):
    """Линии полосы у края (3 клетки), меняющиеся в >=50% обучающих переходов: счётчики ходов, полосы энергии."""
    H, W = len(train[0][0]), len(train[0][0][0]); rows = {}; cols = {}
    for b, a, af in train:
        rr = set(); cc = set()
        for r in range(H):
            for c in range(W):
                if b[r][c] != af[r][c]:
                    if r <= 2 or r >= H - 3: rr.add(r)
                    if c <= 2 or c >= W - 3: cc.add(c)
        for r in rr: rows[r] = rows.get(r, 0) + 1
        for c in cc: cols[c] = cols.get(c, 0) + 1
    need = 0.5 * len(train)
    return sorted(r for r, k in rows.items() if k >= need), sorted(c for c, k in cols.items() if k >= need)


def board_text(g):
    return "\n".join("".join(HEX[v] for v in row) for row in g)


def diff_text(b, a):
    """Изменения — отрезками по строке: (r,c1-c2):старый>новый. Полную доску не вставляем: окно 32k (26.09)."""
    cells = [(r, c, b[r][c], a[r][c]) for r in range(len(b)) for c in range(len(b[0])) if b[r][c] != a[r][c]]
    if not cells:
        return "no change"
    runs = []
    for r, c, x, y in cells:
        if runs and runs[-1][0] == r and runs[-1][2] == c - 1 and runs[-1][3] == x and runs[-1][4] == y:
            runs[-1][2] = c
        else:
            runs.append([r, c, c, x, y])
    parts = ["(%d,%s):%s>%s" % (r, str(c1) if c1 == c2 else "%d-%d" % (c1, c2), HEX[x], HEX[y]) for r, c1, c2, x, y in runs]
    more = ""
    if len(parts) > 120:
        more = " ... (+%d more runs)" % (len(parts) - 120); parts = parts[:120]
    return "%d cells changed: %s%s" % (len(cells), " ".join(parts), more)


def prompt(trans, hud=None):
    lines = [
        "You are reverse-engineering a deterministic grid game from observations.",
        "The board is a 64x64 grid of colors 0-15 (shown as hex digits, one row per line, row 0 at top).",
        "Actions: UP, DOWN, LEFT, RIGHT, SPACE, ACTION7, or MOUSE(row=R, col=C) (a click on a cell).",
        "Below: the starting board, then consecutive transitions; each starts from the previous result.",
        "Write a Python function `def step(grid, action):` that takes the board as a list of lists of ints and the",
        "action string, and returns the next board (list of lists of ints). It must reproduce EVERY transition exactly.",
        "Model objects and rules (what moves, what blocks, what a click does), not a lookup table of these transitions.",
        "Only builtins and the modules collections/itertools/copy are available. Reply with one ```python block.",
        "Keep your analysis short (well under 3000 words): form a hypothesis, then write the code; the checker will send counterexamples.",
        "", "STARTING BOARD:", board_text(trans[0][0]), "", "TRANSITIONS:"]
    if hud and (hud[0] or hud[1]):
        lines[-4:-4] = ["Rows %s and columns %s at the board edge look like a status indicator (step counter / energy bar)."
                        " Your step may leave those cells unchanged: the checker ignores them. Focus on the playfield." % (hud[0], hud[1])]
    for i, (b, a, af) in enumerate(trans):
        lines.append("#%d %s -> %s" % (i, a, diff_text(b, af)))
    return "\n".join(lines)


def ask(token, messages, tries=3, think=True):
    budget = max(2000, min(MAX_TOKENS, 32000 - len(json.dumps(messages)) * 9 // 10))
    body = {"model": MODEL, "messages": messages, "max_tokens": budget, "temperature": 0.6, "top_p": 0.95}
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    for k in range(tries):
        try:
            r = requests.post(URL, headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
                              json=body, timeout=900).json()
            if "choices" in r:
                ch = r["choices"][0]; m = ch["message"]; u = dict(r.get("usage") or {}); u["finish"] = ch.get("finish_reason")
                return (m.get("content") or ""), u
            err = json.dumps(r)[:200]
        except Exception as e:
            err = repr(e)
        time.sleep(10 * (k + 1))
    return "", {"error": err}


def extract_code(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return max(m, key=len) if m else ""


VALIDATE = r'''
import json as _j
_T = __TRANS__
_MR, _MC = __MASK__
def _eq(p, q):
    if p == q: return True
    if not _MR and not _MC: return False
    try:
        return all(p[r][c] == q[r][c] for r in range(len(q)) for c in range(len(q[0])) if r not in _MR and c not in _MC)
    except Exception:
        return False
_res = {"ok": 0, "n": len(_T), "cex": []}
for _i, (_b, _a, _af) in enumerate(_T):
    try:
        _p = step([r[:] for r in _b], _a)
        _good = _eq(_p, _af)
    except Exception as _e:
        _good = False; _p = None
        if len(_res["cex"]) < 3: _res["cex"].append("#%d %s: step raised %r" % (_i, _a, _e))
    if _good:
        _res["ok"] += 1
    elif _p is not None and len(_res["cex"]) < 3:
        _d = [(r, c, _p[r][c], _af[r][c]) for r in range(min(len(_p), len(_af))) for c in range(min(len(_p[r]), len(_af[r]))) if _p[r][c] != _af[r][c] and r not in _MR and c not in _MC]
        _res["cex"].append("#%d %s: %d cells wrong, e.g. %s (row,col,predicted,observed)" % (_i, _a, len(_d), _d[:8]))
print("RESULT" + _j.dumps(_res))
'''


def validate(code, trans, mask=((), ())):
    src = code + "\n" + VALIDATE.replace("__TRANS__", json.dumps([[b, a, af] for b, a, af in trans])).replace(
        "__MASK__", json.dumps([list(mask[0]), list(mask[1])]))
    out = run_sandboxed_python(code=src, timeout_seconds=30, initial_state={"current_frame": None, "history": []},
                               action_handler=lambda a: {})
    m = re.search(r"RESULT(\{.*\})", out.get("stdout", ""))
    if not m:
        return {"ok": 0, "n": len(trans), "cex": ["code failed: %s" % (out.get("error") or "no result")[:300]]}
    return json.loads(m.group(1))


def run_game(args):
    gid, history, token, n_train, repairs = args
    trans = level1_transitions(gid, history)
    if len(trans) < 6:
        return {"game": gid[:4], "skip": "мало переходов (%d)" % len(trans)}
    train, test = trans[:n_train], trans[n_train:n_train + 16]
    mask = hud_lines(train) if os.environ.get("WM_MASK_HUD") == "1" else ([], [])
    msgs = [{"role": "user", "content": prompt(train, mask)}]
    hist = []; tokens = 0; best = (-1, ""); texts = []
    for rnd in range(repairs + 1):
        text, usage = ask(token, msgs, think=os.environ.get("WM_THINK", "1") == "1")
        tokens += int(usage.get("total_tokens", 0) or 0)
        code = extract_code(text)
        if not code and "error" not in usage:   # рассуждение съело потолок: просим только код, без рассуждения
            text, u2 = ask(token, msgs + [{"role": "user", "content": "Your analysis ran out of space. Reply now with ONLY the ```python block implementing step, no analysis."}], think=False)
            tokens += int(u2.get("total_tokens", 0) or 0); code = extract_code(text)
            hist.append({"round": rnd, "rescue": True, "finish": usage.get("finish"), "got_code": bool(code)})
        texts.append(text)
        if not code:
            hist.append({"round": rnd, "ok": 0, "note": "нет кода " + str(usage.get("error", ""))[:80]}); break
        v = validate(code, train, mask)
        hist.append({"round": rnd, "ok": v["ok"], "n": v["n"], "cex": v["cex"][:2]})
        if v["ok"] > best[0]:
            best = (v["ok"], code)
        if v["ok"] == v["n"]:
            break
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": "Your step reproduces %d of %d transitions. Counterexamples:\n%s\nFix step and reply with the full corrected ```python block." % (v["ok"], v["n"], "\n".join(v["cex"]))}]
    code = best[1]
    ident = "def step(grid, action):\n    return grid"
    base = validate(ident, test, mask) if test else {"ok": 0}   # нулевой базлайн: «ничего не меняй»
    exact = best[0] == len(train)
    gen = validate(code, test, mask) if (code and test) else {"ok": 0, "n": 0}
    full = validate(code, train) if (code and (mask[0] or mask[1])) else None   # точность вместе с индикатором
    return {"game": gid[:4], "train": len(train), "rounds": len(hist), "train_ok": max(best[0], 0),
            "exact": bool(exact), "test_ok": gen["ok"], "test_n": gen["n"], "test_identity_ok": base["ok"], "tokens": tokens,
            "mask": mask, "train_ok_with_hud": (full or {}).get("ok"), "hist": hist, "code": code, "texts": texts}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--games", default="all"); ap.add_argument("--train", type=int, default=16)
    ap.add_argument("--repairs", type=int, default=3); ap.add_argument("--par", type=int, default=4); a = ap.parse_args()
    token = env_token()
    bench = json.loads((ROOT / "runs/flash_v1_phaseA/benchmark.json").read_text())["game_runs"]
    want = None if a.games == "all" else set(a.games.split(","))
    jobs = [(g["game_id"], g.get("history") or [], token, a.train, a.repairs) for g in bench if want is None or g["game_id"][:4] in want]
    t0 = time.time(); rows = []
    with ThreadPoolExecutor(a.par) as ex:
        for r in ex.map(run_game, jobs):
            rows.append(r)
            print(json.dumps({k: v for k, v in r.items() if k not in ("hist", "code", "texts")}, ensure_ascii=False), flush=True)
    Path(os.environ.get("WM_OUT", str(ROOT / "runs/wm_cegis_probe.json"))).write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    done = [r for r in rows if "skip" not in r]
    ex_ = sum(r["exact"] for r in done); r0 = sum(1 for r in done if [h for h in r["hist"] if "ok" in h][:1] and [h for h in r["hist"] if "ok" in h][0]["ok"] == r["train"])
    tn = sum(r["test_n"] for r in done); tk = sum(r["test_ok"] for r in done)
    ti = sum(r.get("test_identity_ok", 0) for r in done)
    beat = sum(1 for r in done if r["test_ok"] > r.get("test_identity_ok", 0))
    print("базлайн «ничего не меняй» на отложенных: %d | модель лучше базлайна в %d играх" % (ti, beat))
    print("\nигр: %d (пропущено %d) | точно с первой попытки: %d | точно после ремонта: %d (%.0f%%) | отложенные переходы: %d/%d (%.0f%%) | токенов %d | %.0f мин"
          % (len(done), len(rows) - len(done), r0, ex_, 100 * ex_ / max(len(done), 1), tk, tn, 100 * tk / max(tn, 1),
             sum(r["tokens"] for r in done), (time.time() - t0) / 60))
    print("ПОРОГ: точно после ремонта в трети игр и больше ->", "ДОСТИГНУТ" if ex_ * 3 >= len(done) else "НЕ достигнут")


if __name__ == "__main__":
    main()

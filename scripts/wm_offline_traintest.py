"""Офлайн-замер модели мира по всем 25 играм с train/test внутри игры (ChatGPT р.2–3, 12.09).

Вопрос: у каких игр модель вообще способна написать предсказатель переходов, а у каких нет.
Прежний замер (86%) охватывал только 4 игры, где программа была принята в бою.

Постановка. Из записанных партий (несколько прогонов одной игры) берутся уникальные переходы
(доска до, действие, доска после), в хронологическом порядке. Первые N показываются модели
(контекст), остальные скрыты (тест). Модель пишет две функции:
    state_of(grid) -> состояние;  predict(state, action) -> следующее состояние или None.
Оценка на скрытых переходах:
  WM-1  одношаговая: norm(predict(state_of(до), a)) == norm(state_of(после)), считаются только
        НЕТРИВИАЛЬНЫЕ переходы (state_of(до) != state_of(после)) — константное состояние не проходит;
  WM-2  цепочка на k подряд идущих скрытых переходов, сравнение только конечного состояния;
  схлопывание: сколько различных состояний state_of даёт на различных скрытых досках
        (1 — состояние ничего не различает).
Ключ облака — из окружения DASHSCOPE_API_KEY. Программы модели исполняются в подпроцессе с таймаутом.

usage:
  .venv/bin/python scripts/wm_offline_traintest.py --model qwen3.8-flash --seeds 3 --out docs/wm_offline_flash.json
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
RUNS = ["runs/flash_v1_phaseA", "runs/flash_input_v1", "runs/flash_carry_v1", "runs/public_flash_tufa", "runs/flash_dose_conc13"]
ACTION_RE = re.compile(r"MOUSE\(row=(\d+),\s*col=(\d+)\)")


def transitions(game):
    """Уникальные переходы игры по нескольким прогонам, в порядке появления."""
    seen, out = set(), []
    for run in RUNS:
        for p in glob.glob(os.path.join(run, "artifacts", game + "*_events.jsonl")):
            prev = None
            for line in open(p, encoding="utf-8"):
                try:
                    e = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if e.get("type") not in ("initial", "action") or not e.get("board_ascii"):
                    continue
                cur = e["board_ascii"]
                if e["type"] == "action" and prev is not None:
                    a = str(e.get("action_display") or "")
                    key = (prev, a, cur)
                    if key not in seen:
                        seen.add(key)
                        out.append({"before": prev, "action": a, "after": cur, "level": int(e.get("level") or 0),
                                    "level_up": bool(e.get("level_completed")), "consecutive": True})
                prev = cur
    return out


PROMPT = """You are given recorded transitions from an unknown grid game (ARC-AGI-3). The board is a 64x64 grid; each cell is one hex digit (0-f) = a colour. Actions are UP, DOWN, LEFT, RIGHT, SPACE, or MOUSE(row=r, col=c).

Your task: infer the game's mechanics from the examples and write a Python predictor that generalises to UNSEEN transitions of the same game (including later levels with different layouts). It will be scored on transitions you have not seen.

Write exactly two functions in one ```python block, no other code outside them, no imports except from: collections, itertools, math, re, copy, json:

def state_of(grid):
    # grid: list of 64 strings, each 64 hex chars. Return a hashable, JSON-serialisable summary of the game-relevant state
    # (e.g. tuple of object positions/colours). It must change whenever the game state changes and ignore purely cosmetic detail.
    ...

def predict(state, action):
    # state: a value returned by state_of; action: one of the strings above.
    # Return the state after the action (same format as state_of) or None if you cannot predict this case.
    ...

Scoring: a prediction is correct if predict(state_of(before), action) == state_of(after). Only transitions where state_of(before) != state_of(after) count, so a constant state_of scores zero. Cover as many cases as you can; return None only when truly unknown.

EXAMPLES ({n} transitions, in chronological order; "before" of a transition equals "after" of the previous one when consecutive):
{examples}

Now write the two functions."""


def fmt_transition(i, t, show_before):
    s = "### transition %d  action: %s  level: %d%s\n" % (i, t["action"], t["level"], "  (LEVEL COMPLETED after this action)" if t["level_up"] else "")
    if show_before:
        s += "before:\n" + t["before"] + "\n"
    diff = []
    b, a = t["before"].split("\n"), t["after"].split("\n")
    for r, (x, y) in enumerate(zip(b, a)):
        for c, (p, q) in enumerate(zip(x, y)):
            if p != q:
                diff.append((r, c, p, q))
    if len(diff) > 400:
        s += "after:\n" + t["after"] + "\n"
    else:
        s += "after = before with these cell changes (row,col: old->new): " + " ".join("%d,%d:%s>%s" % d for d in diff) + "\n"
    return s


def build_prompt(train):
    parts = []
    prev_after = None
    for i, t in enumerate(train):
        parts.append(fmt_transition(i + 1, t, show_before=(t["before"] != prev_after)))
        prev_after = t["after"]
    return PROMPT.format(n=len(train), examples="\n".join(parts))


def chat(key, model, prompt, seed, max_tokens, timeout, thinking_budget=6000):
    """Потоковый запрос с бюджетом на думание: шлюз рвёт ЛЮБОЙ запрос ровно через 300 с,
    а без бюджета модель думает дольше (проверено 12.09: 5 мин и обрыв)."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.6, "top_p": 0.95, "seed": seed, "stream": True, "stream_options": {"include_usage": True},
            "enable_thinking": True, "thinking_budget": thinking_budget}
    req = urllib.request.Request(URL, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time(); text = []; usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ch = json.loads(payload)
            except Exception:  # noqa: BLE001
                continue
            if ch.get("usage"):
                usage = ch["usage"]
            for c in ch.get("choices") or []:
                d = c.get("delta") or {}
                if d.get("content"):
                    text.append(d["content"])
    text = "".join(text)
    # код: закрытый блок ```python …```; если ответ оборван без закрывающей ограды -- всё после открывающей;
    # если оград нет вовсе -- текст целиком (сбой 13.09: «```python» в первой строке и незакрытые docstring).
    m = re.search(r"```python\s*(.*?)```", text, re.S)
    if m:
        code = m.group(1)
    elif "```python" in text:
        code = text.split("```python", 1)[1]
    else:
        code = text
    code = code.replace("```", "")
    return code, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), round(time.time() - t0, 1)


EVAL = r'''
import json, sys
def _norm(x):
    if isinstance(x, dict): return tuple(sorted((str(k), _norm(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)): return tuple(_norm(v) for v in x)
    if isinstance(x, set): return tuple(sorted(_norm(v) for v in x))
    return x
data = json.load(open(sys.argv[1]))
test = data["test"]; K = data["k"]
res = {"n": len(test), "nontriv": 0, "covered": 0, "ok": 0, "chains": 0, "chain_ok": 0, "chain_nontriv": 0, "chain_nontriv_ok": 0, "errors": 0}
states = set(); boards = set()
S = []
for t in test:
    try:
        sb = state_of(t["before"].split("\n")); sa = state_of(t["after"].split("\n"))
    except Exception:
        res["errors"] += 1; S.append(None); continue
    S.append((sb, sa))
    boards.add(t["after"]); states.add(_norm(sa))
    if _norm(sb) == _norm(sa): continue
    res["nontriv"] += 1
    try:
        p = predict(sb, t["action"])
    except Exception:
        res["errors"] += 1; continue
    if p is None: continue
    res["covered"] += 1
    res["ok"] += int(_norm(p) == _norm(sa))
for i in range(len(test) - K + 1):
    win = test[i:i+K]
    if any(win[j]["after"] != win[j+1]["before"] for j in range(K-1)): continue
    if any(S[i+j] is None for j in range(K)): continue
    res["chains"] += 1
    s = S[i][0]
    try:
        for t in win:
            s = predict(s, t["action"])
            if s is None: break
    except Exception:
        s = None
    target = S[i+K-1][1]
    hit = s is not None and _norm(s) == _norm(target)
    res["chain_ok"] += int(hit)
    if _norm(target) != _norm(S[i][0]):
        res["chain_nontriv"] += 1; res["chain_nontriv_ok"] += int(hit)
res["distinct_states"] = len(states); res["distinct_boards"] = len(boards)
print("WMRESULT " + json.dumps(res))
'''


def evaluate(code, test, k, timeout=120):
    with tempfile.TemporaryDirectory() as d:
        Path(d, "prog.py").write_text(code + "\n" + EVAL, encoding="utf-8")
        Path(d, "data.json").write_text(json.dumps({"test": test, "k": k}), encoding="utf-8")
        try:
            out = subprocess.run([sys.executable, "-I", os.path.join(d, "prog.py"), os.path.join(d, "data.json")],
                                 capture_output=True, text=True, timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return {"error": "timeout"}
        m = re.search(r"WMRESULT (.*)", out.stdout)
        if not m:
            return {"error": (out.stderr or out.stdout)[-300:]}
        return json.loads(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.8-flash")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--train-share", type=float, default=0.4)
    ap.add_argument("--max-train", type=int, default=16)
    ap.add_argument("--max-test", type=int, default=120)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--games", default="")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--thinking-budget", type=int, default=6000)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="docs/wm_offline_results.json")
    a = ap.parse_args()
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise SystemExit("нет ключа DASHSCOPE_API_KEY в окружении")
    games = a.games.split(",") if a.games else sorted({os.path.basename(p)[:4] for p in glob.glob("runs/flash_v1_phaseA/artifacts/*_events.jsonl")})
    jobs = []
    for g in games:
        tr = transitions(g)
        n_train = min(a.max_train, max(4, int(a.train_share * len(tr))))
        train, test = tr[:n_train], tr[n_train:n_train + a.max_test]
        if len(test) < 5:
            print("%s: переходов %d — мало для теста, пропуск" % (g, len(tr))); continue
        for seed in range(a.seeds):
            jobs.append((g, seed, train, test))
    print("игр %d, вызовов %d, модель %s" % (len({j[0] for j in jobs}), len(jobs), a.model))

    def run(job):
        g, seed, train, test = job
        try:
            code, tin, tout, sec = chat(key, a.model, build_prompt(train), seed, a.max_tokens, a.timeout, a.thinking_budget)
        except Exception as e:  # noqa: BLE001
            return {"game": g, "seed": seed, "error": "api: " + repr(e)[:200]}
        r = evaluate(code, test, a.k)
        r.update({"game": g, "seed": seed, "n_train": len(train), "n_test": len(test), "in": tin, "out": tout, "sec": sec, "code": code})
        return r

    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for r in pool.map(run, jobs):
            results.append(r)
            if "error" in r and "nontriv" not in r:
                print("  %s s%d СБОЙ: %s" % (r["game"], r["seed"], r["error"][:120]))
            else:
                acc = 100 * r["ok"] / r["nontriv"] if r["nontriv"] else float("nan")
                cov = 100 * r["covered"] / r["nontriv"] if r["nontriv"] else float("nan")
                ch = 100 * r["chain_nontriv_ok"] / r["chain_nontriv"] if r["chain_nontriv"] else float("nan")
                print("  %s s%d: тест %3d, нетрив. %3d, взялась %3.0f%%, верно %3.0f%% | цепочки k=%d: %3.0f%% (%d) | состояний %d на %d досок | %d вых. ток, %.0f с"
                      % (r["game"], r["seed"], r["n_test"], r["nontriv"], cov, acc, a.k, ch, r["chain_nontriv"], r["distinct_states"], r["distinct_boards"], r["out"], r["sec"]))
    Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = [r for r in results if "nontriv" in r]
    tin = sum(r["in"] for r in ok); tout = sum(r["out"] for r in ok)
    print("готово: %d ответов, вход %d ток, выход %d ток, ~%.2f юаня; результаты в %s" % (len(ok), tin, tout, (tin * 0.8 + tout * 2.7) / 1e6, a.out))


if __name__ == "__main__":
    main()

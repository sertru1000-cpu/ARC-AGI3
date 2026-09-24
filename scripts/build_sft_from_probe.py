"""Обучающий набор из записей стенда: расшифровки боевой обвязки -> примеры в формате диалога (20.09).

Зачем. Августовские записи учителя собраны на СТАРОЙ обвязке (другой промпт, другой протокол инструментов), учить
на них -- учить не тому. Этот сборщик берёт расшифровки любого прогона scripts/run_api_probe.py (боевая обвязка Duck
на локальном движке) и делает из них примеры ровно того вида, который модель видит в бою.

Единица обучения -- ХОД обвязки (наблюдать -> подумать -> сходить), а не отдельный вызов модели: внутри хода модель
сначала запускает питон, смотрит на вывод и только потом ходит. ИЗМЕРЕНО на боевом окне: 1.74 вызова на ход.

Отбор (иначе учим проигрышам): ход берётся, только если уровень, на котором он сделан, БЫЛ ВЗЯТ, и эффективность
этого уровня (базлайн/ходы) не ниже порога. Хвост на непройденном уровне выбрасывается -- по замеру это 48% ходов.

usage:
  .venv/bin/python scripts/build_sft_from_probe.py runs/win32k_tu93 [runs/teacher_gemini_pilot] \
      --out data/sft_v3/train.jsonl --min-efficiency 0.5
"""
import argparse, json, os, re
from pathlib import Path

TAG = re.compile(r"^\[[^\]\n]{2,40}\]", re.M)
STEP = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+)", re.M)
LEVEL = re.compile(r"Current state: step \d+, level (\d+)")


def sections(step_text: str) -> list:
    """[(тег, тело)] в порядке появления"""
    out = []
    marks = list(TAG.finditer(step_text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(step_text)
        out.append((m.group(0), step_text[m.end():end].strip()))
    return out


def tool_calls(meta_body: str) -> list:
    """разбор raw_tool_calls из блока META; подпись мысли Gemini выбрасываем -- это мусор на 700 символов"""
    i = meta_body.find("raw_tool_calls:")
    if i < 0:
        return []
    j = meta_body.find("[", i)
    if j < 0:
        return []
    depth = 0
    for k in range(j, len(meta_body)):
        if meta_body[k] == "[":
            depth += 1
        elif meta_body[k] == "]":
            depth -= 1
            if depth == 0:
                try:
                    calls = json.loads(meta_body[j:k + 1])
                except Exception:
                    return []
                for c in calls:
                    c.pop("extra_content", None)
                return calls
    return []


def parse_transcript(path: Path) -> list:
    s = path.read_text(encoding="utf-8", errors="replace")
    heads = [m.start() for m in STEP.finditer(s)]
    steps = []
    for a, b in zip(heads, heads[1:] + [len(s)]):
        body = s[a:b]
        secs = sections(body)
        sys_txt = next((t for tag, t in secs if tag == "[SYSTEM PROMPT]"), "")
        usr_txt = next((t for tag, t in secs if tag == "[USER PROMPT]"), "")
        if not sys_txt or not usr_txt:
            continue
        lvl = LEVEL.search(usr_txt)
        msgs = [{"role": "system", "content": sys_txt}, {"role": "user", "content": usr_txt}]
        n_calls = 0
        # ЛОВУШКА (24.09): рассуждение учителя лежит ОТДЕЛЬНЫМ разделом [THINKING] сразу после META, а не
        # внутри него. Первая версия сборщика его не читала -- и набор, ради которого писался переходник к
        # родному API, выходил БЕЗ мысли, то есть учил студента ходить не думая (у базы это 2.91 против 9.43).
        # Порядок в ходе: META -> THINKING -> TOOL CALL -> TOOL RESULT, поэтому мысль прикладываем к
        # ПОСЛЕДНЕМУ добавленному ответу модели, в поле reasoning_content -- именно его ждёт шаблон чата
        # модели: '<|im_start|>assistant\n<think>\n' + reasoning_content + '\n</think>\n\n' + content.
        for tag, t in secs:
            if tag == "[MODEL RESPONSE META]":
                calls = tool_calls(t)
                if calls:
                    msgs.append({"role": "assistant", "content": "", "tool_calls": calls}); n_calls += 1
            elif tag == "[THINKING]" and t:
                if msgs and msgs[-1]["role"] == "assistant":
                    msgs[-1]["reasoning_content"] = t
                else:
                    msgs.append({"role": "assistant", "content": "", "reasoning_content": t}); n_calls += 1
            elif tag == "[ASSISTANT]" and t:
                if msgs and msgs[-1]["role"] == "assistant" and not (msgs[-1].get("content") or "").strip():
                    msgs[-1]["content"] = t
                else:
                    msgs.append({"role": "assistant", "content": t}); n_calls += 1
            elif tag == "[TOOL RESULT: python]" and t:
                msgs.append({"role": "tool", "name": "python", "content": t})
        if n_calls == 0:            # ход без ответа модели (сбой доступа) -- учить нечему
            continue
        steps.append({"level": int(lvl.group(1)) if lvl else None, "messages": msgs, "calls": n_calls})
    return steps


def run_stats(run: Path) -> dict:
    """уровни и эффективность по benchmark.json прогона"""
    out = {}
    bj = run / "benchmark.json"
    if not bj.exists():
        return out
    for r in json.load(open(bj, encoding="utf-8")).get("game_runs", []):
        per = list(r.get("actions_per_level") or []); base = r.get("base_actions_per_level") or []
        done = int(r.get("levels_completed") or 0)
        eff = {}
        for i in range(min(done, len(per), len(base))):
            if per[i]:
                eff[i + 1] = base[i] / per[i]
        out[r["game_id"][:4]] = {"done": done, "eff": eff}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="data/sft_v3/train.jsonl")
    ap.add_argument("--min-efficiency", type=float, default=0.5,
                    help="нижняя граница базлайн/ходы у взятого уровня; 0 -- брать все взятые")
    ap.add_argument("--holdout", default="", help="игры через запятую, которые уходят в ОТЛОЖЕННЫЙ файл рядом с train")
    ap.add_argument("--history-tokens", type=int, default=30000,
                    help="сколько токенов прошлых ходов подмешивать перед текущим (в бою обвязка режет историю "
                         "под окно 32768 минус резерв; 0 -- примеры без истории)")
    a = ap.parse_args()
    kept = dropped_tail = dropped_slow = 0
    rows = []
    for rd in a.runs:
        run = Path(rd); stats = run_stats(run)
        for tp in sorted((run / "transcripts").glob("*_p0.txt")):
            game = tp.name[:4]
            st = stats.get(game, {"done": 0, "eff": {}})
            parsed = parse_transcript(tp)
            for k, step in enumerate(parsed):
                lvl = step["level"]
                if lvl is None or lvl > st["done"]:
                    dropped_tail += 1; continue
                e = st["eff"].get(lvl)
                if e is not None and e < a.min_efficiency:
                    dropped_slow += 1; continue
                msgs = step["messages"]
                if a.history_tokens > 0:
                    # ИСТОРИЯ: обвязка в бою шлёт предыдущие ходы того же прогона и режет их под окно.
                    # Воспроизводим это: идём назад от текущего хода, пока хватает бюджета.
                    budget = a.history_tokens * 2.6        # бюджет в символах
                    hist = []
                    for prev in reversed(parsed[:k]):
                        block = [m for m in prev["messages"] if m["role"] != "system"]
                        size = sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or [])) for m in block)
                        if size > budget:
                            break
                        budget -= size
                        hist = block + hist
                    msgs = [msgs[0]] + hist + msgs[1:]
                rows.append({"game": game, "run": run.name, "level": lvl, "efficiency": e,
                             "calls": step["calls"], "messages": msgs})
                kept += 1
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    hold = {g.strip() for g in a.holdout.split(",") if g.strip()}
    train_rows = [r for r in rows if r["game"] not in hold]
    valid_rows = [r for r in rows if r["game"] in hold]
    with open(out, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if hold:
        vp = out.with_name(out.stem.replace("train", "valid") + out.suffix)
        with open(vp, "w", encoding="utf-8") as f:
            for r in valid_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("отложено игр %s: %d ходов -> %s" % (",".join(sorted(hold)), len(valid_rows), vp))
    rows = train_rows
    chars = sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or [])) for r in rows for m in r["messages"])
    print("годных ходов %d | выброшено: хвост непройденных уровней %d, медленные уровни %d"
          % (kept, dropped_tail, dropped_slow))
    print("вызовов модели в наборе %d (%.2f на ход)" % (sum(r["calls"] for r in rows), sum(r["calls"] for r in rows) / max(1, kept)))
    print("объём %.1f млн символов ~ %.1f млн токенов (2.6 симв/токен); файл %s"
          % (chars / 1e6, chars / 2.6 / 1e6, out))
    if rows:
        print("длина примера: медиана %.0f тыс. токенов"
              % (sorted(sum(len(m.get("content") or "") for m in r["messages"]) for r in rows)[len(rows) // 2] / 2.6 / 1000))


if __name__ == "__main__":
    main()

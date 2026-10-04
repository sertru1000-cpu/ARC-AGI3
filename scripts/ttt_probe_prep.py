"""Проба «дообучение во время игры», вариант A, подготовка данных (локально, бесплатно).

Вопрос пробы: если обучить маленькую поправку к выходу модели (низкоранговую добавку к последнему скрытому состоянию
перед lm_head) на НАБЛЮДЕНИЯХ первой половины игры, станет ли модель лучше предсказывать наблюдения второй половины
той же игры — и лучше ли, чем поправка, обученная на ДРУГОЙ игре (контроль: общая подстройка к домену, а не к правилам игры)?
Наблюдения = сообщения tool (вывод кода агента по доске) и user (отчёт обвязки об исходе ходов).

Данные: записанные запросы контроля base2 (runs/kaggle_stand_0310_v1/*_p0_requests.jsonl). Картинки заменяются строкой
"[board image]" (проба текстовая). На игру берутся K запросов, равномерно по игре; из каждого — последние WINDOW токенов.
Обучение: первые K/2 запросов, цели — все наблюдения в окне. Проверка: последние K/2, цели — только наблюдения новее
последнего обучающего запроса (чтобы не проверять на виденном).
Выход: датасет runs/ttt_probe/data/{probe.json} — input_ids и маски целей.
usage: .venv/bin/python scripts/ttt_probe_prep.py
"""
import glob, json, os
from pathlib import Path

from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
TOK = ROOT / "runs/ttt_probe/tok"
OUT = ROOT / "runs/ttt_probe/data"
K, WINDOW = 8, 6000

tok = PreTrainedTokenizerFast(tokenizer_file=str(TOK / "tokenizer.json"))
TEMPLATE = (TOK / "chat_template.jinja").read_text()


def textify(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = "".join(p.get("text", "") if p.get("type") == "text" else "[board image]" for p in c)
        elif c is None:
            m["content"] = ""
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc)); fn = tc.get("function", {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except Exception:
                        fn["arguments"] = {"code": fn["arguments"]}
                tcs.append(tc)
            m["tool_calls"] = tcs
        out.append(m)
    return out


def render(msgs, tools, kw):
    return tok.apply_chat_template(msgs, tools=tools, tokenize=False, chat_template=TEMPLATE, **(kw or {}))


def spans(rendered, msgs):
    """символьные границы содержимого каждого сообщения (поиск по порядку)."""
    res, pos = [], 0
    for i, m in enumerate(msgs):
        c = m["content"]
        if not c:
            res.append(None); continue
        j = rendered.find(c, pos)
        if j < 0:
            res.append(None); continue
        res.append((j, j + len(c))); pos = j + len(c)
    return res


def build(rec, seen):
    msgs = textify(rec["messages"])
    text = render(msgs, rec.get("tools"), rec.get("chat_template_kwargs"))
    sp = spans(text, msgs)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    mask = [0] * len(ids)
    for i, (m, s) in enumerate(zip(msgs, sp)):
        if s is None or m["role"] not in ("tool", "user") or (seen is not None and m["content"] in seen):
            continue
        for t, (a, b) in enumerate(offs):
            if a >= s[0] and b <= s[1] and b > a:
                mask[t] = 1
    cut = max(0, len(ids) - WINDOW)
    return ids[cut:], mask[cut:], {m["content"] for m in msgs if m["role"] in ("tool", "user")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    games = []
    for f in sorted(glob.glob(str(ROOT / "runs/kaggle_stand_0310_v1/*_p0_requests.jsonl"))):
        recs = [json.loads(l) for l in open(f)]
        recs = [r for r in recs if r.get("messages")]
        if len(recs) < 2 * K:
            continue
        idx = [round(i * (len(recs) - 1) / (K - 1)) for i in range(K)]
        sel = [recs[i] for i in idx]
        half = K // 2
        items, seen = [], set()
        for n, r in enumerate(sel):
            if n < half:
                ids, mask, nm = build(r, None); seen |= nm
                items.append({"split": "train", "ids": ids, "mask": mask})
            else:
                ids, mask, nm = build(r, seen)
                items.append({"split": "eval", "ids": ids, "mask": mask})
        n_tr = sum(sum(i["mask"]) for i in items if i["split"] == "train")
        n_ev = sum(sum(i["mask"]) for i in items if i["split"] == "eval")
        games.append({"game": Path(f).name[:4], "items": items})
        print(f"{Path(f).name[:4]}: запросов {len(recs)}, целевых токенов обучение {n_tr}, проверка {n_ev}")
    (OUT / "probe.json").write_text(json.dumps(games))
    print("игр", len(games), "размер", os.path.getsize(OUT / "probe.json") // 1024, "КБ")


if __name__ == "__main__":
    main()

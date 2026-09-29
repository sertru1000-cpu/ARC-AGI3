"""Сторож публикаций к этапу 30.09 (26.09): замечает, что кто-то из лидеров открыл решение, и сразу разбирает его.

Правила (тема 713634 организаторов): открыть решение ради приза этапа надо до 23:59 UTC 30.09 = 02:59 МСК 01.10;
места этапа — по ПУБЛИЧНОЙ таблице. Намерения (темы 742935, 743624, на 26.09):
  Tong Hui Kang (huikang, 1-е, 20.53) — откроет, только если будет первым;
  Lord Han Solo (lordhansolo, 2-е, 19.45) — откроет, если войдёт в тройку среди желающих открыться;
  Tufa Labs (3-е, 18.81) — не откроет (тема 742801).

Что смотрит каждые --every секунд (всё бесплатно, квоту не тратит):
  1. публичные ноутбуки, датасеты и модели наблюдаемых авторов;
  2. 30 самых новых ноутбуков соревнования;
  3. 20 самых новых тем форума соревнования.
Новое сверх прошлого снимка -> строка в runs/pubwatch/alerts.log, уведомление macOS; для новых ноутбуков
наблюдаемых авторов (и любых ноутбуков с подходящими словами в названии) — сразу scripts/graft_inspect.py,
отчёт в runs/pubwatch/<slug>.txt.
usage: .venv/bin/python scripts/watch_publication.py [--every 300] [--once] [--users huikang,lordhansolo]
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
K = str(ROOT / ".venv/bin/kaggle")
OUT = ROOT / "runs/pubwatch"
COMP = "arc-prize-2026-arc-agi-3"
FORUM_ID = 10403401
USERS = ["huikang", "lordhansolo"]
KEYWORDS = re.compile(r"milestone|1st|2nd|first place|second place|solution|write-?up|open.?source|top.?[123]\b", re.I)
MSK = timezone(timedelta(hours=3))


def now():
    return datetime.now(MSK).strftime("%d.%m %H:%M")


def sh(*a, timeout=120):
    try:
        return subprocess.run([K, *a], capture_output=True, text=True, timeout=timeout).stdout
    except Exception as exc:
        return "ERR %r" % (exc,)


def csv_refs(text):
    """Первый столбец таблицы kaggle CLI (ref), без заголовка и разделителя."""
    rows = []
    for line in text.splitlines()[2:]:
        ref = next((p for p in line.split() if "/" in p), None)   # у models первым идёт числовой id
        if ref:
            rows.append((ref, line.strip()))
    return rows


def token():
    p = Path.home() / ".kaggle/access_token"
    return p.read_text().strip() if p.exists() else ""


def forum_topics():
    try:
        r = requests.post("https://www.kaggle.com/api/i/discussions.DiscussionsService/GetTopicListByForumId",
                          headers={"Authorization": "Bearer " + token(), "Content-Type": "application/json"},
                          json={"forumId": FORUM_ID, "sortBy": "RECENT", "page": 1, "pageSize": 20,
                                "sort": "FORUM_TOPIC_LIST_SORT_BY_RECENT"}, timeout=60).json()
        return [("topic/%s" % t["id"], "%s | %s | %s" % (t.get("postDate", "")[:16], (t.get("authorUser") or {}).get("url"), t.get("title")))
                for t in r.get("topics", [])]
    except Exception as exc:
        print("форум: %r" % (exc,), flush=True)
        return []


WATCH_TOPICS = [743723]   # 26.09: наш пост «Harness or model?» — ответы со ссылками на код (так пришла обвязка Скотта)
LINK = re.compile(r"kaggle\.com/(code|datasets|models)/([\w-]+/[\w.-]+)")


def topic_comments(topic_id):
    """Комментарии темы (рекурсивно): ключ comment/<id>, строка = автор | начало текста | ссылки на код."""
    try:
        r = requests.post("https://www.kaggle.com/api/i/discussions.DiscussionsService/GetForumTopicById",
                          headers={"Authorization": "Bearer " + token(), "Content-Type": "application/json"},
                          json={"forumTopicId": topic_id, "includeComments": True}, timeout=60).json()
    except Exception as exc:
        print("тема %s: %r" % (topic_id, exc), flush=True)
        return []
    out = []
    def walk(cs):
        for c in cs or []:
            a = c.get("author") or {}
            text = c.get("rawMarkdown") or ""
            links = ["kaggle.com/%s/%s" % m for m in LINK.findall(text)]
            out.append(("comment/%s/%s" % (topic_id, c.get("id")),
                        "%s | %s | %s | ссылки: %s" % (c.get("postDate", "")[:16], a.get("url") or a.get("displayName"),
                                                       text[:150].replace("\n", " "), " ".join(links) or "-")))
            walk(c.get("replies"))
    walk(((r or {}).get("forumTopic") or {}).get("comments"))
    return out


def snapshot(users):
    items = {}
    for u in users:
        for ref, line in csv_refs(sh("kernels", "list", "--user", u, "--sort-by", "dateCreated", "--page-size", "20")):
            items["kernel/" + ref] = ("watched", line)
        for ref, line in csv_refs(sh("datasets", "list", "--user", u, "--sort-by", "updated")):
            items["dataset/" + ref] = ("watched", line)
        for ref, line in csv_refs(sh("models", "list", "--owner", u)):
            items["model/" + ref] = ("watched", line)
    for ref, line in csv_refs(sh("kernels", "list", "--competition", COMP, "--sort-by", "dateCreated", "--page-size", "30")):
        items.setdefault("kernel/" + ref, ("comp", line))
    for key, line in forum_topics():
        items[key] = ("forum", line)
    for t in WATCH_TOPICS:
        for key, line in topic_comments(t):
            items[key] = ("reply", line)
    return items


def notify(title, text):
    try:
        subprocess.run(["osascript", "-e", 'display notification "%s" with title "%s" sound name "Glass"' % (
            text.replace('"', "'")[:180], title.replace('"', "'"))], timeout=10)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--every", type=int, default=300); ap.add_argument("--once", action="store_true")
    ap.add_argument("--users", default=",".join(USERS)); a = ap.parse_args()
    users = [u for u in a.users.split(",") if u]
    OUT.mkdir(parents=True, exist_ok=True)
    state_path = OUT / "state.json"
    seen = json.loads(state_path.read_text()) if state_path.exists() else None
    while True:
        items = snapshot(users)
        if seen is None:           # первый снимок — точка отсчёта, тревог нет
            seen = {k: v[1] for k, v in items.items()}
            state_path.write_text(json.dumps(seen, ensure_ascii=False, indent=1))
            print("%s точка отсчёта: %d объектов (наблюдаемые: %s)" % (now(), len(seen), ", ".join(users)), flush=True)
        else:
            for src in ("comment/",):
                if not any(k.startswith(src) for k in seen):      # источник добавлен после старта — точка отсчёта без тревог
                    for k, (kind, line) in items.items():
                        if k.startswith(src):
                            seen[k] = line
            new = [(k, kind, line) for k, (kind, line) in items.items() if k not in seen]
            for k, kind, line in new:
                hot = kind in ("watched", "reply") or bool(KEYWORDS.search(line))
                msg = "%s %s %s %s" % (now(), "ВАЖНО" if hot else "новое", k, line[:200])
                print(msg, flush=True)
                with open(OUT / "alerts.log", "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                if hot:
                    notify("ARC: публикация?", k)
                    refs = [k.split("/", 1)[1]] if k.startswith("kernel/") else \
                           [m[1] for m in LINK.findall(line) if m[0] == "code"] if k.startswith("comment/") else []
                    for ref in refs:
                        rep = OUT / (ref.replace("/", "__") + ".txt")
                        r = subprocess.run([sys.executable, str(ROOT / "scripts/graft_inspect.py"), ref], capture_output=True, text=True, timeout=3600)
                        rep.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr[-5000:], encoding="utf-8")
                        print("%s разбор: %s" % (now(), rep), flush=True)
                seen[k] = line
            state_path.write_text(json.dumps(seen, ensure_ascii=False, indent=1))
            if not new:
                print("%s тихо (%d объектов)" % (now(), len(items)), flush=True)
        if a.once:
            break
        time.sleep(a.every)


if __name__ == "__main__":
    main()

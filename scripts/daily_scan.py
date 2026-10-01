"""Ежедневный скан публичной линии (правило владельца: 06.09 еженедельно, с 01.10 — ежедневно).

Снимает состояние источников в runs/scan/<дата>.json, сравнивает с предыдущим снимком и дописывает раздел
с дельтами в начало docs/public_line_scan_daily.md (страница: http://127.0.0.1:8765/doc/public_line_scan_daily.md).

Источники:
  Kaggle       — таблица, кернелы соревнования, модели, датасеты, форум (темы и число комментариев);
  модели       — HuggingFace (упаковки базовой модели, новые модели отслеживаемых авторов, список набирающих
                 популярность), ModelScope (организация Qwen и поиск по именам следующей базы);
  серверы      — версии на PyPI (включая предрелизы), релизы GitHub, образы Docker Hub, слитые в vLLM и SGLang
                 правки, где упомянута наша модель;
  обвязка/игра — репозитории открытых решений, организаций arcprize и Tufalabs, пакеты движка на PyPI;
  статьи       — arXiv.
Текст темы форума читать отдельно через GetForumTopicById (см. scripts/watch_publication.py).
usage: .venv/bin/python scripts/daily_scan.py            # снимок + отчёт
       .venv/bin/python scripts/daily_scan.py --dry      # только напечатать, файлы не трогать
       .venv/bin/python scripts/daily_scan.py --resnap   # пересохранить сегодняшний снимок без отчёта
"""
import csv, datetime as dt, io, json, os, re, subprocess, sys, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "runs" / "scan"
REPORT = ROOT / "docs" / "public_line_scan_daily.md"
COMP = "arc-prize-2026-arc-agi-3"
FORUM_ID = 10403401   # форум соревнования (внутренний API Kaggle, тот же, что в watch_publication.py)
BASE = "Flash-Next"
HF_AUTHORS = ["Qwen", "deepseek-ai", "google", "meta-llama", "microsoft", "mistralai", "openai", "nvidia", "moonshotai",
              "zai-org", "XiaomiMiMo", "tencent", "MiniMaxAI", "baidu", "stepfun-ai", "ByteDance-Seed", "yandex",
              "Intel", "RadixArk", "albucino", "ukisai", "primitive-ai", "local-inference-lab", "huikang"]
HF_NEXT_BASE_Q = ["Qwen4", "Qwen3.9", "Flash-Next-Instruct", "AliceAI"]
HF_TRENDING_TAGS = ["text-generation", "image-text-to-text"]
MODELSCOPE_Q = ["Qwen4", "Qwen3.9", "Flash-Next"]
PYPI = ["vllm", "sglang", "flashinfer-python", "torch", "transformers", "auto-round", "llmcompressor", "compressed-tensors",
        "arc-agi", "arcengine"]
GH_RELEASES = ["vllm-project/vllm", "sgl-project/sglang", "flashinfer-ai/flashinfer", "intel/auto-round",
               "vllm-project/llm-compressor", "NVIDIA/TensorRT-LLM"]
GH_REPOS = ["da-fr/arc-agi-3-solution", "LohitSiriki/arc-agi-3-milestone2-solution", "jpezzulli/sglang-rtxpro6000",
            "Tufalabs/duck-harness", "arcprize/ARC-AGI-3-Agents"]
GH_ORGS = ["arcprize", "Tufalabs"]
GH_PR_REPOS = ["vllm-project/vllm", "sgl-project/sglang"]
GH_PR_Q = ["Flash-Next", "Qwen4Exp"]
DOCKER = {"vllm/vllm-openai": r"^v?\d+\.\d+(\.\d+)?([-.]?rc\d+)?$|qwen|flash",
          "lmsysorg/sglang": r"^v\d+\.\d+\.\d+(-cu\d+)?$|qwen|flash|next"}
KG_DATASET_Q = ["flash-next", "qwen3.8", "sglang", "vllm", "taaf", "arc-agi-3", "pennyroyal"]
KG_MODEL_Q = ["Flash-Next", "Qwen3.8", "Qwen4", "Qwen 4"]
failed = []


def _token():
    for p in (Path.home() / ".kaggle" / "access_token", ROOT / ".kaggle" / "access_token"):
        if p.is_file() and p.stat().st_size:
            return p.read_text().strip()
    return ""


def kaggle(*args):
    env = dict(os.environ, KAGGLE_API_TOKEN=_token())
    out = subprocess.run([str(ROOT / ".venv/bin/kaggle"), *args, "--csv"], capture_output=True, text=True, env=env, timeout=180).stdout
    lines = out.splitlines()
    start = next((i for i, l in enumerate(lines) if re.match(r"^(ref|id|teamId),", l)), None)
    if start is None:
        raise RuntimeError((out or "пустой ответ")[:120])
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def get(url, raw=False, data=None, headers=None, method=None):
    h = {"User-Agent": "arc3-daily-scan"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    body = urllib.request.urlopen(req, timeout=40).read()
    return body.decode("utf-8", "replace") if raw else json.loads(body)


def section(name, fn):
    try:
        return fn()
    except Exception as e:  # источник недоступен — скан продолжается, отказ попадает в отчёт
        failed.append(f"{name}: {str(e)[:100]}")
        return None


def snap():
    s = {}
    # ---------- Kaggle
    s["leaderboard"] = section("таблица Kaggle", lambda: {r["teamName"]: float(r["score"]) for r in kaggle("competitions", "leaderboard", "-c", COMP, "-s")[:25]})

    def kernels():
        d = {}
        for sort, n in (("dateRun", "100"), ("voteCount", "50")):
            for r in kaggle("kernels", "list", "--competition", COMP, "--sort-by", sort, "--page-size", n):
                d[r["ref"]] = {"title": r["title"], "votes": int(r["totalVotes"] or 0), "run": r["lastRunTime"][:16]}
        return d
    s["kernels"] = section("кернелы Kaggle", kernels)

    def kmodels():
        d = {}
        for q in KG_MODEL_Q:
            for r in kaggle("models", "list", "-s", q, "--page-size", "50"):
                d[r["ref"]] = r["title"]
        return d
    s["kaggle_models"] = section("модели Kaggle", kmodels)

    def kdatasets():
        d = {}
        for q in KG_DATASET_Q:
            for r in kaggle("datasets", "list", "-s", q, "--sort-by", "updated"):
                d[r["ref"]] = {"updated": r["lastUpdated"][:16], "gb": round(int(r["size"] or 0) / 1e9, 2)}
        return d
    s["kaggle_datasets"] = section("датасеты Kaggle", kdatasets)

    def forum():
        topics = get("https://www.kaggle.com/api/i/discussions.DiscussionsService/GetTopicListByForumId",
                     data=json.dumps({"forumId": FORUM_ID, "sortBy": "RECENT", "page": 1, "pageSize": 40,
                                      "sort": "FORUM_TOPIC_LIST_SORT_BY_RECENT"}).encode(),
                     headers={"Authorization": "Bearer " + _token(), "Content-Type": "application/json"}).get("topics", [])
        return {str(t["id"]): {"title": t.get("title", ""), "author": (t.get("authorUser") or {}).get("displayName", ""),
                               "date": (t.get("postDate") or "")[:16], "comments": int(t.get("commentCount") or 0)} for t in topics}
    s["forum"] = section("форум соревнования", forum)

    # ---------- модели вне Kaggle
    def hf_base():
        d = {}
        for sort in ("downloads", "createdAt"):
            url = f"https://huggingface.co/api/models?search={urllib.parse.quote(BASE)}&sort={sort}&direction=-1&limit=100"
            for m in get(url):
                d[m["id"]] = {"downloads": m.get("downloads", 0), "created": (m.get("createdAt") or "")[:10]}
        return d
    s["hf_base"] = section("HF: упаковки базы", hf_base)

    def hf_new():
        d = {}
        for a in HF_AUTHORS:
            for m in get(f"https://huggingface.co/api/models?author={a}&sort=createdAt&direction=-1&limit=10"):
                d[m["id"]] = (m.get("createdAt") or "")[:10]
        for q in HF_NEXT_BASE_Q:
            for m in get(f"https://huggingface.co/api/models?search={urllib.parse.quote(q)}&sort=createdAt&direction=-1&limit=10"):
                d[m["id"]] = (m.get("createdAt") or "")[:10]
        return d
    s["hf_new"] = section("HF: новые модели авторов", hf_new)

    def hf_trending():
        d = {}
        for tag in HF_TRENDING_TAGS:
            for m in get(f"https://huggingface.co/api/models?sort=trendingScore&direction=-1&limit=30&pipeline_tag={tag}"):
                d[m["id"]] = (m.get("createdAt") or "")[:10]
        return d
    s["hf_trending"] = section("HF: набирающие популярность", hf_trending)

    def modelscope():
        d = {}

        def search(name, keep=lambda m: True, size=30):
            r = get("https://modelscope.cn/api/v1/dolphin/models", method="PUT", headers={"Content-Type": "application/json"},
                    data=json.dumps({"PageSize": size, "PageNumber": 1, "SortBy": "GmtModified", "Target": "", "Name": name}).encode())
            for m in ((r.get("Data") or {}).get("Model") or {}).get("Models") or []:
                if keep(m):
                    d[f"{m.get('Path')}/{m.get('Name')}"] = time.strftime("%Y-%m-%d", time.gmtime(int(m.get("CreatedTime") or 0)))
        search("Qwen", keep=lambda m: str(m.get("Path")).lower() == "qwen", size=100)   # всё новое у самой Qwen
        for q in MODELSCOPE_Q:
            search(q)
        return d
    s["modelscope"] = section("ModelScope", modelscope)

    # ---------- серверы и библиотеки
    def pypi():
        d = {}
        for pkg in PYPI:
            rel = get(f"https://pypi.org/pypi/{pkg}/json")["releases"]
            dated = sorted(((max(f["upload_time"] for f in files)[:10], v) for v, files in rel.items() if files), reverse=True)[:6]
            d[pkg] = {v: t for t, v in dated}
        return d
    s["pypi"] = section("PyPI", pypi)

    def releases():
        return {repo: [r["tag_name"] for r in get(f"https://api.github.com/repos/{repo}/releases?per_page=5")] for repo in GH_RELEASES}
    s["releases"] = section("релизы GitHub", releases)

    def docker():
        d = {}
        for repo, pattern in DOCKER.items():
            tags = get(f"https://hub.docker.com/v2/repositories/{repo}/tags?page_size=100&ordering=last_updated").get("results", [])
            d[repo] = {t["name"]: (t.get("last_updated") or "")[:10] for t in tags if re.search(pattern, t["name"], re.I)}
        return d
    s["docker"] = section("Docker Hub", docker)

    def server_prs():
        d = {}
        for repo in GH_PR_REPOS:
            for q in GH_PR_Q:
                url = ("https://api.github.com/search/issues?q=" + urllib.parse.quote(f'repo:{repo} "{q}" is:pr is:merged') +
                       "&sort=updated&order=desc&per_page=30")
                for i in get(url).get("items", []):
                    d[f"{repo}#{i['number']}"] = {"title": i["title"][:110], "merged": (i.get("closed_at") or "")[:10]}
                time.sleep(2)   # поиск GitHub без токена: 10 запросов в минуту
        return d
    s["server_prs"] = section("слитые правки серверов", server_prs)

    # ---------- обвязка и игра
    def repos():
        d = {}
        for repo in GH_REPOS:
            d[repo] = get(f"https://api.github.com/repos/{repo}").get("pushed_at", "")[:16]
        for org in GH_ORGS:
            for r in get(f"https://api.github.com/orgs/{org}/repos?sort=pushed&per_page=15"):
                d.setdefault(r["full_name"], (r.get("pushed_at") or "")[:16])
        q = "https://api.github.com/search/repositories?q=arc-agi-3&sort=updated&order=desc&per_page=40"
        for r in get(q).get("items", []):
            d.setdefault(r["full_name"], "search:" + (r.get("created_at") or "")[:10])
        return d
    s["repos"] = section("репозитории GitHub", repos)

    # ---------- статьи
    def arxiv():
        d = {}
        for q, n in (('all:"ARC-AGI-3"', 25), ('all:"ARC-AGI"', 15)):
            url = "https://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(q) + f"&sortBy=submittedDate&sortOrder=descending&max_results={n}"
            x = get(url, raw=True)
            for i, t in re.findall(r"<entry>.*?<id>(.*?)</id>.*?<title>(.*?)</title>", x, re.S):
                d[i.rsplit("/", 1)[-1]] = " ".join(t.split())
            time.sleep(3)
        return d
    s["arxiv"] = section("arXiv", arxiv)
    return s


def diff(old, new):
    out = []

    def added(key, label, fmt=lambda k, v: f"`{k}`", limit=40):
        a, b = old.get(key), new.get(key)
        if a is None or b is None:
            return
        fresh = [k for k in b if k not in a]
        if fresh:
            out.append(f"**{label}: новых {len(fresh)}**")
            out.extend(f"- {fmt(k, b[k])}" for k in fresh[:limit])
            if len(fresh) > limit:
                out.append(f"- … и ещё {len(fresh) - limit}")

    def nested(key, label, fmt):
        a, b = old.get(key), new.get(key)
        if a is None or b is None:
            return
        rows = [fmt(group, k, v) for group in b for k, v in (b[group].items() if isinstance(b[group], dict) else [(t, "") for t in b[group]])
                if k not in (a.get(group) or {})]
        if rows:
            out.append(f"**{label}**"); out.extend(f"- {r}" for r in rows[:40])

    a, b = old.get("leaderboard"), new.get("leaderboard")
    if a and b:
        moves = [f"- {t}: {a[t]:.2f} → {b[t]:.2f}" for t in b if t in a and abs(b[t] - a[t]) > 1e-9]
        moves += [f"- {t}: вошёл в топ-25 с {b[t]:.2f}" for t in b if t not in a]
        if moves:
            out.append("**Таблица: изменения в топ-25**"); out.extend(moves)
    added("kernels", "Кернелы соревнования", lambda k, v: f"`{k}` — {v['title']} (голосов {v['votes']}, запуск {v['run']})")
    a, b = old.get("kernels"), new.get("kernels")
    if a and b:
        jumps = [f"- `{k}`: голосов {a[k]['votes']} → {b[k]['votes']}" for k in b if k in a and b[k]["votes"] - a[k]["votes"] >= 5]
        if jumps:
            out.append("**Кернелы: рост голосов на 5 и больше**"); out.extend(jumps)
    added("forum", "Форум: новые темы", lambda k, v: f"тема {k} — {v['title']} ({v['author']}, {v['date']}, комментариев {v['comments']})")
    a, b = old.get("forum"), new.get("forum")
    if a and b:
        grown = [f"- тема {k} — {v['title']}: комментариев {a[k]['comments']} → {v['comments']}" for k, v in b.items()
                 if k in a and v["comments"] > a[k]["comments"]]
        if grown:
            out.append("**Форум: новые комментарии**"); out.extend(grown)
    added("kaggle_models", "Модели Kaggle", lambda k, v: f"`{k}` — {v}")
    added("kaggle_datasets", "Датасеты Kaggle", lambda k, v: f"`{k}` — {v['gb']} ГБ, обновлён {v['updated']}")
    a, b = old.get("kaggle_datasets"), new.get("kaggle_datasets")
    if a and b:
        upd = [f"- `{k}`: {a[k]['updated']} → {b[k]['updated']}" for k in b if k in a and a[k]["updated"] != b[k]["updated"]]
        if upd:
            out.append("**Датасеты Kaggle: обновлены**"); out.extend(upd[:30])
    added("hf_base", f"HF: упаковки {BASE}", lambda k, v: f"`{k}` — создана {v['created']}, скачиваний {v['downloads']:,}")
    added("hf_new", "HF: новые модели отслеживаемых авторов", lambda k, v: f"`{k}` — {v}")
    added("hf_trending", "HF: вошли в список набирающих популярность", lambda k, v: f"`{k}` — создана {v}")
    added("modelscope", "ModelScope", lambda k, v: f"`{k}` — создана {v}")
    nested("pypi", "PyPI: новые версии", lambda pkg, ver, date: f"{pkg} {ver} ({date})")
    nested("releases", "GitHub: новые релизы", lambda repo, tag, _: f"{repo}: {tag}")
    nested("docker", "Docker Hub: новые образы", lambda repo, tag, date: f"{repo}:{tag} ({date})")
    added("server_prs", "Слитые правки серверов с упоминанием нашей модели", lambda k, v: f"{k} — {v['title']} (слито {v['merged']})")
    a, b = old.get("repos"), new.get("repos")
    if a and b:
        rp = [f"- `{k}`: новый ({v})" for k, v in b.items() if k not in a]
        rp += [f"- `{k}`: пуш {a[k]} → {v}" for k, v in b.items() if k in a and not v.startswith("search:") and not a[k].startswith("search:") and a[k] != v]
        if rp:
            out.append("**Репозитории GitHub**"); out.extend(rp[:30])
    added("arxiv", "arXiv", lambda k, v: f"{k} — {v}")
    return out


def baseline(s):
    out = ["Первый снимок — базовая линия, сравнивать не с чем. Состояние источников:"]
    if s.get("leaderboard"):
        out.append("**Таблица, топ-12:** " + "; ".join(f"{t} {v:.2f}" for t, v in list(s["leaderboard"].items())[:12]))
    if s.get("kernels"):
        ks = sorted(s["kernels"].items(), key=lambda kv: kv[1]["run"], reverse=True)[:15]
        out.append("**Последние запуски кернелов:**"); out.extend(f"- `{k}` — {v['title']} ({v['run']}, голосов {v['votes']})" for k, v in ks)
    if s.get("forum"):
        out.append("**Форум, свежие темы:**")
        out.extend(f"- тема {k} — {v['title']} ({v['date']}, комментариев {v['comments']})" for k, v in sorted(s["forum"].items(), key=lambda kv: kv[1]["date"], reverse=True)[:8])
    if s.get("hf_base"):
        hb = s["hf_base"]
        out.append(f"**HF, упаковки {BASE}: всего {len(hb)}; новейшие:**")
        out.extend(f"- `{k}` — {v['created']}, скачиваний {v['downloads']:,}" for k, v in sorted(hb.items(), key=lambda kv: kv[1]["created"], reverse=True)[:12])
    if s.get("hf_new"):
        out.append("**HF, новейшие модели отслеживаемых авторов:**")
        out.extend(f"- `{k}` — {v}" for k, v in sorted(s["hf_new"].items(), key=lambda kv: kv[1], reverse=True)[:12])
    if s.get("hf_trending"):
        out.append("**HF, новейшие из набирающих популярность:**")
        out.extend(f"- `{k}` — {v}" for k, v in sorted(s["hf_trending"].items(), key=lambda kv: kv[1], reverse=True)[:10])
    if s.get("modelscope"):
        out.append("**ModelScope, новейшее:**")
        out.extend(f"- `{k}` — {v}" for k, v in sorted(s["modelscope"].items(), key=lambda kv: kv[1], reverse=True)[:8])
    if s.get("pypi"):
        out.append("**PyPI, последние версии:** " + "; ".join(f"{p} {next(iter(v))} ({next(iter(v.values()))})" for p, v in s["pypi"].items() if v))
    if s.get("releases"):
        out.append("**Релизы GitHub:** " + "; ".join(f"{r.split('/')[1]} {', '.join(t[:3])}" for r, t in s["releases"].items()))
    if s.get("docker"):
        out.append("**Docker Hub, свежие образы:** " + "; ".join(f"{r}: {', '.join(list(t)[:4])}" for r, t in s["docker"].items()))
    if s.get("server_prs"):
        out.append("**Слитые правки серверов с упоминанием нашей модели, последние:**")
        out.extend(f"- {k} — {v['title']} ({v['merged']})" for k, v in sorted(s["server_prs"].items(), key=lambda kv: kv[1]["merged"], reverse=True)[:10])
    if s.get("arxiv"):
        out.append("**arXiv, новейшее:**"); out.extend(f"- {k} — {v}" for k, v in list(s["arxiv"].items())[:6])
    return out


if __name__ == "__main__":
    dry = "--dry" in sys.argv
    now = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=3)
    day = now.strftime("%Y-%m-%d")
    cur = snap()
    if "--resnap" in sys.argv:
        SNAP.mkdir(parents=True, exist_ok=True)
        (SNAP / f"{day}.json").write_text(json.dumps(cur, ensure_ascii=False, indent=1))
        print(f"снимок {day} пересохранён; разделов {sum(v is not None for v in cur.values())} из {len(cur)}; не прочитано: {failed or 'нет'}")
        sys.exit(0)
    prev_files = sorted(p for p in SNAP.glob("*.json") if p.stem < day) if SNAP.is_dir() else []
    if prev_files:
        old = json.loads(prev_files[-1].read_text())
        body = diff(old, cur) or ["Изменений нет."]
        head = f"## Скан {now:%d.%m.%Y %H:%M} МСК (против снимка {prev_files[-1].stem})"
    else:
        body = baseline(cur)
        head = f"## Скан {now:%d.%m.%Y %H:%M} МСК"
    if failed:
        body += ["**Не удалось прочитать:**"] + [f"- {f}" for f in failed]
    text = head + "\n\n" + "\n".join(body) + "\n"
    print(text)
    if not dry:
        SNAP.mkdir(parents=True, exist_ok=True)
        (SNAP / f"{day}.json").write_text(json.dumps(cur, ensure_ascii=False, indent=1))
        title = "# Ежедневный скан публичной линии\n\nСкрипт: `scripts/daily_scan.py`. Новые сканы сверху.\n\n"
        prior = REPORT.read_text() if REPORT.is_file() else title
        prior = prior[len(title):] if prior.startswith(title) else prior
        REPORT.write_text(title + text + "\n" + prior)

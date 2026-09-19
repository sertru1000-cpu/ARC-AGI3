"""Скачивание результата пробы с Kaggle и сводка против записанных порогов (19.09).

Зачем отдельный скрипт. После каждой пробы одно и то же: забрать вывод кернела, забрать лог (он идёт
ДРУГИМ вызовом -- `kernels_logs`, в `kernels output` его нет), посчитать уровни и игры с уровнем,
сверить с обрезкой баз и поискать в логе следы механизма. Руками это десять минут и место для ошибки
в точке сравнения.

Точка сравнения для потолка 1800 с на игру (обрезка трёх одинаковых баз на Qwen, scripts/truncate_run.py):
уровней 17 / 16 / 12, игр с уровнем 13 / 13 / 12, балл 3.32 / 2.73 / 1.67.

usage:
    .venv/bin/python scripts/fetch_probe.py sergueimakarov/arc3-dsv4-duck --out runs/dsv4_duck_v7
    .venv/bin/python scripts/fetch_probe.py <kernel> --out <dir> --no-download   # уже скачано, только сводка
"""
import argparse, json, os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE30 = {"levels": [17, 16, 12], "games": [13, 13, 12], "score": [3.32, 2.73, 1.67]}


def kaggle(*args):
    env = dict(os.environ)
    if "KAGGLE_API_TOKEN" not in env:
        tok = ROOT / ".kaggle/access_token"
        if tok.exists():
            env["KAGGLE_API_TOKEN"] = tok.read_text().strip()
    return subprocess.run([str(ROOT / ".venv/bin/kaggle"), *args], capture_output=True, text=True, env=env)


def fetch_log(kernel: str, out: Path) -> Path | None:
    """лог кернела отдаётся отдельным вызовом SDK, в выводе его нет (память arc-agi-3-kaggle-log-retrieval)"""
    tok = ROOT / ".kaggle/access_token"
    if tok.exists():
        os.environ.setdefault("KAGGLE_API_TOKEN", tok.read_text().strip())
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        user, slug = kernel.split("/", 1)
        data = api.kernels_output(user, slug, path=str(out), force=True, quiet=True)
        del data
    except Exception as exc:
        print("лог: %r" % (exc,))
    cands = sorted(out.glob("*.log")) + sorted(out.glob("**/*.log"))
    return cands[0] if cands else None


def summarize(out: Path) -> None:
    bench = out / "benchmark.json"
    if not bench.exists():
        found = list(out.glob("**/benchmark.json"))
        bench = found[0] if found else None
    if not bench:
        print("benchmark.json не найден -- прогон, вероятно, не дошёл до конца игр")
    else:
        b = json.loads(bench.read_text(encoding="utf-8"))
        rows = []
        for r in b["game_runs"]:
            rows.append((r["game_id"][:4], r.get("levels_completed", 0), r.get("number_of_levels"),
                         len(r.get("history") or []), r.get("final_score"), r.get("final_generated_tokens"),
                         r.get("state")))
        rows.sort(key=lambda x: -(x[1] or 0))
        print("%-6s %-8s %-7s %-9s %-7s %s" % ("игра", "уровней", "ходов", "токенов", "балл", "состояние"))
        for g, L, T, A, S, TK, st in rows:
            print("%-6s %d/%-6s %-7d %-9s %-7s %s" % (g, L or 0, T, A, TK, ("%.2f" % S) if S is not None else "-", st))
        lv = sum(r[1] or 0 for r in rows)
        gw = sum(1 for r in rows if (r[1] or 0) >= 1)
        sc = [r[4] for r in rows if r[4] is not None]
        avg = sum(sc) / len(sc) if sc else 0.0
        sj = out / "score.json"
        if not sj.exists():
            found = list(out.glob("**/score.json")); sj = found[0] if found else None
        frozen = json.loads(sj.read_text(encoding="utf-8")).get("score") if sj and sj.exists() else None
        print("\nИТОГ: игр %d, уровней %d, игр с уровнем %d, средний балл %.2f%s"
              % (len(rows), lv, gw, avg,
                 "" if frozen is None else " (замороженный счётчик score.json: %.2f -- судим по нему)" % frozen))
        print("БАЗА-30 (три обрезки): уровней %s, игр с уровнем %s, балл %s" % (BASE30["levels"], BASE30["games"], BASE30["score"]))
        verdict = "СИГНАЛ" if lv >= 21 else ("ВРЕД" if lv <= 11 else "НЕОТЛИЧИМО ОТ БАЗЫ")
        print("ПО ЗАПИСАННОМУ ПОРОГУ (уровни: сигнал >= 21, вред <= 11): %s" % verdict)

    log = None
    for p in list(out.glob("*.log")) + list(out.glob("**/*.log")):
        log = p; break
    if log:
        t = log.read_text(encoding="utf-8", errors="replace")
        print("\nмеханизм по логу (%s, %.1f МБ):" % (log.name, len(t) / 1e6))
        for pat, name in (
            (r"\[\[DSV4\]\] сервер готов: (\w+) за ([\d.]+) с", "сервер"),
            (r"\[\[DSV4\]\] игр в пробе: (\d+)", "игр в пробе"),
        ):
            m = re.search(pat, t)
            print("  %-14s %s" % (name, m.group(0) if m else "НЕ НАЙДЕНО"))
        fails = len(re.findall(r"analyzer request failed", t))
        toolfail = len(re.findall(r"Failed to parse tool call", t))
        print("  %-14s %d (из них «не разобран вызов инструмента» %d)" % ("сбоев анализатора", fails, toolfail))
    else:
        print("\nлога в выводе нет (kernels output его не отдаёт, если прогон ещё идёт)")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("kernel"); ap.add_argument("--out", required=True)
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    if not a.no_download:
        r = kaggle("kernels", "output", a.kernel, "-p", str(out), "--force")
        print((r.stdout or r.stderr or "").strip()[-400:])
        fetch_log(a.kernel, out)
    summarize(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

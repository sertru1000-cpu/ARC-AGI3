"""Разбор пробы слоя «модель мира» (wm v5..v10): техника, механизм, игра.

Зачем отдельно от wm_autopsy.py: тот написан под формат v2 и считает точность
гипотез. Начиная с v5 приём ТОЧНЫЙ (принята/не принята), появились цель, план и
фоновый синтезатор, а вопрос к прогону стал другим: не «какой балл», а
«жив ли механизм и во сколько обошлась интеграция».

Источники, каждый называется в отчёте (см. feedback-claims-need-sources):
  * vllm-metrics-final.prom  — запросы, очередь, вытеснения, токены (ТЕХНИКА);
  * transcripts/*.txt        — пер-игровые отметки WM_CHECK/WM_GOAL/WM_PLAN,
                               потому что они попадают в промпт следующего хода;
  * главный лог кернела      — события фонового синтезатора (v7/v8: его песочница
                               оффлайновая, в транскрипт не попадает);
  * benchmark.json           — действия, уровни, балл, состояние по каждой игре.

Пороги НЕ зашиты: они пишутся до пуска в docs/plan_top10_by_3009.md. Скрипт даёт
числа, вердикт выносит критик (.claude/skills/run-verdict-critic).

usage:
    .venv/bin/python scripts/wm_probe_report.py runs/flash_wm_v9
    .venv/bin/python scripts/wm_probe_report.py runs/flash_wm_v8 --base docs/base30_flash_v1.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MET = re.compile(r"^(?P<name>vllm:[a-z_0-9]+)\{(?P<labels>[^}]*)\}\s+(?P<val>[-\deE.+]+)$", re.M)


def log_text(run: Path) -> str:
    """Главный лог кернела: JSON-массив кусков потоков -> сплошной текст."""
    logs = [p for p in run.glob("*.log") if "vllm" not in p.name]
    if not logs:
        return ""
    raw = logs[0].read_text(encoding="utf-8", errors="replace")
    try:
        return "".join(str(c.get("data", "")) for c in json.loads(raw))
    except Exception:
        return "".join(json.loads('"' + m + '"') for m in re.findall(r'"data":"((?:[^"\\]|\\.){0,400})"', raw)) or raw


def metrics(run: Path) -> dict[str, float]:
    p = run / "vllm-metrics-final.prom"
    if not p.exists():
        return {}
    out: dict[str, float] = {}
    for m in MET.finditer(p.read_text(encoding="utf-8", errors="replace")):
        key = m.group("name")
        for lab in ("finished_reason", "reason", "source"):
            # (?:^|,) — иначе "reason" находится внутри "finished_reason" и метка удваивается
            v = re.search(r'(?:^|,)' + lab + r'="([^"]*)"', m.group("labels"))
            if v:
                key += f"[{lab}={v.group(1)}]"
        out[key] = out.get(key, 0.0) + float(m.group("val"))
    return out


def hist(met: dict[str, float], base: str) -> tuple[float, float]:
    """Среднее по гистограмме vLLM: sum/count."""
    s = met.get(f"vllm:{base}_seconds_sum", 0.0)
    c = met.get(f"vllm:{base}_seconds_count", 0.0)
    return (s / c if c else 0.0, c)


def per_game(run: Path) -> dict[str, dict]:
    """Пер-игровые отметки механизма из транскриптов + счётчики игры из benchmark."""
    games: dict[str, dict] = {}
    b = run / "benchmark.json"
    if b.exists():
        for r in json.load(open(b, encoding="utf-8"))["game_runs"]:
            gid = str(r["game_id"])[:4]
            acts = r.get("actions_per_level") or "[]"
            try:
                acts = sum(int(x) for x in json.loads(str(acts)))
            except Exception:
                acts = 0
            games[gid] = {
                "actions": acts,
                "levels": int(r.get("levels_completed") or 0),
                "total_levels": int(r.get("number_of_levels") or 0),
                "score": float(r.get("final_score") or 0.0),
                "state": str(r.get("state") or ""),
                "seconds": float(r.get("final_wallclock_seconds") or 0.0),
            }
    for t in sorted((run / "transcripts").glob("*.txt")) if (run / "transcripts").exists() else []:
        gid = t.name[:4]
        txt = t.read_text(encoding="utf-8", errors="replace")
        g = games.setdefault(gid, {})
        g["turns"] = txt.count("--- analysis_step=")
        g["admitted"] = len(re.findall(r"WM_CHECK admitted=1", txt))
        g["checks"] = len(re.findall(r"WM_CHECK admitted=", txt))
        g["goal_ok"] = len(re.findall(r"WM_GOAL ok=1", txt))
        g["goal_try"] = len(re.findall(r"WM_GOAL ok=", txt))
        g["plan_found"] = len(re.findall(r"WM_PLAN found", txt))
        g["exec_ok"] = len(re.findall(r"WM_EXEC ok", txt))
        g["exec_bad"] = len(re.findall(r"WM_EXEC mismatch", txt))
        g["cex"] = len(re.findall(r"WM_COUNTEREXAMPLE", txt))
        g["rejected"] = len(re.findall(r"Ход отклонён", txt))
        g["wrote_predict"] = len(re.findall(r"def\s+predict\s*\(", txt))
        g["wrote_goal"] = len(re.findall(r"def\s+goal\s*\(", txt))
    return games


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--base", default="docs/base30_flash_v1.json",
                    help="пер-игровые баллы точки сравнения (по умолчанию база-30, среднее 3.06)")
    a = ap.parse_args()
    run = Path(a.run)
    if not run.exists():
        print(f"{run}: каталога нет — сначала `kaggle kernels output`")
        return 1

    text = log_text(run)
    ver = re.search(r"wm v(\d+) installed", text)
    print(f"ПРОГОН {run}   слой: {'wm v' + ver.group(1) if ver else 'НЕ УСТАНОВЛЕН — дальше всё бессмысленно'}")
    end = re.search(r"WM(\d) PROBE: потолок ([\d.]+) с", text)
    if end:
        print(f"проба: потолок {end.group(2)} с на игру")

    met = metrics(run)
    if met:
        reqs = sum(v for k, v in met.items() if k.startswith("vllm:request_success_total"))
        qavg, qn = hist(met, "request_queue_time")
        eavg, _ = hist(met, "e2e_request_latency")
        print("\nТЕХНИКА (vllm-metrics-final.prom) — цена интеграции")
        print(f"  запросов завершено      {reqs:.0f}")
        for k, v in sorted(met.items()):
            if k.startswith("vllm:request_success_total[") and v:
                reason = k[k.index("=") + 1:].rstrip("]")
                print(f"    причина конца: {reason:<12} {v:.0f}")
        print(f"  очередь, среднее        {qavg:.1f} с   (наблюдений {qn:.0f})")
        print(f"  задержка запроса        {eavg:.1f} с")
        print(f"  вытеснений              {met.get('vllm:num_preemptions_total', 0):.0f}")
        print(f"  токенов промпта         {met.get('vllm:prompt_tokens_total', 0):,.0f}")
        print(f"  токенов генерации       {met.get('vllm:generation_tokens_total', 0):,.0f}")

    ev = {
        "синтезатор: программа принята": len(re.findall(r"\[WM\] СИНТЕЗАТОР: программа принята", text)),
        "синтезатор: не принята": len(re.findall(r"\[WM\] синтезатор: программа не принята", text)),
        "синтезатор: вернул не программу": len(re.findall(r"\[WM\] синтезатор вернул не программу", text)),
        "синтезатор: упал": len(re.findall(r"\[WM\] синтезатор упал", text)),
        "ход без действия отклонён": len(re.findall(r"\[WM\] ход без действия отклонён", text)),
    }
    if any(ev.values()):
        print("\nСОБЫТИЯ ГЛАВНОГО ЛОГА (фоновый синтезатор в транскрипт не попадает)")
        for k, v in ev.items():
            if v:
                print(f"  {k:<34} {v}")

    games = per_game(run)
    if not games:
        print("\nbenchmark.json и транскриптов нет — прогон не отдал результатов")
        return 0

    print(f"\nМЕХАНИЗМ ПО ИГРАМ (транскрипты, {len(games)} игр)")
    hdr = ("игра", "ход", "дейст", "ур", "балл", "прин/пров", "цель", "план", "исп", "кпр", "откл")
    print("  %-5s %4s %6s %4s %7s %10s %6s %5s %4s %4s %5s" % hdr)
    for gid in sorted(games):
        g = games[gid]
        print("  %-5s %4s %6s %4s %7.2f %5s/%-4s %6s %5s %4s %4s %5s" % (
            gid, g.get("turns", "-"), g.get("actions", "-"),
            f"{g.get('levels', 0)}/{g.get('total_levels', 0)}", g.get("score", 0.0),
            g.get("admitted", "-"), g.get("checks", "-"), g.get("goal_ok", "-"),
            g.get("plan_found", "-"), g.get("exec_ok", "-"), g.get("cex", "-"), g.get("rejected", "-")))

    n = len(games)
    with_prog = sum(1 for g in games.values() if g.get("admitted"))
    with_goal = sum(1 for g in games.values() if g.get("goal_ok"))
    with_plan = sum(1 for g in games.values() if g.get("plan_found"))
    with_exec = sum(1 for g in games.values() if g.get("exec_ok"))
    took_l1 = sum(1 for g in games.values() if g.get("levels", 0) >= 1)
    took_l2 = sum(1 for g in games.values() if g.get("levels", 0) >= 2)
    acts = [g.get("actions", 0) for g in games.values()]
    print("\nСВОДКА МЕХАНИЗМА")
    print(f"  принятая программа в    {with_prog}/{n} играх")
    print(f"  проверенная цель в      {with_goal}/{n} играх")
    print(f"  найденный план в        {with_plan}/{n} играх   (исполнен в {with_exec})")
    print(f"  взят первый уровень в   {took_l1}/{n} играх   <- потолок для цели и плана")
    print(f"  взят второй уровень в   {took_l2}/{n} играх")
    print(f"  действий на игру        среднее {sum(acts)/n:.1f}, медиана {sorted(acts)[n//2]}")

    scores = {g: v.get("score", 0.0) for g, v in games.items()}
    mean = sum(scores.values()) / n
    print(f"\nБАЛЛ (benchmark.json): среднее {mean:.2f} по {n} играм; уровней всего "
          f"{sum(v.get('levels', 0) for v in games.values())}")
    bp = Path(a.base)
    if bp.exists():
        base = json.load(open(bp, encoding="utf-8"))
        common = sorted(set(base) & set(scores))
        if common:
            bm = sum(base[g] for g in common) / len(common)
            nm = sum(scores[g] for g in common) / len(common)
            wins = [g for g in common if scores[g] - base[g] > 0.5]
            loss = [g for g in common if base[g] - scores[g] > 0.5]
            print(f"  точка сравнения {bp.name}: {bm:.2f} против нашей {nm:.2f} на {len(common)} общих играх")
            print(f"  выигранных игр {len(wins)} ({' '.join(wins) or '-'}), проигранных {len(loss)} ({' '.join(loss) or '-'})")
            print("  ОДИН прогон: разница меньше 1 балла в этой метрике неразличима от шума")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

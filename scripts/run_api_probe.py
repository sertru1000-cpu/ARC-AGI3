"""Прогон боевой обвязки Duck ЛОКАЛЬНО на внешней модели через API (19.09, по слову владельца «может мы на API
попробуем сначала?»).

Зачем. Вся затея с DeepSeek-V4-Flash в ядре держится на непроверенном допущении: «модель сильнее -> уровней больше».
Проверка этого допущения на Kaggle стоит квоты (30 минут пробы = час машины) и упирается в возню с llama.cpp.
Через API то же самое меряется на маке за копейки: обвязка та же самая (бандл duck_smoke_live, тот, что уходит в бой),
движок тот же (environment_files, наш детерминированный повтор боя), меняется ТОЛЬКО ядро-модель.

Чего эта проба НЕ даёт: модель по API не может быть сабмитом -- в боевом ядре интернета нет. Это измерительный
инструмент, а не путь к баллу. Если выигрыша нет -- линия «сильная модель в ядре» закрывается бесплатно; если есть --
тогда имеет смысл тратить квоту на сборку с llama.cpp.

Сравнивать с базой-30 (1.67 / 2.73 / 3.32) можно только при том же потолке на игру и том же наборе игр.

usage:
  .venv/bin/python scripts/run_api_probe.py --games tu93,ft09,lp85,sp80 --cap 1800 --model gemini-3.6-flash
"""
import argparse, asyncio, json, os, pickle, sys, time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# --bundle harness/duck -- гонять НАШ форк обвязки (scripts/duck_fork.py); по умолчанию -- боевой бандл как есть.
# Путь импорта настраивается ДО импорта модулей обвязки, поэтому ключ читается отсюда, а не из argparse.
_b = sys.argv[sys.argv.index("--bundle") + 1] if "--bundle" in sys.argv else "runs/peer_kernels/duck_smoke_live"
BUNDLE = (ROOT / _b) if not Path(_b).is_absolute() else Path(_b)
SRC = BUNDLE / "src"
for p in (SRC / "ARC3-Inference", SRC / "tufa-arc-agi-framework/src"):
    sys.path.insert(0, str(p))


def load_env_file() -> None:
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="tu93,ft09,lp85,sp80")
    ap.add_argument("--cap", type=float, default=1800.0, help="потолок на игру, секунд")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--base-url", default="https://generativelanguage.googleapis.com/v1beta/openai")
    ap.add_argument("--api-key-env", default="LLM_API_KEY_AISTUDIO")
    ap.add_argument("--provider", default="openrouter")
    # ЛОВУШКА: в бою окно 32768 (serving_setup.ANALYZER_CONTEXT, видно в runs/flash_v1_phaseA/taaf_setup_env.json).
    # Прежний умолчательный 131072 приехал из кернелов DeepSeek, где это ОБЩИЙ контекст llama-server на 4 слота
    # (на разговор там выходило те же 131072/4). Обвязка режет историю под это число: большее окно = длиннее
    # промпт, другая память агента и другая цена, то есть замер не про бой.
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--max-output", type=int, default=2048)
    ap.add_argument("--out", default="")
    ap.add_argument("--bundle", default="runs/peer_kernels/duck_smoke_live", help="бандл обвязки; harness/duck -- наш форк")
    ap.add_argument("--min-balance", type=float, default=1.0, help="минимальный остаток на счёте OpenRouter, $")
    ap.add_argument("--max-calls", type=int, default=0, help="жёсткий потолок числа вызовов модели (0 -- без потолка)")
    ap.add_argument("--price-in", type=float, default=0.0, help="$/млн входных токенов -- для оценки трат на ходу")
    ap.add_argument("--price-out", type=float, default=0.0, help="$/млн выходных токенов")
    ap.add_argument("--reasoning", choices=["default", "on", "off"], default="default",
                    help="рассуждение модели у провайдера: off -- дописать в запрос reasoning.enabled=false (OpenRouter)")
    a = ap.parse_args()

    load_env_file()
    key = os.environ.get(a.api_key_env, "").strip()
    if not key:
        print("нет ключа в переменной %s" % a.api_key_env)
        return 2

    # ЛОВУШКА 19.09: на бесплатном уровне OpenRouter (total_credits = 0) сервис отпускает ~$0.15 в долг, а дальше
    # отвечает 402, и обвязка полчаса «играет» отказами -- оба прогона DeepSeek так и испортились. Проверяем счёт ДО старта.
    if "openrouter.ai" in a.base_url:
        import urllib.request as _u
        def _get(path):
            rq = _u.Request("https://openrouter.ai/api/v1/" + path, headers={"Authorization": "Bearer " + key})
            return json.load(_u.urlopen(rq, timeout=30)).get("data", {})
        try:
            credits = _get("credits"); kinfo = _get("key")
            bal = float(credits.get("total_credits", 0)) - float(credits.get("total_usage", 0))
            print("OpenRouter: кредитов $%.2f, потрачено $%.2f, остаток $%.2f, лимит ключа %s"
                  % (float(credits.get("total_credits", 0)), float(credits.get("total_usage", 0)), bal, kinfo.get("limit_remaining")), flush=True)
            if bal < a.min_balance:
                print("ОТКАЗ: остаток на счёте $%.2f меньше порога $%.2f -- пополните OpenRouter (иначе прогон измерит отказы 402)"
                      % (bal, a.min_balance)); return 4
        except Exception as exc:
            print("счёт OpenRouter проверить не удалось: %r" % (exc,))

    # обвязка ходит в модель по совместимому протоколу -- ей всё равно, кто за ним стоит
    os.environ["LOCAL_ANALYZER_BASE_URL"] = a.base_url
    os.environ["LOCAL_ANALYZER_MODEL_ID"] = a.model
    os.environ["INFERENCE_ANALYZER_MODEL"] = a.model
    os.environ["LOCAL_ANALYZER_API_KEY"] = key
    # ЛОВУШКА: режим "openai"/"vllm" дописывает в запрос top_k и chat_template_kwargs -- чужой сервер их не знает
    # и отвечает 400. Режим "openrouter" шлёт голый совместимый запрос, он подходит любому внешнему API.
    os.environ["LOCAL_ANALYZER_PROVIDER"] = a.provider
    os.environ["LOCAL_ANALYZER_CONTEXT_WINDOW"] = str(a.context)
    if a.context != 32768:
        print("ВНИМАНИЕ: окно контекста %d, в бою 32768 -- замер не сравним с боем" % a.context, flush=True)
    os.environ["LOCAL_ANALYZER_MAX_OUTPUT"] = str(a.max_output)
    os.environ.setdefault("RECORDINGS_DIR", str(ROOT / "runs/_api_recordings"))

    out = Path(a.out) if a.out else ROOT / ("runs/api_probe_%s_%s" % (a.model.replace("/", "_"), datetime.now().strftime("%d%m_%H%M")))
    out.mkdir(parents=True, exist_ok=True)

    import arc_agi
    import taaf.game_api

    with open(BUNDLE / "deploy_target.pkl", "rb") as f:
        target = pickle.load(f)
    target.actual_run_as_submission = False
    target.is_competition_rerun = False
    with open(BUNDLE / "benchmark_initial.pkl", "rb") as f:
        bm = pickle.load(f)
    bm.job_dir = out

    # ОГРАНИЧИТЕЛЬ ТРАТ (20.09; пилот учителя ушёл в минус на $3.75 сверх внесённых $5 -- Google дал перерасход).
    # Считаем вызовы и токены сами, поверх сборщика запроса; по достижении потолка запросы прекращаются.
    if a.max_calls > 0 or a.price_in or a.price_out:
        import inference.agent.tool_agent as _ta
        _state = {"calls": 0, "in": 0, "out": 0, "stopped": False}
        _orig_post = _ta.requests.post
        def _counted_post(url, **kw):
            if a.max_calls and _state["calls"] >= a.max_calls:
                if not _state["stopped"]:
                    _state["stopped"] = True
                    print("ОГРАНИЧИТЕЛЬ: достигнут потолок %d вызовов -- дальше запросы не шлём" % a.max_calls, flush=True)
                raise RuntimeError("достигнут потолок вызовов (--max-calls)")
            _state["calls"] += 1
            r = _orig_post(url, **kw)
            try:
                u = r.json().get("usage") or {}
                _state["in"] += int(u.get("prompt_tokens") or 0)
                # ЛОВУШКА (измерено 20.09 родным API Gemini против совместимого): completion_tokens НЕ включает
                # токены размышления, а платим мы за них как за выход. Тот же запрос: completion_tokens 528,
                # total_tokens 8076 при промпте 75 -- 7473 токена спрятаны. Поэтому выход считаем по total.
                _c = int(u.get("completion_tokens") or 0); _t = int(u.get("total_tokens") or 0)
                _state["out"] += max(_c, _t - int(u.get("prompt_tokens") or 0))
                # кэш: сколько входных токенов сервис засчитал как прочитанные из кэша (в 10 раз дешевле)
                _state["cached"] = _state.get("cached", 0) + int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            except Exception:
                pass
            if _state["calls"] % 10 == 0:
                cached = _state.get("cached", 0); fresh = max(0, _state["in"] - cached)
                cost = (fresh * a.price_in + cached * a.price_in * 0.1) / 1e6 + _state["out"] / 1e6 * a.price_out
                # Причина прежнего занижения вдвое найдена (см. выше: спрятанное размышление). Ориентир по факту
                # 20.09: $0.024 за вызов у gemini-3.6-flash при боевом окне.
                print("ОГРАНИЧИТЕЛЬ: вызовов %d, вход %d (из кэша %d = %.0f%%), выход %d%s"
                      % (_state["calls"], _state["in"], cached, 100.0 * cached / max(1, _state["in"]), _state["out"],
                         (", $%.2f (выход с размышлением)" % cost) if (a.price_in or a.price_out) else ""), flush=True)
            return r
        _ta.requests.post = _counted_post

    # Рассуждение (19.09, слово владельца «thinking выключи»). В режиме "openrouter" обвязка шлёт голый запрос
    # без переключателя рассуждения, а DeepSeek на OpenRouter по умолчанию рассуждает. Проверено запросом:
    # "reasoning": {"enabled": false} даёт 0 токенов рассуждения, вызов инструмента сохраняется.
    # Подменяем сборщик запроса только в стенде: боевой код обвязки не трогается.
    if a.reasoning != "default":
        import inference.agent.tool_agent as _ta
        _orig = _ta.build_chat_payload
        def _with_reasoning(*args, **kwargs):
            payload = _orig(*args, **kwargs)
            payload["reasoning"] = {"enabled": a.reasoning == "on"}
            return payload
        _ta.build_chat_payload = _with_reasoning
        print("рассуждение у провайдера: %s" % a.reasoning, flush=True)

    env_dir = str(ROOT / "environment_files")
    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    ids = [e.game_id for e in arcade.available_environments]
    want = set(a.games.split(",")) if a.games != "all" else None
    ids = [g for g in ids if want is None or g[:4] in want]
    if not ids:
        print("игр не выбрано; доступны:", [g[:4] for g in arcade.available_environments])
        return 2
    bm.games = [taaf.game_api.GameAPI(env_name=g, arcade_spec=spec) for g in ids]
    bm.n_passes = 1
    bm.game_weights = None
    bm.solver.max_runtime_s_per_game = a.cap
    bm.solver.concurrency = a.concurrency

    print("игр %d: %s | модель %s | потолок %.0f с | одновременно %d | вывод %s"
          % (len(bm.games), ",".join(g[:4] for g in ids), a.model, a.cap, a.concurrency, out), flush=True)
    t0 = time.time()
    soft_end = datetime.now() + timedelta(seconds=a.cap * max(1, len(bm.games) / max(1, a.concurrency)) + 300)
    asyncio.run(bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=True))
    bm._save_json()
    print("прогон занял %.1f мин" % ((time.time() - t0) / 60), flush=True)

    rows = []
    for run in bm.game_runs:
        rows.append({"game": run.game_id[:4], "state": run.state, "score": run.final_score,
                     "actions": len(run.history), "levels": getattr(run, "levels_completed", None)})
        print("  %s: %s уровней=%s ходов=%d балл=%s" % (rows[-1]["game"], run.state, rows[-1]["levels"],
                                                        rows[-1]["actions"], rows[-1]["score"]), flush=True)
    try:
        from inference.tools.eval import evaluate_runs
        summary = evaluate_runs([out])
        print("СЧЁТ:", summary)
        json.dump({"rows": rows, "summary": repr(summary)}, open(out / "probe_summary.json", "w"), ensure_ascii=False, indent=1)
    except Exception as exc:
        print("счётчик не отработал: %r" % (exc,))
        json.dump({"rows": rows}, open(out / "probe_summary.json", "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

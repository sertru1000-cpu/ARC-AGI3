"""Наш харнес с мозгом wmloop («модель думает, алгоритм ходит») на Kaggle с сервером Flash-Next из стокового
ноутбука Duck (15.09, слово владельца «после этого — готовь сборку и пуш на Kaggle на 30 минут»).

Сборка: стоковый ноутбук (ячейки 3–13: колёса, бандл, setup_commands = запуск vLLM с Qwen3.8-Flash-Next-NVFP4,
загрузка bm) + ячейка 5 с поиском колёс соревнования по /kaggle/input + ячейка 15, где вместо bm.run() идёт наш
раннер: распаковка agent/my_agent.py + harness/* + skills/*, копия фреймворка ARC-AGI-3-Agents, AGENT_BRAIN=wmloop,
LLM_BASE_URL = локальный vLLM, 25 публичных игр параллельно, потолок игры GAME_SECONDS, отчёт + RHAE по журналу
кадров (формула из scripts/build_notebook.py), заглушка submission.parquet; teardown стоковый.
Только оффлайн (TRUE_SUBMISSION -> стоковый bm.run не вызывается, ветка боя не собирается — это проба).
Пороги пробы (30 мин): работает ли цикл на vLLM (вызовы модели без ошибок, программы собираются), уровни против
базы-30 3.32 (13 игр с уровнем); на API за 1 час v4 дала 1 уровень (lp85).
"""
import base64, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "sergueimakarov/arc3-wmloop-flash"
TITLE = "arc3 wmloop flash"
GAME_SECONDS = int(os.getenv("WMLOOP_GAME_SECONDS", "1500"))
PARALLEL = int(os.getenv("WMLOOP_PARALLEL", "8"))   # max_num_seqs сервера = 8 (проба v1: 4 работают, 22 ждут)
VARIANTS = int(os.getenv("WMLOOP_VARIANTS", "2"))
MAX_VARIANTS = int(os.getenv("WMLOOP_MAX_VARIANTS", "4"))
MAX_ACTIONS = int(os.getenv("WMLOOP_MAX_ACTIONS", "1500"))


def collect_sources() -> dict:
    a = ROOT / "agent"
    files = {"my_agent.py": (a / "my_agent.py").read_text(encoding="utf-8")}
    for p in sorted((a / "harness").glob("*.py")):
        files[f"harness/{p.name}"] = p.read_text(encoding="utf-8")
    for p in sorted((a / "skills").glob("*.md")):
        files[f"skills/{p.name}"] = p.read_text(encoding="utf-8")
    return files


RUNNER = r'''
    # ================= wmloop: наш харнес вместо bm.run() (только оффлайн-проба) =================
    import base64 as _b64, json as _json, shutil as _sh, threading as _thr, urllib.request as _ur
    from concurrent.futures import ThreadPoolExecutor as _TPE
    if TRUE_SUBMISSION:
        raise RuntimeError("wmloop-проба не предназначена для боевого сабмита")
    _FILES = _json.loads(_b64.b64decode("__BUNDLE__").decode("utf-8"))
    _src = WORKING_DIR / "agent_src"
    for _rel, _txt in _FILES.items():
        _p = _src / _rel; _p.parent.mkdir(parents=True, exist_ok=True); _p.write_text(_txt, encoding="utf-8")
    print("wmloop: распаковано файлов", len(_FILES), flush=True)
    _fw_src = None
    for _cand in [Path(COMP_WHEELS_DIR).parent / "ARC-AGI-3-Agents"] + [d for d in Path("/kaggle/input").rglob("ARC-AGI-3-Agents") if d.is_dir()]:
        if _cand.is_dir():
            _fw_src = _cand; break
    if _fw_src is None:
        raise RuntimeError("фреймворк ARC-AGI-3-Agents не найден под /kaggle/input")
    _fw = WORKING_DIR / "ARC-AGI-3-Agents"
    if not _fw.exists():
        _sh.copytree(_fw_src, _fw)
    _sh.copy(_src / "my_agent.py", _fw / "agents/templates/my_agent.py")
    _sh.copytree(_src / "harness", _fw / "harness", dirs_exist_ok=True)
    _sh.copytree(_src / "skills", _fw / "skills", dirs_exist_ok=True)
    (_fw / "agents/__init__.py").write_text("from typing import Type\nfrom .agent import Agent, Playback\nfrom .templates.my_agent import MyAgent\nAVAILABLE_AGENTS: dict[str, Type[Agent]] = {'myagent': MyAgent}\n", encoding="utf-8")
    try:
        import dotenv  # noqa: F401
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--find-links", str(COMP_WHEELS_DIR), "python-dotenv"], check=False)
    sys.path.insert(0, str(_fw))
    # сервер: адрес и имя модели из окружения setup_commands (LOCAL_ANALYZER_*), иначе стоковые
    _base = os.environ.get("LOCAL_ANALYZER_BASE_URL") or "http://127.0.0.1:1234/v1"
    _model = os.environ.get("LOCAL_ANALYZER_MODEL") or "Qwen/Qwen3.8-Flash-Next-NVFP4"
    try:
        _rq = _ur.Request(_base.rstrip("/") + "/models", headers={"Authorization": "Bearer " + os.environ.get("LLM_API_KEY", os.environ.get("DASHSCOPE_API_KEY", "EMPTY"))})
        with _ur.urlopen(_rq, timeout=10) as _r:
            _ids = [m.get("id") for m in _json.loads(_r.read().decode("utf-8")).get("data", [])]
        if _ids and _model not in _ids:
            _model = _ids[0]
        print("wmloop: сервер отвечает, модели:", _ids[:5], "-> используем", _model, flush=True)
    except Exception as _e:
        print("wmloop: сервер не отвечает:", repr(_e), flush=True); raise
    os.environ.update({
        "AGENT_BRAIN": "wmloop", "LLM_BASE_URL": _base, "LLM_MODEL": _model, "LLM_API_KEY": "EMPTY", "LLM_AUTH": "key",
        "LLM_TIMEOUT_S": "900", "MY_AGENT_MAX_ACTIONS": "__MAX_ACTIONS__", "MY_AGENT_GAME_SECONDS": "__GAME_SECONDS__",
        "MY_AGENT_WM_VARIANTS": "__VARIANTS__", "MY_AGENT_WM_MAX_VARIANTS": "__MAX_VARIANTS__",
        "MY_AGENT_TRACE_DIR": str(WORKING_DIR / "wmloop_trace"),
        # проба v1 (15.09 00:52): 216 из 240 ответов пустые -- думание без бюджета съело max_tokens; 24 запроса -- 400
        # (промпт до 34K символов hex-досок + max_tokens > 32768). Лечение: без думания, компактный промпт, ответ 6000.
        "LLM_DISABLE_THINKING": "1", "MY_AGENT_WM_COMPACT": "1", "MY_AGENT_WM_TRANS": "20", "MY_AGENT_WM_MAX_TOKENS": "6000",
    })
    os.environ.pop("LLM_THINKING_BUDGET", None)
    import importlib, arc_agi
    from arc_agi import OperationMode
    from agents import MyAgent
    _arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(Path(COMP_WHEELS_DIR).parent / "environment_files"))
    _games = [e.game_id for e in _arc.available_environments]
    print("wmloop: игр", len(_games), "параллельно", __PARALLEL__, "потолок", __GAME_SECONDS__, "с,", "__MAX_ACTIONS__", "ходов", flush=True)
    _results = {}
    def _play(gid):
        t0 = time.time()
        try:
            env_ = _arc.make(gid)
            ag = MyAgent(card_id="wmloop", game_id=gid, agent_name=f"wmloop.{gid}", ROOT_URL="http://localhost", record=False, arc_env=env_, tags=["wmloop"])
            ag.main()
            f = ag.frames[-1]
            summ = {}
            try: summ = ag.explorer.summary()
            except Exception: pass
            res = {"game": gid, "levels": int(f.levels_completed or 0), "win_levels": int(f.win_levels or 0), "actions": ag.action_counter,
                   "state": str(f.state).split(".")[-1], "seconds": round(time.time() - t0), "summary": {k: v for k, v in summ.items() if k != "phase_log"},
                   "phase_log": summ.get("phase_log")}
        except Exception as exc:
            res = {"game": gid, "levels": 0, "win_levels": 0, "actions": 0, "state": "CRASH", "error": repr(exc)[:300], "seconds": round(time.time() - t0)}
        print("[wmloop] %s: уровней %d/%d, ходов %s, %s, %ss, вызовов %s" % (gid[:4], res["levels"], res["win_levels"], res["actions"], res["state"], res["seconds"], (res.get("summary") or {}).get("calls")), flush=True)
        _results[gid] = res
        return res
    with _TPE(max_workers=__PARALLEL__) as _pool:
        list(_pool.map(_play, _games))
    _tot = sum(r["levels"] for r in _results.values()); _win = sum(r["win_levels"] for r in _results.values())
    print("[wmloop] ИТОГ: уровней %d из %d, игр с уровнем %d/%d" % (_tot, _win, sum(1 for r in _results.values() if r["levels"] > 0), len(_results)), flush=True)
    (WORKING_DIR / "wmloop_results.json").write_text(_json.dumps({"game_seconds": __GAME_SECONDS__, "parallel": __PARALLEL__, "variants": "__VARIANTS__/__MAX_VARIANTS__", "results": list(_results.values())}, ensure_ascii=False, indent=1), encoding="utf-8")
    import pandas as pd
    pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(WORKING_DIR / "submission.parquet", index=False)
'''


def main() -> None:
    src = json.load(open(ROOT / "kernels/notebooks_stockflash/submission.ipynb", encoding="utf-8"))
    nb = json.loads(json.dumps(src))
    # ячейка 5: поиск колёс соревнования по /kaggle/input (как в build_lvfact_reset_notebook.build)
    sys.path.insert(0, str(ROOT / "scripts"))
    c5 = "".join(nb["cells"][5]["source"])
    guard = ("import time as _t5\n"
             "def _c5_find():\n"
             "    fixed = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels')\n"
             "    if fixed.exists():\n        return fixed\n"
             "    for root in (Path('/kaggle/input'), Path('/kaggle/input/competitions')):\n"
             "        if root.exists():\n"
             "            for d in root.rglob('arc_agi_3_wheels'):\n"
             "                if d.is_dir():\n                    return d\n"
             "    return None\n"
             "COMP_WHEELS_DIR = None\n"
             "for _i5 in range(120):\n"
             "    COMP_WHEELS_DIR = _c5_find()\n"
             "    if COMP_WHEELS_DIR is not None:\n        break\n"
             "    print('taaf.kaggle: жду монтирования входов соревнования (%d)' % _i5, flush=True); _t5.sleep(5)\n"
             "print('taaf.kaggle: колёса соревнования:', COMP_WHEELS_DIR, flush=True)\n"
             "if COMP_WHEELS_DIR is None:\n"
             "    raise RuntimeError('входы соревнования не примонтированы')\n")
    lit = '"/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",'
    assert lit in c5
    nb["cells"][5]["source"] = (guard + c5.replace(lit, "str(COMP_WHEELS_DIR),")).splitlines(keepends=True)
    # ячейка 15: наш раннер вместо bm.run()
    c15 = "".join(nb["cells"][15]["source"])
    lit15 = 'str(Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent / "environment_files")'
    assert lit15 in c15
    c15 = c15.replace(lit15, 'str(Path(COMP_WHEELS_DIR).parent / "environment_files")')
    start = c15.index("try:\n    await bm.run(")
    end = c15.index("finally:", start)
    bundle = base64.b64encode(json.dumps(collect_sources()).encode("utf-8")).decode("ascii")
    runner = (RUNNER.replace("__BUNDLE__", bundle).replace("__MAX_ACTIONS__", str(MAX_ACTIONS)).replace("__GAME_SECONDS__", str(GAME_SECONDS))
              .replace("__VARIANTS__", str(VARIANTS)).replace("__MAX_VARIANTS__", str(MAX_VARIANTS)).replace("__PARALLEL__", str(PARALLEL)))
    c15 = c15[:start] + "try:\n" + runner + c15[end:]
    nb["cells"][15]["source"] = c15.splitlines(keepends=True)
    out = ROOT / "kernels/notebooks_wmloop_flash"; out.mkdir(parents=True, exist_ok=True)
    json.dump(nb, open(out / "submission.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    meta = json.load(open(ROOT / "kernels/notebooks_stockflash/kernel-metadata.json"))
    meta["id"] = SLUG; meta["title"] = TITLE
    json.dump(meta, open(out / "kernel-metadata.json", "w"), indent=2)
    open(out / "cell15.py", "w", encoding="utf-8").write(c15)
    import ast
    compile(c15, "c15", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT); compile("".join(nb["cells"][5]["source"]), "c5", "exec")
    size = os.path.getsize(out / "submission.ipynb")
    print("ok   собрано:", SLUG, "| игра %d с, параллельно %d, варианты %d/%d, ходов %d | размер %.0f КБ" % (GAME_SECONDS, PARALLEL, VARIANTS, MAX_VARIANTS, MAX_ACTIONS, size / 1024))


if __name__ == "__main__":
    main()

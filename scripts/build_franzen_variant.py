"""Сборщик вариантов стенда на базе копии Франзена (kernels/graft_dfranzen_m2_stand).

Каждый вариант = стенд базы + правки: переменные окружения обвязки (setup_env в ячейке 4), настройки сервера
(CFG в ячейке 12), строки ячейки 16. Бой в каждом кернеле — копия Франзена с теми же правками.
usage: .venv/bin/python scripts/build_franzen_variant.py            # собрать все варианты из VARIANTS
       .venv/bin/python scripts/build_franzen_variant.py fp4 img4   # только перечисленные
"""
import copy, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# с 03.10 база — base3 (копия Франзена + макро-ходы + планировщик impatient + картинка x4), слово владельца; старая — FRANZEN_BASE=graft_dfranzen_m2_stand
BASE = ROOT / "kernels" / os.environ.get("FRANZEN_BASE", "graft_dfranzen_m2_base3")

# env: значения setup_env (новые ключи добавляются, старые заменяются); cfg: CFG сервера; cell16: замены строк
VARIANTS = {
    # планировщик Франзена: C = 0.25·2^(-(a/115)^2) + 0.75·2^(-(t/62000)^2)
    "sched_nodecay":   {"note": "приоритет без затухания по ходам и токенам (только ценность уровня)",
                        "env": {"ARC3_PRIORITY_ACTION_SCALE": 1000000000, "ARC3_PRIORITY_TOKEN_SCALE": 1000000000000}},
    "sched_patient":   {"note": "затухание вдвое медленнее: 230 ходов, 124 тыс. токенов",
                        "env": {"ARC3_PRIORITY_ACTION_SCALE": 230, "ARC3_PRIORITY_TOKEN_SCALE": 124000}},
    "sched_impatient": {"note": "затухание вдвое быстрее: 57 ходов, 31 тыс. токенов",
                        "env": {"ARC3_PRIORITY_ACTION_SCALE": 57, "ARC3_PRIORITY_TOKEN_SCALE": 31000}},
    "img4":            {"note": "картинка доски x4 вместо x10",
                        "env": {"MULTIMODAL_UPSCALE": "4"}},
    "sum16k":          {"note": "пересказ истории каждые 16k токенов, обрезка останавливается на пересказах (механизм Франзена)",
                        "env": {"ARC3_SUMMARY_INTERVAL_TOKENS": 16384, "ARC3_DRAIN_STOP_AT_SUMMARIES": "1"}},
    "guards":          {"note": "стражи повторов: пустой ход, гибельный маршрут, повтор состояния, журнал гибелей",
                        "env": {"ARC3_NOOP_REPEAT_GUARD": "1", "ARC3_DEATH_REPEAT_GUARD": "1",
                                "ARC3_REPEAT_STATE_GUARD": "1", "ARC3_DEATH_LEDGER": "1"}},
    "fp4":             {"note": "кэш KV nvfp4 вместо fp8 (и у черновика)",
                        "cfg": {"KVDTYPE": '"nvfp4"'}},
    "s14fp4":          {"note": "14 активных игр с полным окном 128k на кэше nvfp4",
                        "cfg": {"KVDTYPE": '"nvfp4"', "MAXREQ": "14", "CUDAGRAPH_MAXBS": "14"},
                        "env": {"ARC3_MAX_ACTIVE_STREAMS": 14}},
}


def build(name, spec):
    nb = json.load(open(BASE / "submission.ipynb"))
    orig = copy.deepcopy(nb)
    c4 = "".join(nb["cells"][4]["source"])
    c12 = "".join(nb["cells"][12]["source"])
    for key, val in (spec.get("env") or {}).items():
        line_val = repr(val) if isinstance(val, str) else str(val)
        import re
        m = re.search(r"^    '%s': [^\n]*\n" % re.escape(key), c4, re.M)
        new = "    '%s': %s,   # вариант %s\n" % (key, line_val, name)
        if m:
            c4 = c4[:m.start()] + new + c4[m.end():]
        else:
            anchor = "    'EXPOSE_UNDO': 'on',\n"
            assert c4.count(anchor) == 1
            c4 = c4.replace(anchor, anchor + new)
    for key, val in (spec.get("cfg") or {}).items():
        import re
        m = re.search(r"^    %s=[^\n]*\n" % re.escape(key), c12, re.M)
        assert m, (name, key)
        c12 = c12[:m.start()] + "    %s=%s,   # вариант %s\n" % (key, val, name) + c12[m.end():]
    nb["cells"][4]["source"] = c4.splitlines(keepends=True)
    nb["cells"][12]["source"] = c12.splitlines(keepends=True)
    d = ROOT / ("kernels/graft_dfranzen_m2_" + name)
    d.mkdir(parents=True, exist_ok=True)
    json.dump(nb, open(d / "submission.ipynb", "w"), ensure_ascii=False, indent=1)
    meta = json.load(open(BASE / "kernel-metadata.json"))
    meta["id"] = "sergueimakarov/arc3-graft-dfranzen-m2-" + name.replace("_", "-")
    meta["title"] = "arc3 graft dfranzen m2 " + name.replace("_", " ")
    json.dump(meta, open(d / "kernel-metadata.json", "w"), indent=2)
    for i in (4, 12):
        compile("".join(nb["cells"][i]["source"]).replace("!rm", "#!rm").replace("!cp", "#!cp"), f"{name}:{i}", "exec")
    changed = [i for i, (a, b) in enumerate(zip(orig["cells"], nb["cells"])) if a["source"] != b["source"]]
    print(f"{d.relative_to(ROOT)}  ячейки {changed}  — {spec['note']}")


if __name__ == "__main__":
    names = sys.argv[1:] or list(VARIANTS)
    for n in names:
        build(n, VARIANTS[n])

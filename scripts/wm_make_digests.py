"""Готовит сводки наблюдений по записанным прогонам — вход для пакетного синтеза на поде.

ЗАЧЕМ. Чтобы проверить приём по горизонту, нужны не игры, а ПРОГРАММЫ: много попыток синтеза
на одних и тех же наблюдениях. Игра для этого только помеха — в прогоне v13 она дала восемь
попыток за полчаса, из них четыре умерли на транспорте.

Здесь из записей делается ровно то, что видит синтезатор в игре: сводка наблюдений (та же
_WM_DIGEST_CODE, что в боевой сборке), плюс сами переходы для последующей оценки. Файл
отправляется на под, там по нему гоняется генерация, программы возвращаются сюда и меряются
теми же скриптами, что и сегодня (wm_horizon, wm_plan_offline).

usage:
    .venv/bin/python scripts/wm_make_digests.py runs/flash_v1_phaseA --out docs/wm_digests.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_acceptance_rules import HELPERS, pts, trajectory  # noqa: E402

DIGEST = r'''
print('ВАЛИДНЫЕ ДЕЙСТВИЯ:', valid_actions)
print('УРОВЕНЬ:', wm_level(current_frame))
print('ТЕКУЩИЙ КАДР:'); print(current_frame.ascii)
print('НАБЛЮДЁННЫЕ ПЕРЕХОДЫ (действие -> что изменилось у объектов):')
_pp = _wm_pairs(12, level=_LEVEL)
for _b, _a, _af in _pp:
    _ob = {}; _oa = {}
    for _o in wm_objects(_b): _ob.setdefault(_o['t'], []).append(_o)
    for _o in wm_objects(_af): _oa.setdefault(_o['t'], []).append(_o)
    _ch = []
    for _t, _lst in _ob.items():
        _bs = sorted(_lst, key=lambda o: (o['y'], o['x']))
        _as_ = sorted(_oa.get(_t, []), key=lambda o: (o['y'], o['x']))
        for _i, _o in enumerate(_bs):
            if _i >= len(_as_): _ch.append((_t, 'исчез')); continue
            _n = _as_[_i]
            _d = (_n['x'] - _o['x'], _n['y'] - _o['y'], (_n['pixels'] or 0) - (_o['pixels'] or 0))
            if any(_d): _ch.append((_t, 'dx=%d dy=%d dpix=%d' % _d))
    print('  %-8s %s' % (_a, _ch[:8] if _ch else 'без изменений'))
print('ЧИСЛО ПЕРЕХОДОВ В БУФЕРЕ:', len(_wm_pairs(400, level=_LEVEL)))
wm_ontology(top=4)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="runs/flash_v1_phaseA")
    ap.add_argument("--out", default="docs/wm_digests.json")
    ap.add_argument("--max-chars", type=int, default=7000)
    a = ap.parse_args()

    out = []
    for f in sorted(glob.glob(os.path.join(a.run, "artifacts", "*_events.jsonl"))):
        game = os.path.basename(f).split("_p0")[0][:4]
        state = trajectory(f)
        if state is None:
            continue
        level = state["history"][0]["frame"]["level"]
        res = pts.run_sandboxed_python(
            code=HELPERS + f"\n_LEVEL = {level}\n" + DIGEST, timeout_seconds=120,
            initial_state=state,
            action_handler=lambda x: (_ for _ in ()).throw(RuntimeError("нельзя")))
        digest = str(res.get("stdout", "") or "")[:a.max_chars]
        if "ЧИСЛО ПЕРЕХОДОВ" not in digest:
            print(f"  {game}: сводка не собралась, пропуск")
            continue
        n = 0
        for line in digest.splitlines():
            if line.startswith("ЧИСЛО ПЕРЕХОДОВ"):
                n = int(line.split(":")[1])
        out.append({"game": game, "level": level, "pairs": n, "digest": digest})
        print(f"  {game}: переходов {n}, сводка {len(digest)} знаков")

    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\nготово: {len(out)} игр -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Что модель делает СРАЗУ ПОСЛЕ пустого хода (25.09, проверка гипотезы критика).

Гипотеза критика: модель умеет строить модель мира (86% предсказаний невиданных переходов), но НЕ
ЗНАЕТ, какие пробы ставить. Если так, после пустого хода она должна вести себя бестолково: повторять
то же самое или тыкать наугад, вместо того чтобы поставить различающий эксперимент.

Это проверяется по записям, без модели и без квоты. Для каждого хода, не изменившего кадр, смотрим
СЛЕДУЮЩИЙ ход и раскладываем по разрядам:
  * ТО ЖЕ САМОЕ            -- повтор пустого действия один в один (худший случай);
  * ДРУГОЙ КЛИК            -- та же кнопка мыши, другая клетка (слепой перебор координат);
  * ДРУГОЕ ДЕЙСТВИЕ        -- сменила орудие: осмысленная проба;
  * ВОЗВРАТ К ПУСТОМУ ПОЗЖЕ -- отдельно считаем, вернулась ли она к этому действию после того,
    как доска изменилась. Это и есть «эксперимент с предусловием», ради которого всё затевается.

Порог для вывода: если «то же самое» и «другой клик» вместе дают больше половины, гипотеза критика
подтверждается — после пустого хода модель не ставит эксперимент, а перебирает.

usage: .venv/bin/python scripts/after_noop_behaviour.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT / "scripts"))
import logging  # noqa: E402

logging.disable(logging.ERROR)
import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402


def frame_of(fr):
    if fr is None or not fr.frame:
        return None
    return np.asarray(fr.frame[-1], dtype=np.int16)


def key_of(a):
    name = a.get("id") or "?"
    d = a.get("data")
    return "%s(%s,%s)" % (name, d["x"], d["y"]) if isinstance(d, dict) and "x" in d else name


def main() -> None:
    bench = json.load(open(ROOT / "runs/flash_v1_phaseA/benchmark.json", encoding="utf-8"))
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL,
                         environments_dir=str(ROOT / "environment_files"))
    kinds = collections.Counter()
    retested = retested_worked = 0
    noops = 0
    for gr in bench["game_runs"]:
        gid = gr["game_id"]
        env = arc.make(gid)
        fr = env.reset()
        g = frame_of(fr)
        if g is None:
            continue
        score = getattr(fr, "score", 0)
        hist = gr.get("history") or []
        seq = []          # (ключ, действие, изменилось ли)
        for rec in hist:
            act = (rec.get("action") or {})
            if not act.get("id"):
                continue
            try:
                fr2 = env.step(GameAction[act["id"]] if act["id"] != "RESET" else GameAction.RESET,
                               data=act.get("data"))
            except Exception:
                break
            g2 = frame_of(fr2)
            sc2 = getattr(fr2, "score", score) if fr2 is not None else score
            changed = (g2 is None) or (g2.shape != g.shape) or bool(np.any(g2 != g)) or sc2 != score
            seq.append((key_of(act), act.get("id"), bool(changed)))
            if g2 is None:
                break
            g, score = g2, sc2

        for i, (key, name, changed) in enumerate(seq):
            if changed or i + 1 >= len(seq):
                continue
            noops += 1
            nkey, nname, _ = seq[i + 1]
            if nkey == key:
                kinds["то же самое"] += 1
            elif nname == name == "ACTION6":
                kinds["другой клик"] += 1
            elif nname != name:
                kinds["другое действие"] += 1
            else:
                kinds["то же действие, иные данные"] += 1
            # вернулась ли к этому действию позже, после изменения доски
            later = [(k, c) for k, _, c in seq[i + 2:] if k == key]
            if later and any(c for _, _, c in seq[i + 1:i + 6]):
                retested += 1
                if any(c for _, c in later):
                    retested_worked += 1

    tot = sum(kinds.values())
    print("пустых ходов, у которых есть следующий: %d" % tot)
    for k, v in kinds.most_common():
        print("  %-28s %5d (%.1f%%)" % (k, v, 100 * v / max(tot, 1)))
    blind = kinds["то же самое"] + kinds["другой клик"]
    print("\nслепой перебор («то же» + «другой клик»): %.1f%%" % (100 * blind / max(tot, 1)))
    print("вернулась к пустому действию после изменения доски: %d (%.1f%% пустых ходов)"
          % (retested, 100 * retested / max(tot, 1)))
    print("  из них действие ЗАРАБОТАЛО: %d" % retested_worked)
    print("\nВЫВОД:", "гипотеза критика ПОДТВЕРЖДАЕТСЯ — после пустого хода идёт перебор, а не эксперимент"
          if blind / max(tot, 1) > 0.5 else "после пустого хода модель чаще меняет орудие, чем перебирает")


if __name__ == "__main__":
    main()

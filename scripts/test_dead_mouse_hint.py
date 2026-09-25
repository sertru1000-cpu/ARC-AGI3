"""Проверка слоя «мышь в этой игре не работает» на синтетической истории (25.09).

Правило проекта: код, который иначе будет впервые исполнен в боевом прогоне, сначала прогоняется
синтетическим тестом (см. память feedback-untestable-code-before-real-submission — версия v7 дала 0.06
именно потому, что этот шаг пропустили).

Проверяем четыре случая:
 1. клики подряд, экран не меняется -> подсказка ЕСТЬ;
 2. клики есть, но экран меняется -> подсказки НЕТ (в таких играх мышь и есть орудие);
 3. серия пустых кликов прервана удачным -> подсказки НЕТ (счётчик обнуляется);
 4. пустых кликов меньше порога -> подсказки НЕТ.

usage: .venv/bin/python scripts/test_dead_mouse_hint.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "atlas_src" / "src" / "ARC3-Inference"
sys.path.insert(0, str(SRC))

from inference.agent.runtime_state import Frame, HistoryEntry  # noqa: E402
from inference.agent.tool_agent import (  # noqa: E402
    _ATLAS_DEAD_MOUSE_AFTER,
    _atlas_dead_mouse_hint,
)

FAILS = []


def check(name, cond, detail=""):
    print("  %-46s %s%s" % (name, "ок" if cond else "ПРОВАЛ", (" — " + detail) if detail else ""))
    if not cond:
        FAILS.append(name)


def grid(v):
    return tuple(tuple(v for _ in range(4)) for _ in range(4))


def frame(v):
    f = Frame.__new__(Frame)
    object.__setattr__(f, "grid", grid(v))
    for fld in getattr(Frame, "__dataclass_fields__", {}):
        if fld != "grid" and not hasattr(f, fld):
            object.__setattr__(f, fld, None)
    return f


def history(actions_and_values):
    return [HistoryEntry(action=a, frame=frame(v)) for a, v in actions_and_values]


def main() -> None:
    n = _ATLAS_DEAD_MOUSE_AFTER
    print("порог слоя: %d пустых кликов подряд" % n)

    dead = history([("", 0)] + [("MOUSE(row=%d, col=1)" % i, 0) for i in range(n + 2)])
    out = _atlas_dead_mouse_hint(dead)
    check("пустые клики -> подсказка есть", bool(out), (out[0][:60] + "…") if out else "нет строки")

    alive = history([("", 0)] + [("MOUSE(row=%d, col=1)" % i, i) for i in range(n + 2)])
    check("клики работают -> подсказки нет", not _atlas_dead_mouse_hint(alive))

    broken = history(
        [("", 0)]
        + [("MOUSE(row=%d, col=1)" % i, 0) for i in range(n)]
        + [("MOUSE(row=99, col=1)", 7)]
        + [("MOUSE(row=%d, col=2)" % i, 7) for i in range(3)]
    )
    check("серия прервана удачным кликом -> подсказки нет", not _atlas_dead_mouse_hint(broken))

    few = history([("", 0)] + [("MOUSE(row=%d, col=1)" % i, 0) for i in range(n - 3)])
    check("пустых меньше порога -> подсказки нет", not _atlas_dead_mouse_hint(few))

    mixed = history(
        [("", 0)]
        + [("ACTION1", 1), ("ACTION2", 2)]
        + [("MOUSE(row=%d, col=1)" % i, 2) for i in range(n + 1)]
    )
    check("простые действия не сбивают счёт кликов", bool(_atlas_dead_mouse_hint(mixed)))

    print("\nИТОГ:", "ПРОЙДЕНО" if not FAILS else "ПРОВАЛЫ: %s" % ", ".join(FAILS))
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()

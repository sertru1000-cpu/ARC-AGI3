"""Запрет повторного пустого хода при побитово той же доске, с предохранителем.

Правило владельца (25.09): ход, который уже дал «ничего» в ЭТОМ состоянии доски,
второй раз в движок не уходит.

ЧЕГО ЭТОТ ЗАПРЕТ НЕ ГАРАНТИРУЕТ. Сначала было записано, что ошибиться он не может: движок
детерминирован (561 повтор из 561), значит та же доска плюс тот же ход дают тот же исход.
Довод неверен. Проверка детерминизма показывает лишь, что та же ПОСЛЕДОВАТЕЛЬНОСТЬ ходов даёт
ту же траекторию; из неё не следует, что доска есть полное состояние игры. Замер на 25 играх:
из 200 срезов 37 (18%) отняли бы работающий ход. Анимация объясняет лишь 4 из них.

ОТКУДА ОШИБКИ. Не из загадочной скрытой переменной, а из конкретной механики: часть действий
срабатывает ЧЕРЕЗ РАЗ — первое нажатие взводит что-то невидимое, второе действует. Пример из
записи: g50t, ход 10 — ACTION2 не меняет доску, ход 11 — тот же ACTION2 при той же доске меняет.
Запрет со второго нажатия такую механику ломает начисто.

СКОЛЬКО ПУСТЫХ ИСХОДОВ ЖДЁМ. Отсюда min_repeats: сколько раз подряд пара (доска, действие)
должна дать пусто, прежде чем следующий такой ход будет срезан. Замер на записи 4380 ходов
(25 игр, детерминированный движок, поэтому это точный контрфакт, а не сравнение прогонов):

    N=1   200 срезов, 37 ошибочных (18%)   — ломает механику «через раз»
    N=2   154 среза,   6 ошибочных (4%)
    N=3   143 среза,   0 ошибочных         <- взято
    N=4   138 срезов,  0 ошибочных

N=3 сохраняет 72% пользы и убирает все ошибки. Оговорка: это колено кривой на НАШИХ играх;
тумблер, которому нужно четыре нажатия, положит и N=3.

Порог другого рода — «включать запрет только после N пустых ходов на уровне» — пробовался
(scripts/noop_guard_fuse.py) и решением владельца отклонён.
"""
from __future__ import annotations

import os

import hashlib
from collections import OrderedDict
from typing import Any


def board_signature(grid: Any) -> str:
    """Stable, compact signature for a 2D integer grid."""
    rows = tuple(tuple(int(cell) for cell in row) for row in grid or ())
    digest = hashlib.blake2b(repr(rows).encode("utf-8"), digest_size=8)
    return digest.hexdigest()


def normalize_action_signature(value: Any) -> str:
    return " ".join(str(value or "").split())


class NoopGuard:
    """Tracks ``(level, board_before_sig, action_sig)`` combos known to be no-ops.

    Bounded per level (oldest states/actions evicted first) so long runs can't
    grow this without limit.
    """

    def __init__(
        self,
        *,
        max_states_per_level: int = 512,
        max_actions_per_state: int = 16,
        min_repeats: int = 3,
    ) -> None:
        self._max_states_per_level = max(1, int(max_states_per_level))
        self._max_actions_per_state = max(1, int(max_actions_per_state))
        # Сколько пустых исходов подряд у пары (доска, действие) нужно накопить,
        # прежде чем резать. См. заголовок модуля: на N=3 ошибок ноль.
        self._min_repeats = max(1, int(min_repeats))
        # Значения словаря действий — счётчик пустых исходов, а не None.
        self._levels: "OrderedDict[int, OrderedDict[str, OrderedDict[str, int]]]" = OrderedDict()

    def observe(
        self,
        *,
        level: Any,
        board_before_sig: str,
        action_sig: str,
        board_changed: bool,
        animated: bool = False,
    ) -> None:
        """Record one executed action and whether it had any effect.

        ``animated`` means the environment returned more than one frame for
        this action. Such an action is never a no-op, even when the visible
        board ends up identical to before: in a type-1 animation (ft09, sb26)
        the first and last frame match by design and the entire signal -- a
        rejected click, a consumed attempt -- lives in the frames in between.
        Recording those as no-ops made the guard hard-block actions that had
        clearly worked, on exactly the games with the most animations.
        """
        try:
            level_num = int(level)
        except (TypeError, ValueError):
            return
        action_sig = normalize_action_signature(action_sig)
        if not action_sig:
            return
        states = self._levels.setdefault(level_num, OrderedDict())

        if board_changed or animated:
            # Evidence now contradicts a previously recorded no-op for this
            # exact state+action: drop the stale entry so we never block a
            # combo that turned out to have an effect after all.
            entry = states.get(board_before_sig)
            if entry is not None and action_sig in entry:
                del entry[action_sig]
                if not entry:
                    del states[board_before_sig]
            return

        entry = states.get(board_before_sig)
        if entry is None:
            entry = OrderedDict()
            states[board_before_sig] = entry
            while len(states) > self._max_states_per_level:
                states.popitem(last=False)
        entry[action_sig] = entry.get(action_sig, 0) + 1
        while len(entry) > self._max_actions_per_state:
            entry.popitem(last=False)

    def is_known_noop(self, level: Any, board_sig: str, action_sig: str) -> bool:
        """Известен ли этот ход как пустой в этом самом состоянии доски."""
        # 27.09: NEXTFORK_NOOPGUARD=0 выключает запрет (ход исполняется всегда). Основание: у Скотта аналогичный отказ
        # выключен — «the v3 regression hunt showed refusals misled the model»; v4 с запретом 1.80 / 2.06 на поде.
        if os.environ.get("NEXTFORK_NOOPGUARD", "1").strip().lower() in {"0", "false", "off"}:
            return False
        try:
            level_num = int(level)
        except (TypeError, ValueError):
            return False
        states = self._levels.get(level_num)
        if not states:
            return False
        entry = states.get(board_sig)
        if not entry:
            return False
        return entry.get(normalize_action_signature(action_sig), 0) >= self._min_repeats

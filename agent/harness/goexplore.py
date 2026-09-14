"""Go-Explore brain: algorithmic search over the real environment, no LLM.

Раунд 4 критиков (14.09): единственная незакрытая ветка — «алгоритм-исследователь с графом переходов».
Нижняя граница (слепой HeuristicExplorer): 5/183 уровней при 3000 ходах. Здесь — поиск с архивом:

  cell     = (уровень, огрублённая доска 16x16 по моде цвета в блоке 4x4)
  archive  = cell -> {path от старта уровня, точный хеш доски, посещений}
  цикл     = выбрать ячейку (вес 1/sqrt(1+посещений)), вернуться в неё (RESET + повтор пути, если мы не там),
             сделать K шагов разведки (стрелки/SPACE или клик по центру объекта), новые доски -> новые ячейки.
  уровень  = при росте levels_completed архив сбрасывается (RESET возвращает на старт ТЕКУЩЕГО уровня,
             взятые уровни сохраняются — семантика проверена ранее, см. память scoring-mechanics).
  сбой     = после повтора пути хеш доски не совпал -> ячейка помечена ненадёжной и выбрасывается.

Интерфейс как у HeuristicExplorer: observe(frame) + decide(frame, simple_actions, has_click) -> (name, payload).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from .perception import frame_hash, latest_grid, segment


def coarse_key(grid: np.ndarray, block: int = 4) -> tuple:
    h, w = grid.shape
    hb, wb = h // block, w // block
    g = grid[: hb * block, : wb * block].reshape(hb, block, wb, block).transpose(0, 2, 1, 3).reshape(hb, wb, block * block)
    out = []
    for r in range(hb):
        row = []
        for c in range(wb):
            v = np.bincount(g[r, c].astype(np.int64), minlength=16)
            row.append(int(v.argmax()))
        out.append(tuple(row))
    return tuple(out)


@dataclass
class Cell:
    path: list[tuple[str, dict | None]]
    exact: str
    visits: int = 0
    unreliable: bool = False


@dataclass
class GoState:
    level: int = 0
    archive: dict[tuple, Cell] = field(default_factory=dict)
    cur_path: list[tuple[str, dict | None]] = field(default_factory=list)
    cur_exact: str | None = None
    queue: list[tuple[str, dict | None]] = field(default_factory=list)   # actions to execute verbatim (replay)
    replay_target: tuple | None = None
    explore_left: int = 0
    resets: int = 0
    replay_fail: int = 0
    new_cells: int = 0
    steps: int = 0
    game_overs: int = 0
    last_action: tuple[str, dict | None] | None = None
    need_reset: bool = True


class GoExplore:
    EXPLORE_STEPS = 20
    MAX_CLICK_TARGETS = 40

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.st = GoState()

    # ---------------------------------------------------------------- observe
    def observe(self, frame) -> None:
        st = self.st
        grid = latest_grid(frame)
        if grid is None:
            return
        level = int(getattr(frame, "levels_completed", 0) or 0)
        state = str(getattr(frame, "state", ""))
        if level != st.level:
            # новый уровень: архив с нуля, путь с нуля (RESET вернёт на старт этого уровня)
            st.level = level
            st.archive.clear(); st.cur_path = []; st.queue = []; st.replay_target = None; st.explore_left = 0
        if state.endswith("GAME_OVER"):
            st.game_overs += 1
            st.need_reset = True; st.queue = []; st.replay_target = None; st.explore_left = 0
            return
        exact = frame_hash(grid)
        st.cur_exact = exact
        if st.last_action is not None and st.last_action[0] == "RESET":
            st.cur_path = []
        key = (level, exact)   # ячейка = точная доска (движок детерминирован, взрыва при 10^4 ходах нет)
        cell = st.archive.get(key)
        if cell is None:
            st.archive[key] = Cell(path=list(st.cur_path), exact=exact)
            st.new_cells += 1
        elif len(st.cur_path) < len(cell.path) and not st.replay_target:
            cell.path = list(st.cur_path); cell.exact = exact
        if st.replay_target is not None and not st.queue:
            # повтор пути закончен: проверяем, что пришли туда же
            target = st.archive.get(st.replay_target)
            if target is None or target.exact != exact:
                st.replay_fail += 1
                if target is not None:
                    target.unreliable = True
            st.replay_target = None
            st.explore_left = self.EXPLORE_STEPS

    # ----------------------------------------------------------------- decide
    def decide(self, frame, simple_actions: list[str], has_click: bool) -> tuple[str, dict | None]:
        st = self.st
        st.steps += 1
        if st.need_reset:
            st.need_reset = False
            return self._emit(("RESET", None))
        if st.queue:
            return self._emit(st.queue.pop(0))
        if st.explore_left <= 0:
            # выбрать ячейку и запланировать возврат
            key = self._select_cell()
            if key is not None and key != (st.level, None):
                cell = st.archive[key]
                cell.visits += 1
                if cell.exact != st.cur_exact:
                    st.replay_target = key
                    st.queue = [("RESET", None)] + list(cell.path)
                    st.resets += 1
                    if st.queue:
                        return self._emit(st.queue.pop(0))
            st.explore_left = self.EXPLORE_STEPS
        st.explore_left -= 1
        return self._emit(self._explore_action(frame, simple_actions, has_click))

    def _emit(self, act: tuple[str, dict | None]) -> tuple[str, dict | None]:
        st = self.st
        st.last_action = act
        if act[0] == "RESET":
            st.cur_path = []
        else:
            st.cur_path = st.cur_path + [act]
        return act

    def _select_cell(self):
        st = self.st
        cands = [(k, c) for k, c in st.archive.items() if not c.unreliable and k[0] == st.level]
        if not cands:
            return None
        weights = [1.0 / math.sqrt(1.0 + c.visits) for _, c in cands]
        return self.rng.choices([k for k, _ in cands], weights=weights, k=1)[0]

    def _explore_action(self, frame, simple_actions: list[str], has_click: bool) -> tuple[str, dict | None]:
        grid = latest_grid(frame)
        options: list[tuple[str, dict | None]] = [(a, None) for a in simple_actions]
        if has_click and grid is not None:
            seg = segment(grid)
            objs = seg.non_background()[: self.MAX_CLICK_TARGETS]
            clicks = [("ACTION6", {"x": int(round(o.centroid[1])), "y": int(round(o.centroid[0]))}) for o in objs]
            if clicks:
                # половина веса стрелкам, половина кликам
                if not options or self.rng.random() < 0.5:
                    return self.rng.choice(clicks)
            if not options:
                return ("ACTION6", {"x": self.rng.randint(0, 63), "y": self.rng.randint(0, 63)})
        if not options:
            return ("RESET", None)
        return self.rng.choice(options)

    def summary(self) -> dict:
        st = self.st
        return {"steps": st.steps, "cells": len(st.archive), "new_cells": st.new_cells, "resets": st.resets,
                "replay_fail": st.replay_fail, "game_overs": st.game_overs, "level": st.level}

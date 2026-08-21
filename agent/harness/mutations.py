"""Generic game mutations: bijective transforms wrapped around an arc env.

Game sources are obfuscated, so instead of editing them we transform the
OBSERVATION (grids) and the ACTION stream at the env boundary. Mechanics are
preserved exactly; the agent sees a genuinely different-looking game — cheap
"new" training material (user's synthetic-games tier 1).

Specs (combinable with '+'):
  colors:<seed>   random permutation of the 16 colors (seeded, bijective)
  mirror_h        horizontal mirror: grids flipped left-right, LEFT<->RIGHT
                  swapped, CLICK x -> width-1-x
  mirror_v        vertical mirror: up-down flip, UP<->DOWN, CLICK y -> h-1-y

Usage:
    env = make_mutated_env(arc.make(gid), "mirror_h+colors:7")
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np

_LEFT, _RIGHT, _UP, _DOWN = 3, 4, 1, 2  # engine action ids


class _Mutation:
    def grids(self, g: np.ndarray) -> np.ndarray:
        return g

    def action(self, action_id: int, data: dict | None) -> tuple[int, dict | None]:
        return action_id, data


class _ColorPerm(_Mutation):
    def __init__(self, seed: int):
        rng = random.Random(seed)
        perm = list(range(16))
        rng.shuffle(perm)
        self.lut = np.array(perm, dtype=np.int8)

    def grids(self, g: np.ndarray) -> np.ndarray:
        return self.lut[g]


class _MirrorH(_Mutation):
    def grids(self, g: np.ndarray) -> np.ndarray:
        return np.fliplr(g)

    def action(self, action_id: int, data: dict | None) -> tuple[int, dict | None]:
        if action_id == _LEFT:
            return _RIGHT, data
        if action_id == _RIGHT:
            return _LEFT, data
        if data and "x" in data:
            data = dict(data, x=63 - int(data["x"]))
        return action_id, data


class _MirrorV(_Mutation):
    def grids(self, g: np.ndarray) -> np.ndarray:
        return np.flipud(g)

    def action(self, action_id: int, data: dict | None) -> tuple[int, dict | None]:
        if action_id == _UP:
            return _DOWN, data
        if action_id == _DOWN:
            return _UP, data
        if data and "y" in data:
            data = dict(data, y=63 - int(data["y"]))
        return action_id, data


def _parse(spec: str) -> list[_Mutation]:
    out: list[_Mutation] = []
    for part in (spec or "").split("+"):
        part = part.strip().lower()
        if not part:
            continue
        if part.startswith("colors:"):
            out.append(_ColorPerm(int(part.split(":", 1)[1])))
        elif part == "mirror_h":
            out.append(_MirrorH())
        elif part == "mirror_v":
            out.append(_MirrorV())
        else:
            raise ValueError(f"unknown mutation spec: {part!r}")
    return out


class MutatedEnv:
    """Duck-typed proxy over an arcade env applying the mutation chain."""

    def __init__(self, env: Any, spec: str):
        self._env = env
        self._chain = _parse(spec)
        self.mutation_spec = spec

    # observation transforms run in order; actions in REVERSE order (the
    # model acts in mutated space, we translate back towards the engine).
    def _fix_raw(self, raw: Any) -> Any:
        if raw is not None and getattr(raw, "frame", None) is not None:
            raw.frame = [self._mut_grid(np.asarray(g)) for g in raw.frame]
        return raw

    def _mut_grid(self, g: np.ndarray) -> np.ndarray:
        for m in self._chain:
            g = m.grids(g)
        return g

    def step(self, action: Any, data: dict | None = None, reasoning: Any = None) -> Any:
        try:
            import arcengine
            aid = int(getattr(action, "value", action))
            for m in reversed(self._chain):
                aid, data = m.action(aid, data)
            action = arcengine.GameAction.from_id(aid)
        except Exception:
            pass  # RESET & friends pass through untouched
        return self._fix_raw(self._env.step(action, data=data, reasoning=reasoning))

    @property
    def observation_space(self) -> Any:
        return self._fix_raw(self._env.observation_space)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def make_mutated_env(env: Any, spec: str | None) -> Any:
    return MutatedEnv(env, spec) if spec else env

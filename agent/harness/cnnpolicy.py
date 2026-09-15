"""Мозг «политика подражания»: свёрточная сеть выбирает ход по доске (15.09). Без модели, без поиска.
Ход = выборка из softmax (температура), клики не обучены -> при отсутствии стрелок клик по центру случайного объекта.
Интерфейс как у GoExplore: observe(frame) + decide(frame, simple_actions, has_click)."""
from __future__ import annotations

import random

import numpy as np
import torch

from .perception import latest_grid, spread_click_targets

ID2NAME = {1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5"}


class CNNPolicy:
    def __init__(self, model_path: str, seed: int = 0, temperature: float = 1.0):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts"))
        from imitation_train import make_model
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = make_model(ck.get("arch", "cnn")); self.model.load_state_dict(ck["state"]); self.model.eval()
        self.rng = random.Random(seed); self.temperature = temperature
        self.stats = {"steps": 0, "policy_moves": 0, "clicks": 0, "resets": 0}
        self.last_hash = None; self.same_count = 0
        import types
        self.st = types.SimpleNamespace(need_reset=False, last_action=None)

    def observe(self, frame) -> None:
        pass

    def _emit(self, act):
        self.st.last_action = act
        return act

    def decide(self, frame, simple_actions, has_click):
        self.stats["steps"] += 1
        grid = latest_grid(frame)
        if grid is None:
            return self._emit(("RESET", None))
        h = hash(grid.tobytes())
        self.same_count = self.same_count + 1 if h == self.last_hash else 0
        self.last_hash = h
        allowed = [a for a in simple_actions if a in ID2NAME.values()]
        if (allowed or has_click) and self.same_count < 6:
            with torch.no_grad():
                logits, cmap = self.model(torch.from_numpy(grid.astype(np.int64))[None], with_click=True)
                logits = logits[0]; cmap = cmap[0]
            mask = torch.full_like(logits, float("-inf"))
            for a in allowed:
                mask[int(a[-1])] = 0.0
            if has_click:
                mask[6] = 0.0
            probs = torch.softmax((logits + mask) / self.temperature, dim=0).numpy()
            aid = int(np.random.choice(len(probs), p=probs / probs.sum()))
            self.stats["policy_moves"] += 1
            if aid == 6:
                cp = torch.softmax(cmap / self.temperature, dim=0).numpy()
                cell = int(np.random.choice(4096, p=cp / cp.sum())); self.stats["clicks"] += 1
                return self._emit(("ACTION6", {"x": cell % 64, "y": cell // 64}))
            return self._emit((ID2NAME[aid], None))
        if has_click:
            t = spread_click_targets(grid, 24)
            if t:
                x, y = self.rng.choice(t); self.stats["clicks"] += 1
                return self._emit(("ACTION6", {"x": x, "y": y}))
        if allowed:
            return self._emit((self.rng.choice(allowed), None))
        return self._emit(("RESET", None))

    def summary(self):
        return dict(self.stats)

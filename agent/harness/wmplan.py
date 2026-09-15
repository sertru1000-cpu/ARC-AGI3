"""«Модель думает, алгоритм ходит» (14.09, слово владельца «придумывай и реализовывай, без вариантов»).

Данные: в tn36 модель написала ПОЛНУЮ модель механики (200/200 контрфактических кликов верно), но уровней 0.
Значит, её надо оставить на том, что она умеет — писать state_of(grid)/predict(state, action) — а ходить
должен алгоритм. Планировщик покрывает пространство АБСТРАКТНЫХ состояний (тех, что различает программа модели):

  1. abstract(grid) = state_of(grid) без «часов» — компонент, меняющихся почти на каждом ходе (счётчики ходов,
     таймеры); они раздувают пространство, но не несут смысла.
  2. Фронтир: (abstract, путь от старта уровня). Из текущего реального состояния для каждого действия алфавита
     (стрелки/SPACE + клики по центрам объектов) predict даёт следующее абстрактное состояние; если оно новое —
     кандидат. Если predict вернул None (модель не знает) — тоже кандидат, но с меньшим приоритетом
     (неизвестное — это и есть то, что стоит проверить).
  3. Исполнение: кандидат с кратчайшим путём исполняется в настоящей среде (если мы не в его родителе —
     RESET + повтор пути). Реальное следующее состояние записывается; расхождение с predict — контрпример
     (счётчик для будущей починки программы моделью, CEGIS).
  4. Взятие уровня — цель; на новом уровне фронтир строится заново с той же программой.

Интерфейс как у GoExplore: observe(frame) + decide(frame, simple_actions, has_click) -> (name, payload).
Программа подаётся снаружи (ProgramProvider): для проверки алгоритма — готовые программы модели из
docs/wm_offline_flash_s1*.json; в бою — синтез моделью по собранным переходам.
"""
from __future__ import annotations

import collections
import heapq
import random
from dataclasses import dataclass, field

import numpy as np

from .perception import frame_hash, latest_grid, segment, spread_click_targets

MODEL_ACTION = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT", "ACTION5": "SPACE"}


def _norm(x):
    if isinstance(x, dict):
        return tuple(sorted((str(k), _norm(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return tuple(_norm(v) for v in x)
    if isinstance(x, set):
        return tuple(sorted(_norm(v) for v in x))
    return x


def action_display(act: tuple[str, dict | None]) -> str:
    name, payload = act
    if name == "ACTION6" and payload:
        return "MOUSE(row=%d, col=%d)" % (int(payload["y"]), int(payload["x"]))
    return MODEL_ACTION.get(name, name)


class Program:
    """Скомпилированные state_of/predict модели + определение «часов»."""

    def __init__(self, source: str, letter_map: dict[int, str]):
        ns: dict = {}
        exec(source, ns)
        self.state_of = ns["state_of"]; self.predict = ns["predict"]
        self.progress = ns.get("progress")   # необязательная: progress(state) -> число, больше = ближе к взятию уровня
        self.is_goal = ns.get("is_goal")     # необязательная: is_goal(state) -> bool, состояние завершает уровень
        self.letters = letter_map
        self.clock_idx: set[int] = set()
        self.frozen = False
        self._seen_pairs = 0
        self._changes: collections.Counter = collections.Counter()
        self._alone: collections.Counter = collections.Counter()
        self._values: dict = {}
        self._mono: dict = {}

    def rows(self, grid: np.ndarray) -> list[str]:
        return ["".join(self.letters.get(int(v), "?") for v in row) for row in grid]

    def raw_state(self, grid: np.ndarray):
        try:
            return _norm(self.state_of(self.rows(grid)))
        except Exception:
            return None

    def note_transition(self, s_before, s_after) -> None:
        """Учёт «часов» -- компонент верхнего уровня, которые (а) меняются почти на каждом ходе (tn36: счётчик
        кликов) или (б) никогда не меняются сами по себе, а только вместе с другими, и принимают много значений
        (ft09: счётчик, убывающий при каждом переключении блока) -- избыточны для поиска."""
        if not (isinstance(s_before, tuple) and isinstance(s_after, tuple) and len(s_before) == len(s_after)):
            return
        if s_before == s_after:
            return
        self._seen_pairs += 1
        changed = [i for i, (a, b) in enumerate(zip(s_before, s_after)) if a != b]
        for i in changed:
            self._changes[i] += 1
            self._values.setdefault(i, set()).add(s_after[i])
            if len(changed) == 1:
                self._alone[i] += 1
            if isinstance(s_before[i], (int, float)) and isinstance(s_after[i], (int, float)) and not isinstance(s_after[i], bool):
                d = s_after[i] - s_before[i]
                self._mono.setdefault(i, set()).add(1 if d > 0 else -1)
        if self._seen_pairs >= 4 and (not self.frozen or self._seen_pairs in (4, 8, 16, 32, 64, 128)):
            clocks = set()
            for i, n in self._changes.items():
                # часы = ЧИСЛОВОЙ МОНОТОННЫЙ компонент с >= 3 значениями, меняющийся в >= 80% переходов
                # (tn36: счётчик кликов; ft09: счётчик, убывающий при каждом переключении). Нечисловые компоненты
                # (цвета блоков, позиции) часами не бывают, даже если меняются каждый ход.
                if (len(self._mono.get(i, ())) == 1 and len(self._values.get(i, ())) >= 3 and n >= 0.8 * self._seen_pairs):
                    clocks.add(i)
            if clocks and len(clocks) >= len(s_after):
                clocks = set()      # не выбрасывать всё
            self.clock_idx = clocks

    def abstract(self, s):
        if s is None:
            return None
        if isinstance(s, tuple) and self.clock_idx:
            return tuple(v for i, v in enumerate(s) if i not in self.clock_idx)
        return s

    def progress_of(self, s_raw) -> float:
        if self.progress is None or s_raw is None:
            return 0.0
        try:
            return float(self.progress(s_raw))
        except Exception:
            return 0.0

    def goal_of(self, s_raw) -> bool:
        if self.is_goal is None or s_raw is None:
            return False
        try:
            return bool(self.is_goal(s_raw))
        except Exception:
            return False

    def predict_raw(self, s_raw, act):
        try:
            p = self.predict(s_raw, action_display(act))
        except Exception:
            return None
        return None if p is None else _norm(p)

    def predict_abstract(self, s_raw, act):
        try:
            p = self.predict(s_raw if not isinstance(s_raw, tuple) else s_raw, action_display(act))
        except Exception:
            return "ERR"
        return None if p is None else self.abstract(_norm(p))


@dataclass
class Node:
    abstract: object
    raw: object
    exact: str
    path: list[tuple[str, dict | None]]
    expanded: bool = False
    board: str = ""       # текст доски (hex) для дайджеста стратегу
    imagined: bool = False


@dataclass
class PlanState:
    level: int = 0
    nodes: dict = field(default_factory=dict)          # abstract -> Node
    heap: list = field(default_factory=list)           # (priority, seq, parent_abstract, action)
    seq: int = 0
    queue: list = field(default_factory=list)          # actions to execute verbatim
    pending: tuple | None = None                       # (parent_abstract, action, predicted_abstract)
    cur_raw: object = None
    cur_abstract: object = None
    cur_exact: str | None = None
    cur_path: list = field(default_factory=list)
    clocks_seen: set = field(default_factory=set)
    need_reset: bool = True
    last_action: tuple | None = None
    stats: collections.Counter = field(default_factory=collections.Counter)


class WMPlanner:
    MAX_CLICK_TARGETS = 40
    WARMUP = 10   # случайных ходов на определение «часов» до начала планирования; после — часы заморожены

    def __init__(self, program: Program, seed: int = 0):
        self.prog = program
        self.rng = random.Random(seed)
        self.st = PlanState()
        self.hybrid = False   # ключ узла = (абстракция, точная доска), когда абстракция склеивает разные доски

    # ---------------------------------------------------------------- observe
    def observe(self, frame) -> None:
        st = self.st
        grid = latest_grid(frame)
        if grid is None:
            return
        level = int(getattr(frame, "levels_completed", 0) or 0)
        state = str(getattr(frame, "state", ""))
        if level != st.level:
            st.level = level
            st.nodes.clear(); st.heap.clear(); st.queue.clear(); st.pending = None; st.cur_path = []
            st.stats["levels"] += 1
        if state.endswith("GAME_OVER"):
            st.stats["game_overs"] += 1
            st.need_reset = True; st.queue.clear(); st.pending = None
            return
        raw = self.prog.raw_state(grid)
        if st.last_action is not None and st.last_action[0] == "RESET":
            st.cur_path = []
        elif st.last_action is not None and st.cur_raw is not None and raw is not None:
            self.prog.note_transition(st.cur_raw, raw)
        abstract = self.prog.abstract(raw)
        exact = frame_hash(grid)
        if (st.last_action is not None and st.last_action[0] != "RESET" and st.cur_exact is not None
                and exact != st.cur_exact and abstract == st.cur_abstract):
            st.stats["state_unchanged"] += 1
            if st.stats["state_unchanged"] >= 3 and not self.hybrid:
                # абстракция модели не различает изменившиеся доски: ключ = (абстракция, точная доска)
                self.hybrid = True
                st.stats["hybrid_at"] = st.stats["steps"]
                st.nodes = {(n.abstract if isinstance(n.abstract, tuple) and len(n.abstract) == 2 and isinstance(n.abstract[1], str) else (n.abstract, n.exact)): n for n in st.nodes.values()}
                for k, n in st.nodes.items():
                    n.abstract = k; n.expanded = False
                st.heap.clear()
        if self.hybrid:
            abstract = (abstract, exact)
        st.cur_raw, st.cur_abstract, st.cur_exact = raw, abstract, exact
        if st.pending is not None and not st.queue:
            parent, act, predicted = st.pending
            st.pending = None
            st.stats["executed"] += 1
            if predicted == "ERR" or predicted is None:
                st.stats["unknown_tested"] += 1
            elif predicted == abstract:
                st.stats["pred_ok"] += 1
            else:
                st.stats["pred_wrong"] += 1
        if abstract not in st.nodes:
            st.nodes[abstract] = Node(abstract=abstract, raw=raw, exact=exact, path=list(st.cur_path),
                                      board="\n".join("".join(self.prog.letters.get(int(v), "?") for v in row) for row in grid))
            st.stats["new_states"] += 1
        elif len(st.cur_path) < len(st.nodes[abstract].path):
            st.nodes[abstract].path = list(st.cur_path); st.nodes[abstract].exact = exact; st.nodes[abstract].raw = raw

    # ----------------------------------------------------------------- decide
    IMAGINE_DEPTH = 8
    IMAGINE_NODES = 3000

    def _imagine(self, node: Node, frame, simple_actions, has_click):
        """BFS в модели от узла: предсказанные состояния (predict != None). Возвращает путь к лучшему состоянию,
        которого ещё нет в реальном графе: цель (is_goal) > прогресс > глубина. Пустой список, если нечего."""
        grid = latest_grid(frame)
        alphabet: list = [(a, None) for a in simple_actions]
        if has_click and grid is not None:
            alphabet += [act for act, _ in self._click_alphabet(node, grid)]
        start_key = self.prog.abstract(node.raw)
        seen = {start_key: (node.raw, [], 0)}
        frontier = [(node.raw, [], 0)]
        best = None
        while frontier and len(seen) < self.IMAGINE_NODES:
            nxt = []
            for raw, path, depth in frontier:
                if depth >= self.IMAGINE_DEPTH:
                    continue
                for act in alphabet:
                    praw = self.prog.predict_raw(raw, act)
                    if praw is None:
                        continue
                    key = self.prog.abstract(praw)
                    if key in seen or key == self.prog.abstract(raw):
                        continue
                    seen[key] = (praw, path + [act], depth + 1)
                    nxt.append((praw, path + [act], depth + 1))
                    real_known = key in st_nodes if (st_nodes := self.st.nodes) is not None else False
                    if real_known:
                        continue
                    cand = (1 if self.prog.goal_of(praw) else 0, self.prog.progress_of(praw), -(depth + 1))
                    if best is None or cand > best[0]:
                        best = (cand, path + [act], key)
                    if cand[0] == 1:
                        self.st.stats["goal_imagined"] += 1
                        return path + [act], key
            frontier = nxt
        self.st.stats["imagined_states"] = max(self.st.stats["imagined_states"], len(seen))
        return (best[1], best[2]) if best else ([], None)

    def decide(self, frame, simple_actions: list[str], has_click: bool) -> tuple[str, dict | None]:
        st = self.st
        st.stats["steps"] += 1
        if st.need_reset:
            st.need_reset = False
            return self._emit(("RESET", None))
        if st.queue:
            return self._emit(st.queue.pop(0))
        # воображение: план из модели к лучшему невиданному состоянию, исполняется с проверкой каждого шага
        node = st.nodes.get(st.cur_abstract)
        if node is not None and not self.hybrid and self.prog.frozen and not getattr(node, "imagined", False):
            node.imagined = True
            path, key = self._imagine(node, frame, simple_actions, has_click)
            if path:
                st.stats["imagined_plans"] += 1
                st.pending = (node.abstract, path[-1], key)
                st.queue = list(path)
                return self._emit(st.queue.pop(0))
        if not self.prog.frozen:
            if st.stats["steps"] <= self.WARMUP:
                return self._emit(self._warmup_action(frame, simple_actions, has_click))
            self.prog.frozen = True
            self._rekey(); st.stats["clocks_frozen_at"] = st.stats["steps"]
        elif self.prog.clock_idx != st.clocks_seen:
            # часы переопределились по новым переходам (Program пересчитывает их до заморозки; после -- по
            # накопленным парам при вызове refresh) -- перестроить ключи
            self._rekey()
        # расширить текущий узел, если ещё не расширен
        node = st.nodes.get(st.cur_abstract)
        if node is not None and not node.expanded:
            node.expanded = True
            self._expand(node, frame, simple_actions, has_click)
        # взять лучший кандидат
        while st.heap:
            prio, _, parent_abs, act, predicted = heapq.heappop(st.heap)
            if predicted not in (None, "ERR") and predicted in st.nodes:
                continue   # уже достигнуто
            parent = st.nodes.get(parent_abs)
            if parent is None:
                continue
            st.pending = (parent_abs, act, predicted)
            if parent.exact == st.cur_exact:
                return self._emit(act)
            st.queue = [("RESET", None)] + list(parent.path) + [act]
            st.stats["resets"] += 1
            return self._emit(st.queue.pop(0))
        # фронтир пуст: случайное действие (шум для выхода из тупика)
        st.stats["frontier_empty"] += 1
        return self._emit(self._random_action(frame, simple_actions, has_click))

    def _click_alphabet(self, node: Node, grid) -> list[tuple[tuple[str, dict | None], object]]:
        """Алфавит кликов из знаний модели: predict пробуется на КАЖДОЙ клетке доски (чистый Python, дёшево),
        клетки группируются по предсказанному абстрактному состоянию, из группы берётся один представитель.
        Группа «модель не знает» (None) представлена центрами объектов сегментации."""
        groups: dict = {}
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                pred = self.prog.predict_abstract(node.raw, ("ACTION6", {"x": c, "y": r}))
                if pred in (None, "ERR"):
                    continue
                if pred not in groups:
                    groups[pred] = ("ACTION6", {"x": c, "y": r})
        out = [(act, pred) for pred, act in groups.items()]
        for x, y in spread_click_targets(grid, self.MAX_CLICK_TARGETS):
            act = ("ACTION6", {"x": x, "y": y})
            pred = self.prog.predict_abstract(node.raw, act)
            if pred in (None, "ERR"):
                out.append((act, pred))
        self.st.stats["click_groups"] = max(self.st.stats["click_groups"], len(groups))
        return out

    def _expand(self, node: Node, frame, simple_actions, has_click) -> None:
        st = self.st
        grid = latest_grid(frame)
        cands: list[tuple[tuple[str, dict | None], object]] = [((a, None), self.prog.predict_abstract(node.raw, (a, None))) for a in simple_actions]
        if has_click and grid is not None:
            cands += self._click_alphabet(node, grid)
        depth = len(node.path)
        base_abs = node.abstract[0] if self.hybrid else node.abstract
        # группы кандидатов: 0 -- модель предсказывает НОВОЕ состояние; 1 -- модель не знает; 2 -- модель говорит
        # «ничего не изменится» (ft09, 14.09: завершающий уровень клик был предсказан как noop и отсечён -- поэтому
        # noop-кандидаты не отсекаются, а идут последними, по одному на клетку-цель); 3 -- предсказано уже виденное
        noop_clicks = set()
        if grid is not None:
            for x, y in spread_click_targets(grid, self.MAX_CLICK_TARGETS):
                noop_clicks.add(("ACTION6", x, y))
        seen_click_reps = set()
        for act, predicted in cands:
            if act[0] == "ACTION6" and act[1] is not None:
                seen_click_reps.add(("ACTION6", act[1]["x"], act[1]["y"]))
            if self.hybrid:
                group = 0 if predicted not in (None, "ERR") and predicted != base_abs else 1
            elif predicted == node.abstract:
                group = 2; st.stats["noop_deferred"] += 1
            elif predicted not in (None, "ERR") and predicted in st.nodes:
                group = 3; st.stats["known_deferred"] += 1
            elif predicted in (None, "ERR"):
                group = 1
            else:
                group = 0
            praw = self.prog.predict_raw(node.raw, act) if (group == 0 and not self.hybrid) else None
            prog_val = self.prog.progress_of(praw if praw is not None else node.raw)
            prio = (group, -prog_val, depth + 1, self.rng.random())
            st.seq += 1
            heapq.heappush(st.heap, (prio, st.seq, node.abstract, act, None if self.hybrid or group >= 2 else predicted))
        if not self.hybrid and grid is not None:
            # представители noop-группы: все цели-клики, которых нет среди кандидатов других групп
            for key in noop_clicks:
                if key in seen_click_reps:
                    continue
                act = ("ACTION6", {"x": key[1], "y": key[2]})
                st.seq += 1
                heapq.heappush(st.heap, ((2, -self.prog.progress_of(node.raw), depth + 1, self.rng.random()), st.seq, node.abstract, act, None))

    def _random_action(self, frame, simple_actions, has_click):
        grid = latest_grid(frame)
        opts = [(a, None) for a in simple_actions]
        if has_click and grid is not None:
            opts += [("ACTION6", {"x": x, "y": y}) for x, y in spread_click_targets(grid, self.MAX_CLICK_TARGETS)]
        return self.rng.choice(opts) if opts else ("RESET", None)

    def _rekey(self) -> None:
        st = self.st
        rebuilt: dict = {}
        for n in st.nodes.values():
            key = self.prog.abstract(n.raw)
            if self.hybrid:
                key = (key, n.exact)
            if key not in rebuilt or len(n.path) < len(rebuilt[key].path):
                n.abstract = key; n.expanded = False; n.imagined = False; rebuilt[key] = n
        st.nodes = rebuilt; st.heap.clear()
        st.cur_abstract = self.prog.abstract(st.cur_raw)
        if self.hybrid:
            st.cur_abstract = (st.cur_abstract, st.cur_exact)
        st.clocks_seen = set(self.prog.clock_idx)
        st.stats["rekeys"] += 1

    def _warmup_action(self, frame, simple_actions, has_click):
        """Разминка: сначала клики по разнообразным целям (малые объекты), затем стрелки -- чтобы часы проявились."""
        grid = latest_grid(frame)
        i = self.st.stats["steps"] - 1
        targets = spread_click_targets(grid, self.WARMUP) if (has_click and grid is not None) else []
        opts = [("ACTION6", {"x": x, "y": y}) for x, y in targets] + [(a, None) for a in simple_actions]
        if not opts:
            return ("RESET", None)
        return opts[i % len(opts)]

    def _emit(self, act):
        st = self.st
        st.last_action = act
        st.cur_path = [] if act[0] == "RESET" else st.cur_path + [act]
        return act

    def set_goal_functions(self, progress=None, is_goal=None) -> None:
        """Стратег дал progress/is_goal: заменить в программе и перестроить фронтир."""
        if progress is not None:
            self.prog.progress = progress
        if is_goal is not None:
            self.prog.is_goal = is_goal
        for n in self.st.nodes.values():
            n.expanded = False; n.imagined = False
        self.st.heap.clear(); self.st.stats["goal_updates"] += 1

    def suggest_moves(self, moves: list[tuple[str, dict | None]]) -> None:
        """Стратег подсказал конкретные ходы: исполнить их первыми из текущего состояния."""
        self.st.queue = list(moves) + list(self.st.queue)
        self.st.pending = None; self.st.stats["suggested"] += len(moves)

    def digest(self, max_states: int = 4) -> dict:
        """Дайджест для стратега: стартовая доска, характерные достигнутые состояния, число состояний."""
        nodes = list(self.st.nodes.values())
        start = next((n for n in nodes if not n.path), None) or (nodes[0] if nodes else None)
        picks = []
        if nodes:
            by_prog = sorted(nodes, key=lambda n: (-self.prog.progress_of(n.raw), -len(n.path)))
            longest = max(nodes, key=lambda n: len(n.path))
            for n in [by_prog[0], longest] + by_prog[1:max_states]:
                if n is not start and n not in picks and len(picks) < max_states:
                    picks.append(n)
        return {"n_states": len(nodes), "start": start, "picks": picks, "stats": dict(self.st.stats)}

    def summary(self) -> dict:
        d = dict(self.st.stats)
        d["states"] = len(self.st.nodes); d["clocks"] = sorted(self.prog.clock_idx)
        return d


# ---------------------------------------------------------------- поставщик программ для локальной проверки
def load_recorded_program(game4: str, json_paths: list[str], events_glob: str) -> Program | None:
    """Лучшая программа модели для игры из записей оффлайн-теста + карта цветов из журнала событий."""
    import glob, json
    best = None
    for f in json_paths:
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for x in data:
            if x.get("game") == game4 and x.get("code") and x.get("ok") is not None:
                if best is None or x["ok"] > best["ok"]:
                    best = x
    if best is None:
        return None
    letters: dict[int, str] = {}
    for p in glob.glob(events_glob.replace("GAME", game4)):
        for line in open(p, encoding="utf-8"):
            e = json.loads(line)
            if e.get("board") and e.get("board_ascii"):
                rows = e["board_ascii"].split("\n")
                for r, row in enumerate(e["board"]):
                    for c, v in enumerate(row):
                        if r < len(rows) and c < len(rows[r]):
                            letters[int(v)] = rows[r][c]
                if len(letters) >= 12:
                    break
        break
    return Program(best["code"], letters)
